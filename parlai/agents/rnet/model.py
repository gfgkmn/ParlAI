# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging

from torch.autograd import Variable
from .utils import load_embeddings, AverageMeter
from .rnn_reader import RnnDocReader

logger = logging.getLogger('RNET')


class DocReaderModel(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    def __init__(self, opt, word_dict, feature_dict, state_dict=None):
        # Book-keeping.
        self.opt = opt
        self.word_dict = word_dict
        self.feature_dict = feature_dict
        self.updates = 0
        self.train_loss = AverageMeter()

        # Building network.
        self.network = RnnDocReader(opt)
        if opt['multi_gpu']:
            self.gpu_device_id = [int(i) for i in opt['multi_gpu']]
        if state_dict:
            new_state = set(self.network.state_dict().keys())
            for k in list(state_dict['network'].keys()):
                if k not in new_state:
                    del state_dict['network'][k]
            self.network.load_state_dict(state_dict['network'])
            # what's state_dict, netword's weight?

        # Building optimizer.
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if opt['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, opt['learning_rate'],
                                       momentum=opt['momentum'],
                                       weight_decay=opt['weight_decay'])
        elif opt['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=opt['weight_decay'])
        elif opt['optimizer'] == 'adadelta':
            self.optimizer = optim.Adadelta(
                parameters,
                opt['learning_rate'],
                rho=opt['rho'],
                eps=opt['eps'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' % opt['optimizer'])

    def set_embeddings(self):
        # Read word embeddings.
        if not self.opt.get('embedding_file'):
            logger.warning('[ WARNING: No embeddings provided. '
                           'Keeping random initialization. ]')
            return
        logger.info('[ Loading pre-trained embeddings ]')
        embeddings = load_embeddings(self.opt, self.word_dict)
        logger.info('[ Num embeddings = %d ]' % embeddings.size(0))

        # Sanity check dimensions
        new_size = embeddings.size()
        old_size = self.network.embedding.weight.size()
        if new_size[1] != old_size[1]:
            raise RuntimeError('Embedding dimensions do not match.')
        if new_size[0] != old_size[0]:
            logger.warning(
                '[ WARNING: Number of embeddings changed (%d->%d) ]' %
                (old_size[0], new_size[0])
            )

        # Swap weights
        self.network.embedding.weight.data = embeddings

        # If partially tuning the embeddings, keep the old values
        if self.opt['tune_partial'] > 0:
            if self.opt['tune_partial'] + 2 < embeddings.size(0):
                fixed_embedding = embeddings[self.opt['tune_partial'] + 2:]
                self.network.fixed_embedding = fixed_embedding

    def update(self, ex):
        # Train mode
        self.network.train()

        # Transfer to GPU
        if self.opt['cuda']:
            inputs = [Variable(e.cuda(async=True)) for e in ex[:9]]
            target_s = Variable(ex[9].cuda(async=True))
            target_e = Variable(ex[10].cuda(async=True))
        else:
            inputs = [Variable(e) for e in ex[:9]]
            target_s = Variable(ex[9])
            target_e = Variable(ex[10])

        # Run forward
        # score_s, score_e = self.network(*inputs)
        net = torch.nn.DataParallel(self.network, device_ids=self.gpu_device_id)
        score_s, score_e = net(*inputs)
        

        # Compute loss and accuracies
        loss = F.nll_loss(score_s, target_s) + F.nll_loss(score_e, target_e)
        self.train_loss.update(loss.data[0], ex[0].size(0))

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm(self.network.parameters(),
                                      self.opt['grad_clipping'])

        # Update parameters
        self.optimizer.step()
        self.updates += 1

        # Reset any partially fixed parameters (e.g. rare words)
        self.reset_parameters()

    def predict(self, ex):
        # Eval mode
        self.network.eval()

        # Transfer to GPU
        if self.opt['cuda']:
            inputs = [Variable(e.cuda(async=True), volatile=True)
                      for e in ex[:9]]
        else:
            inputs = [Variable(e, volatile=True) for e in ex[:9]]

        # Run forward
        # score_s, score_e = self.network(*inputs)
        net = torch.nn.DataParallel(self.network, device_ids=self.gpu_device_id)
        score_s, score_e = net(*inputs)

        # Transfer to CPU/normal tensors for numpy ops
        score_s = score_s.data.cpu()
        score_e = score_e.data.cpu()

        # Get argmax text spans
        text = ex[-2]
        spans = ex[-1]
        predictions = []
        max_len = self.opt['max_len'] or score_s.size(1)
        for i in range(score_s.size(0)):
            scores = torch.ger(score_s[i], score_e[i])
            scores.triu_().tril_(max_len - 1)
            scores = scores.numpy()
            s_idx, e_idx = np.unravel_index(np.argmax(scores), scores.shape)
            s_offset, e_offset = spans[i][s_idx][0], spans[i][e_idx][1]
            predictions.append(text[i][s_offset:e_offset])

        return predictions

    def reset_parameters(self):
        # Reset fixed embeddings to original value
        if self.opt['tune_partial'] > 0:
            offset = self.opt['tune_partial'] + 2
            if offset < self.network.embedding.weight.data.size(0):
                self.network.embedding.weight.data[offset:] \
                    = self.network.fixed_embedding

    def save(self, filename):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
            },
            'word_dict': self.word_dict,
            'feature_dict': self.feature_dict,
            'config': self.opt,
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warn('[ WARN: Saving failed... continuing anyway. ]')

    def cuda(self):
        self.network.cuda()
