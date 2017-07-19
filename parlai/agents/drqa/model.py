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
import math

from torch.autograd import Variable
from .utils import load_embeddings, AverageMeter
from .rnn_reader import RnnDocReader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

logger = logging.getLogger('DrQA')


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
            inputs = [Variable(e.cuda(async=True)) for e in ex[:5]]
            target_s = Variable(ex[5].cuda(async=True))
            target_e = Variable(ex[6].cuda(async=True))
        else:
            inputs = [Variable(e) for e in ex[:5]]
            target_s = Variable(ex[5])
            target_e = Variable(ex[6])

        # Run forward
        score_s, score_e = self.network(*inputs)

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
                      for e in ex[:5]]
        else:
            inputs = [Variable(e, volatile=True) for e in ex[:5]]

        # Run forward
        score_s, score_e, atten_softmax, q_merge_weights = self.network(
            *inputs)

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

            if self.opt['visualize_attention']:
                # visualize question-aware document attention.
                qticks = ex[-3][i]
                dticks = ex[-4][i]
                n_plots = math.ceil(len(dticks) / 60.0)
                fig = plt.figure(figsize=(15, 5 * (n_plots * 3 + 1)))
                gs = gridspec.GridSpec(n_plots * 3 + 1, 1)

                atten_data = atten_softmax[i].data.tolist()
                max_v = atten_softmax[i].data.max()
                min_v = atten_softmax[i].data.min()
                dticks.extend(['' for i in range(n_plots * 60 - len(dticks))])
                for pi in range(n_plots):
                    # plt.subplot(n_plots, 1, i+1)
                    fig.add_subplot(gs[pi])
                    sns.heatmap(
                        [j[pi * 60:(pi + 1) * 60] for j in atten_data],
                        cmap=sns.cubehelix_palette(64),
                        vmax=max_v,
                        vmin=min_v,
                        xticklabels=dticks[pi * 60:(pi + 1) * 60],
                        yticklabels=qticks)
                    plt.xticks(rotation=85)
                    plt.yticks(rotation=0)

                start_data = score_s[i].unsqueeze(0).tolist()
                # visualize start position softmax weight
                max_v = score_s[i].max()
                min_v = score_s[i].min()
                for si in range(n_plots):
                    fig.add_subplot(gs[n_plots + si])
                    sns.heatmap(
                        [j[si * 60: (si + 1) * 60] for j in start_data],
                        cmap=sns.cubehelix_palette(64),
                        vmax=max_v,
                        vmin=min_v,
                        xticklabels=dticks[si * 60:(si + 1) * 60],
                        yticklabels=['weight'])
                    plt.xticks(rotation=85)

                end_data = score_e[i].unsqueeze(0).tolist()
                # visualize end position softmax weight
                max_v = score_e[i].max()
                min_v = score_e[i].min()
                for ei in range(n_plots):
                    fig.add_subplot(gs[n_plots * 2 + ei])
                    sns.heatmap(
                        [j[ei * 60: (ei + 1) * 60] for j in end_data],
                        cmap=sns.cubehelix_palette(64),
                        vmax=max_v,
                        vmin=min_v,
                        xticklabels=dticks[ei * 60:(ei + 1) * 60],
                        yticklabels=['weight'])
                    plt.xticks(rotation=85)

                fig.add_subplot(gs[n_plots * 3])
                # visualize question merge weight
                sns.heatmap(
                    q_merge_weights[i].unsqueeze(0).data.tolist(),
                    cmap=sns.cubehelix_palette(64),
                    xticklabels=qticks,
                    yticklabels=['weight'])
                plt.xticks(rotation=85)

                gs.update(wspace=.5, hspace=0.7)
                plt.show()
                plt.clf()

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
