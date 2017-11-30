# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""(A partial) implementation of the Bidaf Document Reader from:

Danqi Chen, Adam Fisch, Jason Weston, Antoine Bordes. 2017.
Reading Wikipedia to Answer Open-Domain Questions.
In Association for Computational Linguistics (ACL).

Link: https://arxiv.org/abs/1704.00051

Note:
To use pretrained word embeddings, set the --embeddings_file path argument.
GloVe is recommended, see http://nlp.stanford.edu/data/glove.840B.300d.zip.
"""

try:
    import torch
except ModuleNotFoundError:
    raise ModuleNotFoundError('Need to install pytorch: go to pytorch.org')

import os
import numpy as np
import logging
import pickle
import copy
try:
    import spacy
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Please install spacy and spacy 'en' model: go to spacy.io")

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from . import config
from .utils import build_feature_dict, vectorize, batchify, normalize_text
from .model import DocReaderModel
from collections import defaultdict
import re

# ------------------------------------------------------------------------------
# Dictionary.
# ------------------------------------------------------------------------------

NLP = spacy.load('en_core_web_sm')


class SimpleDictionaryAgent(DictionaryAgent):
    """Override DictionaryAgent to use spaCy tokenizer."""

    @staticmethod
    def add_cmdline_args(argparser):
        group = DictionaryAgent.add_cmdline_args(argparser)
        group.add_argument(
            '--pretrained_words',
            type='bool',
            default=True,
            help='Use only words found in provided embedding_file')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if 'shared' in kwargs:
            self.feature_dict = kwargs['shared'].get('feature_dict', {
                'pos': set(),
                'ner': set()
            })
            self.char_dict = kwargs['shared'].get('char_dict', set())
            self.char2ind = kwargs['shared'].get('char2ind', {})
            self.ind2char = kwargs['shared'].get('ind2char', {})
            self.charfreq = kwargs['shared'].get('charfreq', {})
            self.posfreq = kwargs['shared'].get('posfreq', {})
            self.nerfreq = kwargs['shared'].get('nerfreq', {})
            self.pos2ind = kwargs['shared'].get('pos2ind', {})
            self.ind2pos = kwargs['shared'].get('ind2pos', {})
            self.ner2ind = kwargs['shared'].get('ner2ind', {})
            self.ind2ner = kwargs['shared'].get('ind2ner', {})
        else:
            if not hasattr(self, 'feature_dict'):
                # if feature_dict defined, so every variable will be defined.
                self.feature_dict = {'pos': set(), 'ner': set()}
                self.char_dict = set()
                self.posfreq = defaultdict(int)
                self.nerfreq = defaultdict(int)
                self.charfreq = defaultdict(int)
                self.char2ind = {}
                self.ind2char = {}
                self.pos2ind = {}
                self.ind2pos = {}
                self.ner2ind = {}
                self.ind2ner = {}
            # Index words in embedding file
        if self.opt['pretrained_words'] and self.opt.get('embedding_file'):
            print('[ Indexing words with embeddings... ]')
            self.embedding_words = set()
            with open(self.opt['embedding_file']) as f:
                for line in f:
                    w = normalize_text(line.rstrip().split(' ')[0])
                    self.embedding_words.add(w)
            print('[ Num words in set = %d ]' % len(self.embedding_words))
        else:
            self.embedding_words = None

    def tokenize(self, text, **kwargs):
        text = self.pre_proc(text)
        tokens = NLP(text)
        # return [t.text for t in tokens]
        return tokens

    def pre_proc(self, text):
        '''normalize spaces in a string.'''
        text = re.sub('\s+', ' ', text)
        return text

    def span_tokenize(self, text):
        """
        self.dict_misc.span_tokenize('what if i do')
        [(0, 4), (5, 7), (8, 9), (10, 12)]
        """
        tokens = NLP.tokenizer(text)
        return [(t.idx, t.idx + len(t.text)) for t in tokens]

    def load(self, filename):
        """Load pre-existing dictionary in 'token[<TAB>count]' format.
        Initialize counts from other dictionary, or 0 if they aren't included.
        """
        print(
            'Dictionary: loading existing dictionary from {}'.format(filename))
        dics = pickle.load(open(filename, 'rb'))
        self.tok2ind, self.ind2tok, self.feature_dict, \
            self.posfreq, self.pos2ind, self.ind2pos, \
            self.nerfreq, self.ner2ind, self.ind2ner, self.char_dict,\
            self.char2ind, self.ind2char, self.charfreq = dics
        print('[ num words =  %d ]' % len(self))

    def save(self, filename=None, append=False, sort=True):
        """Save dictionary to file.  Format is 'token<TAB>count' for every
        token in the dictionary, sorted by count with the most frequent words
        first.

        If ``append`` (default ``False``) is set to ``True``, appends instead
        of overwriting.

        If ``sort`` (default ``True``), then first sort the dictionary before
        saving.
        """
        filename = self.opt['model_file'] if filename is None else filename
        print('Dictionary: saving dictionary to {}'.format(filename))
        if sort:
            self.sort()
        pickle.dump((self.tok2ind, self.ind2tok, self.feature_dict,
                     self.posfreq, self.pos2ind, self.ind2pos, self.nerfreq,
                     self.ner2ind, self.ind2ner, self.char_dict, self.char2ind,
                     self.ind2char, self.charfreq), open(filename, 'wb'))

    def add_to_dict(self, tokens):
        """Builds dictionary from the list of provided tokens.
        Only adds words contained in self.embedding_words, if not None.
        So only take those word in embeddings
        """
        for token in tokens:
            if (self.embedding_words is not None
                    and token not in self.embedding_words):
                continue
            self.freq[token] += 1
            # in DictionaryAgent init as defaultdict
            if token not in self.tok2ind:
                index = len(self.tok2ind)
                self.tok2ind[token] = index
                self.ind2tok[index] = token

    def add_to_pos(self, features):
        for key in features:
            self.posfreq[key] += 1
            if key not in self.feature_dict['pos']:
                index = len(self.pos2ind)
                self.pos2ind[key] = index
                self.ind2pos[index] = key
                self.feature_dict['pos'].add(key)

    def add_to_char(self, chars):
        for char in chars:
            self.charfreq[char] += 1
            # in DictionaryAgent init as defaultdict
            if char not in self.char2ind:
                index = len(self.char2ind)
                self.char2ind[char] = index
                self.ind2char[index] = char

    def add_to_ner(self, features):
        for key in features:
            self.nerfreq[key] += 1
            if key not in self.feature_dict['ner']:
                index = len(self.ner2ind)
                self.ner2ind[key] = index
                self.ind2ner[index] = key
                self.feature_dict['ner'].add(key)

    def act(self):
        """add word to dictionary and add pos and ner to pos and ner dict"""
        for source in ([self.observation.get('text')],
                       self.observation.get('labels')):
            if source:
                for text in source:
                    if text:
                        # self.add_to_dict(self.tokenize(text))
                        sentence = NLP(text)
                        self.add_to_dict([t.text for t in sentence])
                        self.add_to_pos([t.pos_ for t in sentence])
                        self.add_to_ner([t.ent_type_ for t in sentence])
                        self.add_to_char([i for t in sentence for i in t.text])
        return {'id': 'Dictionary'}

    def shared(self):
        shared_dict = super().share()
        shared_dict['feature_dict'] = self.feature_dict
        shared_dict['posfreq'] = self.posfreq
        shared_dict['nerfreq'] = self.nerfreq
        shared_dict['pos2ind'] = self.pos2ind
        shared_dict['ind2pos'] = self.ind2pos
        shared_dict['ner2ind'] = self.ner2ind
        shared_dict['ind2ner'] = self.ind2ner
        shared_dict['char_dict'] = self.char_dict
        shared_dict['charfreq'] = self.charfreq
        shared_dict['char2ind'] = self.char2ind
        shared_dict['ind2char'] = self.ind2char
        return shared_dict


# ------------------------------------------------------------------------------
# Document Reader.
# ------------------------------------------------------------------------------


class BidafAgent(Agent):
    @staticmethod
    def add_cmdline_args(argparser):
        config.add_cmdline_args(argparser)
        BidafAgent.dictionary_class().add_cmdline_args(argparser)
        # dictionary_class is SimpleDictionaryAgent so support other
        # diction_class
        # so every model have three part of config. for task config.
        # for model config, and for data config. so you split it.
        # and then you add these config to argparser.

    @staticmethod
    def dictionary_class():
        return SimpleDictionaryAgent

    def __init__(self, opt, shared=None):
        if opt['numthreads'] > 1:
            raise RuntimeError("numthreads > 1 not supported for this model.")

        # Load dict.
        if not shared:
            dict_misc = BidafAgent.dictionary_class()(opt)
        # All agents keep track of the episode (for multiple questions)
        self.episode_done = True

        # Only create an empty dummy class when sharing
        # used when batchsize > 1 when used in batchworld
        if shared is not None:
            self.is_shared = True
            return

        # Set up params/logging/dicts
        self.is_shared = False
        self.id = self.__class__.__name__
        self.dict_misc = dict_misc
        self.opt = copy.deepcopy(opt)
        config.set_defaults(self.opt)

        if self.opt.get('model_file') and os.path.isfile(opt['model_file']):
            self._init_from_saved(opt['model_file'])
        else:
            if self.opt.get('pretrained_model'):
                self._init_from_saved(opt['pretrained_model'])
            else:
                self._init_from_scratch()
        self.opt[
            'cuda'] = not self.opt['no_cuda'] and torch.cuda.is_available()
        if self.opt['cuda']:
            print('[ Using CUDA (GPU %d) ]' % opt['gpu'])
            torch.cuda.set_device(opt['gpu'])
            self.model.cuda()
            # model.cuda, network.cuda? what this mean?
        self.n_examples = 0

    def _init_from_scratch(self):
        # self.feature_dict = build_feature_dict(self.opt,
        #                                        self.dict_misc.feature_dict)
        self.feature_dict = build_feature_dict(self.opt)
        self.opt['num_features'] = len(self.feature_dict)
        self.opt['pos_size'] = len(self.dict_misc.feature_dict['pos'])
        self.opt['ner_size'] = len(self.dict_misc.feature_dict['ner'])
        self.opt['vocab_size'] = len(self.dict_misc)
        self.opt['char_size'] = len(self.dict_misc.char2ind)

        print('[ Initializing model from scratch ]')
        self.model = DocReaderModel(self.opt, self.dict_misc,
                                    self.feature_dict)
        self.model.set_embeddings()

    def _init_from_saved(self, fname):
        print('[ Loading model %s ]' % fname)
        saved_params = torch.load(
            fname, map_location=lambda storage, loc: storage)

        # TODO expand dict and embeddings for new data
        self.dict_misc = saved_params['word_dict']
        self.feature_dict = saved_params['feature_dict']
        self.state_dict = saved_params['state_dict']
        config.override_args(self.opt, saved_params['config'])
        self.model = DocReaderModel(self.opt, self.dict_misc,
                                    self.feature_dict, self.state_dict)

    def observe(self, observation):
        # shallow copy observation (deep copy can be expensive)
        observation = observation.copy()
        if not self.episode_done:
            dialogue = self.observation['text'].split('\n')[:-1]
            dialogue.extend(observation['text'].split('\n'))
            observation['text'] = '\n'.join(dialogue)
        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation

    def act(self):
        """Update or predict on a single example (batchsize = 1)."""
        if self.is_shared:
            raise RuntimeError("Parallel act is not supported.")

        reply = {'id': self.getID()}

        ex = self._build_ex(self.observation)
        if ex is None:
            return reply
        batch = batchify(
            [ex],
            null=self.dict_misc[self.dict_misc.null_token],
            cuda=self.opt['cuda'])

        # Either train or predict
        if 'labels' in self.observation:
            self.n_examples += 1
            self.model.update(batch)
        else:
            reply['text'] = self.model.predict(batch)[0]

        return reply

    def batch_act(self, observations):
        """Update or predict on a batch of examples.
        More efficient than act().
        """
        if self.is_shared:
            raise RuntimeError("Parallel act is not supported.")

        batchsize = len(observations)
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        # Some examples will be None (no answer found). Filter them.
        examples = [self._build_ex(obs) for obs in observations]
        valid_inds = [i for i in range(batchsize) if examples[i] is not None]
        # examples[i] is None, when in example not text fiele(document), or
        # target field(answer), or epoch done
        examples = [ex for ex in examples if ex is not None]

        # If all examples are invalid, return an empty batch.
        if len(examples) == 0:
            return batch_reply

        # Else, use what we have (hopefully everything).
        batch = batchify(
            examples,
            null=self.dict_misc[self.dict_misc.null_token],
            cuda=self.opt['cuda'])

        # Either train or predict
        if 'labels' in observations[0]:
            self.n_examples += len(examples)
            self.model.update(batch)
        else:
            predictions = self.model.predict(batch)
            # assert len(examples) == len(valid_inds)
            # cause you have already filter invalid examples
            for i in range(len(predictions)):
                batch_reply[valid_inds[i]]['text'] = predictions[i]

        return batch_reply

    def save(self, fname=None):
        """Save the parameters of the agent to a file."""
        fname = self.opt.get('model_file', None) if fname is None else fname
        if fname:
            print("[ saving model: " + fname + " ]")
            self.model.save(fname)

    # --------------------------------------------------------------------------
    # Helper functions.
    # --------------------------------------------------------------------------

    def _build_ex(self, ex):
        """Find the token span of the answer in the context for this example.
        If a token span cannot be found, return None. Otherwise, torchify.
        """
        # Check if empty input (end of epoch)
        if 'text' not in ex:
            return

        # Split out document + question
        inputs = {}
        fields = ex['text'].strip().split('\n')

        # Data is expected to be text + '\n' + question
        if len(fields) < 2:
            raise RuntimeError('Invalid input. Is task a QA task?')

        document, question = ' '.join(fields[:-1]), fields[-1]
        inputs['document'] = self.dict_misc.tokenize(document)
        # notice in tokenize we use spacy to deal with document, to return a
        # spacy.token.doc.Doc
        inputs['question'] = self.dict_misc.tokenize(question)
        inputs['target'] = None
        token_doc = inputs['document']
        token_ques = inputs['question']

        # Find targets (if labels provided).
        # Return if we were unable to find an answer.
        if 'labels' in ex:
            # so in actually we didn't provide labels in squad DefaultTeacher.
            # this condition didn't satisfy
            inputs['target'] = self._find_target(inputs['document'],
                                                 ex['labels'])
            if inputs['target'] is None:
                return

        # Vectorize.
        inputs = vectorize(self.opt, inputs, self.dict_misc, self.feature_dict)
        # return document, features, question, start, end all torch.LongTensor

        # Return inputs with original text + spans (keep for prediction)
        return inputs + (token_doc, token_ques, document,
                         self.dict_misc.span_tokenize(document))

    def _find_target(self, document, labels):
        """Find the start/end token span for all labels in document.
        Return a random one for training.
        """

        def _positions(d, l):
            for i in range(len(d)):
                for j in range(i, min(len(d) - 1, i + len(l))):
                    if l.text == d[i:j + 1].text:
                        yield (i, j)

        targets = []
        for label in labels:
            targets.extend(
                _positions(document, self.dict_misc.tokenize(label)))
        if len(targets) == 0:
            return
        return targets[np.random.choice(len(targets))]
        # so we could know Bidaf's hypothesis is still squad-like dataset.
        # answer should be sub-string in document.

    def report(self):
        return ('[train] updates = %d | train loss = %.2f | exs = %d' %
                (self.model.updates, self.model.train_loss.avg,
                 self.n_examples))
