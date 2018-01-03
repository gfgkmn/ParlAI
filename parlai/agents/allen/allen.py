# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Agent does gets the local keyboard input in the act() function.
   Example: python examples/eval_model.py -m local_human -t babi:Task1k:1 -dt valid
"""

from parlai.core.agents import Agent
from parlai.core.worlds import display_messages
from parlai.core.utils import translate
import sys
from os import path
sys.path.append(path.join(path.dirname(path.abspath(__file__)), 'allennlp'))
from allennlp.models.archival import load_archive
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

class AllenAgent(Agent):

    @staticmethod
    def add_cmdline_args(argparser):
        allen = argparser.add_argument_group('AllenNlp Agent Args')
        allen.add_argument(
            '--archive_file', help='archive to load trained model')
        allen.add_argument(
            '--cuda_device',
            help='specify which gpu device to use.',
            default=-1)
        allen.add_argument(
            '--overrides', help='overrides arguments', default='')

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'Allen'
        self.episodeDone = False
        archive = load_archive(opt['archive_file'], opt['cuda_device'], opt['overrides'])
        self.config = archive.config
        self.model = archive.model
        self.data_reader = DatasetReader.from_params(
            self.config.pop('dataset_reader'))

    def observe(self, observation):
        # shallow copy observation (deep copy can be expensive)
        observation = observation.copy()
        print(display_messages([observation]))
        if not self.episode_done:
            dialogue = self.observation['text'].split('\n')[:-1]
            dialogue.extend(observation['text'].split('\n'))
            observation['text'] = '\n'.join(dialogue)
        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation


    def act(self):
        obs = self.observation
        instance = self.text_to_instance(obs)
        output = self.model.forward_on_instance(instance, -1)
        reply_text = output['best_span_str']
        reply = {}
        reply['id'] = self.getID()
        # reply_text = input("Enter Your Message: ")
        print('answer is %s' % reply_text)
        print('translate to %s' % (translate(reply_text) + '\n'))
        reply_text = reply_text.replace('\\n', '\n')
        reply['episode_done'] = False
        if '[DONE]' in reply_text:
            reply['episode_done'] = True
            self.episodeDone = True
            reply_text = reply_text.replace('[DONE]', '')
        reply['text'] = reply_text
        return reply

    def episode_done(self):
        return self.episodeDone

    def text_to_instance(self, msg):
        fields = msg['text'].strip().split('\n')

        # Data is expected to be text + '\n' + question
        if len(fields) < 2:
            raise RuntimeError('Invalid input. Is task a QA task?')

        document, question = ' '.join(fields[:-1]), fields[-1]
        answer_texts = list(msg['eval_labels'])
        span_starts = [document.index(answer) for answer in answer_texts]
        span_ends = [s + len(ans) for s, ans in zip(span_starts, answer_texts)]
        tokenized_paragraph = self.data_reader._tokenizer.tokenize(document)
        instance = self.data_reader.text_to_instance(question,
                                                     document,
                                                     zip(span_starts, span_ends),
                                                     answer_texts,
                                                     tokenized_paragraph)
        return instance 
