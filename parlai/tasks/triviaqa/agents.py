# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.agents import Teacher
from parlai.core.dialog_teacher import DialogTeacher
from .build import build

import json
import random
import os


class HandwrittenTeacher(Teacher):
    """Hand-written TRIVIAQA teacher, which loads the json triviaqa data and
    implements its own `act()` method for interacting with student agent,
    rather than inheriting from the core Dialog Teacher. This code is here as
    an example of rolling your own without inheritance.  """

    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        build(opt)
        if opt['datatype'].startswith('train'):
            suffix = 'train'
        else:
            suffix = 'dev'
        datapath = os.path.join(opt['datapath'], 'TriviaQA',
                                suffix + '-v1.1.json')
        self.data = self._setup_data(datapath)
        self.episode_idx = -1
        super().__init__(opt, shared)

    def __len__(self):
        return self.len

    # return state/action dict based upon passed state
    def act(self):
        if self.datatype == 'train':
            self.episode_idx = random.randrange(len(self.examples))
        else:
            self.episode_idx = (self.episode_idx + 1) % len(self.examples)
        article_idx, paragraph_idx, qa_idx = self.examples[self.episode_idx]
        article = self.triviaqa[article_idx]
        paragraph = article['paragraphs'][paragraph_idx]
        qa = paragraph['qas'][qa_idx]
        question = qa['question']
        answers = [a['text'] for a in qa['answers']]
        context = paragraph['context']

        if (self.episode_idx == (len(self.examples) - 1) and
                self.datatype != 'train'):
            self.epochDone = True

        return {
            'text': context + '\n' + question,
            'labels': answers,
            'episode_done': True
        }

    def _setup_data(self, path):
        print('loading: ' + path)
        with open(path) as data_file:
            self.triviaqa = json.load(data_file)['data']
        self.len = 0
        self.examples = []
        for article_idx in range(len(self.triviaqa)):
            article = self.triviaqa[article_idx]
            for paragraph_idx in range(len(article['paragraphs'])):
                paragraph = article['paragraphs'][paragraph_idx]
                num_questions = len(paragraph['qas'])
                self.len += num_questions
                for qa_idx in range(num_questions):
                    self.examples.append((article_idx, paragraph_idx, qa_idx))


class DefaultTeacher(DialogTeacher):
    """This version of TRIVIAQA inherits from the core Dialog Teacher, which just
    requires it to define an iterator over its data `setup_data` in order to
    inherit basic metrics, a default `act` function, and enables
    Hogwild training with shared memory with no extra work.
    For TRIVIAQA, this does not efficiently store the paragraphs in memory.
    """

    def __init__(self, opt, shared=None):
        # this version just support wikipedia training.
        # TODO: support web wikipedia training and add verified version
        self.datatype = opt['datatype']
        build(opt)
        # just check whether data is exist, and download it.
        if opt['datatype'].startswith('train'):
            suffix = 'train'
        else:
            suffix = 'dev'
        opt['datafile'] = os.path.join(opt['datapath'], 'TriviaQA', 'qa',
                                       'wikipedia-' + suffix + '.json')
        self.id = 'triviaqa'
        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading: ' + path)
        evidence_path = os.path.join(
            os.path.split(os.path.split(path)[0])[0], 'evidence')
        with open(path) as data_file:
            self.triviaqa = json.load(data_file)['Data']
        for qa_pairs in self.triviaqa:
            question = qa_pairs['Question']
            context = ''
            if qa_pairs['EntityPages']:
                for page in qa_pairs['EntityPages']:
                    page_path = os.path.join(evidence_path, 'wikipedia',
                                             page['Filename'])
                    with open(page_path) as fevds:
                        page_context = fevds.read()
                    context += page_context
                    context += '\n'
                # wikipedia data have no searchresults.
            assert context != ''
            answers = (qa_pairs['Answer']['Value'],)
            yield (context + '\n' + question, answers), True
