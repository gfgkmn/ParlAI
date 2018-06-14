# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.teachers import FixedDialogTeacher, DialogTeacher
from .build import build

import json
import os

class IndexTeacher(FixedDialogTeacher):
    """Hand-written CIPS teacher, which loads the json cips data and
    implements its own `act()` method for interacting with student agent,
    rather than inheriting from the core Dialog Teacher. This code is here as
    an example of rolling your own without inheritance.

    This teacher also provides access to the "answer_start" indices that
    specify the location of the answer in the context.
    """

    def __init__(self, opt, shared=None):
        build(opt)
        super().__init__(opt, shared)

        if self.datatype.startswith('train'):
            suffix = 'train.all'
        else:
            suffix = 'valid'
        datapath = os.path.join(
            opt['datapath'],
            'Cips',
            suffix + '.json'
        )
        self.data = self._setup_data(datapath)
        # self.data is (35002, 2) mean this qa pair is from query 35002's second qa pair

        self.id = 'cips'
        self.reset()

    def num_examples(self):
        return len(self.examples)

    def num_episodes(self):
        return self.num_examples()

    def get(self, episode_idx, entry_idx=None):
        query_idx, passage_answer_idx = self.examples[episode_idx]
        item = self.cips[query_idx]
        passage_id = item['answer'][passage_answer_idx]['from_passage'] - 1
        passage = item['passages'][passage_id]['passage_text']
        query = item['query']
        answers = [item['answer'][passage_anser_idx]['answer_text']]
        answer_starts = [passage.index(answer)]

        action = {
            'id': 'cips',
            'text': passage + '\n' + query,
            'labels': answers,
            'episode_done': True,
            'answer_starts': answer_starts
        }
        return action


    def _setup_data(self, path):
        self.cips = []
        with open(path) as data_file:
            ajson = data_file.readline()
            while ajson:
                self.cips.append(json.loads(ajson))
                ajson = data_file.readline()

        self.examples = []
        for query_idx, item in enumerate(self.cips):
            if item['answer']:
                for passage_anser_idx, pas in enumerate(item['answer']):
                    self.examples.append(query_idx, passage_answer_idx)


class DefaultTeacher(DialogTeacher):
    """This version of CIPS inherits from the core Dialog Teacher, which just
    requires it to define an iterator over its data `setup_data` in order to
    inherit basic metrics, a default `act` function.
    For CIPS, this does not efficiently store the paragraphs in memory.
    """

    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        build(opt)
        # just check whether data is exist, and download it.
        if opt['datatype'].startswith('train'):
            suffix = 'train.all'
        else:
            suffix = 'valid'
        opt['datafile'] = os.path.join(opt['datapath'], 'Cips',
                                       suffix + '.json')
        self.id = 'cips'
        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading: ' + path)

        self.cips = []
        with open(path) as data_file:
            ajson = data_file.readline()
            while ajson:
                self.cips.append(json.loads(ajson))
                ajson = data_file.readline()

        for item in self.cips:
            # each paragraph is a context for the attached questions
            if item['answer']:
                for passage_answer in item['answer']:
                    passage_id = passage_answer['from_passage'] - 1
                    passage = item['passages'][passage_id]['passage_text']
                    query = item['query']
                    answers = [passage_answer['answer_text']]
                    yield (passage + '\n' + query, answers), True


class TitleTeacher(DefaultTeacher):
    """This version of CIPS inherits from the Default Teacher. The only
    difference is that the 'text' field of an observation will contain
    the title of the article separated by a newline from the paragraph and the
    query.
    Note: The title will contain underscores, as it is the part of the link for
    the Wikipedia page; i.e., the article is at the site:
    https://en.wikipedia.org/wiki/{TITLE}
    Depending on your task, you may wish to remove underscores.
    """

    def __init__(self, opt, shared=None):
        self.id = 'cips_title'
        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading: ' + path)

        self.cips = []
        with open(path) as data_file:
            ajson = data_file.readline()
            while ajson:
                self.cips.append(json.loads(ajson))
                ajson = data_file.readline()

        for item in self.cips:
            # each paragraph is a context for the attached questions
            title = item['query']
            if item['answer']:
                for passage_answer in item['answer']:
                    passage_id = passage_answer['from_passage'] - 1
                    passage = item['passages'][passage_id]['passage_text']
                    query = item['query']
                    answers = [passage_answer['answer_text']]
                    yield (
                        '\n'.join([title, passage, answer]),
                        answers
                    ), True
