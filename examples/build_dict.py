# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Generates a dictionary file from the training data."""

from parlai.core.dict import DictionaryAgent
from parlai.core.params import ParlaiParser, str2class
from parlai.core.worlds import create_task
import copy
import os


def build_dict(opt):
    if not opt.get('dict_file'):
        print('Tried to build dictionary but `--dict-file` is not set. Set ' +
              'this param so the dictionary can be saved.')
        return
    print('[ setting up dictionary. ]')
    if os.path.isfile(opt['dict_file']):
        # Dictionary already built
        print("[ dictionary already built .]")
        return
    if opt.get('dict_class'):
        # Custom dictionary class
        dictionary = str2class(opt['dict_class'])(opt)
        # because ParlaiParser'parameter model == true.
        # use self.add_cmdline_args in self.add_model_args
        # to add model's config and data's config through
        # dictionary config.
        # SimpleDictionaryAgent from parlai/agents/drqa/drqa.py
        # user_agent initialize.
    else:
        # Default dictionary class
        dictionary = DictionaryAgent(opt)
    ordered_opt = copy.deepcopy(opt)
    cnt = 0
    # we use train set to build dictionary
    ordered_opt['datatype'] = 'train:ordered:stream'
    # DialogTeacher parent class used when initialize.
    ordered_opt['numthreads'] = 1
    ordered_opt['batchsize'] = 1
    ordered_opt['image_mode'] = 'none'
    world_dict = create_task(ordered_opt, dictionary)
    # when use squad drqa task.
    # DialogPartnerWorld(opt, DefaultTeacher + SimpleDictionaryAgent)
    # world_dict, look like some step between build world
    # environment.
    # pass examples to dictionary
    while not world_dict.epoch_done():
        cnt += 1
        if cnt > opt['dict_maxexs'] and opt['dict_maxexs'] > 0:
            print('Processed {} exs, moving on.'.format(opt['dict_maxexs']))
            # don't wait too long...
            break
        world_dict.parley()
        # looks like, default teacher give the data to dictionary.
        # so always use world.parley to generate next iteration.
        # and in agents[0], indicate whether data is already done.
    print('[ dictionary built. ]')
    dictionary.save(opt['dict_file'], sort=True)


def main():
    # Get command line arguments
    argparser = ParlaiParser()
    DictionaryAgent.add_cmdline_args(argparser)
    opt = argparser.parse_args()
    build_dict(opt)


if __name__ == '__main__':
    main()
