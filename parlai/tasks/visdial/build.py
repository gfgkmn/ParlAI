# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os
import json
import random


def buildImage(opt):
    dpath = os.path.join(opt['datapath'], 'COCO-IMG')

    if not build_data.built(dpath):
        print('[building image data: ' + dpath + ']')
        build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # download the image data.
        fname1 = 'train2014.zip'
        fname2 = 'val2014.zip'
        fname3 = 'test2014.zip'

        url = 'http://msvocds.blob.core.windows.net/coco2014/'

        build_data.download(url + fname1, dpath, fname1)
        build_data.download(url + fname2, dpath, fname2)
        build_data.download(url + fname3, dpath, fname3)

        build_data.untar(dpath, fname1)
        build_data.untar(dpath, fname2)
        build_data.untar(dpath, fname3)

        # Mark the data as built.
        build_data.mark_done(dpath)


def build(opt):
    dpath = os.path.join(opt['datapath'], 'VisDial-v0.9')

    if not build_data.built(dpath):
        print('[building data: ' + dpath + ']')
        build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        fname1 = 'visdial_0.9_train.zip'
        fname2 = 'visdial_0.9_val.zip'

        url = 'https://computing.ece.vt.edu/~abhshkdz/data/visdial/'
        build_data.download(url + fname1, dpath, fname1)
        build_data.download(url + fname2, dpath, fname2)

        build_data.untar(dpath, fname1)
        build_data.untar(dpath, fname2)

        # Use 1000 examples from training set as validation.
        print('processing unpacked files')
        json1 = os.path.join(dpath, fname1.rsplit('.', 1)[0] + '.json')
        with open(json1) as t_json:
            train_data = json.load(t_json)

        valid_idx = random.choices(range(len(train_data['data']['dialogs'])), k=1000)
        valid_data = train_data.copy()
        valid_data['data'] = train_data['data'].copy()
        valid_data['data']['dialogs'] = []
        for i in sorted(valid_idx, reverse=True):
            valid_data['data']['dialogs'].append(train_data['data']['dialogs'][i])
            del train_data['data']['dialogs'][i]

        train_json = json1.rsplit('.', 1)[0] + '_train.json'
        valid_json = json1.rsplit('.', 1)[0] + '_valid.json'
        with open(train_json, 'w') as t_out, open(valid_json, 'w') as v_out:
            json.dump(train_data, t_out)
            json.dump(valid_data, v_out)
        os.remove(json1)

        # Use validation data as test.
        json2 = os.path.join(dpath, fname2.rsplit('.', 1)[0] + '.json')
        test_json = json2.rsplit('.', 1)[0] + '_test.json'
        build_data.move(json2, test_json)

        # Mark the data as built.
        build_data.mark_done(dpath)
