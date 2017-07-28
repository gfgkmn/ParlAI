# Copyright 2004-present Facebook. All Rights Reserved.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Basic example which iterates through the tasks specified and runs the given
model on them.

For example:
`python examples/display_model.py -t babi:task1k:1 -m "repeat_label"`
or:
`python examples/display_model.py -t "#MovieDD-Reddit" -m "ir_baseline" -mp "-lp 0.5" -dt test`
"""

from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task

import random
import json
import requests

translate_token = ""


def translate(astr):
    origin_data = {
        # "source": str(sys.argv[1]),
        "source": astr,
        "trans_type": "en2zh",
        "request_id": "a11111",
        "replaced": True,
        "cached": True
    }
    json_data = json.dumps(origin_data)

    return_data = requests.post(
        'http://api.interpreter.caiyunai.com/v1/translator',
        data=json_data,
        headers={
            "Content-type": "application/json",
            "X-Authorization": "token " % translate_token,
        })

    # print json.loads(return_data.content)['target']
    return json.loads(return_data.content)['target']


def main():
    random.seed(42)

    # Get command line arguments
    parser = ParlaiParser(True, True)
    parser.add_argument('-n', '--num-examples', default=10)
    opt = parser.parse_args()

    # Create model and assign it to the specified task
    agent = create_agent(opt)
    world = create_task(opt, agent)

    # Show some example dialogs.
    with world:
        for k in range(int(opt['num_examples'])):
            world.parley()
            msgs = world.display()
            document, question, answer, _ = msgs.split('\n')
            print(document + '\n')
            print(translate(document) + '\n')
            print(question + '\n')
            print(translate(question) + '\n')
            print(answer + '\n')
            print(translate(answer) + '\n')
            # print(world.display() + "\n~~")
            input("Input any key to continue:")
            if world.epoch_done():
                print("EPOCH DONE")
                break


if __name__ == '__main__':
    main()
