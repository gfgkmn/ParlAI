# import zmq
import json


def sanitize(obs):
    if 'image' in obs and type(obs['image']) != str:
        # can't json serialize images, unless they're in ascii format
        obs.pop('image', None)
    for k, v in obs.items():
        if type(v) == set:
            obs[k] = list(v)
    return obs


def wrap_input(text):
    dic = {"id": "myself", "episode_done": True, "text": text}
    return dic


# address = '192.168.1.55'
# socket_type = zmq.REQ
# # request connection
# port = '5555'

# context = zmq.Context()
# socket = context.socket(socket_type)
# socket.setsockopt(zmq.LINGER, 1)
# host = 'tcp://{}:{}'.format(address, port)

# if socket_type == zmq.REP:
#     socket.bind(host)
# else:
#     socket.connect(host)


def reply(text, socket):
    if text:
        content = json.dumps(sanitize(text))
        socket.send_unicode(content)
    reply = socket.recv_unicode()
    return json.loads(reply)


# a = wrap_input(input("please input:"))

# while a:
#     response = reply(a)
#     print(response['text'])
#     a = wrap_input(input("please input:"))

# try:
#     socket.send_unicode('<END>', zmq.NOBLOCK)
# except zmq.error.ZMQError:
#     # may need to listen first
#     try:
#         socket.recv_unicode(zmq.NOBLOCK)
#         socket.send_unicode('<END>', zmq.NOBLOCK)
#     except zmq.error.ZMQError:
#         # paired process is probably dead already
#         pass
