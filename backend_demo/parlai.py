# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from flask_cors import CORS
import zmq
import json
from send_to_remote import wrap_input, reply

app = Flask(__name__)
CORS(app)


address = 'localhost'
socket_type = zmq.REQ
# request connection
port = '5555'
host = 'tcp://{}:{}'.format(address, port)

pre_dataset = json.load(open('pre_data.json', 'r'))


@app.route('/select')
def select():
    data = dict()
    data["titles"] = [i['title'] for i in pre_dataset]
    data["contextss"] = [i['context'] for i in pre_dataset]
    data["context_questions"] = [i['context_question'] for i in pre_dataset]
    result = {'result': data}
    return jsonify(result)


@app.route('/submit')
def submit():
    # a = request.args.get('paragraph')
    b = request.args.get('question')
    # finaltext = a + '\n' + b
    finaltext = b
    response = reply(wrap_input(finaltext), socket)
    return jsonify(result=response['text'])


if __name__ == '__main__':
    context = zmq.Context()
    socket = context.socket(socket_type)
    socket.setsockopt(zmq.LINGER, 1)

    if socket_type == zmq.REP:
        socket.bind(host)
    else:
        socket.connect(host)
    app.Debug = True
    app.run(host='0.0.0.0', port=3377)
    try:
        socket.send_unicode('<END>', zmq.NOBLOCK)
    except zmq.error.ZMQError:
        # may need to listen first
        try:
            socket.recv_unicode(zmq.NOBLOCK)
            socket.send_unicode('<END>', zmq.NOBLOCK)
        except zmq.error.ZMQError:
            # paired process is probably dead already
            pass
    # app.run(host='0.0.0.0')
