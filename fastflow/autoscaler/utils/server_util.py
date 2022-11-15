# Copyright 2022 Samsung Electronics Co., Ltd. All Rights Reserved
import json


def loads_json(handler):
    content_length = int(handler.headers.get('content-length'))
    return json.loads(handler.rfile.read(content_length).decode('utf-8'))


def dumps_json(python_dict):
    return json.dumps(python_dict).encode('utf-8')


# only response json
def set_headers(handler, content_length):
    handler.send_header('Content-type', 'application/json')
    handler.send_header('Content-length', str(content_length))
    handler.end_headers()
