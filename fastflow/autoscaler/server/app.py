# Copyright 2022 Samsung Electronics Co., Ltd. All Rights Reserved
import argparse

from http.server import ThreadingHTTPServer

from fastflow.autoscaler.server.handler import FastFlowHTTPRequestHandler


def create_server(port):
    return ThreadingHTTPServer(('0.0.0.0', port),
                               FastFlowHTTPRequestHandler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('port', action='store',
                        default=1515, type=int,
                        nargs='?',
                        help='Specify alternate port [default: 1515]')
    args = parser.parse_args()
    server = create_server(args.port)
    # TODO use Logging tool
    print("server run 0.0.0.0:" + str(server.server_port))
    print("uuid_token: " + FastFlowHTTPRequestHandler.uuid_token)
    server.serve_forever()
