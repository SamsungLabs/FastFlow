# Copyright 2022 Samsung Electronics Co., Ltd. All Rights Reserved
from http.server import BaseHTTPRequestHandler
from threading import Lock
from uuid import uuid4

from fastflow.autoscaler.server.api import routes
from fastflow.platform.logger import get_logger

logger = get_logger()


class FastFlowHTTPRequestHandler(BaseHTTPRequestHandler):
    def fastflow_state_get_all(self):
        with self._fastflow_state_lock:
            return self._fastflow_state.copy()

    def fastflow_state_get(self, key):
        with self._fastflow_state_lock:
            return self._fastflow_state.get(key)

    def fastflow_state_set(self, key, val):
        with self._fastflow_state_lock:
            self._fastflow_state[key] = val

    def _do_route(self):
        routing_function = routes.get(self.path)
        if routing_function is None:
            self.send_response(404, "API NOT FOUND")
            self.end_headers()
        else:
            routing_function(self)

    def log_message(self, format, *args):  # pylint: disable=redefined-builtin
        logger.debug("{} - - [{}] {}\n".format(self.address_string(),
                                               self.log_date_time_string(),
                                               format % args))

    # routing command
    do_GET = _do_route
    do_POST = _do_route

    # FastFlow State
    uuid_token = str(uuid4())
    _fastflow_state_lock = Lock()
    _fastflow_state = dict()
