# Copyright 2022 Samsung Electronics Co., Ltd. All Rights Reserved
import signal
import time

import tensorflow as tf

from fastflow.autoscaler.client.fastflow_client import AutoScalerServerClient
from fastflow.autoscaler.utils.common_util import get_hostname
from fastflow.autoscaler.utils.manager_util import bind_port_safety
from fastflow.platform.logger import get_logger

logger = get_logger()


def _run_worker(port, dispatcher_address):
    port = bind_port_safety(port)
    w_config = tf.data.experimental.service.WorkerConfig(
        dispatcher_address=dispatcher_address,
        worker_address=get_hostname() + ":" + str(port),
        port=port
    )
    signal.alarm(30)  # start sigkill with timeout
    worker = tf.data.experimental.service.WorkerServer(w_config)
    signal.alarm(0)  # stop sigkill with timeout
    return worker, get_hostname() + ":" + str(port)


def _run_dispatcher(port):
    port = bind_port_safety(port)
    d_config = tf.data.experimental.service.DispatcherConfig(port=port)
    dispatcher = tf.data.experimental.service.DispatchServer(d_config)
    return dispatcher, get_hostname() + ":" + str(port)


class TFDataServiceManager:
    def __init__(self,
                 uuid_token,
                 server_address,
                 instance_id,
                 tf_data_service_type,
                 dispatcher_address,
                 port):
        self._uuid_token = uuid_token
        self._server_address = server_address
        self._instance_id = instance_id
        self._tf_data_service_type = tf_data_service_type
        if tf_data_service_type == "worker" and not dispatcher_address:
            raise ValueError("need dispatcher address argument "
                             "--dispatcher_address")
        self._dispatcher_address = dispatcher_address
        self._port = port
        if tf_data_service_type == "dispatcher":
            self._tf_data_service, self._tf_data_service_address = \
                _run_dispatcher(port)
        else:
            self._tf_data_service, self._tf_data_service_address = \
                _run_worker(port, dispatcher_address)

    def watch(self):
        # TODO: Add user-defiend timeout argument
        client = AutoScalerServerClient(self._server_address)

        while True:
            logger.debug("post heartbeat")
            # TODO: Add cpu and memory util
            resp = client.post_heartbeat(self._tf_data_service_address,
                                         self._instance_id,
                                         self._tf_data_service_type,
                                         self._uuid_token)
            if not resp:
                logger.error("Server is dead.")
                self._tf_data_service._stop()  # pylint: disable=protected-access
                break
            time.sleep(1)
