# Copyright 2022 Samsung Electronics Co., Ltd. All Rights Reserved
import grpc
import requests

from fastflow.autoscaler.framework.grpc import dispatcher_pb2
from fastflow.autoscaler.utils.client_util import request_and_retry


# AutoScalerServerClient for InstanceManager and TFDataServiceManager
class AutoScalerServerClient:
    def __init__(self,
                 server_address,
                 sleep_secs=3,
                 timeout=9,
                 max_retry=5):
        '''
        server_address:   <server_ip>:<port>
        sleep_secs:       sleep seconds
        timeout:          timeout for requests of server
        max_retry:        a number of retry count
        '''
        self._server_address = server_address
        self._sleep_secs = sleep_secs
        self._timeout = timeout
        self._max_retry = max_retry

    def _api_address(self, api_path):
        # api_path: Path of API ex) "/api/heartbeat"
        return "http://" + self._server_address + api_path

    def post_heartbeat(self,
                       tf_data_service_address,
                       instance_id,
                       tf_data_service_type,
                       uuid_token):
        # TODO: More information add
        request_json = {
            "address": tf_data_service_address,
            "instance_id": instance_id,
            "tf_data_service_type": tf_data_service_type,
            "uuid_token": uuid_token
        }
        return request_and_retry(
            requests.post,
            self._timeout,
            self._max_retry,
            self._sleep_secs,
            self._api_address("/api/heartbeat"),
            json=request_json
        )

    def get_instance_infos(self):
        return request_and_retry(
            requests.get,
            self._timeout,
            self._max_retry,
            self._sleep_secs,
            self._api_address("/api/instance_infos")
        )


# DispatcherClient for InstanceManager or AutoScaler
class DispatcherClient:
    def __init__(self,
                 dispatcher_address,
                 sleep_secs=3,
                 timeout=9,
                 max_retry=5):
        '''
        dispatcher_address:   <server_ip>:<port>
        sleep_secs:           sleep seconds
        timeout:              timeout for requests of server
        max_retry:            a number of retry count
        '''
        self._dispatcher_address = dispatcher_address
        self._sleep_secs = sleep_secs
        self._timeout = timeout
        self._max_retry = max_retry

    def get_workers(self):
        with grpc.insecure_channel(self._dispatcher_address) as channel:
            stub = channel.unary_unary(
                '/tensorflow.data.DispatcherService/GetWorkers',
                request_serializer=dispatcher_pb2
                    .GetWorkersRequest
                    .SerializeToString,
                response_deserializer=dispatcher_pb2
                    .GetWorkersResponse
                    .FromString
            )
            return request_and_retry(stub,
                                     self._timeout,
                                     self._max_retry,
                                     self._sleep_secs,
                                     dispatcher_pb2.GetWorkersRequest())
