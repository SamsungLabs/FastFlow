# Copyright 2022 Samsung Electronics Co., Ltd. All Rights Reserved
import time

import grpc
import requests

from fastflow.platform.logger import get_logger

logger = get_logger()


def request_and_retry(request_function,
                      timeout,
                      max_retry,
                      sleep_secs,
                      *args, **kwargs):
    '''
    request_function: requests.get or requets.post or stub
    timeout:          timeout for requests of server
    max_retry:        a number of retry count
    sleep_secs:       sleep seconds


    args:             url, request_message etc..
    kwargs:           json, etc..
    '''
    kwargs['timeout'] = timeout
    for _ in range(max_retry):
        try:
            response = request_function(*args, **kwargs)
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectTimeout,
                requests.exceptions.Timeout,
                requests.exceptions.InvalidSchema):
            logger.error("requests connection Error!")
            time.sleep(sleep_secs)
            continue
        except grpc.RpcError:
            logger.error("gRPC connection Error!")
            time.sleep(sleep_secs)
            continue
        if not isinstance(response, requests.models.Response):
            return response
        if response.ok:
            return response
        logger.debug("response not ok")
        time.sleep(sleep_secs)
    return None
