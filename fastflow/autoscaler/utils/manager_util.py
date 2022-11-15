# Copyright 2022 Samsung Electronics Co., Ltd. All Rights Reserved
import socket
import threading
import time

from fastflow.core.exceptions import DispatcherNotReadyError
from fastflow.core.exceptions import WorkersNotReadyError
from fastflow.platform.logger import get_logger

logger = get_logger()


def bind_port_safety(port):
    """
    port: string | int
    :return: safety_port, int
    """
    with socket.socket() as dummy_socket:
        try:
            dummy_socket.bind(('', int(port)))
        except OSError:
            dummy_socket.bind(('', 0))
        port = dummy_socket.getsockname()[1]
        dummy_socket = None
        return port


def retry_until_function_ready(instance_manager_thread, server_helper,
                               target_function, timeout,
                               timeout_message="timeout", *args, **kwargs):
    """
    instance_manager_thread: Required to detect
                             if the instance_manager is dead.
    server_helper: Required to detect if the server is dead.
    target_function: A function that retries until ready.
    """
    start_time = time.time()
    while True:
        if not instance_manager_thread.is_alive():
            raise RuntimeError("instance manager is dead")
        if not server_helper.is_alive():
            raise RuntimeError("server is dead")
        last_try_time = time.time()
        if last_try_time - start_time > timeout:
            raise TimeoutError(timeout_message)
        try:
            target_retv = target_function(*args, **kwargs)
            return target_retv
        except DispatcherNotReadyError:
            logger.debug("Dispatcher dose not ready.")
        except WorkersNotReadyError:
            logger.debug("Worker dose not ready.")
        time.sleep(1)


def threaded(original_function):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=original_function,
                                  args=args,
                                  kwargs=kwargs)
        return thread

    return wrapper
