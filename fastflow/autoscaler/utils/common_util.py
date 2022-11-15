# Copyright 2022 Samsung Electronics Co., Ltd. All Rights Reserved
import socket


def get_hostname():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as dummy_socket:
        try:
            dummy_socket.connect(('10.255.255.255', 1))
            return dummy_socket.getsockname()[0]
        except Exception:
            return '127.0.0.1'
