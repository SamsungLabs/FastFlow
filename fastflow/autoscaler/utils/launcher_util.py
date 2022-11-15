# Copyright 2022 Samsung Electronics Co., Ltd. All Rights Reserved
import paramiko

WORKER_ENTRYPOINT_SSH_FORM = '''#!/bin/bash
export INSTANCE_ID={instance_id_script}
python -m fastflow.autoscaler.manager.run_tf_data_service_manager \
       --uuid_token {uuid_token} \
       --server_address \
            {server_address} \
       --instance_id \\$INSTANCE_ID \
       --tf_data_service_type worker \
       --dispatcher_address \
            {dispatcher_address}
'''

DISPATCHER_ENTRYPOINT_SSH_FORM = '''#!/bin/bash
export INSTANCE_ID={instance_id_script}
python -m fastflow.autoscaler.manager.run_tf_data_service_manager \
       --uuid_token {uuid_token} \
       --server_address \
            {server_address} \
       --instance_id \\$INSTANCE_ID \
       --tf_data_service_type dispatcher
'''

WORKER_ENTRYPOINT_FORM = '''#!/bin/bash
export INSTANCE_ID={instance_id_script}
python -m fastflow.autoscaler.manager.run_tf_data_service_manager \
       --uuid_token {uuid_token} \
       --server_address \
            {server_address} \
       --instance_id $INSTANCE_ID \
       --tf_data_service_type worker \
       --dispatcher_address \
            {dispatcher_address}
'''

DISPATCHER_ENTRYPOINT_FORM = '''#!/bin/bash
export INSTANCE_ID={instance_id_script}
python -m fastflow.autoscaler.manager.run_tf_data_service_manager \
       --uuid_token {uuid_token} \
       --server_address \
            {server_address} \
       --instance_id $INSTANCE_ID \
       --tf_data_service_type dispatcher
'''

SSH_PROCESS_TERMINATION_FORM = '''#!/bin/bash
export WORKER_PID=`ps -ef | grep -v grep | grep "tf_data_service_manager" \
    | awk '{print $2}' | tail -1`
kill -9 \\$WORKER_PID
'''


def build_multiline_command(command):
    return "bash <<EOF\n" + command + "EOF\n"


class SSHClient:
    def __init__(
            self,
            node_ip,
            user_name,
            password=None,
            keyfile_path=None):
        self._node_ip = node_ip
        self._user_name = user_name
        self._password = password
        self._keyfile_path = keyfile_path
        self._ssh_client = None

    def __del__(self):
        self._terminate_ssh_session()

    def get_ip(self):
        return self._node_ip

    def _make_ssh_connection(self):
        assert (self._password is not None) or (self._keyfile_path is not None)
        if self._ssh_client is not None:
            self._terminate_ssh_session()

        self._ssh_client = paramiko.SSHClient()
        self._ssh_client.load_system_host_keys()
        self._ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            self._ssh_client.connect(
                hostname=self._node_ip,
                username=self._user_name,
                key_filename=self._keyfile_path,
                password=self._password,
                allow_agent=False,
                look_for_keys=False)
        except Exception as e:
            print("[!] Cannot connect to the specified SSH server: " + e)
            raise e

    def _terminate_ssh_session(self):
        if self._ssh_client is not None:
            self._ssh_client.close()
            self._ssh_client = None

    def exec_command(self, cmd_str, **kwargs):
        if self._ssh_client is None:
            self._make_ssh_connection()

        try:
            self._ssh_client.exec_command(cmd_str, **kwargs)
        except Exception as e:
            print("[!] Unable to connect to the specified SSH server: " + e)
            raise e
