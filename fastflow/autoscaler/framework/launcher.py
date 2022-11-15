# Copyright 2022 Samsung Electronics Co., Ltd. All Rights Reserved
import uuid
import boto3

import fastflow.autoscaler.utils.launcher_util as utils

from fastflow.autoscaler.framework import instance as instance_framework


class LauncherFactory:
    @classmethod
    def get_launcher(cls, option):
        if isinstance(option, instance_framework.AWSEC2InstanceOption):
            return AWSEC2InstanceLauncher(option.get_option_dictionary())
        if isinstance(option, instance_framework.SSHInstanceOption):
            return SSHInstanceLauncher(option.get_option_dictionary())
        if isinstance(option,
                      instance_framework.SelfHostedClusterInstanceOption):
            return SelfHostedClusterInstanceLauncher(
                option.get_option_dictionary()
            )
        raise ValueError("Invalid Instance Option")


class Launcher:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def gen_unique_id():
        return uuid.uuid4().int & (1 << 64) - 1

    def execute_dispatcher(self, comm_data):
        raise NotImplementedError

    def execute_remote_workers(self, num, comm_data):
        raise NotImplementedError


class AWSEC2InstanceLauncher(Launcher):
    INSTANCE_ID_SCRIPT = \
        "$(curl http://169.254.169.254/latest/meta-data/instance-id)"

    def __init__(self, config):
        boto3_config = config.get('boto3_credential')
        self._ec2_resource = boto3.resource('ec2', **boto3_config)
        super().__init__(config)

    def execute_dispatcher(self, comm_data):
        dispatcher_config = self.config \
            .get('ec2_instance_option') \
            .get('dispatcher')

        command = utils.DISPATCHER_ENTRYPOINT_FORM.format(
            instance_id_script=self.INSTANCE_ID_SCRIPT,
            uuid_token=comm_data.get('uuid_token'),
            server_address=comm_data.get('server_address')
        )

        instances = self._ec2_resource.create_instances(
            **dispatcher_config,
            UserData=command)

        # We currently assume that
        # there is only a single dispatcher for a tf.data service.
        return list(map(self._create_instance_object, instances))[0]

    def execute_remote_workers(self, num, comm_data):
        command = utils.WORKER_ENTRYPOINT_FORM.format(
            instance_id_script=self.INSTANCE_ID_SCRIPT,
            uuid_token=comm_data.get('uuid_token'),
            server_address=comm_data.get('server_address'),
            dispatcher_address=comm_data.get('dispatcher_address'))

        worker_config = self.config.get('ec2_instance_option').get('worker')
        instances = self._ec2_resource.create_instances(
            **worker_config,
            MaxCount=num,
            UserData=command)

        return list(map(self._create_instance_object, instances))

    @staticmethod
    def _create_instance_object(instance):
        return instance_framework.AWSEC2Instance(
            instance=instance,
            instance_id=instance.id)


# TODO: Implement SelfHostedClusterInstanceLauncher
# for autoscaling with self-hosted cluster
class SelfHostedClusterInstanceLauncher(Launcher):
    def __init__(self, config):
        super().__init__(config)


class SSHInstanceLauncher(Launcher):
    def __init__(self, config):
        self._num_workers_launched = 0
        super().__init__(config)

    def execute_dispatcher(self, comm_data):
        instance_id = Launcher.gen_unique_id()
        ssh_client = utils.SSHClient(
            node_ip=self.config.get('dispatcher'),
            user_name=self.config.get('ssh_user_name'),
            password=None,
            keyfile_path=self.config.get('ssh_private_key_path'))

        command = utils.DISPATCHER_ENTRYPOINT_SSH_FORM.format(
            instance_id_script=instance_id,
            uuid_token=comm_data.get('uuid_token'),
            server_address=comm_data.get('server_address')
        )
        command = utils.build_multiline_command(command)
        ssh_client.exec_command(command)

        return instance_framework.SSHInstance(
            instance=ssh_client,
            instance_id=instance_id)

    def execute_remote_workers(self, num, comm_data):
        workers = []
        for _ in range(num):
            instance_id = Launcher.gen_unique_id()
            ssh_client = utils.SSHClient(
                node_ip=self.config.get('workers')[self._num_workers_launched],
                user_name=self.config.get('ssh_user_name'),
                password=None,
                keyfile_path=self.config.get('ssh_private_key_path'))

            command = utils.WORKER_ENTRYPOINT_SSH_FORM.format(
                instance_id_script=instance_id,
                uuid_token=comm_data.get('uuid_token'),
                server_address=comm_data.get('server_address'),
                dispatcher_address=comm_data.get('dispatcher_address')
            )
            command = utils.build_multiline_command(command)
            ssh_client.exec_command(command)

            instance = instance_framework.SSHInstance(
                instance=ssh_client,
                instance_id=instance_id
            )
            workers.append(instance)

            self._num_workers_launched += 1
        return workers
