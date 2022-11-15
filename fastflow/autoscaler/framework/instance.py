# Copyright 2022 Samsung Electronics Co., Ltd. All Rights Reserved
# TODO: Document API wiki
from enum import Enum

import yaml

from fastflow.autoscaler.utils import launcher_util


class InstanceType(Enum):
    AWS_EC2 = "aws_ec2"
    SSH = "ssh"
    SELF_HOSTED_CLUSTER = "self_hosted_cluster"


def instance_option_from_yaml(yaml_path,
                              cluster_type,
                              self_hosted_cluster_option=None):
    """
    Convert yaml type config to InstanceOption.
    Yaml key must be equal to the keyword
    argument of InstanceOption.__init__
    e.g.) To set image_type="fastflow",
        yaml format should be image_type: "fastflow".

    check doc string about
    "fastflow.autoscaler.framework.instance.AWSEC2InstanceOption"
    "fastflow.autoscaler.framework.instance.SSHInstanceOption"
    "fastflow.autoscaler.framework.instance.SelfHostedClusterInstanceOption"

    self_hosted_cluster_option must have "self_hosted_cluster_creator",
    "self_hosted_cluster_terminator", "dispatcher", "worker".
    Also, "dispatcher" and "worker" must have "InstanceType"

    self_hosted_cluster_option:
        each specific field MUST filled
        this is dictionary for your own cluster system

        {
            "self_hosted_cluster_creator": your_python_function,
            "self_hosted_cluster_terminator": your_python_function,
            "dispatcher": {
                "InstanceType": dispatcher_instance_type,
                also_your_needed_option: ...
            },
            "worker": {
                "InstanceType": worker_instance_type,
                also_your_needed_option: ...
            },
            also_your_needed_option: ...
        }

    :this is form about creator and terminator

    self_hosted_cluster_creator form:
        >>> def my_instance_creator(option_dictionary, entrypoint):
        >>>     option = your_option_creator(option_dictionary,
        ...                                  entrypoint)
        >>>     instance = my_cluster_cli.create_instance(option)
        >>>     return instance

    self_hosted_cluster_terminator form:
        >>> def my_instance_terminator(instance):
        >>>     instance.terminate()

    yaml_path: yaml path
    cluster_type: cluster type, "aws_ec2" | "ssh" | "self_hosted_cluster"
    self_hosted_cluster_option: self_hosted_cluster_option
    :return: {cluster_type}InstanceOption
    """
    with open(yaml_path) as yaml_file:
        yaml_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
    if cluster_type == InstanceType.AWS_EC2.value:
        return AWSEC2InstanceOption(**yaml_dict)
    if cluster_type == InstanceType.SSH.value:
        return SSHInstanceOption(**yaml_dict)
    if cluster_type == InstanceType.SELF_HOSTED_CLUSTER.value:
        return SelfHostedClusterInstanceOption(
            self_hosted_cluster_option=self_hosted_cluster_option,
            **yaml_dict
        )
    raise ValueError("Unrecognized Cluster Type")


class InstanceOption:
    def __init__(self, **kwargs):
        if not kwargs.get("image_type") in {"fastflow", "python"}:
            raise ValueError("unsupported image type")
        self._option_dictionary = kwargs.copy()

    def get_option_dictionary(self):
        return self._option_dictionary.copy()


class AWSEC2InstanceOption(InstanceOption):
    def __init__(self,
                 image_type="fastflow",
                 boto3_credential=None,
                 ec2_instance_option=None):
        """
        image_type:
            "fastflow" | "python"

            Type of AMI or docker image that installed fastflow or not.
        boto3_credential:
            {
                "region_name": your_aws_region,
                "endpoint_url": your_aws_endpoint_url,
                "aws_access_key_id": your_aws_access_key_id,
                "aws_secret_access": your_aws_secret_access_key,
                "aws_session_token": your_aws_session_token
            }
        ec2_instance_option:
            {
                "dispatcher": {
                    "ImageId": your_AMI_id,
                    "MinCount": 1,                                     # fixed
                    "MaxCount": 1,                                     # fixed
                    "InstanceType": your_AWS_instance_type,
                    "KeyName": your_key_pair_name,
                    "InstanceInitiatedShutdownBehavior": "terminate",  # fixed
                    "SecurityGroupIds": [your_security_group_list]
                },
                "worker": {
                    "ImageId": your_AMI_id,
                    "MinCount": 1,                                     # fixed
                    "InstanceType": your_AWS_instance_type,
                    "KeyName": your_key_pair_name,
                    "InstanceInitiatedShutdownBehavior": "terminate",  # fixed
                    "SecurityGroupIds": [your_security_group_list]
                }

            }
        """
        if not boto3_credential:
            raise ValueError("detect aws_ec2, boto3_credential "
                             "must need")
        if not ec2_instance_option:
            raise ValueError("detect aws_ec2, ec2_instance_option "
                             "must need")
        if not ec2_instance_option.get("dispatcher") \
                or not ec2_instance_option.get("worker"):
            raise ValueError("ec2_instance_option must "
                             "have dispatcher and worker")
        super().__init__(image_type=image_type,
                         boto3_credential=boto3_credential,
                         ec2_instance_option=ec2_instance_option)


class SSHInstanceOption(InstanceOption):
    """
    An SSH cluster is a cluster environment
    that can access already running instances through ssh.
    """

    def __init__(self,
                 dispatcher,
                 workers,
                 ssh_user_name=None,
                 ssh_private_key_path=None,
                 image_type="fastflow"):
        """
        dispatcher:
            dispatcher address, ex) 10.0.0.68
        workers:
            worker address lists, ex) [10.0.0.67, 10.0.0.66]
        ssh_user_name:
            ssh_user_name, if you need specifying username
        ssh_private_key_path:
            ssh_private_key_path, if you need specifying ssh private key
        image_type:
            "fastflow" | "python"

            Type of AMI or docker image that installed fastflow or not.
            in SSH Cluster, it mean your already running instance
            have installed fastflow or not.
        """
        super().__init__(image_type=image_type,
                         dispatcher=dispatcher,
                         workers=workers,
                         ssh_user_name=ssh_user_name,
                         ssh_private_key_path=ssh_private_key_path)


class SelfHostedClusterInstanceOption(InstanceOption):
    def __init__(self,
                 image_type="fastflow",
                 self_hosted_cluster_option=None):
        """
        image_type:
            "fastflow" | "python"

            Type of AMI or docker image that installed fastflow or not.
        self_hosted_cluster_option:
            each specific field MUST filled.
            this is dictionary for your own cluster system

            {
                "self_hosted_cluster_creator": your_python_function,
                "self_hosted_cluster_terminator": your_python_function,
                "dispatcher": {
                    "InstanceType": dispatcher_instance_type,
                    also_your_needed_option: ...
                },
                "worker": {
                    "InstanceType": worker_instance_type,
                    also_your_needed_option: ...
                },
                also_your_needed_option: ...
            }

            self_hosted_cluster_creator MUST return instance or process

        :this is form about creator and terminator

        self_hosted_cluster_creator form:
            >>> def my_instance_creator(option_dictionary, entrypoint):
            >>>     option = your_option_creator(option_dictionary,
            ...                                  entrypoint)
            >>>     instance = my_cluster_cli.create_instance(option)
            >>>     return instance

        self_hosted_cluster_terminator form:
            >>> def my_instance_terminator(instance):
            >>>     instance.terminate()
        """
        if not self_hosted_cluster_option:
            raise ValueError("detect self_hosted_cluster, "
                             "self_hosted_cluster_option "
                             "must need")
        if not self_hosted_cluster_option.get("self_hosted_cluster_creator") \
                or not self_hosted_cluster_option \
                .get("self_hosted_cluster_terminator") \
                or not self_hosted_cluster_option.get("dispatcher") \
                or not self_hosted_cluster_option.get("worker"):
            raise ValueError("self_hosted_cluster_creator "
                             "and self_hosted_cluster_terminator "
                             "and dispatcher "
                             "and worker must need")
        if not self_hosted_cluster_option \
                .get("dispatcher").get("InstanceType") \
                or not self_hosted_cluster_option \
                .get("worker").get("InstanceType"):
            raise ValueError("dispatcher and worker must have InstanceType")
        super().__init__(image_type=image_type,
                         self_hosted_cluster_option=self_hosted_cluster_option)


class Instance:
    def __init__(self,
                 instance,
                 instance_id
                 ):
        self._instance = instance
        self._instance_id = instance_id
        self._instance_info = None

    def terminate(self):
        raise NotImplementedError

    def get_info(self):
        return self._instance_info

    def set_info(self, info):
        self._instance_info = info

    def get_id(self):
        return str(self._instance_id)


class AWSEC2Instance(Instance):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def terminate(self):
        self._instance.terminate()


class SSHInstance(Instance):
    """
    SSHInstance for terminating process
    """

    def __init__(self, *args, **kwargs):
        self._is_running = True
        super().__init__(*args, **kwargs)

    def terminate(self):
        if not self._is_running:
            return
        command = launcher_util.build_multiline_command(
            launcher_util.SSH_PROCESS_TERMINATION_FORM)
        self._instance.exec_command(command)
        self._is_running = False


# TODO: Implement SelfHostedClusterInstance
# for autoscaling with self-hosted cluster
class SelfHostedClusterInstance(Instance):
    """
    SelfHostedClusterInstance for terminating instance and process
    """
