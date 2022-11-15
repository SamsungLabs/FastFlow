# Copyright 2022 Samsung Electronics Co., Ltd. All Rights Reserved
import argparse

from fastflow.autoscaler.manager.tf_data_service_manager import TFDataServiceManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uuid_token",
                        type=str,
                        required=True,
                        help="Specify UUID_TOKEN for host Server")
    parser.add_argument("--server_address",
                        type=str,
                        required=True,
                        help="Specify '<ADDRESS>:<PORT>' for host Server")
    parser.add_argument("--instance_id",
                        type=str,
                        required=True,
                        help="Specify INSTANCE_ID "
                             "i.e private_address, ec2 instance id")
    parser.add_argument("--tf_data_service_type",
                        type=str,
                        required=True,
                        choices=["dispatcher", "worker"],
                        help="Specify TF_DATA_SERVICE_TYPE "
                             "dispatcher or worker")
    parser.add_argument("--dispatcher_address",
                        type=str,
                        required=False,
                        default="",
                        help="Specify DISPATCHER_ADDRESS for worker "
                             "[default: empty string]")
    parser.add_argument('port', action='store',
                        default=5000, type=int,
                        nargs='?',
                        help='Specify alternate port [default: 5000]')
    args = parser.parse_args()
    tf_data_service_manager = TFDataServiceManager(args.uuid_token,
                                                   args.server_address,
                                                   args.instance_id,
                                                   args.tf_data_service_type,
                                                   args.dispatcher_address,
                                                   args.port)
    tf_data_service_manager.watch()
