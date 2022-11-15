#!/bin/bash

python -m grpc_tools.protoc -Ifastflow/autoscaler/protobufs \
                            --python_out=fastflow/autoscaler/framework/grpc \
                            fastflow/autoscaler/protobufs/*.proto
