# Copyright 2022 Samsung Electronics Co., Ltd. All Rights Reserved
import time
import traceback

from enum import Enum

import yaml

from tensorflow import keras
from tensorflow._api.v2.errors import InvalidArgumentError
from tensorflow.python.data.ops import dataset_ops as ops

from fastflow import keras_utils
from fastflow import pipeline_traverse as pt
from fastflow import utils
from fastflow.autoscaler import scaler
from fastflow.autoscaler.manager.instance_manager import create_instance_manager
from fastflow.metric_store import ProfileMetrics, ProfileMetricManager


class CandidatePipeline(Enum):
    ELEMENT = "element"
    BATCH = "batch"
    MIDDLE = "middle"


class FastFlowConfig:

    @staticmethod
    def from_yaml(yaml_path):
        """
        Convert yaml type config to FastFlowConfig.
        Yaml key must be equal to the keyword argument of FastFlow.__init__.
        e.g.) To set num_profile_steps in FastFlowConfig,
         yaml format should be num_profile_steps: 3.
        :param yaml_path: yaml path
        :return: FastFlowConfig
        """
        with open(yaml_path) as yaml_file:
            yaml_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
        return FastFlowConfig(**yaml_dict)

    def __init__(self,
                 num_profile_steps=3,
                 num_initial_steps=2,
                 verbose_level=0,
                 dispatcher_addr=None,
                 dispatcher_port='5000',
                 server_port='1515',
                 partial_offload_enabled=True,
                 ratio_local=0.0,
                 job_name=None,
                 cpu_bottleneck_threshold=1.3,
                 offloading_speed_up_threshold=1.3,
                 metric_cache_enabled=False,
                 handle_offloading_exception=True,
                 autoscale_enabled=False,
                 instance_option=None,
                 **kwargs):
        self.num_profile_steps = num_profile_steps
        self.num_initial_steps = num_initial_steps
        self.verbose_level = verbose_level
        self.dispatcher_addr = dispatcher_addr
        self.dispatcher_port = dispatcher_port
        self.server_port = server_port
        self.partial_offload_enabled = partial_offload_enabled
        self.ratio_local = ratio_local
        self.job_name = job_name
        self.cpu_bottleneck_threshold = cpu_bottleneck_threshold
        self.offloading_speed_up_threshold = offloading_speed_up_threshold
        self.metric_cache_enabled = metric_cache_enabled
        self.handle_offloading_exception = handle_offloading_exception
        self.autoscale_enabled = autoscale_enabled
        self.instance_option = instance_option


class FastFlowModel(keras.models.Model):
    def __init__(self, *args, **kwargs):
        super(FastFlowModel, self).__init__(*args, **kwargs)

    def compile(self, *args, **kwargs):
        super(FastFlowModel, self).compile(*args, **kwargs)
        print(args, kwargs)
        self.compile_args = args
        self.compile_kwargs = kwargs

    def fit(self,
            auto_offload_conf=None,
            **kwargs):

        if auto_offload_conf is None:
            # No offloading
            return super(FastFlowModel, self).fit(**kwargs)

        kwargs['callbacks'] = \
            kwargs.get('callbacks', []) + [utils.EpochTimeCallback()]

        prep_pipeline = kwargs.get('x')
        prep_pipeline_valid = kwargs.get('validation_data')

        profile_time_start = time.time()

        # Prepare a fresh model
        with self.distribute_strategy.scope():
            model = self.__deepcopy__()
            print(self.compile_args)
            print(self.compile_kwargs)
            model.compile(*self.compile_args, **self.compile_kwargs)
            print(model._compiled_trainable_state)

        # Get Cached Object
        metric_manager = ProfileMetricManager(prep_pipeline,
                                              model,
                                              auto_offload_conf)

        # Try a dummy training to remove model compilation overhead from the measurement
        if not metric_manager.is_all_metrics_cached():
            print("0. Dummy training")
            super(FastFlowModel, model).fit(
                prep_pipeline,
                epochs=1,
                steps_per_epoch=1,
                # validation_data=prep_pipeline_valid,
                validation_steps=1)
        else:
            print("0. Skipping Dummy training")

        # Measure the training & epoch time with the original version of input pipeline
        # 1. lthp
        lthp, lthp_cycles = metric_manager.get_or_profile_metric(
            ProfileMetrics.LTHP
        )
        # 2. gthp
        gthp, _ = metric_manager.get_or_profile_metric(
            ProfileMetrics.GTHP
        )

        speed_up = gthp / lthp
        has_bottleneck = speed_up > auto_offload_conf.cpu_bottleneck_threshold
        print("Does this app have a cpu bottleneck? ",
              'Yes' if has_bottleneck else 'No')

        if not has_bottleneck or auto_offload_conf.dispatcher_addr is None:
            # End of metric profiling.
            profile_time = time.time() - profile_time_start
            print(f"lthp: {lthp}")
            print(f"gthp: {gthp}")
            print(f"profile time: {profile_time}")

            metric_manager.write_all_metrics()
            return super(FastFlowModel, self).fit(**kwargs)

        # Start metric profiling of pcycle, dcycle, and rthp
        # and estimate offloading ratio

        # 3. Estimate pcycle
        pcycle = lthp_cycles

        instance_manager = create_instance_manager(auto_offload_conf)
        dispatcher = instance_manager.launch_and_get_dispatcher()

        dispatcher_address = \
            dispatcher.get_dispatcher_address_or_wait_until_ready()

        instance_manager.launch_workers(n_workers=1)
        auto_offload_conf.dispatcher_addr = \
            dispatcher_address.split(":")[0]
        auto_offload_conf.dispatcher_port = \
            dispatcher_address.split(":")[1]

        instance_manager.wait_until_workers_ready()  # TODO: set timeout

        local_worker = utils.start_local_worker(auto_offload_conf)

        try:
            # 4. rthp
            rthp = {}

            rthp[CandidatePipeline.ELEMENT], rthp_elem_cycles = \
                metric_manager.get_or_profile_metric(
                    ProfileMetrics.RTHP
                )

            rthp[CandidatePipeline.BATCH], rthp_batch_cycles = \
                metric_manager.get_or_profile_metric(
                    ProfileMetrics.RTHP_BATCH
                )

            ((rthp[CandidatePipeline.MIDDLE], rthp_mid_cycles),
             config_mid_local,
             config_mid_remote) = metric_manager.get_or_profile_metric(
                ProfileMetrics.RTHP_MID
            )

            print(f"rthp of candidate pipelines (original): {rthp}")

            local_overhead_elem = lthp * (1 - rthp_elem_cycles / lthp_cycles)
            local_overhead_batch = lthp * (1 - rthp_batch_cycles / lthp_cycles)
            local_overhead_mid = lthp * (1 - rthp_mid_cycles / lthp_cycles)
            rthp[CandidatePipeline.ELEMENT] += local_overhead_elem
            rthp[CandidatePipeline.BATCH] += local_overhead_batch
            rthp[CandidatePipeline.MIDDLE] += local_overhead_mid

            print(f"rthp of candidate pipelines (calibrated): {rthp}")

            pipeline_optimal = max(rthp, key=rthp.get)
            rthp[CandidatePipeline.ELEMENT] -= local_overhead_elem
            rthp[CandidatePipeline.BATCH] -= local_overhead_batch
            rthp[CandidatePipeline.MIDDLE] -= local_overhead_mid

            rthp_optimal = rthp[pipeline_optimal]
            print(f"Optimal pipeline: {pipeline_optimal}")

            # 5. dcycle
            if pipeline_optimal == CandidatePipeline.ELEMENT:
                dcycle = rthp_elem_cycles
            elif pipeline_optimal == CandidatePipeline.BATCH:
                dcycle = rthp_batch_cycles
            else:
                dcycle = rthp_mid_cycles
        except (NotImplementedError, InvalidArgumentError) as error:
            print("Failed to measure stats with " +
                  type(error).__name__ + " error. " +
                  "Error message: " + str(error))
            traceback.print_exc()
            instance_manager.shutdown_all()
            if auto_offload_conf.handle_offloading_exception:
                return super(FastFlowModel, self).fit(**kwargs)
            else:
                raise error

        print(f"lthp: {lthp}")
        print(f"gthp: {gthp}")
        print(f"rthp: {rthp_optimal}")
        print(f"pcycle: {pcycle}")
        print(f"dcycle: {dcycle}")

        # Initial scaling
        if auto_offload_conf.autoscale_enabled:
            init_num_workers = \
                scaler.estimate_initial_workers(lthp,
                                                gthp,
                                                rthp_optimal)
        else:
            init_num_workers = 0
        instance_manager.launch_workers(n_workers=init_num_workers)
        # TODO: set timeout and save current workers
        instance_manager.wait_until_workers_and_local_worker_ready(
            local_worker
        )

        # TODO: fix ratio based on initial scaling
        # TODO: fix ratio considering mid-offloading
        # "init_num_workers + 1" means
        # "initial number of workers after Rthp profiling + first worker created by instance manager"
        ratio_local_optimal = utils.calculate_ratio_local_optimal(
            gthp, lthp, rthp_optimal, pcycle, dcycle, init_num_workers + 1)

        profile_time = time.time() - profile_time_start
        print(f"profile time: {profile_time}")
        print(f"ratio_local_optimal: {ratio_local_optimal}")
        # Create an offloading pipeline with `partial_offload_enabled=True`
        # and `ratio_local=ratio_local_optimal`.
        if ratio_local_optimal >= 0.0:
            print("6. Actual training")
            ops_batch_ds = {ops.BatchDataset, ops.PaddedBatchDataset}
            if pipeline_optimal is not CandidatePipeline.MIDDLE:
                target_op = (ops_batch_ds
                             if pipeline_optimal is CandidatePipeline.ELEMENT
                             else ops.PrefetchDataset)
                auto_offload_conf.ratio_local = ratio_local_optimal
                kwargs['x'] = pt.traverse_and_insert_ops(kwargs.get('x'),
                                                         pt.p_offload,
                                                         target_op,
                                                         auto_offload_conf)
                kwargs['validation_data'] = pt.traverse_and_insert_ops(
                    kwargs.get('validation_data'),
                    pt.p_offload,
                    target_op,
                    auto_offload_conf)
            else:
                job_name = "mid-offload-train-" + time.strftime("%Y%m%d-%H%M%S")
                config_mid_local.job_name = job_name
                config_mid_remote.job_name = job_name
                config_mid_remote.ratio_local = ratio_local_optimal
                pipeline_train = kwargs.get('x')
                kwargs['x'] = pt.traverse_and_insert_ops_mid_offload(
                    pipeline_train,
                    pt.p_offload,
                    {ops.ParallelMapDataset, ops.ShuffleDataset},
                    pt.p_offload,
                    ops_batch_ds,
                    config_mid_local,
                    config_mid_remote)
                pipeline_validation = kwargs.get('validation_data')
                kwargs['validation_data'] = \
                    pt.traverse_and_insert_ops_mid_offload(
                        pipeline_validation,
                        pt.p_offload,
                        {ops.ParallelMapDataset, ops.ShuffleDataset},
                        pt.p_offload,
                        ops_batch_ds,
                        config_mid_local,
                        config_mid_remote
                    )

        metric_manager.write_all_metrics()

        # TODO: scaler implementation
        # Scaler will automatically increase/decrease workers at runtime
        autoscaler = scaler.AutoScaler(instance_manager)
        # based on the metrics provided by the callback
        auto_scaling_callback = scaler.AutoScalingCallback(autoscaler)
        try:
            return keras_utils.fit(
                self,
                iterator_callback=auto_scaling_callback,
                **kwargs
            )
        except Exception as error:
            print("Failed to training..." + type(error).__name__ +
                  " error. " + "Error message: " + str(error))
            traceback.print_exc()
            raise error
        finally:
            local_worker._stop()  # pylint: disable=protected-access
            instance_manager.shutdown_all()
