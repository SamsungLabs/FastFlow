# Copyright 2022 Samsung Electronics Co., Ltd. All Rights Reserved
"""
Code related to MetricStore that saves and loads profiled metrics.
"""
import hashlib
import os
import pickle
import time

from enum import Enum, auto

from tensorflow.core.framework import types_pb2
from tensorflow.python.data.ops import dataset_ops as ops
from tensorflow.python.keras.engine.training_utils_v1 import get_dataset_graph_def

import fastflow as ff
from fastflow import utils
from fastflow import pipeline_traverse as pt


class ProfileMetrics(Enum):
    LTHP = auto()
    GTHP = auto()
    RTHP = auto()
    RTHP_BATCH = auto()
    RTHP_MID = auto()


class ProfileMetricManager:

    def __init__(self, pipeline, model, auto_offload_conf):
        self.metric_store = MetricStoreFactory.get_instance(
            model,
            pipeline,
            auto_offload_conf.metric_cache_enabled,
            auto_offload_conf.dispatcher_addr)

        self.pipeline = pipeline
        self.model = model
        self.auto_offload_conf = auto_offload_conf
        self.metric_profile_func_map = \
            {
                ProfileMetrics.LTHP: self._profile_lthp,
                ProfileMetrics.GTHP: self._profile_gthp,
                ProfileMetrics.RTHP: self._profile_rthp,
                ProfileMetrics.RTHP_BATCH: self._profile_rthp_batch,
                ProfileMetrics.RTHP_MID: self._profile_rthp_mid,
            }

    def is_all_metrics_cached(self):
        return self.metric_store.is_fully_cached()

    def write_all_metrics(self):
        self.metric_store.write_all_metrics()

    def get_or_profile_metric(self, metric_type):
        if not self.metric_store.get_profile_item(metric_type):
            print(f"Measure {metric_type}")
            profile_func = self.metric_profile_func_map.get(metric_type)
            metric = profile_func()
            self.metric_store.set_profile_item(metric_type, metric)
            return metric
        print(f"Skipping Measuring {metric_type}")
        return self.metric_store.get_profile_item(metric_type)

    def _profile_lthp(self):
        return utils.estimate_thp(
            self.pipeline,
            self.model,
            self.auto_offload_conf
        )

    def _profile_gthp(self):
        # Insert `take(1).cache().repeat()` before `prefetch()` to measure
        # the training & epoch time with the cached version of input pipeline.
        m_pipeline = pt.traverse_and_insert_ops(
            self.pipeline,
            pt.p_take_cache_repeat,
            {ops.PrefetchDataset})

        return utils.estimate_thp(
            m_pipeline,
            self.model,
            self.auto_offload_conf
        )

    def _profile_rthp(self):
        # Create an offloading pipeline:
        # Insert `apply(distribute)` before `batch`-`prefetch`.
        ops_batch_ds = {ops.BatchDataset, ops.PaddedBatchDataset}
        m_pipeline = pt.traverse_and_insert_ops(
            self.pipeline,
            pt.p_offload,
            ops_batch_ds,
            self.auto_offload_conf)

        return utils.estimate_thp(
            m_pipeline,
            self.model,
            self.auto_offload_conf
        )

    def _profile_rthp_batch(self):
        # Create an offloading pipeline:
        # Insert `apply(distribute)` before `prefetch`.
        m_pipeline = pt.traverse_and_insert_ops(
            self.pipeline,
            pt.p_offload,
            ops.PrefetchDataset,
            self.auto_offload_conf)

        return utils.estimate_thp(
            m_pipeline,
            self.model,
            self.auto_offload_conf
        )

    def _profile_rthp_mid(self):
        job_name = "mid-offload-profile-" + time.strftime("%Y%m%d-%H%M%S")
        config_mid_local = ff.FastFlowConfig(
            num_profile_steps=self.auto_offload_conf.num_profile_steps,
            num_initial_steps=self.auto_offload_conf.num_initial_steps,
            dispatcher_addr=utils.get_address(),
            partial_offload_enabled=False,
            job_name=job_name)
        config_mid_remote = ff.FastFlowConfig(
            num_profile_steps=self.auto_offload_conf.num_profile_steps,
            num_initial_steps=self.auto_offload_conf.num_initial_steps,
            dispatcher_addr=self.auto_offload_conf.dispatcher_addr,
            dispatcher_port=self.auto_offload_conf.dispatcher_port,
            partial_offload_enabled=True,
            job_name=job_name)
        ops_batch_ds = {ops.BatchDataset, ops.PaddedBatchDataset}
        m_pipeline = pt.traverse_and_insert_ops_mid_offload(
            self.pipeline,
            pt.p_offload,
            {ops.ParallelMapDataset, ops.ShuffleDataset},
            pt.p_offload,
            ops_batch_ds,
            config_mid_local,
            config_mid_remote)
        return utils.estimate_thp(
            m_pipeline,
            self.model,
            self.auto_offload_conf
        ), config_mid_local, config_mid_remote


class MetricStoreFactory:
    @classmethod
    def get_instance(cls,
                     model,
                     pipeline,
                     metric_cache_enabled,
                     dispatcher_addr):
        # Get Cached Object
        if metric_cache_enabled:
            return _FileSystemMetricStore(model, pipeline,
                                          dispatcher_addr)
        return _DummyMetricStore(model, pipeline, dispatcher_addr)


class _MetricStore:
    METRICS = {ProfileMetrics.GTHP,
               ProfileMetrics.LTHP,
               ProfileMetrics.RTHP,
               ProfileMetrics.RTHP_BATCH,
               ProfileMetrics.RTHP_MID}

    def __init__(self, model, pipeline, nodeinfo):
        if _is_run_eagerly(model):
            # Extract model info in eagerly mode.
            # when eagerly mode is enabled, TF does not compile the model
            # and does not generate tf nodes.
            #
            # To extract model info and TF nodes, we copyand compile the model
            # by compile function
            eager_model = model.__deepcopy__()
            eager_compile_args = list(model.compile_args)
            eager_compile_kwargs = model.compile_kwargs.copy()
            # Change run_eagerly to False
            if len(eager_compile_args) > 5:
                eager_compile_args[5] = False
            else:
                eager_compile_kwargs["run_eagerly"] = False
            eager_compile_args = tuple(eager_compile_args)
            eager_model.compile(*eager_compile_args, **eager_compile_kwargs)
            model = eager_model
        self.model = model
        self.pipeline = pipeline
        self.nodeinfo = nodeinfo
        self.profile_dict = {}
        self.cached_model_path = None
        self.cached_pipeline_path = None
        self.cached_model_pipeline_path = None
        self.cached_model_pipeline_nodeinfo_path = None
        self._init_cached_model_path()
        self._init_cached_pipeline_path()
        self._init_cached_model_pipeline_path()
        self._init_cached_model_pipeline_nodeinfo_path()
        self._batch_size_decorator()

    def is_fully_cached(self):
        return self.METRICS.issubset(set(self.profile_dict.keys()))

    def get_profile_item(self, key):
        return self.profile_dict.get(key)

    def set_profile_item(self, key, item):
        self.profile_dict[key] = item

    def write_all_metrics(self):
        raise NotImplementedError

    def _init_cached_model_path(self):
        self.model.make_train_function()
        model_graph_def = self.model.train_function.get_concrete_function(
            iter(self.pipeline)).graph.as_graph_def()

        _remove_unimportant_attr_model_graph(model_graph_def)

        library_list = [
            (fn.signature.name, fn)
            for fn in model_graph_def.library.function
        ]

        node_def_list_string = b''.join(
            bytes(str(nd), 'utf-8') for nd in model_graph_def.node
        )
        # replace inference to function_def for deterministic
        for key, fn in library_list:
            fn.signature.name = ""
            node_def_list_string = node_def_list_string.replace(
                bytes(key, 'utf-8'),
                bytes(str(fn), 'utf-8')
            )
        self.cached_model_path = \
            hashlib.sha512(node_def_list_string).hexdigest()

    def _init_cached_pipeline_path(self):
        pipeline_graph_def = get_dataset_graph_def(self.pipeline)

        _remove_unimportant_attr_pipeline_graph(pipeline_graph_def)

        node_def_list_string = b''.join(
            bytes(str(nd), 'utf-8') for nd in pipeline_graph_def.node
        )
        # replace inference to function_def for deterministic
        library_list = [
            (fn.signature.name, fn)
            for fn in pipeline_graph_def.library.function
        ]
        for key, fn in library_list:
            fn.signature.name = ""
            node_def_list_string = node_def_list_string.replace(
                bytes(key, 'utf-8'),
                bytes(str(fn), 'utf-8')
            )
        for nd in pipeline_graph_def.node:
            if nd.op in {"MutableHashTableOfTensorsV2",
                         "MutableHashTableV2",
                         "HashTableV2"}:
                # Clear HashTableName for deterministic graph def
                node_def_list_string = node_def_list_string.replace(
                    bytes(nd.name, 'utf-8'),
                    b''
                )

        self.cached_pipeline_path = hashlib.sha512(
            node_def_list_string
        ).hexdigest()

    def _init_cached_model_pipeline_path(self):
        self.cached_model_pipeline_path = hashlib.sha512(
            bytes(self.cached_model_path + self.cached_pipeline_path, 'utf-8')
        ).hexdigest()

    def _init_cached_model_pipeline_nodeinfo_path(self):
        self.cached_model_pipeline_nodeinfo_path = hashlib.sha512(
            bytes(self.cached_model_pipeline_path + self.nodeinfo, 'utf-8')
        ).hexdigest()

    def _batch_size_decorator(self):
        # TODO: Batch Size Decorating for model and pipeline caching
        pass


# pylint: disable=unused-argument
class _DummyMetricStore:
    def __init__(self, *args, **kwargs):
        pass

    def is_fully_cached(self, *args, **kwargs):
        pass

    def get_profile_item(self, *args, **kwargs):
        pass

    def set_profile_item(self, *args, **kwargs):
        pass

    def write_all_metrics(self, *args, **kwargs):
        pass


# pylint: enable=unused-argument


class _FileSystemMetricStore(_MetricStore):
    def __init__(self, model, pipeline, nodeinfo):
        super().__init__(model, pipeline, nodeinfo)
        self.root_path = "/tmp/" + hashlib.sha512(
            bytes(os.uname().nodename, 'utf-8')).hexdigest() + "/"
        self.cached_model_path = self.root_path + self.cached_model_path
        self.cached_pipeline_path = self.root_path + self.cached_pipeline_path
        self.cached_model_pipeline_path = (
                self.root_path + self.cached_model_pipeline_path
        )
        self.cached_model_pipeline_nodeinfo_path = (
                self.root_path + self.cached_model_pipeline_nodeinfo_path
        )
        self._load_profile()

    def write_all_metrics(self):
        def _check_val_and_write(key, vals):
            """
            Do not write if the val list has none value.
            :param key: path
            :param vals: list
            """
            if all(map(lambda v: v is not None, vals)):
                _pickle_write(key, vals)

        # Remove model from callback object for saving with pickle
        # model should be not used
        gthp = self.get_profile_item(ProfileMetrics.GTHP)
        _check_val_and_write(self.cached_model_path, [gthp])

        lthp = self.get_profile_item(ProfileMetrics.LTHP)
        _check_val_and_write(self.cached_model_pipeline_path, [lthp])

        rthp_metric_list = [ProfileMetrics.RTHP,
                            ProfileMetrics.RTHP_BATCH,
                            ProfileMetrics.RTHP_MID]
        rthp_metric_list = [self.get_profile_item(rthp_metric)
                            for rthp_metric in rthp_metric_list]
        _check_val_and_write(self.cached_model_pipeline_nodeinfo_path,
                             rthp_metric_list)

    def _load_profile(self):
        not_cached_error_tuple = (
            pickle.UnpicklingError,
            AttributeError,
            EOFError,
            ImportError,
            IndexError,
            ValueError
        )

        try:
            print("Get Gthp from cached model")
            gthp, = _pickle_read(self.cached_model_path)
            self.profile_dict.update({ProfileMetrics.GTHP: gthp})
        except FileNotFoundError as _:
            print("Model did not be cached")
        except not_cached_error_tuple as _:
            print("Cached model cannot be read")

        try:
            print("Get lthp cached model-pipeline")
            lthp, = _pickle_read(self.cached_model_pipeline_path)
            self.profile_dict.update({ProfileMetrics.LTHP: lthp})
        except FileNotFoundError as _:
            print("Model-pipeline did not be cached")
        except not_cached_error_tuple as _:
            print("Cached model-pipeline cannot be read")

        try:
            print("Get rthp cached model-pipeline-nodeinfo")
            (rthp,
             rthp_batch,
             rthp_mid) = _pickle_read(self.cached_model_pipeline_nodeinfo_path)
            self.profile_dict.update({
                                         ProfileMetrics.RTHP: rthp,
                                         ProfileMetrics.RTHP_BATCH: rthp_batch,
                                         ProfileMetrics.RTHP_MID: rthp_mid
                                     })
        except FileNotFoundError as _:
            print("Model-pipeline-nodeinfo did not be cached")
        except not_cached_error_tuple as _:
            print("Cached model-pipeline-nodeinfo cannot be read")


def _pickle_read(path):
    with open(path, "rb") as pickle_file:
        return pickle.load(pickle_file)


def _pickle_write(path, obj):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, "wb") as pickle_file:
        pickle.dump(obj, pickle_file)


def _clear_floating_point(attr):
    # Integer content (ie. Window size, Tensor shapes, Strides, etc..)
    # affect thp but Floating point content does not affect thp
    if attr.tensor is not None:
        attr.tensor.ClearField("scomplex_val")
        attr.tensor.ClearField("float_val")
        attr.tensor.ClearField("double_val")
    if attr.f is not None:
        attr.ClearField("f")


def _clear_floating_point_from_node_def(node_def):
    for nd in node_def:
        # Clear floating point that unaffect thp
        if hasattr(nd, "attr"):
            for key in nd.attr:
                _clear_floating_point(nd.attr[key])


def _remove_unimportant_attr_model_graph(model_graph_def):
    # Graph Node always guarantee deterministic
    # (if compiler guarantee deterministic)
    for nd in model_graph_def.node:
        if hasattr(nd, "attr"):
            # Clear floating point that unaffect thp
            for key in nd.attr:
                _clear_floating_point(nd.attr[key])
                if key == "_gradient_op_type":
                    # Clear gradient op type for deterministic graph def
                    nd.attr[key].ClearField("s")

    for fn in model_graph_def.library.function:
        _clear_floating_point_from_node_def(fn.node_def)
        if hasattr(fn, "attr"):
            for key in fn.attr:
                if key in {"api_implements",
                           "forward_function_name",
                           "backward_function_name"}:
                    # Clear api implements id, forward function name,
                    # and backward_function_name that unaffect thp
                    fn.attr[key].ClearField("s")


def _remove_unimportant_attr_pipeline_graph(pipeline_graph_def):
    # remove tensor_content if it is in-memory tensor float, sorting another type
    # Hash table remove because it does not affect thp
    for nd in pipeline_graph_def.node:
        if hasattr(nd, "attr"):
            for key in nd.attr:
                # Clear floating point that unaffect thp
                _clear_floating_point(nd.attr[key])
                if nd.attr[key].tensor is not None:
                    if nd.op == "Const" and nd.attr[key].tensor is not None:
                        if nd.attr[key].tensor.dtype in {types_pb2.DT_FLOAT,
                                                         types_pb2.DT_DOUBLE,
                                                         types_pb2.DT_BFLOAT16,
                                                         types_pb2.DT_STRING}:
                            # String content is not necessary compute thp
                            # Clear tensor content that unaffect thp
                            nd.attr[key].tensor.ClearField("tensor_content")
                        elif nd.attr[key].tensor.tensor_content is not None:
                            # sorted another contents info for deterministic
                            nd.attr[key].tensor.tensor_content = bytes(
                                sorted(nd.attr[key].tensor.tensor_content)
                            )

    for fn in pipeline_graph_def.library.function:
        _clear_floating_point_from_node_def(fn.node_def)


def _is_run_eagerly(model):
    return model.compile_kwargs.get("run_eagerly") is True \
           or (len(model.compile_args) > 5 and model.compile_args[5] is True)
