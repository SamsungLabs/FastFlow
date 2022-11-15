# Copyright 2022 Samsung Electronics Co., Ltd. All Rights Reserved
from collections.abc import Iterable
from enum import Enum, auto

import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops as ops


class StateMidOffload(Enum):
    FINDING_END_TARGET = auto()
    FINDING_START_TARGET = auto()
    END = auto()


class InvalidStateException(Exception):
    pass


class DatasetBuilder:
    def __init__(self, builder_func):
        self._builder_func = builder_func

    def build(self, dataset):
        return self._builder_func(target_ds=dataset)


def _get_builder(dataset):
    def prefetch_builder_func(prefetch_ds, target_ds):
        print("prefetch is being applied.")
        return target_ds.prefetch(buffer_size=prefetch_ds._buffer_size)

    def batch_builder_func(batch_ds, target_ds):
        print("batch is being applied.")
        return target_ds.batch(
            batch_size=batch_ds._batch_size,
            drop_remainder=batch_ds._drop_remainder
        )

    def padded_batch_builder_func(padded_batch_ds, target_ds):
        print("padded batch is being applied.")
        return target_ds.padded_batch(
            batch_size=padded_batch_ds._batch_size,
            drop_remainder=padded_batch_ds._drop_remainder
        )

    def parallel_map_builder_func(parallel_map_ds, target_ds):
        print("parallel map is being applied.")
        return target_ds.map(
            map_func=parallel_map_ds._map_func._func,
            num_parallel_calls=parallel_map_ds._num_parallel_calls
        )

    def repeat_builder_func(repeat_ds, target_ds):
        print("repeat is being applied.")
        return target_ds.repeat(count=repeat_ds._count)

    def shuffle_builder_func(shuffle_ds, target_ds):
        print("shuffle is being applied.")
        return target_ds.shuffle(buffer_size=shuffle_ds._buffer_size)

    from functools import partial

    if isinstance(dataset, ops.PrefetchDataset):
        print("A builder instance for a PrefechDataset is being created.")
        return DatasetBuilder(
            builder_func=partial(prefetch_builder_func, prefetch_ds=dataset)
        )
    elif isinstance(dataset, ops.BatchDataset):
        print("A builder instance for a BatchDataset is being created.")
        return DatasetBuilder(
            builder_func=partial(batch_builder_func, batch_ds=dataset)
        )
    elif isinstance(dataset, ops.PaddedBatchDataset):
        print("A builder instance for a PaddedBatchDataset is being created.")
        return DatasetBuilder(
            builder_func=partial(
                padded_batch_builder_func, padded_batch_ds=dataset)
        )
    elif isinstance(dataset, ops.ParallelMapDataset):
        print("A builder instance for a ParallelMapDataset is being created.")
        return DatasetBuilder(builder_func=partial(
            parallel_map_builder_func, parallel_map_ds=dataset
        ))
    elif isinstance(dataset, ops.RepeatDataset):
        print("A builder instance for a RepeatDataset is being created.")
        return DatasetBuilder(
            builder_func=partial(repeat_builder_func, repeat_ds=dataset)
        )
    elif isinstance(dataset, ops.ShuffleDataset):
        print("A builder instance for a ShuffleDataset is being created.")
        return DatasetBuilder(
            builder_func=partial(shuffle_builder_func, shuffle_ds=dataset)
        )
    else:
        print("The builder for the given dataset is not supported.")
        raise NotImplementedError(
            "The builder for " + type(dataset).__name__ + " is not supported.")


def p_take_cache_repeat(p):
    return p.take(1).cache().repeat()


def p_offload(p, offload_config):
    return p.apply(tf.data.experimental.service.distribute(
        processing_mode='distributed_epoch',
        service='grpc://' + offload_config.dispatcher_addr +
                ':' + str(offload_config.dispatcher_port),
        partial_offload_enabled=offload_config.partial_offload_enabled,
        ratio_local=offload_config.ratio_local,
        job_name=offload_config.job_name))


def p_take_cache_repeat_offload(p, offload_config):
    p = p.take(1).cache().repeat()
    return p.apply(tf.data.experimental.service.distribute(
        processing_mode='distributed_epoch',
        service='grpc://' + offload_config.dispatcher_addr +
                ':' + str(offload_config.dispatcher_port)))


def traverse_and_mark_first_prep(p):
    # Assume that the pipeline consists of multiple map functions (at least 2) 
    # of which the first map reads data (e.g., `tf.io.read_file` op).
    while p is not None:
        if isinstance(p, ops.ParallelMapDataset) and \
                p._metadata.name == b'prep_begin':
            return True

        if (isinstance(p, ops.ParallelMapDataset)
                and isinstance(p._input_dataset, ops.ParallelMapDataset)
                and not isinstance(
                    p._input_dataset._input_dataset, ops.ParallelMapDataset
                )):
            p._metadata.name = b'prep_begin'
            return True
        p = p._input_dataset
    return False


def traverse_and_insert_ops(p_origin,
                            p_modification_func,
                            op_targets,
                            offload_config=None):
    # Insert new operators before `op_target` in `p_origin` by applying `p_modification_func`.
    # `op_target`: `ops.PrefetchDataset` / `ops.BatchDataset` / `ops.ParallelBatchDataset` / etc.
    # TODO: Need to figure out how to represent `apply(distribute)` transformation for `op_target`.
    # Workaround: Insert `take(1).cache().repeat().apply(distribute)` together.

    def _traverse_and_insert_ops(p, is_prev_op_target, prev_ops_builders):
        if p is None:
            return None

        if is_prev_op_target:
            if offload_config is not None:
                modified_pipeline = p_modification_func(p, offload_config)
            else:
                modified_pipeline = p_modification_func(p)

            for builder in prev_ops_builders:
                modified_pipeline = builder.build(modified_pipeline)
            return modified_pipeline

        is_p_op_target = any(map(
            lambda op_target: isinstance(p, op_target), op_targets
        ))

        return _traverse_and_insert_ops(
            p._input_dataset,
            is_p_op_target,
            [_get_builder(p)] + prev_ops_builders)

    if not isinstance(op_targets, Iterable):
        op_targets = {op_targets}
    return _traverse_and_insert_ops(p_origin, False, [])


def traverse_and_insert_ops_mid_offload(p_origin,
                                        p_modify_func_start,
                                        op_target_start,
                                        p_modify_func_end,
                                        op_target_end,
                                        offload_config_start=None,
                                        offload_config_end=None):
    # pylint: disable=line-too-long
    """
    Pipeline modification for mid-offloading:
    Insert two operators to `p_origin` by sequentially applying `p_modify_func_end` and `p_modify_func_start` 
    in front of `op_target_end` and `op_target_start`, respectively (traverse from rear of the pipeline).
    Assumptions: 
    - `p_origin` contains both `op_target_start` and `op_target_end`
    - In `p_origin`, `op_target_start` (map named w/ 'prep_begin') appears before `op_target_end` (batch)
    
    :param p_origin: original input pipeline (DAG) to be modified.
    :param p_modify_func_start: modification function indicating start of mid-offloading.
    :param op_target_start: first op target before which new op w/ `p_modify_func_start` is inserted.
    :param p_modify_func_end: modification function indicating end of mid-offloading.
    :param op_target_end: second op target before which new op w/ `p_modify_func_end` is inserted.
    :param offload_config_start: offloading config used when applying p_modify_func_start
    :param offload_config_end: offloading config used when applying p_modify_func_end
    :return: Modified input pipeline
    """

    # pylint: enable=line-too-long

    def _traverse_and_insert_ops(p,
                                 state,
                                 prev_ops_builders_start,
                                 prev_ops_builders_end):
        """
        State transitions: 
        FINDING_END_TARGET
        (init) --(Found)--> FINDING_START_TARGET --(Found)--> END
        """
        if p is None:
            return None

        if state == StateMidOffload.END:
            # Now that all traversals are done, start building the new pipeline.
            modified_pipeline = p_modify_func_start(p, offload_config_start)
            for builder in prev_ops_builders_start:
                modified_pipeline = builder.build(modified_pipeline)

            modified_pipeline = p_modify_func_end(modified_pipeline,
                                                  offload_config_end)
            for builder in prev_ops_builders_end:
                modified_pipeline = builder.build(modified_pipeline)

            return modified_pipeline
        elif state == StateMidOffload.FINDING_END_TARGET:
            if any(map(lambda op_target: isinstance(p, op_target),
                       op_target_end)):
                state = StateMidOffload.FINDING_START_TARGET
            prev_ops_builders_end = [_get_builder(p)] + prev_ops_builders_end
        elif state == StateMidOffload.FINDING_START_TARGET:
            if any(map(lambda op_target: isinstance(p, op_target),
                       op_target_start)) and p._metadata.name == b'prep_begin':
                state = StateMidOffload.END
            prev_ops_builders_start = \
                [_get_builder(p)] + prev_ops_builders_start
        else:
            raise InvalidStateException('Invalid state error')

        return _traverse_and_insert_ops(
            p._input_dataset,
            state,
            prev_ops_builders_start,
            prev_ops_builders_end)

    return _traverse_and_insert_ops(p_origin,
                                    StateMidOffload.FINDING_END_TARGET,
                                    [],
                                    [])
