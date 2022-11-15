# Copyright 2022 Samsung Electronics Co., Ltd. All Rights Reserved
import os
import subprocess
import time

import psutil

import tensorflow as tf

from tensorflow import keras

from fastflow import keras_utils


class IteratorCallback(keras_utils.IteratorCallback):
    def __init__(self, num_initial_steps=0):
        self.thp = 0
        self.initial_total_bytes = 0
        self.initial_total_processing_time = 0
        self._num_initial_steps = num_initial_steps

    def on_train_batch_end(self, step, iterator):
        if step == self._num_initial_steps:
            # Skip initial data processing
            self.initial_total_bytes = iterator.get_bytes()
            self.initial_total_processing_time = iterator.get_processing_time()

    def on_epoch_end(self, epoch, iterator):
        # Calculate thp
        self.thp = (
                (iterator.get_bytes()
                 - self.initial_total_bytes)
                / (iterator.get_processing_time()
                   - self.initial_total_processing_time)
        )

    def on_test_batch_begin(self, step, iterator):
        pass

    def on_test_batch_end(self, step, iterator):
        pass

    def get_thp(self):
        return self.thp


class EpochTimeCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"{epoch + 1} Epoch Begin. ({time.time()})")

    def on_epoch_end(self, epoch, logs=None):
        print(f"\n{epoch + 1} Epoch End. ({time.time()})")


def calculate_ratio_local_optimal(gthp,
                                  lthp,
                                  rthp,
                                  pcycle,
                                  dcycle,
                                  num_workers):
    dc_ratio = float(dcycle) / float(pcycle)
    ideal_off = gthp - lthp
    decode_off = ideal_off + ideal_off * dc_ratio
    if lthp > rthp * num_workers:
        off_data = (rthp * num_workers + decode_off) / 2
    else:
        off_data = max(rthp * num_workers, decode_off)
    if lthp + rthp * num_workers < gthp:
        ratio_local_optimal = lthp / (lthp + rthp * num_workers)
    else:
        ratio_local_optimal = (gthp - off_data) / gthp
    if ratio_local_optimal < 0.0:
        return 0.0
    return ratio_local_optimal


def estimate_thp(prep_pipeline,
                 model,
                 config):
    iterator_callback = IteratorCallback(config.num_initial_steps)
    pid = os.getpid()
    current_process = psutil.Process(pid)
    cpu_begin = current_process.cpu_times()
    keras_utils.fit(
        model,
        prep_pipeline,
        epochs=1,
        steps_per_epoch=config.num_profile_steps,
        # validation_data=prep_pipeline_valid,
        # validation_steps=auto_offload_config.num_steps,
        iterator_callback=iterator_callback)
    cpu_end = current_process.cpu_times()
    thp = iterator_callback.get_thp()
    return thp, cpu_end.user - cpu_begin.user


def estimate_cycles(prep_pipeline, config):
    it = iter(prep_pipeline)
    # Skip `config.num_initial_steps` and start measuring cpu cycles
    for step in range(0, config.num_initial_steps):
        elem = it.get_next()

    start = time.process_time_ns()
    for step in range(config.num_initial_steps, config.num_profile_steps):
        elem = it.get_next()

    return time.process_time_ns() - start


class DummyLocalWorker:
    def _stop(self):
        pass


def start_local_worker(config):
    if not config.autoscale_enabled:
        return DummyLocalWorker()

    # TODO: use bind_port_safety
    w_config = tf.data.experimental.service.WorkerConfig(
        dispatcher_address=config.dispatcher_addr +
                           ':' + str(config.dispatcher_port),
        port=5001)

    worker = tf.data.experimental.service.WorkerServer(w_config)
    print("Started local worker")
    return worker


def get_address():
    return subprocess.check_output(["hostname", "-I"]).strip().decode("utf-8")
