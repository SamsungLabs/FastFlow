# Copyright 2022 Samsung Electronics Co., Ltd. All Rights Reserved
import math

from enum import Enum, auto

from fastflow import keras_utils


def estimate_initial_workers(lthp, gthp, rthp, threshold=0.1):
    return math.ceil((gthp - lthp) / rthp - threshold)


class AutoScalingCallback(keras_utils.IteratorCallback):
    def __init__(self, autoscaler, step_period=100):
        self.autoscaler = autoscaler
        self.step_period = step_period
        self.prev_total_bytes = 0
        self.prev_total_processing_time = 0

    def on_train_batch_end(self, step, iterator):
        if step % self.step_period == 0:
            curr_bytes = iterator.get_bytes()
            curr_time = iterator.get_processing_time()

            # Notify the current thp every step_period
            self.autoscaler.notify_current_thp(
                self._calculate_thp(curr_bytes, curr_time)
            )

            self.prev_total_bytes = curr_bytes
            self.prev_total_processing_time = curr_time

    def _calculate_thp(self, curr_bytes, curr_time):
        return (curr_bytes - self.prev_total_bytes) / \
               (curr_time - self.prev_total_processing_time)

    def on_epoch_end(self, epoch, iterator):
        pass

    def on_test_batch_begin(self, step, iterator):
        pass

    def on_test_batch_end(self, step, iterator):
        pass

    def get_thp(self):
        raise NotImplementedError


class ScalingDecision(Enum):
    SCALE_UP = auto()
    SCALE_DOWN = auto()


class AutoScaler:
    def __init__(self, instance_manager):
        self.instance_manager = instance_manager
        self.prev_thp = 0
        self.prev_decision = ScalingDecision.SCALE_UP

    def _change_action(self):
        if self.prev_decision is ScalingDecision.SCALE_UP:
            return ScalingDecision.SCALE_DOWN
        else:
            return ScalingDecision.SCALE_UP

    # TODO: change logic
    def notify_current_thp(self, thp):
        if thp * 1.1 < self.prev_thp:
            # scale up or down
            decision = self._change_action()
        elif thp > self.prev_thp * 1.1:
            # keep decision
            decision = self.prev_decision
        else:
            # Do nothing
            self.prev_thp = thp
            return

        # create or destroy nodes through the launcher
        self.scaling_action(decision)
        self.prev_decision = decision
        self.prev_thp = thp

    def scaling_action(self, decision):
        # TODO: scaling
        pass
