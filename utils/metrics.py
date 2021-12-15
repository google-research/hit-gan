# coding=utf-8
# Copyright 2021 The HiT-GAN Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific HiT-GAN governing permissions and
# limitations under the License.
# ==============================================================================
"""Training utilities."""

from absl import logging

import tensorflow as tf


def reset_metrics(metrics):
  for _, metric in metrics.items():
    metric.reset_states()


def update_metrics(metrics, **kwargs):
  for metric_name, metric_value in kwargs.items():
    metrics[metric_name].update_state(metric_value)


def log_and_write_metrics_to_summary(metrics, step):
  for _, metric in metrics.items():
    metric_value = metric.result()
    logging.info('Step: [%d] %s = %f', step.numpy(), metric.name,
                 metric_value.numpy().astype(float))
    tf.summary.scalar(metric.name, metric_value, step=step)


def log_images_to_summary(name, images, step, max_outputs):
  num_images = images.shape[0]
  splits = tf.split(images, num_or_size_splits=num_images, axis=0)
  for i in range(min(num_images, max_outputs)):
    tf.summary.image('{}/{}'.format(name, i), splits[i], step=step)
