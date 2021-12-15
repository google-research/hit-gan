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
"""Data preprocessing pipeline functions."""

import functools
from typing import Any, Text

from absl import logging
from utils import data_utils

import tensorflow as tf


def preprocess_image(image: tf.Tensor,
                     crop_size: int,
                     aspect_ratio: float = 1.0,
                     crop_mode: Text = 'fixed',
                     crop_proportion: float = 1.0,
                     method: Text = tf.image.ResizeMethod.BICUBIC,
                     flip: bool = False,
                     normalize: bool = True,
                     antialias: bool = True) -> tf.Tensor:
  """Preprocesses the given image.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    crop_size: Width of output image.
    aspect_ratio: The ratio of image width/height.
    crop_mode: Mode for cropping images, including 'fixed', 'random_crop', and
      'center_crop'.
    crop_proportion: Proportion of image to retain along the less-cropped side.
    method: Resize method for image processing.
    flip: Whether or not to flip left and right of an image.
    normalize: Whether to normalize the output image.
    antialias: Whether to use an anti-aliasing filter.

  Returns:
    A preprocessed image `Tensor` of range [-1, 1] if normalized.
  """
  crop_width = crop_size
  crop_height = round(crop_size / aspect_ratio)

  if crop_mode == 'fixed':
    resized = tf.image.resize(
        image, (crop_height, crop_width), method=method, antialias=antialias)
  elif crop_mode == 'center_crop':
    resized = data_utils.center_crop(
        image,
        crop_height,
        crop_width,
        crop_proportion=crop_proportion,
        method=method,
        antialias=antialias)
  elif crop_mode == 'random_crop':
    resized = data_utils.random_crop(
        image,
        crop_height,
        crop_width,
        crop_proportion=crop_proportion,
        method=method,
        antialias=antialias)
  else:
    raise NotImplementedError

  if flip:
    resized = tf.image.random_flip_left_right(resized)

  resized = tf.cast(resized, dtype=image.dtype)
  if normalize:
    resized = tf.image.convert_image_dtype(resized, dtype=tf.float32)
    resized = (resized - 0.5) / 0.5
  return resized


def get_preprocess_fn(crop_size: int,
                      aspect_ratio: float = 1.0,
                      crop_mode: Text = 'fixed',
                      crop_proportion: float = 1.0,
                      method: Text = tf.image.ResizeMethod.BICUBIC,
                      flip: bool = False,
                      normalize: bool = True):
  """Gets function that accepts an image and returns a preprocessed image."""
  return functools.partial(
      preprocess_image,
      crop_size=crop_size,
      aspect_ratio=aspect_ratio,
      crop_mode=crop_mode,
      crop_proportion=crop_proportion,
      method=method,
      flip=flip,
      normalize=normalize)


def build_distributed_dataset(
    builder, strategy: tf.distribute.Strategy,
    **kwargs: Any) -> tf.distribute.DistributedDataset:
  """Builds the distributed dataset."""
  input_fn = build_input_fn(builder, **kwargs)
  return strategy.experimental_distribute_datasets_from_function(input_fn)


def build_input_fn(builder,
                   global_batch_size: int,
                   image_crop_size: int,
                   image_aspect_ratio: int,
                   image_crop_proportion: int,
                   random_flip: bool,
                   training: bool = False,
                   cache: bool = False):
  """Builds input function.

  Args:
    builder: Dataset builder for the specified dataset.
    global_batch_size: Global batch size.
    image_crop_size: Width of output image.
    image_aspect_ratio: The ratio of image width/height.
    image_crop_proportion: Proportion of image to retain along the less-cropped
      side.
    random_flip: Whether or not to flip left and right of an image.
    training: Whether the data are used for training.
    cache: Whether to cache the elements in this dataset.

  Returns:
    A function that accepts a dict of params and returns a tuple of images and
    features.
  """

  def input_fn(input_context):
    """Inner input function."""
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    logging.info('Global batch size: %d', global_batch_size)
    logging.info('Per-replica batch size: %d', batch_size)

    dataset = builder.get_dataset(training=training)
    logging.info('num_input_pipelines: %d', input_context.num_input_pipelines)
    # The dataset is always sharded by number of hosts.
    # num_input_pipelines is the number of hosts rather than number of cores.
    if input_context.num_input_pipelines > 1:
      dataset = dataset.shard(input_context.num_input_pipelines,
                              input_context.input_pipeline_id)
    if cache:
      dataset = dataset.cache()
    if training:
      buffer_multiplier = 10
      dataset = dataset.shuffle(batch_size * buffer_multiplier)
      dataset = dataset.repeat(-1)
    map_fn = builder.get_dataset_map_fn(
        image_crop_size=image_crop_size,
        image_aspect_ratio=image_aspect_ratio,
        image_crop_proportion=image_crop_proportion,
        random_flip=random_flip,
        training=training)
    dataset = dataset.map(
        map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=training)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

  return input_fn
