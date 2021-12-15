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
"""Data preprocessing and augmentation."""

from typing import Optional, Text

import tensorflow as tf


def to_images(inputs: tf.Tensor) -> tf.Tensor:
  """Converts input tensors into images.

  Args:
    inputs: The input tensors ranging from [-1.0, 1.0].

  Returns:
    Output images ranging from [0.0, 1.0].
  """
  images = tf.clip_by_value(inputs, -1.0, 1.0)
  images = (images + 1.0) * 0.5
  return images


def _compute_crop_shape(image_height: int, image_width: int,
                        aspect_ratio: float, crop_proportion: float):
  """Computes aspect ratio-preserving shape for image crop.

  The resulting shape retains `crop_proportion` along one side and a proportion
  less than or equal to `crop_proportion` along the other side.

  Args:
    image_height: Height of image to be cropped.
    image_width: Width of image to be cropped.
    aspect_ratio: Desired aspect ratio (width / height) of output.
    crop_proportion: Proportion of image to retain along the less-cropped side.

  Returns:
    crop_height: Height of image after cropping.
    crop_width: Width of image after cropping.
  """
  image_width_float = tf.cast(image_width, tf.float32)
  image_height_float = tf.cast(image_height, tf.float32)

  def _requested_aspect_ratio_wider_than_image():
    crop_height = tf.cast(
        tf.math.rint(crop_proportion / aspect_ratio * image_width_float),
        tf.int32)
    crop_width = tf.cast(
        tf.math.rint(crop_proportion * image_width_float), tf.int32)
    return crop_height, crop_width

  def _image_wider_than_requested_aspect_ratio():
    crop_height = tf.cast(
        tf.math.rint(crop_proportion * image_height_float), tf.int32)
    crop_width = tf.cast(
        tf.math.rint(crop_proportion * aspect_ratio * image_height_float),
        tf.int32)
    return crop_height, crop_width

  return tf.cond(aspect_ratio > image_width_float / image_height_float,
                 _requested_aspect_ratio_wider_than_image,
                 _image_wider_than_requested_aspect_ratio)


def center_crop(image: tf.Tensor,
                height: int,
                width: int,
                crop_proportion: float,
                method: Text = tf.image.ResizeMethod.BICUBIC,
                antialias: bool = False) -> tf.Tensor:
  """Crops to center of image and rescales to desired size.

  Args:
    image: Image Tensor to crop.
    height: Height of image to be cropped.
    width: Width of image to be cropped.
    crop_proportion: Proportion of image to retain along the less-cropped side.
    method: Resize method for image processing.
    antialias: Whether to use an anti-aliasing filter.

  Returns:
    A `height` x `width` x channels Tensor holding a central crop of `image`.
  """
  shape = tf.shape(image)
  image_height = shape[0]
  image_width = shape[1]
  crop_height, crop_width = _compute_crop_shape(
      image_height, image_width, height / width, crop_proportion)
  offset_height = ((image_height - crop_height) + 1) // 2
  offset_width = ((image_width - crop_width) + 1) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_height, offset_width, crop_height, crop_width)

  image = tf.image.resize(
      image, (height, width), method=method, antialias=antialias)
  return image


def random_crop(image: tf.Tensor,
                height: int,
                width: int,
                crop_proportion: float,
                method: Text = tf.image.ResizeMethod.BICUBIC,
                antialias: bool = False,
                seed: Optional[int] = None) -> tf.Tensor:
  """Randomly crops image and rescales to desired size.

  Args:
    image: Image Tensor to crop.
    height: Height of image to be cropped.
    width: Width of image to be cropped.
    crop_proportion: Proportion of image to retain along the less-cropped side.
    method: Resize method for image processing.
    antialias: Whether to use an anti-aliasing filter.
    seed: An integer used to create a random seed.

  Returns:
    A `height` x `width` x channels Tensor holding a central crop of `image`.
  """
  shape = tf.shape(image)
  image_height = shape[0]
  image_width = shape[1]
  crop_height, crop_width = _compute_crop_shape(image_height, image_width,
                                                height / width, crop_proportion)
  image = tf.image.random_crop(image, (crop_height, crop_width, 3), seed=seed)
  image = tf.image.resize(
      image, (height, width), method=method, antialias=antialias)
  return image
