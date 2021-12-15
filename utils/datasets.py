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
"""Dataset utilities."""

import abc
import math
import os
from typing import Any, Optional, Text

from utils import dataset_utils

import tensorflow as tf
import tensorflow_datasets as tfds


class DatasetBuilder(abc.ABC):
  """Basic class for dataset builders."""

  def __init__(self,
               dataset_name: Text,
               data_dir: Optional[Text] = None,
               train_crop_mode: Text = "fixed",
               eval_crop_mode: Text = "fixed") -> None:
    self._dataset_name = dataset_name
    self._data_dir = data_dir
    self._train_crop_mode = train_crop_mode
    self._eval_crop_mode = eval_crop_mode

  @abc.abstractmethod
  def get_num_examples(self, training: bool = False) -> int:
    raise NotImplementedError

  @abc.abstractmethod
  def get_dataset(self, training: bool = False) -> tf.data.Dataset:
    raise NotImplementedError

  def get_dataset_map_fn(self,
                         image_crop_size: int,
                         image_aspect_ratio: float = 1.0,
                         image_crop_proportion: float = 1.0,
                         random_flip: bool = False,
                         training: bool = False):
    """Gets dataset mapping function."""
    crop_mode = self._train_crop_mode if training else self._eval_crop_mode
    preprocess_fn = dataset_utils.get_preprocess_fn(
        image_crop_size,
        aspect_ratio=image_aspect_ratio,
        crop_mode=crop_mode,
        crop_proportion=image_crop_proportion,
        method=tf.image.ResizeMethod.BICUBIC,
        flip=random_flip,
        normalize=True)

    def map_fn(features):
      return dict(images=preprocess_fn(features["image"]))

    return map_fn


class TFDSBuilder(DatasetBuilder):
  """Dataset builder for TFDS."""

  def __init__(self,
               train_split: Text = "train",
               eval_split: Text = "test",
               **kwargs: Any) -> None:
    super().__init__(**kwargs)
    builder = tfds.builder(self._dataset_name, data_dir=self._data_dir)
    builder.download_and_prepare()
    self._builder = builder
    self._train_split = train_split
    self._eval_split = eval_split

  def get_num_examples(self, training: bool = False) -> int:
    split = self._train_split if training else self._eval_split
    return self._builder.info.splits[split].num_examples

  def get_dataset(self, training: bool = False) -> tf.data.Dataset:
    split = self._train_split if training else self._eval_split
    dataset = self._builder.as_dataset(split, shuffle_files=training)
    return dataset


class FFHQ(DatasetBuilder):
  """FFHQ (Flickr-Faces-HQ) Dataset.

  Reference:
    Karras et al. A Style-Based Generator Architecture for Generative
    Adversarial Networks. https://arxiv.org/pdf/1812.04948.pdf.
  """
  RESOLUTIONS = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
  DATASET_SIZE = 70000
  EVAL_DATASET_SIZE = 50000

  def __init__(self, **kwargs: Any) -> None:
    super().__init__(**kwargs)
    resolution = int(self._dataset_name.split("/")[-1])
    log2_resolution = int(math.log2(resolution))
    data_dir = self._data_dir
    
    self._resolution = resolution
    self._tfrecord_path = os.path.join(
        data_dir, "ffhq-r{:02d}.tfrecords".format(log2_resolution))

  def get_num_examples(self, training: bool = False) -> int:
    return self.DATASET_SIZE if training else self.EVAL_DATASET_SIZE

  def get_dataset(self, training: bool = False) -> tf.data.Dataset:
    dataset = tf.data.TFRecordDataset(
        self._tfrecord_path, buffer_size=256 << 20)
    dataset = dataset.take(
        self.DATASET_SIZE if training else self.EVAL_DATASET_SIZE)
    return dataset

  def get_dataset_map_fn(self,
                         image_crop_size: int,
                         image_aspect_ratio: float = 1.0,
                         image_crop_proportion: float = 1.0,
                         random_flip: bool = False,
                         training: bool = False):
    """Gets dataset mapping function."""
    preprocess_fn = super().get_dataset_map_fn(
        image_crop_size=image_crop_size,
        image_aspect_ratio=image_aspect_ratio,
        image_crop_proportion=image_crop_proportion,
        random_flip=random_flip,
        training=training)

    def map_fn(record):
      features = {"data": tf.io.FixedLenFeature([], tf.string)}
      parsed_example = tf.io.parse_single_example(record, features)
      data = tf.io.decode_raw(parsed_example["data"], tf.uint8)
      image = tf.reshape(data, shape=(3, self._resolution, self._resolution))
      image = tf.transpose(image, (1, 2, 0))
      return preprocess_fn(dict(image=image))

    return map_fn


def get_dataset(dataset_name: Text,
                data_dir: Optional[Text] = None) -> DatasetBuilder:
  """Gets the DatasetBuilder object by the dataset name."""
  if dataset_name == "cifar10":
    dataset = TFDSBuilder(
        dataset_name=dataset_name,
        data_dir=data_dir,
        train_crop_mode="fixed",
        eval_crop_mode="fixed",
        train_split="train",
        eval_split="test")
  elif dataset_name == "imagenet2012":
    dataset = TFDSBuilder(
        dataset_name=dataset_name,
        data_dir=data_dir,
        train_crop_mode="random_crop",
        eval_crop_mode="center_crop",
        train_split="train",
        eval_split="validation[:50000]")
  elif dataset_name.startswith("celeb_a_hq"):
    dataset = TFDSBuilder(
        dataset_name=dataset_name,
        data_dir=data_dir,
        train_crop_mode="fixed",
        eval_crop_mode="fixed",
        train_split="train",
        eval_split="train")
  elif dataset_name in ["ffhq/{}".format(r) for r in FFHQ.RESOLUTIONS]:
    dataset = FFHQ(
        dataset_name=dataset_name,
        data_dir=data_dir,
        train_crop_mode="fixed",
        eval_crop_mode="fixed")
  else:
    raise ValueError("{} is not a recognized dataset".format(dataset_name))
  return dataset
