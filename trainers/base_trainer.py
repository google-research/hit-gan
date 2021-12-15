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
"""Basic trainer class."""

import abc
import math
import os
import time

from typing import Text, Optional

from absl import logging
from utils import data_utils
from utils import dataset_utils
from utils import datasets
from utils import inception_utils
from utils import metrics

import tensorflow as tf


class BaseTrainer(abc.ABC):
  """Basic class for managing training and evaluation."""

  def __init__(self,
               strategy,
               model_dir: Text,
               train_batch_size: int,
               eval_batch_size: int,
               dataset: Text,
               train_steps: int,
               data_dir: Optional[Text] = None,
               image_crop_size: int = 256,
               image_aspect_ratio: float = 1.0,
               image_crop_proportion: float = 1.0,
               random_flip: bool = False,
               record_every_n_steps: int = 100,
               save_every_n_steps: int = 1000,
               batch_every_n_steps: Optional[int] = None,
               keep_checkpoint_max: int = 50):
    """Initializer.

    Args:
      strategy: A tf.distribute.Strategy object for the dsitributed strategy.
      model_dir: A string for the save path of trained models.
      train_batch_size: An integer for the training batch size.
      eval_batch_size: An integer for the evaluation batch size.
      dataset: A string for the dataset name.
      train_steps: An integer for the number of training steps.
      data_dir: A string for the path of the dataset.
      image_crop_size: An integer for the size of cropped images.
      image_aspect_ratio: A float for the aspect ratio of images.
      image_crop_proportion: A float for the crop proportion of images.
      random_flip: Whether to use random flip.
      record_every_n_steps: An integer for the number of steps to record.
      save_every_n_steps: An integer for the number of steps to save models.
      batch_every_n_steps: An integer for the number of steps to batch.
      keep_checkpoint_max: An integer for the maximum number of checkpoints to
        keep.
    """
    self.train_batch_size = train_batch_size
    self.eval_batch_size = eval_batch_size
    self.train_steps = train_steps
    self.keep_checkpoint_max = keep_checkpoint_max
    self.record_every_n_steps = record_every_n_steps
    self.save_every_n_steps = save_every_n_steps
    self.batch_every_n_steps = batch_every_n_steps

    if self.batch_every_n_steps is None:
      self.batch_every_n_steps = self.record_every_n_steps

    self.objects = {}
    self.train_metrics = {}
    self.eval_metrics = {}
    self.global_step = None

    self.strategy = strategy
    self.model_dir = model_dir

    self.dataset = dataset
    self.data_dir = data_dir
    self.image_crop_size = image_crop_size
    self.image_aspect_ratio = image_aspect_ratio
    self.image_crop_proportion = image_crop_proportion
    self.random_flip = random_flip

  def build(self):
    self._build_dataset()
    with self.strategy.scope():
      self._build_models()
      self._build_optimizers()
      self._build_metrics()
    self.summary_writer = tf.summary.create_file_writer(self.model_dir)

  @abc.abstractmethod
  def _build_models(self):
    raise NotImplementedError

  @abc.abstractmethod
  def _build_optimizers(self):
    raise NotImplementedError

  @abc.abstractmethod
  def _build_metrics(self):
    raise NotImplementedError

  def _build_dataset(self):
    """Builds the training or evaluation dataset."""
    builder = datasets.get_dataset(self.dataset, data_dir=self.data_dir)
    num_train_examples = builder.get_num_examples(training=True)
    num_eval_examples = builder.get_num_examples(training=False)

    self.steps_per_epoch = int(num_train_examples // self.train_batch_size)
    self.eval_steps = int(math.ceil(num_eval_examples / self.eval_batch_size))
    self.builder = builder
    self.num_train_examples = num_train_examples
    self.num_eval_examples = num_eval_examples

    logging.info('# train examples: %d', num_train_examples)
    logging.info('# train_steps: %d', self.train_steps)
    logging.info('# eval examples: %d', num_eval_examples)
    logging.info('# eval steps: %d', self.eval_steps)

  @abc.abstractmethod
  def _train_one_step(self, inputs):
    raise NotImplementedError

  @abc.abstractmethod
  def _evaluate_one_step(self, inputs):
    raise NotImplementedError

  def _should_record(self, step):
    last_step = tf.greater_equal(step, self.train_steps)
    should_record = tf.equal(step % self.record_every_n_steps, 0)
    return tf.logical_or(should_record, last_step)

  def _should_save(self, step):
    last_step = tf.greater_equal(step, self.train_steps)
    should_save = tf.equal(step % self.save_every_n_steps, 0)
    return tf.logical_or(should_save, last_step)

  def _log_images(self, name, ds, max_outputs=64, step_fn=None):
    """Logs images to tf.summary."""
    @tf.function
    def run_one_step(inputs):
      if step_fn is None:
        return self._evaluate_one_step(inputs)
      else:
        return step_fn(inputs)

    outputs = []
    num_outputs = 0
    iterator = iter(ds)
    while num_outputs < max_outputs:
      inputs = next(iterator)
      batch_images = self.strategy.gather(inputs['images'], axis=0)
      batch_images = data_utils.to_images(batch_images)

      batch_outputs = self.strategy.run(run_one_step, args=(inputs,))
      batch_outputs = self.strategy.gather(batch_outputs, axis=0)
      batch_outputs = data_utils.to_images(batch_outputs)

      batch_outputs = tf.concat((batch_images, batch_outputs), axis=2)
      outputs.append(batch_outputs)
      num_outputs += batch_outputs.shape[0]

    outputs = tf.concat(outputs, axis=0)
    metrics.log_images_to_summary(name, outputs, self.global_step, max_outputs)

  def train(self):
    """Trains the model."""
    with self.strategy.scope():
      train_ds = dataset_utils.build_distributed_dataset(
          self.builder,
          self.strategy,
          global_batch_size=self.train_batch_size,
          image_crop_size=self.image_crop_size,
          image_aspect_ratio=self.image_aspect_ratio,
          image_crop_proportion=self.image_crop_proportion,
          random_flip=self.random_flip,
          training=True,
          cache=False)
      checkpoint_manager = self.restore_from_checkpoint()

    @tf.function
    def train_multiple_steps(iterator):
      for _ in tf.range(self.batch_every_n_steps):
        self.strategy.run(self._train_one_step, args=(next(iterator),))

    current_step = self.global_step
    iterator = iter(train_ds)
    metrics.reset_metrics(self.train_metrics)

    while current_step < self.train_steps:
      with self.summary_writer.as_default():
        start = time.time()
        train_multiple_steps(iterator)
        duration = (time.time() - start)

        with tf.summary.record_if(self._should_record(current_step)):
          tf.summary.scalar('train/batch_time', duration, step=current_step)
          metrics.log_and_write_metrics_to_summary(self.train_metrics,
                                                   current_step)

        if self._should_record(current_step):
          metrics.reset_metrics(self.train_metrics)

        if self._should_save(current_step):
          checkpoint_manager.save(current_step)
          logging.info('Completed: %d / %d steps', current_step.numpy(),
                       self.train_steps)
        self.summary_writer.flush()
    logging.info('Training complete...')

  def evaluate(self):
    """Evaluates the model."""
    evaluated_last_checkpoint = False
    metrics.reset_metrics(self.eval_metrics)

    with self.strategy.scope():
      eval_ds = dataset_utils.build_distributed_dataset(
          self.builder,
          self.strategy,
          global_batch_size=self.eval_batch_size,
          image_crop_size=self.image_crop_size,
          image_aspect_ratio=self.image_aspect_ratio,
          image_crop_proportion=self.image_crop_proportion,
          random_flip=False,
          training=False,
          cache=False)
      inception_model = inception_utils.restore_inception_model()

    activations, _ = inception_utils.run_inception_model(
        eval_ds, inception_model, steps=self.eval_steps, strategy=self.strategy)

    def timeout_fn():
      """Timeout function to stop the evaluation."""
      return evaluated_last_checkpoint

    for _ in tf.train.checkpoints_iterator(
        self.model_dir, timeout=1, timeout_fn=timeout_fn):

      with self.strategy.scope():
        self.restore_from_checkpoint()
        logging.info('Last checkpoint [iteration: %d] restored at %s.',
                     self.global_step.numpy(), self.model_dir)

      global_step = self.global_step
      if global_step >= self.train_steps:
        evaluated_last_checkpoint = True

      with self.summary_writer.as_default():
        start = time.time()
        gen_activations, gen_logits = inception_utils.run_inception_model(
            eval_ds,
            inception_model,
            steps=self.eval_steps,
            strategy=self.strategy,
            map_fn=self._evaluate_one_step)
        frechet_inception_distance = inception_utils.frechet_inception_distance(
            activations, gen_activations)
        inception_score, _ = inception_utils.inception_score(gen_logits)
        duration = (time.time() - start)
        tf.summary.scalar('eval/eval_time', duration, step=global_step)
        tf.summary.scalar(
            'eval/frechet_inception_distance',
            frechet_inception_distance,
            step=global_step)
        tf.summary.scalar(
            'eval/inception_score', inception_score, step=global_step)

        self._log_images('eval/reconstructions', eval_ds)
        metrics.log_and_write_metrics_to_summary(self.eval_metrics, global_step)
        metrics.reset_metrics(self.eval_metrics)
        self.summary_writer.flush()
        self.save_best_checkpoint(frechet_inception_distance)
      logging.info('Finished evaluation for step %d', global_step.numpy())
    logging.info('Evaluation complete...')

  def restore_from_checkpoint(self, checkpoint_path=None):
    """Restores the checkpoint (if one exists on the path).

    Args:
      checkpoint_path: The path where checkpoints are restored.

    Returns:
      The tf.train.CheckpointManager object.
    """
    checkpoint = tf.train.Checkpoint(**self.objects)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=self.model_dir,
        max_to_keep=self.keep_checkpoint_max)

    if checkpoint_path is None:
      checkpoint_path = checkpoint_manager.latest_checkpoint
    checkpoint.restore(checkpoint_path).expect_partial()
    return checkpoint_manager

  def save_best_checkpoint(self, current_frechet_inception_distance):
    """Saves the current checkpoint (if it is the best one).

    Args:
      current_frechet_inception_distance: FID of the checkpoint.

    Returns:
      Whether the current checkpoint has been saved.
    """
    checkpoint_dir = os.path.join(self.model_dir, 'best_checkpoint')
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)

    if checkpoint_path is not None:
      frechet_inception_distance = tf.Variable(
          initial_value=0, dtype=tf.float32, trainable=False)
      checkpoint = tf.train.Checkpoint(
          frechet_inception_distance=frechet_inception_distance)
      checkpoint.restore(checkpoint_path).expect_partial()
      if current_frechet_inception_distance >= frechet_inception_distance:
        return False

    frechet_inception_distance = tf.Variable(
        initial_value=current_frechet_inception_distance,
        dtype=tf.float32,
        trainable=False)
    checkpoint = tf.train.Checkpoint(
        frechet_inception_distance=frechet_inception_distance, **self.objects)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, directory=checkpoint_dir, max_to_keep=1)
    checkpoint_manager.save(self.global_step)
    return True
