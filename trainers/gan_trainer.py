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
"""GAN trainer class."""

from typing import Any, Dict, Text

from models import discriminators
from models import generators
from trainers import base_trainer
from utils import diff_augment
from utils import losses
from utils import metrics
from utils import moving_averages

import tensorflow as tf


class GANTrainer(base_trainer.BaseTrainer):
  """Trainer for GAN."""

  def __init__(self,
               generator_kwargs: Dict[Text, Any],
               discriminator_kwargs: Dict[Text, Any],
               latent_dim: int,
               generator_lr: float,
               discriminator_lr: float,
               beta1: float,
               beta2: float,
               use_ema_model: bool = False,
               ema_decay: float = 0.999,
               ema_inital_step: int = 1,
               use_consistency_regularization: bool = False,
               consistency_regularization_cost: float = 10.0,
               augment_policy: Text = 'color,translation,cutout',
               gan_loss_type: Text = 'non-saturating',
               grad_penalty_type: Text = 'r1',
               grad_penalty_cost: float = 10.0,
               **kwargs: Any) -> None:
    """Initializer.

    Args:
      generator_kwargs: Arguments for the generator.
      discriminator_kwargs: Arguments for the discriminator.
      latent_dim: An integer for the input latent dimension.
      generator_lr: A float for the learning rate of the generator.
      discriminator_lr: A float for the learning rate of the discriminator.
      beta1: A float for the beta1 value of Adam optimizer.
      beta2: A float for the beta2 value of Adam optimizer.
      use_ema_model: Whether to use EMA weights for the generator.
      ema_decay: A float of the decay value of EMA.
      ema_inital_step: An integer for the initial step of EMA.
      use_consistency_regularization: Whether to use bCR for the discriminator.
      consistency_regularization_cost: A float for the weight of the bCR.
      augment_policy: A string for the policy of augmentation.
      gan_loss_type: A string for the GAN loss type (non-saturating or hinge).
      grad_penalty_type: A string for the gradieht penalty type (None or r1 or
        wgan).
      grad_penalty_cost: A float for the weight of the gradieht penalty.
      **kwargs: Any other arguments for keras Layer.
    """
    super().__init__(**kwargs)
    self._generator_kwargs = generator_kwargs
    self._discriminator_kwargs = discriminator_kwargs
    self._latent_dim = latent_dim
    self._generator_lr = generator_lr
    self._discriminator_lr = discriminator_lr
    self._beta1 = beta1
    self._beta2 = beta2
    self._use_ema_model = use_ema_model
    self._ema_decay = ema_decay
    self._ema_inital_step = ema_inital_step
    self._use_consistency_regularization = use_consistency_regularization
    self._consistency_regularization_cost = consistency_regularization_cost
    self._augment_policy = augment_policy
    self._gan_loss_type = gan_loss_type
    self._grad_penalty_type = grad_penalty_type
    self._grad_penalty_cost = grad_penalty_cost

  def _build_models(self):
    self.generator = generators.HiTGenerator(**self._generator_kwargs)
    self.discriminator = discriminators.StyleGANDiscriminator(
        **self._discriminator_kwargs)
    self.objects['generator'] = self.generator
    self.objects['discriminator'] = self.discriminator

    if self._use_ema_model:
      self.ema_generator = generators.HiTGenerator(**self._generator_kwargs)
      self.objects['ema_generator'] = self.ema_generator

  def _build_optimizers(self):
    self.generator_optim = tf.keras.optimizers.Adam(
        self._generator_lr, beta_1=self._beta1, beta_2=self._beta2)
    self.discriminator_optim = tf.keras.optimizers.Adam(
        self._discriminator_lr, beta_1=self._beta1, beta_2=self._beta2)
    self.global_step = self.generator_optim.iterations

    self.objects['generator_optim'] = self.generator_optim
    self.objects['discriminator_optim'] = self.discriminator_optim
    self.objects['global_step'] = self.global_step

  def _build_metrics(self):
    metric_names = [
        'generator_lr',
        'generator_total_loss',
        'discriminator_lr',
        'discriminator_total_loss',
        'gradient_penalty',
        'consistency_loss',
    ]

    for metric_name in metric_names:
      self.train_metrics[metric_name] = tf.keras.metrics.Mean(
          'train/{}'.format(metric_name))

  def _update_ema_model(self):
    if tf.greater_equal(self.global_step, self._ema_inital_step):
      if tf.greater(self.global_step, self._ema_inital_step):
        moving_averages.update_ema_variables(self.ema_generator.variables,
                                             self.generator.variables,
                                             self._ema_decay)
      else:
        moving_averages.assign_ema_vars_from_initial_values(
            self.ema_generator.variables, self.generator.variables)

  def _train_one_step(self, inputs):
    real_images = inputs['images']
    batch_size = tf.shape(real_images)[0]

    with tf.GradientTape() as generator_tape, tf.GradientTape(
    ) as discriminator_tape:
      latent_codes = tf.random.normal(shape=(batch_size, self._latent_dim))
      fake_images = self.generator(latent_codes, training=True)
      fake_logits = self.discriminator(fake_images, training=True)

      if self._grad_penalty_type == 'r1':
        real_logits, grad_penalty = losses.r1_gradient_penalty(
            self.discriminator,
            real_images,
            penalty_cost=self._grad_penalty_cost)
      elif self._grad_penalty_type == 'wgan':
        real_logits = self.discriminator(real_images, training=True)
        _, grad_penalty = losses.wgan_gradient_penalty(
            self.discriminator,
            real_images,
            fake_images,
            penalty_cost=self._grad_penalty_cost)
      elif self._grad_penalty_type is None:
        real_logits = self.discriminator(real_images, training=True)
        grad_penalty = tf.constant(0.0, dtype=tf.float32)
      else:
        raise ValueError('{} is not a recognized gradient penalty type'.format(
            self._grad_penalty_type))

      if self._use_consistency_regularization:
        real_augmented_images = diff_augment.augment(
            real_images, policy=self._augment_policy)
        fake_augmented_images = diff_augment.augment(
            fake_images, policy=self._augment_policy)
        real_augmented_images = tf.stop_gradient(real_augmented_images)
        fake_augmented_images = tf.stop_gradient(fake_augmented_images)

        augmented_images = tf.concat(
            (real_augmented_images, fake_augmented_images), axis=0)
        augmented_logits = self.discriminator(augmented_images, training=True)
        real_augmented_logits, fake_augmented_logits = tf.split(
            augmented_logits, num_or_size_splits=2, axis=0)
        consistency_loss = self._consistency_regularization_cost * (
            losses.l2_loss(real_logits, real_augmented_logits) +
            losses.l2_loss(fake_logits, fake_augmented_logits))
      else:
        consistency_loss = tf.constant(0.0, dtype=tf.float32)

      generator_total_loss = losses.generator_loss(
          fake_logits, loss_type=self._gan_loss_type)
      discriminator_total_loss = losses.discriminator_loss(
          real_logits, fake_logits,
          loss_type=self._gan_loss_type) + grad_penalty + consistency_loss

      metrics.update_metrics(
          self.train_metrics,
          generator_lr=self.generator_optim.lr,
          generator_total_loss=generator_total_loss,
          discriminator_lr=self.discriminator_optim.lr,
          gradient_penalty=grad_penalty,
          consistency_loss=consistency_loss,
          discriminator_total_loss=discriminator_total_loss,
      )

      # The default behavior of `apply_gradients` is to sum gradients from all
      # replicas so we divide the loss by the number of replicas so that the
      # mean gradient is applied.
      generator_total_loss = (
          generator_total_loss / self.strategy.num_replicas_in_sync)
      discriminator_total_loss = (
          discriminator_total_loss / self.strategy.num_replicas_in_sync)

    generator_grads = generator_tape.gradient(
        generator_total_loss, self.generator.trainable_variables)
    discriminator_grads = discriminator_tape.gradient(
        discriminator_total_loss, self.discriminator.trainable_variables)
    self.generator_optim.apply_gradients(
        zip(generator_grads, self.generator.trainable_variables))
    self.discriminator_optim.apply_gradients(
        zip(discriminator_grads, self.discriminator.trainable_variables))

    if self._use_ema_model:
      if not self.ema_generator.built:
        self.ema_generator.build(latent_codes.shape)
      self._update_ema_model()

  def _evaluate_one_step(self, inputs):
    batch_size = tf.shape(inputs['images'])[0]

    latent_codes = tf.random.normal(shape=(batch_size, self._latent_dim))
    if self._use_ema_model:
      outputs = self.ema_generator(latent_codes, training=False)
    else:
      outputs = self.generator(latent_codes, training=False)
    outputs = tf.clip_by_value(outputs, -1.0, 1.0)
    return outputs
