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
"""Loss functions."""

import tensorflow as tf


def l1_loss(y_true, y_pred):
  return tf.reduce_mean(tf.keras.losses.mean_absolute_error(y_true, y_pred))


def l2_loss(y_true, y_pred):
  return tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true, y_pred))


def r1_gradient_penalty(discriminator, inputs, penalty_cost=1.0):
  """Calculates R1 gradient penalty for the discriminator."""
  batch_size = tf.shape(inputs)[0]

  with tf.GradientTape() as tape:
    tape.watch(inputs)
    outputs = discriminator(inputs, training=True)

  gradients = tape.gradient(outputs, inputs)
  gradients = tf.reshape(gradients, (batch_size, -1))
  penalty = tf.reduce_sum(tf.square(gradients), axis=-1)
  penalty = tf.reduce_mean(penalty) * penalty_cost
  return outputs, penalty


def wgan_gradient_penalty(discriminator,
                          real_inputs,
                          fake_inputs,
                          penalty_cost=1.0):
  """Calculates WGAN gradient penalty for the discriminator."""
  batch_size = tf.shape(real_inputs)[0]

  eps = tf.random.uniform(shape=(batch_size, *real_inputs.shape[1:]))
  rand_inputs = eps * real_inputs + (1 - eps) * fake_inputs
  with tf.GradientTape() as tape:
    tape.watch(rand_inputs)
    rand_outputs = discriminator(rand_inputs, training=True)

  gradients = tape.gradient(rand_outputs, rand_inputs)
  gradients = tf.reshape(gradients, (batch_size, -1))
  penalty = tf.norm(gradients, axis=-1)
  penalty = tf.reduce_mean((penalty - 1.0)**2)
  penalty = tf.reduce_mean(penalty) * penalty_cost
  return rand_outputs, penalty


def discriminator_loss(real_logits, fake_logits, loss_type='hinge'):
  """Calculates losses for the discriminator."""
  if loss_type == 'hinge':
    real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_logits))
    fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_logits))
  elif loss_type == 'non-saturating':
    real_loss = tf.reduce_mean(tf.math.softplus(-real_logits))
    fake_loss = tf.reduce_mean(tf.math.softplus(fake_logits))
  else:
    raise ValueError('Discriminator loss {} not supported'.format(loss_type))
  return real_loss + fake_loss


def generator_loss(fake_logits, loss_type='hinge'):
  """Calculates losses for the generator."""
  if loss_type == 'hinge':
    fake_loss = -tf.reduce_mean(fake_logits)
  elif loss_type == 'non-saturating':
    fake_loss = tf.reduce_mean(tf.math.softplus(-fake_logits))
  else:
    raise ValueError('Generator loss {} not supported'.format(loss_type))
  return fake_loss
