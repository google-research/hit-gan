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
"""Utility function for calculating Inception Score and FID."""

import tensorflow as tf
import tensorflow_probability as tfp

INCEPTION_DEFAULT_IMAGE_SIZE = (299, 299)


def restore_inception_model(weights="imagenet"):
  """Loads Inception V3 Model."""
  inception_v3 = tf.keras.applications.InceptionV3(
      include_top=True,
      weights=weights,
      input_shape=(299, 299, 3),
      classifier_activation=None)

  layer_names = ["avg_pool", "predictions"]
  layers = [inception_v3.get_layer(name).output for name in layer_names]
  model = tf.keras.Model(inputs=inception_v3.input, outputs=layers)
  model.trainable = False
  return model


def run_inception_model(ds, model, steps, strategy, map_fn=None):
  """Runs Inception V3 Model to get activations and logits."""
  def get_images(inputs):
    return inputs["images"]

  if map_fn is None:
    map_fn = get_images

  def step_fn(inputs):
    images = map_fn(inputs)
    images = tf.image.resize(
        images,
        INCEPTION_DEFAULT_IMAGE_SIZE,
        method=tf.image.ResizeMethod.BILINEAR)
    batch_activations, batch_logits = model(images, training=False)
    return batch_activations, batch_logits

  @tf.function
  def run(iterator):
    activations = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    logits = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for i in tf.range(steps):
      batch_activations, batch_logits = strategy.run(
          step_fn, args=(next(iterator),))
      batch_activations = strategy.gather(batch_activations, axis=0)
      batch_logits = strategy.gather(batch_logits, axis=0)
      activations = activations.write(i, batch_activations)
      logits = logits.write(i, batch_logits)
    return activations.concat(), logits.concat()

  iterator = iter(ds)
  activations, logits = run(iterator)
  return activations, logits


def _symmetric_matrix_square_root(mat, eps=1e-10):
  """Computes square root of a symmetric matrix."""
  # Unlike numpy, tensorflow's return order is (s, u, v)
  s, u, v = tf.linalg.svd(mat)
  # sqrt is unstable around 0, just use 0 in such case
  si = tf.compat.v1.where(tf.less(s, eps), s, tf.sqrt(s))
  # Note that the v returned by Tensorflow is v = V
  # (when referencing the equation A = U S V^T)
  # This is unlike Numpy which returns v = V^T
  return tf.matmul(tf.matmul(u, tf.linalg.tensor_diag(si)), v, transpose_b=True)


def _trace_sqrt_product(sigma, sigma_v):
  sqrt_sigma = _symmetric_matrix_square_root(sigma)
  sqrt_a_sigmav_a = tf.matmul(sqrt_sigma, tf.matmul(sigma_v, sqrt_sigma))
  return tf.linalg.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))


def _kl_divergence(p, p_logits, q):
  """Computes the Kullback-Liebler divergence between p and q.

  This function uses p's logits in some places to improve numerical stability.
  Specifically:
  KL(p || q) = sum[ p * log(p / q) ]
    = sum[ p * ( log(p)                - log(q) ) ]
    = sum[ p * ( log_softmax(p_logits) - log(q) ) ]

  Args:
    p: A 2-D floating-point Tensor p_ij, where `i` corresponds to the minibatch
      example and `j` corresponds to the probability of being in class `j`.
    p_logits: A 2-D floating-point Tensor corresponding to logits for `p`.
    q: A 1-D floating-point Tensor, where q_j corresponds to the probability of
      class `j`.
  Returns:
    KL divergence between two distributions. Output dimension is 1D, one entry
    per distribution in `p`.
  """
  return tf.reduce_sum(
      p * (tf.nn.log_softmax(p_logits) - tf.math.log(q)), axis=1)


def frechet_inception_distance(avg_pools1, avg_pools2):
  """FID for evaluating a generative model."""
  m = tf.reduce_mean(avg_pools1, axis=0)
  m_w = tf.reduce_mean(avg_pools2, axis=0)

  # Calculate the unbiased covariance matrix of first activations.
  num_examples_real = tf.cast(tf.shape(input=avg_pools1)[0], tf.float32)
  sigma = (
      num_examples_real / (num_examples_real - 1) *
      tfp.stats.covariance(avg_pools1))

  # Calculate the unbiased covariance matrix of second activations.
  num_examples_generated = tf.cast(tf.shape(input=avg_pools2)[0], tf.float32)
  sigma_w = (
      num_examples_generated / (num_examples_generated - 1) *
      tfp.stats.covariance(avg_pools2))

  # Find the Tr(sqrt(sigma sigma_w)) component of FID
  sqrt_trace_component = _trace_sqrt_product(sigma, sigma_w)

  # First the covariance component.
  # Here, note that trace(A + B) = trace(A) + trace(B)
  trace = tf.linalg.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component

  # Next the distance between means.
  mean = tf.reduce_sum(tf.math.squared_difference(m, m_w))
  fid = trace + mean
  return fid


def inception_score(logits, num_splits=None, num_images_per_split=5000):
  """Inception score for evaluating a generative model."""
  if num_splits is None:
    num_samples = tf.shape(logits)[0]
    num_splits = int(num_samples // num_images_per_split)

  scores = []
  splits = tf.split(logits, num_or_size_splits=num_splits, axis=0)
  for p_logits in splits:
    p = tf.nn.softmax(p_logits, axis=-1)
    q = tf.math.reduce_mean(p, axis=0)
    kl = _kl_divergence(p, p_logits, q)
    log_scores = tf.math.reduce_mean(kl)
    scores.append(tf.exp(log_scores))
  return tf.math.reduce_mean(scores), tf.math.reduce_std(scores)
