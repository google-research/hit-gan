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
"""Model layers for HiT-GAN."""

import math
import string
from typing import Any, Callable, List, Optional, Text

import numpy as np
import tensorflow as tf

_CHR_IDX = string.ascii_lowercase


def _build_attention_equation(rank, attn_axes):
  """Builds einsum equations for the attention computation.

  Query, key, value inputs after projection are expected to have the shape as:
  (bs, <non-attention dims>, <attention dims>, num_heads, channels).
  bs and <non-attention dims> are treated as <batch dims>.
  The attention operations can be generalized:
  (1) Query-key dot product:
  (<batch dims>, <query attention dims>, num_heads, channels), (<batch dims>,
  <key attention dims>, num_heads, channels) -> (<batch dims>,
  num_heads, <query attention dims>, <key attention dims>)
  (2) Combination:
  (<batch dims>, num_heads, <query attention dims>, <key attention dims>),
  (<batch dims>, <value attention dims>, num_heads, channels) -> (<batch dims>,
  <query attention dims>, num_heads, channels)

  Args:
    rank: the rank of query, key, value tensors.
    attn_axes: a list/tuple of axes, [-1, rank), that will do attention.

  Returns:
    Einsum equations.
  """
  target_notation = _CHR_IDX[:rank]
  # `batch_dims` includes the head dim.
  batch_dims = tuple(np.delete(range(rank), attn_axes + (rank - 1,)))
  letter_offset = rank
  source_notation = ""
  for i in range(rank):
    if i in batch_dims or i == rank - 1:
      source_notation += target_notation[i]
    else:
      source_notation += _CHR_IDX[letter_offset]
      letter_offset += 1

  product_notation = "".join([target_notation[i] for i in batch_dims] +
                             [target_notation[i] for i in attn_axes] +
                             [source_notation[i] for i in attn_axes])
  dot_product_equation = "%s,%s->%s" % (source_notation, target_notation,
                                        product_notation)
  attn_scores_rank = len(product_notation)
  combine_equation = "%s,%s->%s" % (product_notation, source_notation,
                                    target_notation)
  return dot_product_equation, combine_equation, attn_scores_rank


def _build_proj_equation(free_dims, bound_dims, output_dims):
  """Builds an einsum equation for projections inside multi-head attention."""
  input_str = ""
  kernel_str = ""
  output_str = ""
  bias_axes = ""
  letter_offset = 0
  for i in range(free_dims):
    char = _CHR_IDX[i + letter_offset]
    input_str += char
    output_str += char

  letter_offset += free_dims
  for i in range(bound_dims):
    char = _CHR_IDX[i + letter_offset]
    input_str += char
    kernel_str += char

  letter_offset += bound_dims
  for i in range(output_dims):
    char = _CHR_IDX[i + letter_offset]
    kernel_str += char
    output_str += char
    bias_axes += char
  equation = "%s,%s->%s" % (input_str, kernel_str, output_str)

  return equation, bias_axes, len(output_str)


def _get_output_shape(output_rank, known_last_dims):
  return [None] * (output_rank - len(known_last_dims)) + list(known_last_dims)


def make_norm_layer(
    norm_type: Optional[Text] = "batch") -> tf.keras.layers.Layer:
  """Makes the normalization layer.

  Args:
    norm_type: A string for the type of normalization.

  Returns:
    A `tf.keras.layers.Layer` instance.
  """
  if norm_type is None:
    return tf.keras.layers.Layer()  # Identity.
  elif norm_type == "batch":
    return tf.keras.layers.BatchNormalization()
  elif norm_type == "syncbatch":
    return tf.keras.layers.experimental.SyncBatchNormalization()
  elif norm_type == "layer":
    return tf.keras.layers.LayerNormalization(epsilon=1e-6)
  else:
    raise ValueError("{} is not a recognized norm type".format(norm_type))


class PositionEmbedding(tf.keras.layers.Layer):
  """Defines learnable positional embeddings."""

  def build(self, input_shape: tf.TensorShape) -> None:
    input_dim = input_shape[-1]
    input_height = input_shape[-3]
    input_width = input_shape[-2]

    self.embedding_weight = self.add_weight(
        "embedding_weight",
        shape=(1, input_height, input_width, input_dim),
        initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        trainable=True)
    super().build(input_shape)

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    return inputs + self.embedding_weight


class SkipToRGB(tf.keras.layers.Layer):
  """Converts skip inputs to RGB images."""

  def __init__(self,
               output_dim: int = 3,
               norm_type: Text = "layer",
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               **kwargs: Any) -> None:
    """Initializer.

    Args:
      output_dim: An integer for the output channel dimension.
      norm_type: A string for the type of normalization.
      kernel_initializer: Initialization function of dense kenrels.
      bias_initializer: Initialization function of dense biases.
      **kwargs: Additional arguments for `tf.keras.layers.Layer`.
    """
    super().__init__(**kwargs)
    self.output_layer = tf.keras.Sequential([
        make_norm_layer(norm_type),
        tf.keras.layers.Dense(
            output_dim,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer)
    ])
    self.upsample = tf.keras.layers.UpSampling2D()

  def call(self,
           inputs: tf.Tensor,
           skip_inputs: Optional[tf.Tensor],
           training: Optional[bool] = None) -> tf.Tensor:
    outputs = self.output_layer(inputs, training=training)
    if skip_inputs is not None:
      skip_outputs = self.upsample(skip_inputs)
      outputs = skip_outputs + outputs
    return outputs


class PixelShuffle(tf.keras.layers.Layer):
  """Up-sampling layer using pixel shuffle."""

  def __init__(self,
               output_dim: int,
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               **kwargs: Any) -> None:
    """Initializer.

    Args:
      output_dim: An integer for the output channel dimension.
      kernel_initializer: Initialization function of dense kenrels.
      bias_initializer: Initialization function of dense biases.
      **kwargs: Additional arguments for `tf.keras.layers.Layer`.
    """
    super().__init__(**kwargs)
    self._output_dim = output_dim
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  def build(self, input_shape: tf.TensorShape) -> None:
    if input_shape[-1] // 4 == self._output_dim:
      self.dense_layer = None
    else:
      self.dense_layer = tf.keras.layers.Dense(
          self._output_dim,
          kernel_initializer=self._kernel_initializer,
          bias_initializer=self._bias_initializer)
    super().build(input_shape)

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    outputs = tf.nn.depth_to_space(inputs, 2)
    if self.dense_layer is not None:
      outputs = self.dense_layer(outputs)
    return outputs


class MLP(tf.keras.layers.Layer):
  """Defines MLP layer with normalization and residual connection."""

  def __init__(self,
               expansion: int = 4,
               dropout: float = 0.,
               norm_type: Text = "batch",
               activation: Callable[..., tf.Tensor] = tf.nn.relu,
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               **kwargs: Any) -> None:
    """Initializer.

    Args:
      expansion: An integer for the expansion ratio of the hidden dimension.
      dropout: A float for the dropout rate after dense layers.
      norm_type: A string for the type of normalization.
      activation: Activation function.
      kernel_initializer: Initialization function of dense kenrels.
      bias_initializer: Initialization function of dense biases.
      **kwargs: Additional arguments for `tf.keras.layers.Layer`.
    """
    super().__init__(**kwargs)
    self._expansion = expansion
    self._dropout = dropout
    self._norm_type = norm_type
    self._activation = activation
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  def build(self, input_shape: tf.TensorShape) -> None:
    input_dim = input_shape[-1]
    common_kwargs = dict(
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer)

    self.norm_layer = make_norm_layer(self._norm_type)
    self.mlp_block = tf.keras.Sequential([
        tf.keras.layers.Dense(
            input_dim * self._expansion,
            activation=self._activation,
            **common_kwargs),
        tf.keras.layers.Dropout(self._dropout),
        tf.keras.layers.Dense(input_dim, **common_kwargs),
        tf.keras.layers.Dropout(self._dropout)
    ])
    super().build(input_shape)

  def call(self,
           inputs: tf.Tensor,
           training: Optional[bool] = None) -> tf.Tensor:
    outputs = self.norm_layer(inputs, training=training)
    outputs = self.mlp_block(outputs, training=training)
    return outputs + inputs


class MultiAxisAttention(tf.keras.layers.Layer):
  """MultiAxisAttention performs attentions along multiple axes."""

  def __init__(self,
               num_heads: int,
               key_dim: int,
               attn_axes: List[List[int]],
               attn_type: Text = "multi_head",
               use_bias: bool = True,
               dropout: float = 0.0,
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               **kwargs: Any) -> None:
    """Initializer.

    Args:
      num_heads: An integer for the number of attention heads.
      key_dim: An integer for the size of each attention head.
      attn_axes: A list for the list of axes over which the attention is
        applied.
      attn_type: A string for attention type ("multi_head" or "multi_query").
      use_bias: A boolean for whether the dense layers use biases.
      dropout: A float for the dropout rate after dense layers.
      kernel_initializer: Initialization function of dense kenrels.
      bias_initializer: Initialization function of dense biases.
      **kwargs: Additional arguments for `tf.keras.layers.Layer`.
    """
    super().__init__(**kwargs)
    self._num_heads = num_heads
    self._key_dim = key_dim
    self._attn_axes = attn_axes
    self._attn_type = attn_type
    self._use_bias = use_bias
    self._dropout = dropout
    self._scale = math.sqrt(float(key_dim))
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  def build(self, input_shape: tf.TensorShape) -> None:
    input_dim = input_shape[-1]
    free_dims = input_shape.rank - 1
    common_kwargs = dict(
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer)

    einsum_equation, bias_axes, output_rank = _build_proj_equation(
        free_dims, bound_dims=1, output_dims=2)
    self.query_dense = tf.keras.layers.experimental.EinsumDense(
        einsum_equation,
        output_shape=_get_output_shape(output_rank - 1,
                                       [self._num_heads, self._key_dim]),
        bias_axes=bias_axes if self._use_bias else None,
        **common_kwargs)

    if self._attn_type == "multi_head":
      num_heads = self._num_heads
    elif self._attn_type == "multi_query":
      num_heads = 1
    else:
      raise ValueError(
          "{} is not a recognized attention type".format(self._attn_type))
    self.key_dense = tf.keras.layers.experimental.EinsumDense(
        einsum_equation,
        output_shape=_get_output_shape(output_rank - 1,
                                       [num_heads, self._key_dim]),
        bias_axes=bias_axes if self._use_bias else None,
        **common_kwargs)
    self.value_dense = tf.keras.layers.experimental.EinsumDense(
        einsum_equation,
        output_shape=_get_output_shape(output_rank - 1,
                                       [num_heads, self._key_dim]),
        bias_axes=bias_axes if self._use_bias else None,
        **common_kwargs)

    self._dot_product_equations = []
    self._combine_equations = []
    self.softmax_layers = []
    for attn_axes in self._attn_axes:
      attn_axes = tuple(attn_axes)
      (dot_product_equation, combine_equation,
       attn_scores_rank) = _build_attention_equation(output_rank, attn_axes)
      norm_axes = tuple(
          range(attn_scores_rank - len(attn_axes), attn_scores_rank))
      self._dot_product_equations.append(dot_product_equation)
      self._combine_equations.append(combine_equation)
      self.softmax_layers.append(tf.keras.layers.Softmax(axis=norm_axes))

    output_shape = [input_dim]
    einsum_equation, bias_axes, output_rank = _build_proj_equation(
        free_dims, bound_dims=2, output_dims=len(output_shape))
    self.output_dense = tf.keras.layers.experimental.EinsumDense(
        einsum_equation,
        output_shape=_get_output_shape(output_rank - 1, output_shape),
        bias_axes=bias_axes if self._use_bias else None,
        **common_kwargs)

    self.dropout_layer = tf.keras.layers.Dropout(self._dropout)
    super().build(input_shape)

  def call(self,
           queries: tf.Tensor,
           values: tf.Tensor,
           training: Optional[bool] = None) -> tf.Tensor:
    queries = self.query_dense(queries)
    keys = self.key_dense(values)
    values = self.value_dense(values)
    if self._attn_type == "multi_query":
      keys = tf.repeat(keys, [self._num_heads], axis=-2)
      values = tf.repeat(values, [self._num_heads], axis=-2)

    num_axes = len(self._attn_axes)
    queries = tf.split(queries, num_or_size_splits=num_axes, axis=-2)
    keys = tf.split(keys, num_or_size_splits=num_axes, axis=-2)
    values = tf.split(values, num_or_size_splits=num_axes, axis=-2)

    outputs = []
    for i in range(num_axes):
      attn_scores = tf.einsum(self._dot_product_equations[i], keys[i],
                              queries[i]) / self._scale
      attn_scores = self.softmax_layers[i](attn_scores)
      attn_scores = self.dropout_layer(attn_scores, training=training)
      outputs.append(
          tf.einsum(self._combine_equations[i], attn_scores, values[i]))

    outputs = tf.concat(outputs, axis=-2)
    outputs = self.output_dense(outputs)
    return outputs
