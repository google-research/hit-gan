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
"""Generators in HiT-GAN."""

from typing import Any, Callable, List, Optional, Text, Tuple, Union

from models import layers

import tensorflow as tf


def block_images(inputs: tf.Tensor, patch_size: int) -> tf.Tensor:
  """Converts the image to blocked patches."""
  # inputs: (batch_size, height, width, channels)
  _, height, width, channel_dim = inputs.shape
  patch_length = patch_size**2

  outputs = tf.nn.space_to_depth(inputs, patch_size)
  outputs = tf.reshape(
      outputs,
      shape=(-1, height * width // patch_length, patch_length, channel_dim))
  # outputs: (batch_size, grid_h * grid_w, patch_h * patch_w, channels)
  return outputs


def unblock_images(inputs: tf.Tensor, grid_size: int,
                   patch_size: int) -> tf.Tensor:
  """Converts blocked patches to the image."""
  # inputs: (batch_size, grid_h * grid_w, patch_h * patch_w, channels)
  grid_width = grid_size
  grid_height = inputs.shape[1] // grid_width
  channel_dim = inputs.shape[3]

  outputs = tf.reshape(
      inputs,
      shape=(-1, grid_height, grid_width, patch_size**2 * channel_dim))
  outputs = tf.nn.depth_to_space(outputs, patch_size)
  # outputs: (batch_size, height, width, channels)
  return outputs


class Block(tf.keras.layers.Layer):
  """Aattention block."""

  def __init__(self,
               attn_axes: List[List[int]],
               num_heads: int = 4,
               dropout: float = 0.0,
               attn_dropout: float = 0.0,
               attn_type: Text = "multi_head",
               norm_type: Text = "layer",
               activation: Callable[..., tf.Tensor] = tf.nn.gelu,
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               **kwargs: Any) -> None:
    """Initializer.

    Args:
      attn_axes: A list for the list of axes over which the attention is
        applied.
      num_heads: An integer for the number of attention heads.
      dropout: A float for the dropout rate for MLP.
      attn_dropout: A float for the dropout for attention.
      attn_type: A string for attention type ("multi_head" or "multi_query").
      norm_type: A string for the type of normalization.
      activation: Activation function.
      kernel_initializer: Initialization function of dense kenrels.
      bias_initializer: Initialization function of dense biases.
      **kwargs: Additional arguments for `tf.keras.layers.Layer`.
    """
    super().__init__(**kwargs)
    self._attn_axes = attn_axes
    self._num_heads = num_heads
    self._dropout = dropout
    self._attn_dropout = attn_dropout
    self._attn_type = attn_type
    self._norm_type = norm_type
    self._activation = activation
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  def build(self, input_shapes: Union[tf.TensorShape, Tuple[tf.TensorShape,
                                                            tf.TensorShape]]):
    if isinstance(input_shapes, tuple):
      input_dim = input_shapes[0][-1]
    else:
      input_dim = input_shapes[-1]

    common_kwargs = dict(
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer)

    self.attention_layer = layers.MultiAxisAttention(
        num_heads=self._num_heads,
        key_dim=input_dim // self._num_heads,
        attn_axes=self._attn_axes,
        attn_type=self._attn_type,
        dropout=self._attn_dropout,
        **common_kwargs)

    self.norm = layers.make_norm_layer(self._norm_type)
    self.dropout_layer = tf.keras.layers.Dropout(self._dropout)
    self.mlp_block = layers.MLP(
        dropout=self._dropout,
        norm_type=self._norm_type,
        activation=self._activation,
        **common_kwargs)
    super().build(input_shapes)

  def call(self,
           inputs: Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]],
           training: Optional[bool] = None) -> tf.Tensor:
    if isinstance(inputs, tuple):
      queries, values = inputs
    else:
      queries = inputs
      values = None

    outputs = self.norm(queries, training=training)
    if values is None:
      values = outputs

    outputs = self.attention_layer(outputs, values, training=training)
    outputs = self.dropout_layer(outputs, training=training)
    outputs = outputs + queries

    outputs = self.mlp_block(outputs, training=training)
    return outputs


class HiTGenerator(tf.keras.Model):
  """HiT generator architecture."""

  def __init__(self,
               output_size: int,
               output_dim: int = 3,
               attn_type: Text = "multi_query",
               norm_type: Text = "batch",
               activation: Callable[..., tf.Tensor] = tf.nn.gelu,
               **kwargs: Any) -> None:
    """Initializer.

    Args:
      output_size: An integer for the output size.
      output_dim: An integer for the output channel dimension.
      attn_type: A string for attention type ("multi_head" or "multi_query").
      norm_type: A string for the type of normalization.
      activation: Activation function.
      **kwargs: Additional arguments for `tf.keras.layers.Layer`.
    """
    super().__init__(**kwargs)
    if output_size == 128:
      num_layers_per_block = [2, 2, 2, 2, 2]
      channel_dim_per_block = [512, 512, 256, 128, 128]
      num_heads_per_block = [16, 8, 4, 4, 4]
      patch_size_per_block = [4, 4, 8, 8, None]
    elif output_size == 256:
      num_layers_per_block = [2, 2, 2, 2, 1, 1]
      channel_dim_per_block = [512, 512, 256, 128, 64, 64]
      num_heads_per_block = [16, 8, 4, 4, 4, 4]
      patch_size_per_block = [4, 4, 8, 8, None, None]
    elif output_size == 1024:
      num_layers_per_block = [2, 2, 2, 2, 1, 1, 1, 1]
      channel_dim_per_block = [512, 512, 256, 128, 64, 64, 32, 32]
      num_heads_per_block = [16, 8, 4, 4, 4, 4, 4, 4]
      patch_size_per_block = [4, 4, 8, 8, None, None, None, None]
    else:
      raise ValueError("Only input_size of 128, 256 or 1024 is supported.")

    self._patch_size_per_block = patch_size_per_block
    self._num_blocks = len(num_layers_per_block)

    initial_patch_size = 8
    initial_patch_dim = channel_dim_per_block[0]
    embedding_dim = initial_patch_dim
    embedding_size = 4

    dilated_attention_axes = [1]
    local_attention_axes = [2]
    full_attention_axes = dilated_attention_axes + local_attention_axes

    common_kwargs = dict(
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        bias_initializer="zeros")
    block_kwargs = dict(norm_type=norm_type, activation=activation)

    self.dense_layer = tf.keras.Sequential([
        tf.keras.layers.Dense(initial_patch_dim * initial_patch_size**2,
                              **common_kwargs),
        tf.keras.layers.Reshape(
            (initial_patch_size, initial_patch_size, initial_patch_dim))
    ])
    self.embedding_layer = tf.keras.Sequential([
        tf.keras.layers.Dense(embedding_dim * embedding_size**2,
                              **common_kwargs),
        tf.keras.layers.Reshape((1, embedding_size**2, embedding_dim)),
        layers.PositionEmbedding(),
        layers.make_norm_layer(norm_type)
    ])

    self.position_embeddings = []
    self.blocks = []
    self.umsamplings = []
    self.to_rgb_layers = []
    for i in range(self._num_blocks):
      num_heads = num_heads_per_block[i]
      patch_size = self._patch_size_per_block[i]

      self.position_embeddings.append(layers.PositionEmbedding())
      block = tf.keras.Sequential()
      block.add(
          Block(
              num_heads=num_heads,
              attn_axes=[full_attention_axes],
              attn_type=attn_type,
              **block_kwargs,
              **common_kwargs))

      for _ in range(num_layers_per_block[i]):
        if patch_size is None:
          block.add(layers.MLP(**block_kwargs, **common_kwargs))
        else:
          block.add(
              Block(
                  num_heads=num_heads,
                  attn_axes=[local_attention_axes, dilated_attention_axes],
                  attn_type=attn_type,
                  **block_kwargs,
                  **common_kwargs))
      self.blocks.append(block)

      if patch_size is None:
        self.to_rgb_layers.append(
            layers.SkipToRGB(output_dim, norm_type=norm_type, **common_kwargs))
      else:
        self.to_rgb_layers.append(None)

      if i < self._num_blocks - 1:
        self.umsamplings.append(
            layers.PixelShuffle(
                output_dim=channel_dim_per_block[i + 1], **common_kwargs))

  def call(self, inputs: tf.Tensor, training: Optional[bool] = None):
    """Computes a forward pass of the generator block.

    Args:
      inputs: The input latent codes with the shape (batch_size, channel_dim).
      training: Boolean, whether training or not.

    Returns:
      The output feature map.
    """
    outputs = self.dense_layer(inputs)
    embeddings = self.embedding_layer(inputs, training=training)
    images = None

    for i in range(self._num_blocks):
      outputs = self.position_embeddings[i](outputs)
      patch_size = self._patch_size_per_block[i]
      if patch_size is not None:
        grid_size = outputs.shape[2] // patch_size
        outputs = block_images(outputs, patch_size)
        outputs = self.blocks[i]((outputs, embeddings), training=training)
        outputs = unblock_images(outputs, grid_size, patch_size)
      else:
        outputs = self.blocks[i]((outputs, embeddings), training=training)

      if self.to_rgb_layers[i] is not None:
        images = self.to_rgb_layers[i](outputs, images, training=training)

      if i < self._num_blocks - 1:
        outputs = self.umsamplings[i](outputs)

    return images
