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
"""Discriminators in HiT-GAN."""

import math
from typing import Any, Text, Tuple, Type, Union

import tensorflow as tf


class BlurPool2D(tf.keras.layers.Layer):
  """A layer to do channel-wise blurring + subsampling on 2D inputs.

  Reference:
    Zhang et al. Making Convolutional Networks Shift-Invariant Again.
    https://arxiv.org/pdf/1904.11486.pdf.
  """

  def __init__(self,
               filter_size: int = 3,
               strides: Tuple[int, int, int, int] = (1, 2, 2, 1),
               padding: Text = "SAME",
               **kwargs: Any) -> None:
    """Initializes the BlurPool2D layer.

    Args:
      filter_size: The size (height and width) of the blurring filter.
      strides: The stride for convolution of the blurring filter for each
        dimension of the inputs.
      padding: One of 'VALID' or 'SAME', specifying the padding type used for
        convolution.
      **kwargs: Keyword arguments forwarded to super().__init__().

    Raises:
      ValueError: If filter_size is not 3, 4, 5, 6 or 7.
    """
    if filter_size not in [3, 4, 5, 6, 7]:
      raise ValueError("Only filter_size of 3, 4, 5, 6 or 7 is supported.")
    super().__init__(**kwargs)
    self._strides = strides
    self._padding = padding

    if filter_size == 3:
      self._filter = [1., 2., 1.]
    elif filter_size == 4:
      self._filter = [1., 3., 3., 1.]
    elif filter_size == 5:
      self._filter = [1., 4., 6., 4., 1.]
    elif filter_size == 6:
      self._filter = [1., 5., 10., 10., 5., 1.]
    elif filter_size == 7:
      self._filter = [1., 6., 15., 20., 15., 6., 1.]

    self._filter = tf.constant(self._filter, dtype=tf.float32)
    self._filter = self._filter[:, None] * self._filter[None, :]
    self._filter /= tf.reduce_sum(self._filter)
    self._filter = tf.reshape(
        self._filter, [self._filter.shape[0], self._filter.shape[1], 1, 1])

  def build(self, input_shape: tf.TensorShape) -> None:
    self._filter = tf.tile(self._filter, [1, 1, input_shape[-1], 1])
    super().build(input_shape)

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    """Calls the BlurPool2D layer on the given inputs."""
    return tf.nn.depthwise_conv2d(
        input=inputs,
        filter=self._filter,
        strides=self._strides,
        padding=self._padding)


class MinibatchStddev(tf.keras.layers.Layer):
  """Minibatch standard deviation layer.

  It calculates the standard deviation for each feature map,
  averages them to `num_features` values and appends them to all the channels.
  """

  def __init__(self,
               group_size: int = 4,
               num_features: int = 1,
               **kwargs: Any) -> None:
    """Initializer.

    Args:
      group_size: The size of the group to split the input batch to.
      num_features: The number to split the input channels to.
      **kwargs: Any other arguments for keras Layer.
    """
    super().__init__(**kwargs)
    self._group_size = group_size
    self._num_features = num_features

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    batch_size = tf.shape(inputs)[0]
    group_size = tf.minimum(batch_size, self._group_size)

    outputs = tf.reshape(
        inputs,
        shape=(group_size, -1, inputs.shape[1], inputs.shape[2],
               self._num_features, inputs.shape[3] // self._num_features))
    outputs = tf.math.reduce_variance(outputs, axis=0)
    outputs = tf.math.sqrt(outputs + 1e-8)
    outputs = tf.reduce_mean(outputs, axis=(1, 2, 4), keepdims=True)
    outputs = tf.squeeze(outputs, axis=-1)
    outputs = tf.tile(outputs,
                      (group_size, inputs.shape[1], inputs.shape[2], 1))
    outputs = tf.concat((inputs, outputs), axis=-1)
    return outputs


class EqualDense(tf.keras.layers.Layer):
  """Dense layer with equalized learning rate."""

  def __init__(self,
               units: int,
               use_bias: bool = True,
               lr_multiplier: float = 1.,
               **kwargs: Any) -> None:
    """Initializer.

    Args:
      units: The number of output channels.
      use_bias: Whether the layer uses a bias vector.
      lr_multiplier: learning rate multiplier.
      **kwargs: Any other arguments for keras Layer.
    """
    super().__init__(**kwargs)
    self._units = units
    self._use_bias = use_bias
    self._lr_multiplier = lr_multiplier

  def build(self, input_shape: tf.TensorShape) -> None:
    input_dim = input_shape[-1]
    self._scale = (1. / math.sqrt(input_dim)) * self._lr_multiplier
    self.kernel = self.add_weight(
        "kernel",
        shape=(input_dim, self._units),
        initializer=tf.keras.initializers.RandomNormal(
            mean=0., stddev=1. / self._lr_multiplier))
    if self._use_bias:
      self.bias = self.add_weight(
          "bias", shape=(self._units,), initializer="zeros")
    super().build(input_shape)

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    outputs = tf.matmul(inputs, self.kernel * self._scale)
    if self._use_bias:
      outputs = tf.nn.bias_add(outputs, self.bias * self._lr_multiplier)
    return outputs


class EqualConv2D(tf.keras.layers.Layer):
  """Conv2D layer with equalized learning rate."""

  def __init__(self,
               filters: int,
               kernel_size: int,
               strides: Tuple[int, int] = (1, 1),
               padding: Text = "VALID",
               use_bias: bool = True,
               **kwargs: Any) -> None:
    """Initializer.

    Args:
      filters: The number of output channels.
      kernel_size: An integer for the height and width of the 2D convolution
        window.
      strides: An integer for the strides of the convolution along the height
        and width.
      padding: The type of padding ("VALID" or "SAME").
      use_bias: Whether the layer uses a bias vector.
      **kwargs: Any other arguments for keras Layer.
    """
    super().__init__(**kwargs)
    self._filters = filters
    self._kernel_size = kernel_size
    self._strides = strides
    self._padding = padding
    self._use_bias = use_bias

  def build(self, input_shape: tf.TensorShape) -> None:
    input_dim = input_shape[-1]
    self._scale = 1. / math.sqrt(input_dim * self._kernel_size**2)
    self.kernel = self.add_weight(
        "kernel",
        shape=(self._kernel_size, self._kernel_size, input_dim, self._filters),
        initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.))
    if self._use_bias:
      self.bias = self.add_weight(
          "bias", shape=(self._filters,), initializer="zeros")
    super().build(input_shape)

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    outputs = tf.nn.conv2d(
        inputs,
        self.kernel * self._scale,
        strides=self._strides,
        padding=self._padding)
    if self._use_bias:
      outputs = tf.nn.bias_add(outputs, self.bias)
    return outputs


class Block(tf.keras.layers.Layer):
  """Residual block for StyleGANDiscriminator."""

  def __init__(self,
               filters: int,
               relu_slope: float,
               blur_resample: bool = False,
               conv2d_cls: Type[Union[tf.keras.layers.Conv2D,
                                      EqualConv2D]] = tf.keras.layers.Conv2D,
               **kwargs: Any):
    """Initializer.

    Args:
      filters: The number of output channels.
      relu_slope: A float for the negative slope coefficient.
      blur_resample: Whether the blur downsample is used.
      conv2d_cls: The type of Conv2D layer (`EqualConv2D` or
        `tf.keras.layers.Conv2D`).
      **kwargs: Any other arguments for keras Layer.
    """
    super().__init__(**kwargs)
    self._filters = filters
    self._relu_slope = relu_slope
    self._blur_resample = blur_resample
    self._conv2d_cls = conv2d_cls

  def _make_pooling_layer(self):
    return BlurPool2D(
        filter_size=4
    ) if self._blur_resample else tf.keras.layers.AveragePooling2D()

  def build(self, input_shape: tf.TensorShape) -> None:
    input_dim = input_shape[-1]

    self.skip_layers = tf.keras.Sequential([
        self._make_pooling_layer(),
        self._conv2d_cls(self._filters, 1, use_bias=False)
    ])
    self.conv_layers = tf.keras.Sequential([
        self._conv2d_cls(input_dim, 3, padding="SAME"),
        tf.keras.layers.LeakyReLU(self._relu_slope),
        self._make_pooling_layer(),
        self._conv2d_cls(self._filters, 3, padding="SAME"),
        tf.keras.layers.LeakyReLU(self._relu_slope)
    ])
    super().build(input_shape)

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    outputs = self.conv_layers(inputs)
    inputs = self.skip_layers(inputs)
    outputs = (outputs + inputs) / math.sqrt(2.)
    return outputs


class StyleGANDiscriminator(tf.keras.Model):
  """StyleGAN discriminator architecture.

  Reference:
    Karras et al. Analyzing and Improving the Image Quality of StyleGAN.
    https://arxiv.org/pdf/1912.04958.pdf.
  """

  def __init__(self,
               input_size: int,
               channel_multiplier: int = 1,
               blur_resample: bool = False,
               use_equalized_lr: bool = False,
               use_batch_stddev: bool = False,
               **kwargs: Any):
    """Initializer.

    Args:
      input_size: An integer for the input image size.
      channel_multiplier: An integer for the factor of channel dimensions.
      blur_resample: Whether the blur downsample is used.
      use_equalized_lr: Whether the equalized learning rate is used.
      use_batch_stddev: Whether the minibatch standard deviation is used.
      **kwargs: Any other arguments for keras Layer.
    """
    super().__init__(**kwargs)

    relu_slope = 0.2
    channel_dim = {
        4: 512,
        8: 512,
        16: 512,
        32: 512,
        64: 256 * channel_multiplier,
        128: 128 * channel_multiplier,
        256: 64 * channel_multiplier,
        512: 32 * channel_multiplier,
        1024: 16 * channel_multiplier,
    }

    dense_cls = EqualDense if use_equalized_lr else tf.keras.layers.Dense
    conv2d_cls = EqualConv2D if use_equalized_lr else tf.keras.layers.Conv2D

    self.conv_blocks = tf.keras.Sequential([
        conv2d_cls(channel_dim[input_size], 3, padding="SAME"),
        tf.keras.layers.LeakyReLU(relu_slope)
    ])
    log_size = int(math.log2(input_size))
    for i in range(log_size, 2, -1):
      self.conv_blocks.add(
          Block(channel_dim[2**(i - 1)], relu_slope, blur_resample, conv2d_cls))

    if use_batch_stddev:
      self.conv_blocks.add(MinibatchStddev())

    self.final_layers = tf.keras.Sequential([
        conv2d_cls(channel_dim[4], 3, padding="SAME"),
        tf.keras.layers.LeakyReLU(relu_slope),
        tf.keras.layers.Flatten(),
        dense_cls(channel_dim[4]),
        tf.keras.layers.LeakyReLU(relu_slope),
        dense_cls(1)
    ])

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    outputs = self.conv_blocks(inputs)
    outputs = self.final_layers(outputs)
    return outputs
