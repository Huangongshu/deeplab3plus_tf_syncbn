# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for Residual Networks.

Residual networks ('v1' ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant was introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer
rather than after.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.nccl.ops import gen_nccl_ops
from tensorflow.contrib.framework import add_model_variable
from tensorflow.contrib.nccl.python.ops.nccl_ops import _validate_and_load_nccl_so
_validate_and_load_nccl_so()

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_DTYPE = tf.float32

bn_counter = 0

################################################################################
# Convenience functions for building the ResNet model.
################################################################################
def batch_norm(inputs,
               training,
               data_format='channels_last',
               num_dev=4,
               decay=_BATCH_NORM_DECAY,
               epsilon=_BATCH_NORM_EPSILON,
               activation_fn=None,
               updates_collections=tf.GraphKeys.UPDATE_OPS,
               reuse=None,
               variables_collections=None,
               trainable=True,):

  '''
  num_dev is how many gpus you use.
  '''

  red_axises = [0, 1, 2]
  num_outputs = inputs.get_shape().as_list()[-1]

  global bn_counter
  if bn_counter == 0:
    current_scope = 'batch_normalization'
  else:
    current_scope = 'batch_normalization_%d'%bn_counter
  bn_counter = bn_counter + 1

  with tf.variable_scope(current_scope, reuse=reuse):

    gamma = tf.get_variable(
        name='gamma', shape=[num_outputs], dtype=tf.float32,
        initializer=tf.constant_initializer(1.0), trainable=trainable,
        collections=variables_collections)

    beta  = tf.get_variable(
        name='beta', shape=[num_outputs], dtype=tf.float32,
        initializer=tf.constant_initializer(0.0), trainable=trainable,
        collections=variables_collections)

    moving_mean = tf.get_variable(
        name='moving_mean', shape=[num_outputs], dtype=tf.float32,
        initializer=tf.constant_initializer(0.0), trainable=False,
        collections=variables_collections)

    moving_var = tf.get_variable(
        name='moving_variance', shape=[num_outputs], dtype=tf.float32,
        initializer=tf.constant_initializer(1.0), trainable=False,
        collections=variables_collections)

    if training and trainable:

      if num_dev == 1:
        mean, var = tf.nn.moments(inputs, red_axises)
      else:
        shared_name = tf.get_variable_scope().name
        batch_mean = tf.reduce_mean(inputs, axis=red_axises)
        batch_mean_square = tf.reduce_mean(tf.square(inputs), axis=red_axises)
        batch_mean = gen_nccl_ops.nccl_all_reduce(
            input=batch_mean,
            reduction='sum',
            num_devices=num_dev,
            shared_name=shared_name + '_NCCL_mean') * (1.0 / num_dev)
        batch_mean_square = gen_nccl_ops.nccl_all_reduce(
            input=batch_mean_square,
            reduction='sum',
            num_devices=num_dev,
            shared_name=shared_name + '_NCCL_mean_square') * (1.0 / num_dev)
        mean = batch_mean
        var = batch_mean_square - tf.square(batch_mean)

      outputs = tf.nn.batch_normalization(
          inputs, mean, var, beta, gamma, epsilon)

      if int(outputs.device[-1])== 0:
        update_moving_mean_op = tf.assign(
            moving_mean, moving_mean * decay + mean * (1 - decay))
        update_moving_var_op  = tf.assign(
            moving_var,  moving_var  * decay + var  * (1 - decay))
        add_model_variable(moving_mean)
        add_model_variable(moving_var)

        if updates_collections is None:
          with tf.control_dependencies(
              [update_moving_mean_op, update_moving_var_op]):
            outputs = tf.identity(outputs)
        else:
          tf.add_to_collections(updates_collections, update_moving_mean_op)
          tf.add_to_collections(updates_collections, update_moving_var_op)
          outputs = tf.identity(outputs)
      else:
        outputs = tf.identity(outputs)

    else:
      outputs, _, _ = tf.nn.fused_batch_norm(
          inputs, gamma, beta, mean=moving_mean, variance=moving_var,
          epsilon=epsilon, is_training=False)

    if activation_fn is not None:
      outputs = activation_fn(outputs)

    return outputs


def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides,
                         data_format, dilation_rate=1):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format, dilation_rate=dilation_rate)


################################################################################
# ResNet block definitions.
################################################################################
def _bottleneck_block_v2(inputs, filters, training, projection_shortcut,
                         strides, data_format, dilation_rate=1):
  """A single block for ResNet v2, with a bottleneck.

  Similar to _building_block_v2(), except using the "bottleneck" blocks
  described in:
    Convolution then batch normalization then ReLU as described by:
      Deep Residual Learning for Image Recognition
      https://arxiv.org/pdf/1512.03385.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

  Adapted to the ordering conventions of:
    Batch normalization then ReLu then convolution as described by:
      Identity Mappings in Deep Residual Networks
      https://arxiv.org/pdf/1603.05027.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block; shape should match inputs.
  """
  shortcut = inputs
  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1,
      data_format=data_format)

  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format, dilation_rate=dilation_rate)

  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
      data_format=data_format)

  return inputs + shortcut


def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                training, name, data_format, dilation_rate):
  """Creates one layer of blocks for the ResNet model.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    bottleneck: Is the block created a bottleneck block.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block layer.
  """

  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = filters * 4 if bottleneck else filters

  def projection_shortcut(inputs):
    return conv2d_fixed_padding(
        inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
        data_format=data_format)

  # Only the first block per block_layer uses projection_shortcut and strides
  inputs = block_fn(inputs, filters, training, projection_shortcut, strides,
                    data_format, dilation_rate)

  for _ in range(1, blocks):
    inputs = block_fn(inputs, filters, training, None, 1, data_format,
                      dilation_rate)

  return tf.identity(inputs, name)


def resnet_aspp(inputs, training, num_classes, data_format):
  """Add operations to classify a batch of input images.

  Args:
    inputs: A Tensor representing a batch of input images.
    training: A boolean. Set to True to add operations required only when
      training the classifier.
    data_format: default is channels_first set by flags in main.

  Returns:
    A logits Tensor with shape [<batch_size>, num_classes].
  """

  resnet_size = 101
  block_fn = _bottleneck_block_v2
  num_filters = 64
  kernel_size = 7
  conv_stride = 2
  first_pool_size = 3
  first_pool_stride = 2
  block_sizes = [3, 4, 23, 3]
  block_strides = [1, 2, 2, 1]
  block_rates = [1, 1, 1, 2]

  global bn_counter
  bn_counter = 0

  with tf.variable_scope('resnet_model'):
    inputs_size = tf.shape(inputs)[1:3]

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=num_filters, kernel_size=kernel_size,
        strides=conv_stride, data_format=data_format)
    inputs = tf.identity(inputs, 'initial_conv')

    if first_pool_size:
      inputs = tf.layers.max_pooling2d(
          inputs=inputs, pool_size=first_pool_size,
          strides=first_pool_stride, padding='SAME',
          data_format=data_format)
      inputs = tf.identity(inputs, 'initial_max_pool')

    low_level_feature = None
    for i, num_blocks in enumerate(block_sizes):
      if i==1:
        low_level_feature = inputs

      inputs = block_layer(
          inputs=inputs, filters=(num_filters * (2**i)), bottleneck=True,
          block_fn=block_fn, blocks=num_blocks,
          strides=block_strides[i], training=training,
          name='block_layer{}'.format(i + 1), data_format=data_format,
          dilation_rate=block_rates[i])


    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    root = inputs

    root_size = tf.shape(inputs)[1:3]
    atrous_rates = [6, 12, 18]
    aspp_depth = 256
    with tf.variable_scope("dense"):
      conv1x1 = tf.layers.conv2d(inputs=root, filters=aspp_depth, kernel_size=1)

      pool = tf.reduce_mean(root, [1, 2], keepdims=True)
      pool = tf.layers.conv2d(pool, aspp_depth, [1, 1], use_bias=False)
      pool = tf.image.resize_bilinear(pool, root_size, name='upsample')

      aspps = [pool, conv1x1]

      for r in atrous_rates:
        inputs = tf.layers.conv2d(
            inputs=root, filters=aspp_depth, kernel_size=3,
            dilation_rate=r, padding='same', use_bias=False)
        aspps.append(inputs)

      aspp = tf.concat(aspps, axis=3, name='concat')

      aspp = batch_norm(aspp, training, data_format)
      aspp = tf.nn.relu(aspp)
      aspp = tf.layers.conv2d(inputs=aspp, filters=aspp_depth, kernel_size=1)

      low_level_feature_size = tf.shape(low_level_feature)[1:3]
      low_level_feature = batch_norm(low_level_feature, training, data_format)
      low_level_feature = tf.nn.relu(low_level_feature)
      low_level_feature = tf.layers.conv2d(
          inputs=low_level_feature, filters=48, kernel_size=1)
      aspp = tf.image.resize_bilinear(aspp, low_level_feature_size)
      mix_feature = tf.concat([aspp, low_level_feature], axis=3)

      for i in range(2):
        mix_feature = batch_norm(mix_feature, training, data_format)
        mix_feature = tf.nn.relu(mix_feature)
        mix_feature = tf.layers.conv2d(
            inputs=mix_feature, filters=256, kernel_size=3)

      logits  = tf.layers.conv2d(
          inputs=mix_feature, filters=num_classes, kernel_size=1)
      logits = tf.image.resize_bilinear(logits, inputs_size, name='upsample')

    return logits
