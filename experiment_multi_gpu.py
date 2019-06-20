"""Model adaptor for segmentation."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import deeplab_plus_sync
import preprocessing

IGNORE_LABEL = 255
_MOMENTUM = 0.9
_POWER = 0.9
_END_LEARNING_RATE = 1e-6

# colour map
LABEL_COLOURS = [
    # 0=background
    (0, 0, 0),
    # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
    (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
    # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
    (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
    # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
    (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
    # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
    (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]


def compute_mean_iou(total_cm, name='mean_iou'):
  """Compute the mean intersection-over-union via the confusion matrix."""
  sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
  sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
  cm_diag = tf.to_float(tf.diag_part(total_cm))
  denominator = sum_over_row + sum_over_col - cm_diag

  num_valid_entries = tf.reduce_sum(tf.cast(
      tf.not_equal(denominator, 0), dtype=tf.float32))

  denominator = tf.where(
      tf.greater(denominator, 0),
      denominator,
      tf.ones_like(denominator))
  iou = tf.div(cm_diag, denominator)

  result = tf.where(
      tf.greater(num_valid_entries, 0),
      tf.reduce_sum(iou, name=name) / num_valid_entries,
      0)
  return result


def _average_gradients(tower_grads):
  """Calculates average of gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list is
      over individual gradients. The inner list is over the gradient calculation
      for each tower.

  Returns:
     List of pairs of (gradient, variable) where the gradient has been summed
       across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads, variables = zip(*grad_and_vars)
    grad = tf.reduce_mean(tf.stack(grads, axis=0), axis=0)

    # All vars are of the same value, using the first tower here.
    average_grads.append((grad, variables[0]))

  return average_grads


def _tower_fn(features, labels, training, num_classes, weight_decay, optimizer):
  """Build computation tower (Resnet).

  Args:
    is_training: true if is training graph.
    weight_decay: weight regularization strength, a float.
    feature: a Tensor.
    label: a Tensor.
    data_format: channels_last (NHWC) or channels_first (NCHW).
    num_layers: number of layers, an int.
    batch_norm_decay: decay for batch normalization, a float.
    batch_norm_epsilon: epsilon for batch normalization, a float.

  Returns:
    A tuple with the loss for the tower, the gradients and parameters, and
    predictions.

  """
  logits = deeplab_plus_sync.resnet_aspp(
      features, training, num_classes, 'channels_last')

  valid_mask = tf.to_float(tf.not_equal(labels, IGNORE_LABEL)) * 1.0
  labels = labels * tf.to_int32(valid_mask)

  preds = tf.expand_dims(
      tf.argmax(logits, axis=3, output_type=tf.int32), axis=3)

  mean_iou = tf.metrics.mean_iou(
      labels, preds, num_classes, valid_mask)

  tower_pred = {
      'classes': preds,
      'probabilities': tf.nn.softmax(logits),
      'mean_iou': mean_iou,
  }

  tower_loss = tf.losses.sparse_softmax_cross_entropy(
      labels=labels, logits=logits, weights=valid_mask)

  model_params = [v for v in tf.trainable_variables()
                  if 'batch_normalization' not in v.name]
  tower_loss += weight_decay * tf.add_n(
      [tf.nn.l2_loss(v) for v in model_params])

  tower_grad = optimizer.compute_gradients(tower_loss)

  return tower_loss, tower_grad, tower_pred


def model_adaptor(features, labels, mode, params):
  """Model function for PASCAL VOC."""
  training = mode == tf.estimator.ModeKeys.TRAIN

  global_step = tf.train.get_or_create_global_step()

  learning_rate = tf.train.polynomial_decay(
      params['initial_learning_rate'],
      tf.cast(global_step, tf.int32),
      params['max_iter'], _END_LEARNING_RATE, power=_POWER)

  optimizer = tf.train.MomentumOptimizer(
      learning_rate=learning_rate,
      momentum=_MOMENTUM)

  tower_losses = []
  tower_grads = []
  tower_preds = []
  num_clones = params['num_gpus']
  if not training:
    num_clones = 1
  for i in range(num_clones):
    with tf.variable_scope(tf.get_variable_scope(), reuse=bool(i != 0)):
      with tf.name_scope('clone_%d' % i) as name_scope:
        with tf.device('/gpu:%d' % i):
          loss, grad, pred = _tower_fn(
              features[i], labels[i], training, params['num_classes'],
              params['weight_decay'], optimizer)
          tower_losses.append(loss)
          tower_grads.append(grad)
          tower_preds.append(pred)

  with tf.device('/cpu:0'):
    total_loss = tf.add_n(tower_losses)

  if mode == tf.estimator.ModeKeys.EVAL:
    vis_hooks = None
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=total_loss,
        eval_metric_ops={'mean_iou': tower_preds[0]['mean_iou']},
        evaluation_hooks=vis_hooks)
  elif mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions={})

  with tf.device('/cpu:0'):
    grads_and_vars = _average_gradients(tower_grads)
    grad_updates = optimizer.apply_gradients(
        grads_and_vars, global_step=global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    update_ops.append(grad_updates)
    update_op = tf.group(*update_ops)
    with tf.control_dependencies([update_op]):
      train_op = tf.identity(total_loss, name='total_loss')

    tower_mious = []
    for pred in tower_preds:
      tower_miou = compute_mean_iou(pred['mean_iou'][1])
      tower_mious.append(tower_miou)
    train_mean_iou = tf.add_n(tower_mious) / params['num_gpus']

    tf.identity(train_mean_iou, name='train_mean_iou')
    tf.summary.scalar('train_mean_iou', train_mean_iou)

    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    tf.summary.scalar('total_loss', total_loss)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=total_loss,
        train_op=train_op)
