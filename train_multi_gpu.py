"""Train a DeepLab v3 model using tf.estimator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
import experiment_multi_gpu
import preprocessing
import pascal_voc_dataset

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', type=str, default='shit',
                    help='Base directory for the model.')

parser.add_argument('--bs_per_gpu', type=int, default=8,
                    help='Number of examples per batch.')

parser.add_argument('--max_iter', type=int, default=30000,
                    help='maximum iteration num for "poly" lr policy.')

parser.add_argument('--model_variant', type=str, default='deeplab_v3_aspp',
                    choices=['deeplab_v3_aspp'],
                    help='The architecture of base Resnet building block.')

parser.add_argument('--output_stride', type=int, default=16,
                    choices=[8, 16],
                    help='Currently 8 or 16 is supported.')

parser.add_argument('--freeze_batch_norm', action='store_true',
                    help='Freeze BN parameters during the training.')

parser.add_argument('--initial_learning_rate', type=float, default=7e-3,
                    help='Initial learning rate for the optimizer.')

parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='The weight decay to use for regularizing the model.')

parser.add_argument('--dataset', type=str, default='pascal_voc',
                    choices=['pascal_voc', 'cityscapes'],
                    help='Path to the pre-trained model checkpoint.')

parser.add_argument('--num_gpus', type=int, default=2,
                    help='How many GPUs to use.')

parser.add_argument('--eval_only', action='store_true',
                    help='Freeze BN parameters during the training.')

parser.add_argument('--visualize_outcome', action='store_true',
                    help='Stor visualization in vis_out directory')

_NUM_CLASSES = {'cityscapes': 19, 'pascal_voc': 21}
_DATASETS = {'cityscapes': cityscapes_dataset, 'pascal_voc': pascal_voc_dataset}
_MODELS_ROOT_PATH = '/home/wuboxi/deep/segmentation_v3/logdir/'

def main(unused_argv):

  if FLAGS.dataset not in ['cityscapes', 'pascal_voc']:
    tf.logging.info("Unknown dataset!")
    return

  model_dir = _MODELS_ROOT_PATH + FLAGS.model_dir

  input_fn = _DATASETS[FLAGS.dataset].input_fn

  session_config = tf.ConfigProto(allow_soft_placement=True)
  run_config = tf.estimator.RunConfig(
      session_config=session_config,
      save_checkpoints_secs=60*60*24)

  warm_start_settings = tf.estimator.WarmStartSettings(
      '/home/wuboxi/deep/classification/logdir/base',
      vars_to_warm_start='^(?!.*dense)')

  model = tf.estimator.Estimator(
      model_fn=experiment_multi_gpu.model_adaptor,
      model_dir=model_dir,
      config=run_config,
      warm_start_from=warm_start_settings,
      params={
          'num_gpus': FLAGS.num_gpus,
          'output_stride': FLAGS.output_stride,
          'model_variant': FLAGS.model_variant,
          'num_classes': _NUM_CLASSES[FLAGS.dataset],
          'weight_decay': FLAGS.weight_decay,
          'initial_learning_rate': FLAGS.initial_learning_rate,
          'max_iter': FLAGS.max_iter,
          'freeze_batch_norm': FLAGS.freeze_batch_norm,
          'visualize_outcome': False,
      })

  if not FLAGS.eval_only:
    tf.logging.info("Start training.")

    tensors_to_log = {
      'learning_rate': 'learning_rate',
      'total_loss': 'total_loss',
      'train_mean_iou': 'train_mean_iou',
    }

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10)
    train_hooks = [logging_hook]

    model.train(
        input_fn=lambda: input_fn(
            True, 'train', FLAGS.num_gpus, FLAGS.bs_per_gpu, None),
        hooks=train_hooks,
        steps=30000
    )

  tf.logging.info("Start evaluation.")
  eval_results = model.evaluate(
      input_fn=lambda: input_fn(False, 'val', 1, 1, 1)
  )
  print(eval_results)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
