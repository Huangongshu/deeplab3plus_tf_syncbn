import os
import tensorflow as tf
import preprocessing

_HEIGHT = 513
_WIDTH = 513
_IGNORE_LABEL = 255
_MIN_SCALE = 0.5
_MAX_SCALE = 2.0
_NUM_IMAGES = {
    'train': 10582,
    'validation': 1449,
}

_TF_RECORDS_DIR = '/home/wuboxi/pascal_voc_data/tf_records/'

def get_filenames(split):
  if split not in ['train', 'val']:
    tf.logging.info("Unknown pascal voc split!")
    return [os.path.join(_TF_RECORDS_DIR, 'voc_train.record')]
  else:
    return [os.path.join(_TF_RECORDS_DIR, 'voc_%s.record' % split)]


def parse_record(raw_record):
  """Parse PASCAL image and label from a tf record."""
  keys_to_features = {
      'image/height':
      tf.FixedLenFeature((), tf.int64),
      'image/width':
      tf.FixedLenFeature((), tf.int64),
      'image/encoded':
      tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format':
      tf.FixedLenFeature((), tf.string, default_value='jpeg'),
      'label/encoded':
      tf.FixedLenFeature((), tf.string, default_value=''),
      'label/format':
      tf.FixedLenFeature((), tf.string, default_value='png'),
  }

  parsed = tf.parse_single_example(raw_record, keys_to_features)

  image = tf.image.decode_image(
      tf.reshape(parsed['image/encoded'], shape=[]), 3)
  image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
  image.set_shape([None, None, 3])

  label = tf.image.decode_image(
      tf.reshape(parsed['label/encoded'], shape=[]), 1)
  label = tf.to_int32(tf.image.convert_image_dtype(label, dtype=tf.uint8))
  label.set_shape([None, None, 1])

  return image, label


def preprocess_image(image, label, is_training):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training:
    # Randomly scale the image and label.
    image, label = preprocessing.random_rescale_image_and_label(
        image, label, _MIN_SCALE, _MAX_SCALE)

  # Randomly crop or pad a [_HEIGHT, _WIDTH] section of the image and label.
  image, label = preprocessing.random_crop_or_pad_image_and_label(
      image, label, _HEIGHT, _WIDTH, _IGNORE_LABEL)

  if is_training:
    # Randomly flip the image and label horizontally.
    image, label = preprocessing.random_flip_left_right_image_and_label(
        image, label)

  image.set_shape([_HEIGHT, _WIDTH, 3])
  label.set_shape([_HEIGHT, _WIDTH, 1])

  original_image = image
  processed_image = preprocessing.mean_image_subtraction(image)

  return processed_image, label


def input_fn(is_training,
             split,
             num_gpus,
             bs_per_gpu,
             num_epochs):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A tuple of images and labels.
  """
  total_batch_size = num_gpus * bs_per_gpu
  dataset = tf.data.Dataset.from_tensor_slices(get_filenames(split))
  dataset = dataset.flat_map(tf.data.TFRecordDataset)

  if is_training:
    dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'])

  dataset = dataset.map(parse_record)
  dataset = dataset.map(
      lambda image, label: preprocess_image(image, label, is_training))
  dataset = dataset.prefetch(total_batch_size)
  if num_epochs is not None:
    dataset = dataset.repeat(num_epochs)
  else:
    dataset = dataset.repeat()
  dataset = dataset.batch(total_batch_size)

  image_batch, label_batch = dataset.make_one_shot_iterator().get_next()
  if num_gpus == 1:
    return [image_batch], [label_batch]

  image_batch = tf.unstack(image_batch, num=total_batch_size, axis=0)
  label_batch = tf.unstack(label_batch, num=total_batch_size, axis=0)
  image_shards = [[] for i in range(num_gpus)]
  label_shards = [[] for i in range(num_gpus)]
  for i, val in enumerate(zip(image_batch,label_batch)):
    image, label = val
    idx = int(i / bs_per_gpu)
    image_shards[idx].append(image)
    label_shards[idx].append(label)
  image_shards = [tf.parallel_stack(x) for x in image_shards]
  label_shards = [tf.parallel_stack(x) for x in label_shards]

  return image_shards, label_shards
