#!/usr/bin/env python
# coding: utf-8

# Code taken from https://github.com/tensorflow/tensorflow/blob/85c8b2a817f95a3e979ecd1ed95bff1dc1335cff/tensorflow/python/keras/preprocessing/image_dataset.py
# Modified to add random crop and resize OR center crop and resize to preserve aspect ratio

import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.keras.layers.preprocessing import image_preprocessing
from tensorflow.python.keras.preprocessing import dataset_utils
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.framework import ops
from tensorflow.python.keras.preprocessing.image import smart_resize

def random_crop_resize(x, size, interpolation='bilinear'):
  img = ops.convert_to_tensor_v2_with_dispatch(x)
  shape = array_ops.shape(img)
  height, width, ch = shape[0], shape[1], shape[2]
  crop_size = math_ops.minimum(height, width)
  img = random_ops.random_crop(img, [crop_size, crop_size, ch])
  img = image_ops.resize_images_v2(images=img, size=size, method=interpolation)
  if isinstance(x, np.ndarray):
    return img.numpy()
  return img

ALLOWLIST_FORMATS = ('.bmp', '.gif', '.jpeg', '.jpg', '.png')

def image_dataset_from_directory(directory,
                                 labels='inferred',
                                 label_mode='int',
                                 class_names=None,
                                 color_mode='rgb',
                                 batch_size=32,
                                 image_size=(256, 256),
                                 crop_method='center',
                                 shuffle=True,
                                 seed=None,
                                 validation_split=None,
                                 subset=None,
                                 interpolation='bilinear',
                                 follow_links=False,
                                 preserve_aspect_ratio=True):

  if labels != 'inferred':
    if not isinstance(labels, (list, tuple)):
      raise ValueError(
          '`labels` argument should be a list/tuple of integer labels, of '
          'the same size as the number of image files in the target '
          'directory. If you wish to infer the labels from the subdirectory '
          'names in the target directory, pass `labels="inferred"`. '
          'If you wish to get a dataset that only contains images '
          '(no labels), pass `label_mode=None`.')
    if class_names:
      raise ValueError('You can only pass `class_names` if the labels are '
                       'inferred from the subdirectory names in the target '
                       'directory (`labels="inferred"`).')
  if label_mode not in {'int', 'categorical', 'binary', None}:
    raise ValueError(
        '`label_mode` argument must be one of "int", "categorical", "binary", '
        'or None. Received: %s' % (label_mode,))
  if color_mode == 'rgb':
    num_channels = 3
  elif color_mode == 'rgba':
    num_channels = 4
  elif color_mode == 'grayscale':
    num_channels = 1
  else:
    raise ValueError(
        '`color_mode` must be one of {"rbg", "rgba", "grayscale"}. '
        'Received: %s' % (color_mode,))
  interpolation = image_preprocessing.get_interpolation(interpolation)
  dataset_utils.check_validation_split_arg(
      validation_split, subset, shuffle, seed)

  if seed is None:
    seed = np.random.randint(1e6)
  image_paths, labels, class_names = dataset_utils.index_directory(
      directory,
      labels,
      formats=ALLOWLIST_FORMATS,
      class_names=class_names,
      shuffle=shuffle,
      seed=seed,
      follow_links=follow_links)

  if label_mode == 'binary' and len(class_names) != 2:
    raise ValueError(
        'When passing `label_mode="binary", there must exactly 2 classes. '
        'Found the following classes: %s' % (class_names,))

  image_paths, labels = dataset_utils.get_training_or_validation_split(
      image_paths, labels, validation_split, subset)

  dataset = paths_and_labels_to_dataset(
      image_paths=image_paths,
      image_size=image_size,
      crop_method=crop_method,
      num_channels=num_channels,
      labels=labels,
      label_mode=label_mode,
      num_classes=len(class_names),
      interpolation=interpolation,
      preserve_aspect_ratio=preserve_aspect_ratio)
  if shuffle:
    # Shuffle locally at each iteration
    dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
  dataset = dataset.batch(batch_size)
  # Users may need to reference `class_names`.
  dataset.class_names = class_names
  # Include file paths for images as attribute.
  dataset.file_paths = image_paths
  return dataset


def paths_and_labels_to_dataset(image_paths,
                                image_size,
                                crop_method,
                                num_channels,
                                labels,
                                label_mode,
                                num_classes,
                                interpolation,
                                preserve_aspect_ratio):
  """Constructs a dataset of images and labels."""

  path_ds = dataset_ops.Dataset.from_tensor_slices(image_paths)
  img_ds = path_ds.map(
      lambda x: path_to_image(x, image_size, num_channels, interpolation, crop_method, preserve_aspect_ratio))
  if label_mode:
    label_ds = dataset_utils.labels_to_dataset(labels, label_mode, num_classes)
    img_ds = dataset_ops.Dataset.zip((img_ds, label_ds))
  return img_ds


def path_to_image(path, image_size, num_channels, interpolation, crop_method, preserve_aspect_ratio):
  img = io_ops.read_file(path)
  img = image_ops.decode_image(img, channels=num_channels, expand_animations=False)

  if preserve_aspect_ratio:
    if crop_method=='center':
        img = smart_resize(img, image_size, interpolation=interpolation)
    else:
        img = random_crop_resize(img, image_size, interpolation=interpolation)
  else:
    img = image_ops.resize_images_v2(img, image_size, method=interpolation)

  img.set_shape((image_size[0], image_size[1], num_channels))
  return img
