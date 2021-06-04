#!/usr/bin/env python
# coding: utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print(os.environ["CUDA_VISIBLE_DEVICES"])

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import preprocess_input
from vgg16 import loadVGG16

physical_devices = tf.config.list_physical_devices('GPU')
for dev in physical_devices:
    tf.config.experimental.set_memory_growth(dev, True)

for dev in physical_devices:
    print(dev, tf.config.experimental.get_memory_growth(dev))

train_dir = pathlib.Path("./mnist_datasets/train/")
val_dir = pathlib.Path("./mnist_datasets/val/")

batch_size = 32
img_height = 224
img_width = 224

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class PreprocessVGG16(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return preprocess_input(x)

data_preprocess = tf.keras.Sequential(
    [
        PreprocessVGG16(),
    ]
)


preproccesed_train_ds = train_ds.map(lambda x, y: (data_preprocess(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
preproccesed_val_ds = val_ds.map(lambda x, y: (data_preprocess(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)

preproccesed_train_ds = preproccesed_train_ds.prefetch(tf.data.experimental.AUTOTUNE)
preproccesed_val_ds = preproccesed_val_ds.prefetch(tf.data.experimental.AUTOTUNE)


model = loadVGG16((224, 224, 3))
opt = tf.keras.optimizers.Adam(lr=1e-4)
model.compile(
    optimizer=opt,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint('./saved_model/VGG16_MNIST.h5',
                                                     monitor='val_sparse_categorical_accuracy',
                                                     save_best_only=True,
                                                     save_weights_only=True)

EPOCHS = 20
history = model.fit(preproccesed_train_ds, validation_data=preproccesed_val_ds, epochs=EPOCHS)
model.load_weights('./saved_model/VGG16_MNIST.h5')
model.evaluate(preproccesed_val_ds)
