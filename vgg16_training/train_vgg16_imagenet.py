#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import logging
import pathlib

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomCrop

from utils import plotSamples, printMsg, disableTFlogging, SaveHistory
from utils import PreprocessVGG16, Rotate90
from image_data import image_dataset_from_directory
from vgg16 import loadVGG16

disableTFlogging()

# Set memory growth option for tf
physical_devices = tf.config.list_physical_devices('GPU')
for dev in physical_devices:
    tf.config.experimental.set_memory_growth(dev, True)

for dev in physical_devices:
    print(dev, tf.config.experimental.get_memory_growth(dev))

printMsg("Loading data...")
train_dir = pathlib.Path("./imagenet_dataset/train/") # ImageNet2012 train data path
val_dir = pathlib.Path("./imagenet_dataset/val/") # ImageNet2012 val data path

# batch size and image resize size
batch_size = 150
img_height = 256
img_width = 256

# For train data it resize the image such that smallest side is 256 and then take a random crop of 256 x 256 [Simonyan, K., & Zisserman, A. (2014)]
# Resize operation keeps the aspect ratio intact
train_ds = image_dataset_from_directory(
    train_dir,
    image_size=(img_height, img_width),
    crop_method='random',
    shuffle=True,
    preserve_aspect_ratio=True,
    batch_size=batch_size)

# For validation data it resize the image such that smallest side is 256 and then take a centre crop of 256 x 256 [Simonyan, K., & Zisserman, A. (2014)]
# Resize operation keeps the aspect ratio intact
val_ds = image_dataset_from_directory(
    val_dir,
    image_size=(img_height, img_width),
    crop_method='center',
    shuffle=False,
    preserve_aspect_ratio=True,
    batch_size=batch_size)

# Apply data augmentation and preprocessing
# When training flag is off RandomCrop resize and then centre crop
data_augmentation = tf.keras.Sequential([RandomFlip("horizontal"), RandomCrop(224, 224), Rotate90()])
data_preprocess = tf.keras.Sequential([PreprocessVGG16()])

augmented_train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
preproccesed_train_ds = augmented_train_ds.map(lambda x, y: (data_preprocess(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
preproccesed_train_ds = preproccesed_train_ds.prefetch(tf.data.experimental.AUTOTUNE)

# training flag false therefore RandomCrop will resize and then centre crop
augmented_val_ds = val_ds.map(lambda x, y: (data_augmentation(x, training=False), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
preproccesed_val_ds = augmented_val_ds.map(lambda x, y: (data_preprocess(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
preproccesed_val_ds = preproccesed_val_ds.prefetch(tf.data.experimental.AUTOTUNE)

# Set tf MirroredStrategy for multi GPU training
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2"])
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


printMsg("Creating Model...")
with strategy.scope():
    model = loadVGG16((224, 224, 3))
    opt = tfa.optimizers.AdamW(lr=1e-4, weight_decay=1e-5) #Adam with weight decay
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

save_dir = "./saved_model/VGG16_Imagenet_WD_Adam/"
save_ver = 0
os.makedirs(save_dir, exist_ok=True)

history_callback = SaveHistory(save_dir, ver=save_ver, opt='adam')
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir + 'model_' + str(save_ver) + '.h5',
                                                     monitor='val_sparse_categorical_accuracy',
                                                     save_best_only=True,
                                                     save_weights_only=True)

if save_ver > 0:
    printMsg("Loading latest weights")
    model.load_weights(save_dir + 'model_' + str(save_ver-1) + '.h5')
    model.evaluate(preproccesed_val_ds)


printMsg("Start training...")
epochs = 60
history = model.fit(preproccesed_train_ds, validation_data=preproccesed_val_ds, epochs=epochs, callbacks=[model_checkpoint, history_callback])
