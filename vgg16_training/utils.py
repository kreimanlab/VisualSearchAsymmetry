import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
import os
import logging
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

# History logger
class SaveHistory(tf.keras.callbacks.Callback):
    def __init__(self, save_dir, ver=0, opt='sgd'):
        super(SaveHistory, self).__init__()
        self.historye = {}
        self.save_dir = save_dir
        self.ver = ver
        self.iteration = 0
        self.history = {}
        self.opt = opt

    def on_train_begin(self, logs=None):
        self.epoch = []

    def on_batch_end(self, epoch, logs=None):
        self.iteration += 1
        if self.iteration%100 == 0:
            logs = logs or {}

            self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
            if self.opt == 'sgd':
                self.history.setdefault('momentum', []).append(K.get_value(self.model.optimizer.momentum))
            elif self.opt == 'adam':
                self.history.setdefault('beta_1', []).append(K.get_value(self.model.optimizer.beta_1))

            self.history.setdefault('iterations', []).append(self.iteration)

            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)

            if self.save_dir != None:
                hisMat = []
                for keys in self.history:
                    tmp = [keys] + self.history[keys]
                    hisMat.append(tmp)

                np.savetxt(self.save_dir + "batch_history_" + str(self.ver) + ".csv", hisMat, fmt="%s")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.historye.setdefault(k, []).append(v)

        hisMat = []
        tmp = ['epoch'] + self.epoch
        hisMat.append(tmp)
        for keys in self.historye:
            tmp = [keys] + self.historye[keys]
            hisMat.append(tmp)

        np.savetxt(self.save_dir + "history_" + str(self.ver) + ".csv", hisMat, fmt="%s")

# Defining 90 degree Rotate operation as tf.keras layers to apply this as a data augmentation to the datasets.
def rotate_img(x):
    return tf.image.rot90(x)

class Rotate90(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return rotate_img(x)

# VGG16 Preprocessing
class PreprocessVGG16(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return preprocess_input(x)


def disableTFlogging():
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    os.environ["KMP_AFFINITY"] = "noverbose"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(3)

def plotSamples(ds, class_names):
    plt.figure(figsize=(10, 10))
    for images, labels in ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")

def printMsg(txtmsg):
    print()
    print("".join(["#"]*100))
    print("".join(["#"]*11), "".join([" "]*(len(txtmsg)+2)), "".join(["#"]*(98-10-3-len(txtmsg))))
    print("".join(["#"]*10), " ", txtmsg, " ", "".join(["#"]*(98-10-4-len(txtmsg))))
    print("".join(["#"]*11), "".join([" "]*(len(txtmsg)+2)), "".join(["#"]*(98-10-3-len(txtmsg))))
    print("".join(["#"]*100))
    print()

class LRFinder(Callback):
    '''
    A simple callback for finding the optimal learning rate range for your model + dataset.

    # Usage
        ```python
            lr_finder = LRFinder(min_lr=1e-5,
                                 max_lr=1e-2,
                                 steps_per_epoch=np.ceil(epoch_size/batch_size),
                                 epochs=3)
            model.fit(X_train, Y_train, callbacks=[lr_finder])

            lr_finder.plot_loss()
        ```

    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`.
        epochs: Number of epochs to run experiment. Usually between 2 and 4 epochs is sufficient.

    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: https://arxiv.org/abs/1506.01186
    '''

    def __init__(self, min_lr=1e-5, max_lr=1e-2, steps_per_epoch=None, epochs=None):
        super().__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iterations = steps_per_epoch * epochs
        self.iteration = 0
        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        x = self.iteration / self.total_iterations
        return self.min_lr + (self.max_lr-self.min_lr) * x

    def on_train_begin(self, logs=None):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.min_lr)

    def on_batch_end(self, epoch, logs=None):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.iteration += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.iteration)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())

    def plot_lr(self):
        '''Helper function to quickly inspect the learning rate schedule.'''
        plt.plot(self.history['iterations'], self.history['lr'])
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')
        plt.show()

    def plot_loss(self):
        '''Helper function to quickly observe the learning rate experiment results.'''
        plt.plot(self.history['lr'], self.history['loss'])
        plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')
        plt.show()
