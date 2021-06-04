import tensorflow as tf
from tensorflow.keras import layers

# VGG16 architecture as per the Simonyan, K., & Zisserman, A. (2014)
initializer = tf.keras.initializers.GlorotNormal()
def loadVGG16(input_shape=None):
    img_input = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(64, (3, 3), kernel_initializer=initializer, activation='relu', padding='same', name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3), kernel_initializer=initializer, activation='relu', padding='same', name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3), kernel_initializer=initializer, activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3), kernel_initializer=initializer, activation='relu', padding='same', name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3), kernel_initializer=initializer, activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3), kernel_initializer=initializer, activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3), kernel_initializer=initializer, activation='relu', padding='same', name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3), kernel_initializer=initializer, activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3), kernel_initializer=initializer, activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3), kernel_initializer=initializer, activation='relu', padding='same', name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3), kernel_initializer=initializer, activation='relu', padding='same', name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3), kernel_initializer=initializer, activation='relu', padding='same', name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3), kernel_initializer=initializer, activation='relu', padding='same', name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(4096, kernel_initializer=initializer, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, kernel_initializer=initializer, activation='relu', name='fc2')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1000, kernel_initializer=initializer, activation='softmax', name='predictions')(x)

    model = tf.keras.Model(inputs=img_input, outputs=x)

    return model
