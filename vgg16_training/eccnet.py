import tensorflow as tf
from tensorflow.keras import layers

import numpy as np
from skimage.draw import circle

# These are the default eccNET model parameters to achieve the eccentricity dependent sampling similar to the data found in macaque (see Fig # in the paper)
eccParamDefault = {}
eccParamDefault['rf_min'] = [2]*5
eccParamDefault['stride'] = [2]*5
eccParamDefault['ecc_slope'] = [0, 0, 3.5*0.02, 8*0.02, 16*0.02]
eccParamDefault['deg2px'] = [round(30.0), round(30.0/2), round(30.0/4), round(30.0/8), round(30.0/16)]
eccParamDefault['fovea_size'] = 4
eccParamDefault['rf_quant'] = 1
eccParamDefault['pool_type'] = 'avg'

class CentreDependentPooling2D(tf.keras.layers.Layer):
    """
    Eccentricity dependent pooling operation for 2D spatial data.
    It simply creates a multiple number of pooling layers of different sizes.
    And also creates a mask corresponding to each of the pooling layers were the specified pooling window size should be used.
    To get the ouput of eccentricity dependent pooling layer. It computes the sum of Mask[i]*Pooling_Lays[i]

    # Arguments
    - rf_min: minimum receptive field size in px
    - ecc_slope: rate at which the size of pooling window will increase w.r.t eccentricity
    - deg2px: number of pixels in 1 degree visual angle (dva)
    - fovea_size: size of fovea region in degrees
    - rf_quant: dva from the centre at which the next pooling layer window will be created.
    - stride: stride of the pooling layers
    - pool_type: type of the pooling operation 'max' or 'avg'
    - layer: not a necessary parameter was used in defining eccNET architecture where layer = the pooling layer number of VGG16 architecture.

    """

    def __init__(self, rf_min=2, ecc_slope=0.2, deg2px=30, fovea_size=4, rf_quant=1, stride=2, pool_type='max', layer=0, **kwargs):
        super(CentreDependentPooling2D, self).__init__(**kwargs)
        self.stride = stride
        self.pool_type = pool_type
        self.data_format = tf.keras.backend.image_data_format()
        self.fovea_size = fovea_size
        self.rf_min = rf_min
        self.ecc_slope = ecc_slope
        self.deg2px = deg2px
        self.rf_quant = rf_quant
        self.layer = layer
        self.ecc_border = []

    def build(self, input_shape):
        """
        This will create the list of receptive field size corresponding to each eccentricity intervals.
        Also, creates the binary mask corresponding to each of the rf sizes and eccentricity intervals.
        These rf size were used during the pooling operation to perform corresponding pooling size operation
        only in the region ariund the corresponding eccentricity level
        """

        if self.data_format == 'channels_first':
            (self.batch_size, self.channels, self.rows, self.cols) = input_shape.as_list()
        elif self.data_format == 'channels_last':
            (self.batch_size, self.rows, self.cols, self.channels) = input_shape.as_list()

        self.base = max(self.rows, self.cols)
        if self.stride > 1:
            self.out_shape = np.int((self.base+1)/(self.stride))
        else:
            self.out_shape = np.int(self.base)

        c = int(self.out_shape/2)

        mask_size = np.zeros((self.out_shape, self.out_shape, self.channels))

        self.mask = []
        self.rf_sizes = []

        ecc = round((self.fovea_size*self.deg2px)/2)

        if ecc > self.out_shape/2:
            temp_mask = np.ones(mask_size.shape)
            (self.mask).append(tf.constant(temp_mask, tf.float32))
            (self.ecc_border).append(2*ecc)
            (self.rf_sizes).append(self.rf_min)
        else:
            # first mask -----
            temp_mask = np.copy(mask_size)
            rr, cc = circle(c, c, ecc)
            temp_mask[rr, cc, :] = 1
            (self.mask).append(tf.constant(temp_mask, tf.float32))
            (self.ecc_border).append(2*ecc)
            (self.rf_sizes).append(self.rf_min)

            # mid mask ------
            ecc += round((self.rf_quant*self.deg2px)/2)
            while ecc < self.out_shape/2:
                temp_mask = np.copy(mask_size)
                rr, cc = circle(c, c, ecc)
                temp_mask[rr, cc, :] = 1
                rr, cc = circle(c, c, ecc-round((self.rf_quant*self.deg2px)/2))
                temp_mask[rr, cc, :] = 0
                (self.mask).append(tf.constant(temp_mask, tf.float32))
                (self.ecc_border).append(2*ecc)
                (self.rf_sizes).append(self.rf_min + round(self.ecc_slope*(ecc*2 - self.fovea_size*self.deg2px)))
                ecc += round((self.rf_quant*self.deg2px)/2)

            # end mask --------
            temp_mask = np.ones(mask_size.shape)
            rr, cc = circle(c, c, ecc-round((self.rf_quant*self.deg2px)/2))
            temp_mask[rr, cc, :] = 0
            (self.mask).append(tf.constant(temp_mask, tf.float32))
            (self.ecc_border).append(2*ecc)
            (self.rf_sizes).append(self.rf_min + round(self.ecc_slope*(ecc*2 - self.fovea_size*self.deg2px)))

        super(CentreDependentPooling2D, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            return tf.TensorShape((input_shape[0], input_shape[1], tf.TensorShape(self.out_shape)[0], tf.TensorShape(self.out_shape)[0]))
        elif self.data_format == 'channels_last':
            return tf.TensorShape((input_shape[0], tf.TensorShape(self.out_shape)[0], tf.TensorShape(self.out_shape)[0], input_shape[3]))

    def call(self, input):
        """
        Perform the eccentricity dependent pooling operation
        """

        paddings = tf.constant([[1, 1,], [2, 2]])
        pad_top, pad_bot, pad_left, pad_right = 0, 0, 0, 0
        if self.rows<self.cols:
            pad_top = int((self.cols-self.rows)/2)
            pad_bot = self.cols - self.rows - pad_top
        else:
            pad_left = int((self.rows-self.cols)/2)
            pad_right = self.rows - self.cols - pad_left

        paddings = tf.constant([[0, 0], [pad_top, pad_bot,], [pad_left, pad_right], [0, 0]])

        img_new = tf.pad(input, paddings, "CONSTANT")

        if self.pool_type == 'avg':
            pool_2D = tf.keras.layers.AveragePooling2D(pool_size=self.rf_min, strides=self.stride, padding='same')
        else:
            pool_2D = tf.keras.layers.MaxPooling2D(pool_size=self.rf_min, strides=self.stride, padding='same')

        temp_pool = pool_2D(img_new)
        curr_mask = tf.broadcast_to(self.mask[0], tf.shape(temp_pool))
        out = tf.math.multiply(tf.dtypes.cast(curr_mask, tf.float32), tf.dtypes.cast(temp_pool, tf.float32))
        for i in range(1, len(self.mask)):
            if self.pool_type == 'avg':
                pool_2D = tf.keras.layers.AveragePooling2D(pool_size=self.rf_sizes[i], strides=self.stride, padding='same')
            else:
                pool_2D = tf.keras.layers.MaxPooling2D(pool_size=self.rf_sizes[i], strides=self.stride, padding='same')

            temp_pool = pool_2D(img_new)
            curr_mask = tf.broadcast_to(self.mask[i], tf.shape(temp_pool))
            out = tf.math.add(tf.math.multiply(tf.dtypes.cast(curr_mask, tf.float32), tf.dtypes.cast(temp_pool, tf.float32)), out)

        return out

def pool_layer(x, eccParam, layer, ecc_depth, depth, name=None):
    rf_min_l = eccParam['rf_min']
    stride_l = eccParam['stride']
    deg2px_l = eccParam['deg2px']
    ecc_slope = eccParam['ecc_slope']
    pool_type = eccParam['pool_type']
    fovea_size = eccParam['fovea_size']
    rf_quant = eccParam['rf_quant']

    if ecc_depth < depth:
        if pool_type == 'max':
            x = tf.keras.layers.MaxPooling2D((rf_min_l[depth-1], rf_min_l[depth-1]), strides=(stride_l[depth-1], stride_l[depth-1]), padding='same', name=name)(x)
        elif pool_type == 'avg':
            x = tf.keras.layers.AveragePooling2D((rf_min_l[depth-1], rf_min_l[depth-1]), strides=(stride_l[depth-1], stride_l[depth-1]), padding='same', name=name)(x)
    else:
        x = CentreDependentPooling2D(rf_min=rf_min_l[depth-1], ecc_slope=ecc_slope[depth-1], deg2px=deg2px_l[depth-1], fovea_size=fovea_size, rf_quant=rf_quant, pool_type=pool_type, layer=layer, name=name, stride=stride_l[depth-1])(x)

    return x

# VGG16 architecture as per the Simonyan, K., & Zisserman, A. (2014)
initializer = tf.keras.initializers.GlorotNormal()

def block1_conv(x):
    x = tf.keras.layers.Conv2D(64, (3, 3),
                      kernel_initializer=initializer,
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3),
                      kernel_initializer=initializer,
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)

    return x

def block2_conv(x):
    x = tf.keras.layers.Conv2D(128, (3, 3),
                      kernel_initializer=initializer,
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3),
                      kernel_initializer=initializer,
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)

    return x

def block3_conv(x):
    x = tf.keras.layers.Conv2D(256, (3, 3),
                      kernel_initializer=initializer,
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3),
                      kernel_initializer=initializer,
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3),
                      kernel_initializer=initializer,
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)

    return x

def block4_conv(x):
    x = tf.keras.layers.Conv2D(512, (3, 3),
                      kernel_initializer=initializer,
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3),
                      kernel_initializer=initializer,
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3),
                      kernel_initializer=initializer,
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)

    return x

def block5_conv(x):
    x = tf.keras.layers.Conv2D(512, (3, 3),
                      kernel_initializer=initializer,
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3),
                      kernel_initializer=initializer,
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3),
                      kernel_initializer=initializer,
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)

    return x


def load_eccNET(vgg_model_path=None, input_shape=None, eccParam=eccParamDefault, ecc_depth=5, freeze_filters=True):

    img_input = tf.keras.layers.Input(shape=input_shape)
    depth = 1

    # Block 1 #############################################################
    x = block1_conv(img_input)
    x = pool_layer(x, eccParam=eccParam, layer=depth, ecc_depth=ecc_depth, depth=depth, name="Layer" + str(depth))
    depth += 1

    # Block 2 #############################################################
    x = block2_conv(x)
    x = pool_layer(x, eccParam=eccParam, layer=depth, ecc_depth=ecc_depth, depth=depth, name="Layer" + str(depth))
    depth += 1

    # Block 3 #############################################################
    x = block3_conv(x)
    x = pool_layer(x, eccParam=eccParam, layer=depth, ecc_depth=ecc_depth, depth=depth, name="Layer" + str(depth))
    depth += 1

    # Block 4 #############################################################
    x = block4_conv(x)
    x = pool_layer(x, eccParam=eccParam, layer=depth, ecc_depth=ecc_depth, depth=depth, name="Layer" + str(depth))
    depth += 1

    # Block 5 #############################################################
    x = block5_conv(x)
    x = pool_layer(x, eccParam=eccParam, layer=depth, ecc_depth=ecc_depth, depth=depth, name="Layer" + str(depth))
    depth += 1

    # Classification block
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(4096, kernel_initializer=initializer, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, kernel_initializer=initializer, activation='relu', name='fc2')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1000, kernel_initializer=initializer, activation='softmax', name='predictions')(x)

    model = tf.keras.Model(inputs=img_input, outputs=x)

    if vgg_model_path:
        model.load_weights(vgg_model_path)

        if freeze_filters:
            for layer in model.layers:
                if 'conv' in layer.name:
                    layer.trainable = False

    return model