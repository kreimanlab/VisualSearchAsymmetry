import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

def get_act(stim_img, tar_img, back_fill):
    op_stimuli = np.expand_dims(stim_img, axis=0)
    op_target = np.expand_dims(tar_img, axis=0)

    # pad_size = op_target.shape[1]
    # temp = np.uint8(back_fill*np.ones((1, op_stimuli.shape[1]+2*pad_size, op_stimuli.shape[2]+2*pad_size, 3)))
    # temp[:, pad_size:op_stimuli.shape[1]+pad_size, pad_size:op_stimuli.shape[2]+pad_size, :] = np.copy(op_stimuli)
    # op_stimuli = np.copy(temp)

    op_stimuli = preprocess_input(np.array(op_stimuli, dtype=np.float32))
    op_target = preprocess_input(np.array(op_target, dtype=np.float32))

    MMconv = tf.keras.layers.Conv2D(1, kernel_size=(op_target.shape[1], op_target.shape[2]),
                         input_shape=(op_stimuli.shape[1], op_stimuli.shape[2], op_stimuli.shape[3]),
                         padding='same',
                         use_bias=False)

    init = MMconv(tf.constant(op_stimuli))

    layer_weight = []
    layer_weight.append(op_target.reshape(MMconv.get_weights()[0].shape))
    MMconv.set_weights(layer_weight)

    out = MMconv(tf.constant(op_stimuli))
    out = out.numpy().reshape((init.shape[1], init.shape[2]))

    out = cv2.resize(out, (op_stimuli.shape[2], op_stimuli.shape[1]), interpolation = cv2.INTER_AREA)
    out = cv2.GaussianBlur(out,(7,7),3)
    return out

def recog(x, y, gt):
    ior_size = 10

    fxt_xtop = x-int(ior_size/2)
    fxt_ytop = y-int(ior_size/2)
    fxt_place = gt[fxt_xtop:(fxt_xtop+ior_size), fxt_ytop:(fxt_ytop+ior_size)]

    if (np.sum(fxt_place)>0):
        return 1
    else:
        return 0

def remove_attn(mask, x, y, gt_mask):
    for i in range(len(gt_mask)):
        gt = np.copy(gt_mask[i])
        if gt[x, y] > 0:
            mask = np.copy((1-gt)*mask)
            break

    return mask
