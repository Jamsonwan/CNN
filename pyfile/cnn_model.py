import os
import numpy as np

import tensorflow as tf


def initialize_convolution_kernel(conv_layers):
    """
    initialize convolution kernel
    :param conv_layers:
    :return:
    """
    l = len(conv_layers)
    conv_kernel = {}

    for i in range(l):
        W = tf.compat.v1.get_variable('W'+str(i+1), conv_layers.get('W'+str(i+1)), initializer=tf.glorot_uniform_initializer())
        conv_kernel['W'+str(i+1)] = W

    return conv_kernel


if __name__ == '__main__':
    pass