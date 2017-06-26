#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x28d04466

# Compiled with Coconut version 1.2.3-post_dev1 [Colonel]

# Coconut Header: --------------------------------------------------------

from __future__ import print_function, absolute_import, unicode_literals, division

import sys as _coconut_sys, os.path as _coconut_os_path
_coconut_file_path = _coconut_os_path.dirname(_coconut_os_path.abspath(__file__))
_coconut_sys.path.insert(0, _coconut_file_path)
from __coconut__ import _coconut, _coconut_MatchError, _coconut_tail_call, _coconut_tco, _coconut_igetitem, _coconut_compose, _coconut_pipe, _coconut_starpipe, _coconut_backpipe, _coconut_backstarpipe, _coconut_bool_and, _coconut_bool_or, _coconut_minus, _coconut_map, _coconut_partial
from __coconut__ import *
_coconut_sys.path.remove(_coconut_file_path)

# Compiled Coconut: ------------------------------------------------------

import tensorflow as tf

#####################################
# batch_norm
#####################################

def dense_batch_norm(*args, **kwargs):

    name = kwargs.pop("name", None)
    activation = kwargs.pop("activation", None)
    bn_kwargs = kwargs.pop("bn_kwargs", {})

    with tf.variable_scope(name, "DenseBatchNorm"):
        net = tf.layers.dense(*args, **kwargs)
        net = tf.layers.batch_normalization(net, **bn_kwargs)

        return activation(net) if activation else net

def conv2d_batch_norm(*args, **kwargs):

    name = kwargs.pop("name", None)
    activation = kwargs.pop("activation", None)
    bn_kwargs = kwargs.pop("bn_kwargs", {})

    with tf.variable_scope(name, default_name="Conv2dBatchNorm"):
        net = tf.layers.conv2d(*args, **kwargs)
        net = tf.layers.batch_normalization(net, **bn_kwargs)

        return activation(net) if activation else net

#####################################
# fire
#####################################
def fire(inputs, squeeze_filters, expand_1x1_filters, expand_3x3_filters, **kwargs):

    name = kwargs.pop("name", None)

    with tf.variable_scope(name, default_name="Fire"):
# squeeze
        squeeze = tf.layers.conv2d(inputs, squeeze_filters, [1, 1], **kwargs)

# expand
        kwargs["padding"] = "same"
        expand_1x1 = tf.layers.conv2d(squeeze, expand_1x1_filters, [1, 1], **kwargs)
        expand_3x3 = tf.layers.conv2d(squeeze, expand_3x3_filters, [3, 3], **kwargs)

        return tf.concat([expand_1x1, expand_3x3], axis=3)


def fire_batch_norm(inputs, squeeze_filters, expand_1x1_filters, expand_3x3_filters, **kwargs):

    name = kwargs.pop("name", None)

    with tf.variable_scope(name, default_name="FireBatchNorm"):
# squeeze
        squeeze = conv2d_batch_norm(inputs, squeeze_filters, [1, 1], **kwargs)

# expand
        kwargs["padding"] = "same"
        expand_1x1 = conv2d_batch_norm(squeeze, expand_1x1_filters, [1, 1], **kwargs)
        expand_3x3 = conv2d_batch_norm(squeeze, expand_3x3_filters, [3, 3], **kwargs)

        return tf.concat([expand_1x1, expand_3x3], axis=3)

if __name__ == '__main__':
    x = tf.random_uniform(shape=(16, 32, 32, 3))

    f = fire(x, 32, 64, 64, activation=tf.nn.relu)
    fb = fire_batch_norm(x, 32, 64, 64, activation=tf.nn.relu, bn_kwargs=dict(training=True))
    print(f)
    print(fb)
