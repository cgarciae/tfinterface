#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xda59baa9

# Compiled with Coconut version 1.2.3 [Colonel]
from __future__ import print_function, absolute_import, unicode_literals, division

import tensorflow as tf
import itertools
import numpy as np


#####################################
# flatten
#####################################

def flatten(net, name = None):
    dims = net.get_shape().as_list()[1:]
    last_dim = np.prod(dims)
    shape = [-1, last_dim]
    net = tf.reshape(net, shape, name = name)

    return net
    


#####################################
# global_average_pooling
#####################################

def global_average_pooling4d(net, keepdims = False, name = None):

    with tf.name_scope("GlobalAveragePooling4D"):
        net = tf.reduce_mean(net, axis = [1, 2, 3, 4], keepdims = keepdims, name = name)

    return net

def global_average_pooling3d(net, keepdims = False, name = None):

    with tf.name_scope("GlobalAveragePooling3D"):
        net = tf.reduce_mean(net, axis = [1, 2, 3], keepdims = keepdims, name = name)

    return net

def global_average_pooling2d(net, keepdims = False, name = None):

    with tf.name_scope("GlobalAveragePooling2D"):
        net = tf.reduce_mean(net, axis = [1, 2], keepdims = keepdims, name = name)

    return net


def global_average_pooling1d(net, keepdims = False, name = None):

    with tf.name_scope("GlobalAveragePooling1D"):
        net = tf.reduce_mean(net, axis = [1], keepdims = keepdims, name = name)

    return net

#####################################
# batch_norm
#####################################

def dense_batch_norm(*args, **kwargs):

    name = kwargs.pop("name", None)
    activation = kwargs.pop("activation", None)
    batch_norm = kwargs.pop("batch_norm", {})

    with tf.variable_scope(name, "DenseBatchNorm"):
        net = tf.layers.dense(*args, **kwargs)
        net = tf.layers.batch_normalization(net, **batch_norm)

        return activation(net) if activation else net


def conv3d_batch_norm(*args, **kwargs):

    name = kwargs.pop("name", None)
    activation = kwargs.pop("activation", None)
    batch_norm = kwargs.pop("batch_norm", {})

    with tf.variable_scope(name, default_name="Conv3dBatchNorm"):
        net = tf.layers.conv3d(*args, **kwargs)
        net = tf.layers.batch_normalization(net, **batch_norm)

        return activation(net) if activation else net

def conv2d_batch_norm(*args, **kwargs):

    name = kwargs.pop("name", None)
    activation = kwargs.pop("activation", None)
    batch_norm = kwargs.pop("batch_norm", {})

    with tf.variable_scope(name, default_name="Conv2dBatchNorm"):
        net = tf.layers.conv2d(*args, **kwargs)
        net = tf.layers.batch_normalization(net, **batch_norm)

        return activation(net) if activation else net

def conv1d_batch_norm(*args, **kwargs):

    name = kwargs.pop("name", None)
    activation = kwargs.pop("activation", None)
    batch_norm = kwargs.pop("batch_norm", {})

    with tf.variable_scope(name, default_name="Conv1dBatchNorm"):
        net = tf.layers.conv1d(*args, **kwargs)
        net = tf.layers.batch_normalization(net, **batch_norm)

        return activation(net) if activation else net


def conv3d_transpose_batch_norm(*args, **kwargs):

    name = kwargs.pop("name", None)
    activation = kwargs.pop("activation", None)
    batch_norm = kwargs.pop("batch_norm", {})

    with tf.variable_scope(name, default_name="Conv3dTransposeBatchNorm"):
        net = tf.layers.conv3d_transpose(*args, **kwargs)
        net = tf.layers.batch_normalization(net, **batch_norm)

        return activation(net) if activation else net

def conv2d_transpose_batch_norm(*args, **kwargs):

    name = kwargs.pop("name", None)
    activation = kwargs.pop("activation", None)
    batch_norm = kwargs.pop("batch_norm", {})

    with tf.variable_scope(name, default_name="Conv2dTransposeBatchNorm"):
        net = tf.layers.conv2d_transpose(*args, **kwargs)
        net = tf.layers.batch_normalization(net, **batch_norm)

        return activation(net) if activation else net


def conv1d_transpose_batch_norm(*args, **kwargs):

    name = kwargs.pop("name", None)
    activation = kwargs.pop("activation", None)
    batch_norm = kwargs.pop("batch_norm", {})

    with tf.variable_scope(name, default_name="Conv1dTransposeBatchNorm"):
        net = tf.layers.conv1d_transpose(*args, **kwargs)
        net = tf.layers.batch_normalization(net, **batch_norm)

        return activation(net) if activation else net

#####################################
# fire2d
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

fire2d = fire
fire2d_batch_norm = fire_batch_norm

#####################################
# fire1d
#####################################
def fire1d(inputs, squeeze_filters, expand_1x1_filters, expand_3x3_filters, **kwargs):

    name = kwargs.pop("name", None)

    with tf.variable_scope(name, default_name="Fire1D"):
# squeeze
        squeeze = tf.layers.conv1d(inputs, squeeze_filters, [1], **kwargs)

# expand
        kwargs["padding"] = "same"
        expand_1x1 = tf.layers.conv1d(squeeze, expand_1x1_filters, [1], **kwargs)
        expand_3x3 = tf.layers.conv1d(squeeze, expand_3x3_filters, [3], **kwargs)

        return tf.concat([expand_1x1, expand_3x3], axis=2)


def fire1d_batch_norm(inputs, squeeze_filters, expand_1x1_filters, expand_3x3_filters, **kwargs):

    name = kwargs.pop("name", None)

    with tf.variable_scope(name, default_name="Fire1DBatchNorm"):
# squeeze
        squeeze = conv1d_batch_norm(inputs, squeeze_filters, [1], **kwargs)

# expand
        kwargs["padding"] = "same"
        expand_1x1 = conv1d_batch_norm(squeeze, expand_1x1_filters, [1], **kwargs)
        expand_3x3 = conv1d_batch_norm(squeeze, expand_3x3_filters, [3], **kwargs)

        return tf.concat([expand_1x1, expand_3x3], axis=2)



#####################################
# conv2d_dense_block
#####################################

def conv2d_densenet_layer(net, growth_rate, bottleneck, batch_norm, dropout, activation, **kwargs):


    kwargs.setdefault("kernel_regularizer")

    with tf.variable_scope(None, default_name="Conv2dDenseNetlayer"):

        net = tf.layers.batch_normalization(net, **batch_norm)
        net = activation(net) if activation else net

        if bottleneck:
            net = tf.layers.conv2d(net, bottleneck, [1, 1], **kwargs)
            net = tf.layers.dropout(net, **dropout) if dropout else net
            net = tf.layers.batch_normalization(net, **batch_norm)
            net = activation(net) if activation else net

        net = tf.layers.conv2d(net, growth_rate, [3, 3], **kwargs)
        net = tf.layers.dropout(net, **dropout) if dropout else net

        return net


def conv2d_densenet_transition(net, compression, batch_norm, dropout, activation, **kwargs):

    filters = int(net.get_shape()[-1])

    if compression:
        if compression <= 1:
            filters = int(filters * compression)
        else:
            filters = compression

    with tf.variable_scope(None, default_name="TransitionLayer"):

        net = tf.layers.batch_normalization(net, **batch_norm)
        net = activation(net) if activation else net
        net = tf.layers.conv2d(net, filters, [1, 1], **kwargs)
        net = tf.layers.dropout(net, **dropout)

        return net


def conv2d_dense_block(net, growth_rate, n_layers, **kwargs):
    name = kwargs.pop("name", None)
    bottleneck = kwargs.pop("bottleneck", None)
    compression = kwargs.pop("compression", None)
    batch_norm = kwargs.pop("batch_norm", {})
    dropout = kwargs.pop("dropout", {})
    activation = kwargs.pop("activation", None)
    weight_decay = kwargs.pop("weight_decay", None)

    kwargs.setdefault("use_bias", False)

    if weight_decay:
        kwargs.setdefault("kernel_regularizer", tf.contrib.layers.l2_regularizer(weight_decay))
        batch_norm.setdefault("beta_regularizer", tf.contrib.layers.l2_regularizer(weight_decay))
        batch_norm.setdefault("gamma_regularizer", tf.contrib.layers.l2_regularizer(weight_decay))



    with tf.variable_scope(name, default_name="Conv2dDenseNetBlock"):

        for layers in range(n_layers):
            layer = conv2d_densenet_layer(net, growth_rate, bottleneck, batch_norm, dropout, activation, **kwargs)
            net = tf.concat([net, layer], axis=3)

        net = conv2d_densenet_transition(net, compression, batch_norm, dropout, activation, **kwargs)

    return net



#####################################
# conv1d_dense_block
#####################################

def conv1d_densenet_layer(net, growth_rate, bottleneck, batch_norm, dropout, activation, **kwargs):


    kwargs.setdefault("kernel_regularizer")

    with tf.variable_scope(None, default_name="Conv1dDenseNetlayer"):

        net = tf.layers.batch_normalization(net, **batch_norm)
        net = activation(net) if activation else net

        if bottleneck:
            net = tf.layers.conv1d(net, bottleneck, [1], **kwargs)
            net = tf.layers.dropout(net, **dropout) if dropout else net
            net = tf.layers.batch_normalization(net, **batch_norm)
            net = activation(net) if activation else net

        net = tf.layers.conv1d(net, growth_rate, [3], **kwargs)
        net = tf.layers.dropout(net, **dropout) if dropout else net

        return net


def conv1d_densenet_transition(net, compression, batch_norm, dropout, activation, **kwargs):

    filters = int(net.get_shape()[-1])

    if compression:
        if compression <= 1:
            filters = int(filters * compression)
        else:
            filters = compression

    with tf.variable_scope(None, default_name="TransitionLayer"):

        net = tf.layers.batch_normalization(net, **batch_norm)
        net = activation(net) if activation else net
        net = tf.layers.conv1d(net, filters, [1], **kwargs)
        net = tf.layers.dropout(net, **dropout)

        return net


def conv1d_dense_block(net, growth_rate, n_layers, **kwargs):
    name = kwargs.pop("name", None)
    bottleneck = kwargs.pop("bottleneck", None)
    compression = kwargs.pop("compression", None)
    batch_norm = kwargs.pop("batch_norm", {})
    dropout = kwargs.pop("dropout", {})
    activation = kwargs.pop("activation", None)
    weight_decay = kwargs.pop("weight_decay", None)
    kwargs["padding"] = "SAME"

    kwargs.setdefault("use_bias", False)

    if weight_decay:
        kwargs.setdefault("kernel_regularizer", tf.contrib.layers.l2_regularizer(weight_decay))
        batch_norm.setdefault("beta_regularizer", tf.contrib.layers.l2_regularizer(weight_decay))
        batch_norm.setdefault("gamma_regularizer", tf.contrib.layers.l2_regularizer(weight_decay))



    with tf.variable_scope(name, default_name="Conv1dDenseNetBlock"):

        for layers in range(n_layers):
            layer = conv1d_densenet_layer(net, growth_rate, bottleneck, batch_norm, dropout, activation, **kwargs)
            net = tf.concat([net, layer], axis=2)

        net = conv1d_densenet_transition(net, compression, batch_norm, dropout, activation, **kwargs)

    return net

#####################################
# fc_dense_block
#####################################

def fc_densenet_layer(net, growth_rate, bottleneck, batch_norm, dropout, activation, **kwargs):


    kwargs.setdefault("kernel_regularizer")

    with tf.variable_scope(None, default_name="FCDenseNetlayer"):

        net = tf.layers.batch_normalization(net, **batch_norm)
        net = activation(net) if activation else net

        if bottleneck:
            net = tf.layers.dense(net, bottleneck, **kwargs)
            net = tf.layers.dropout(net, **dropout) if dropout else net
            net = tf.layers.batch_normalization(net, **batch_norm)
            net = activation(net) if activation else net

        net = tf.layers.dense(net, growth_rate, **kwargs)
        net = tf.layers.dropout(net, **dropout) if dropout else net

        return net


def fc_densenet_transition(net, compression, batch_norm, dropout, activation, **kwargs):

    filters = int(net.get_shape()[-1])

    if compression <= 1:
        filters = int(filters * compression)
    else:
        filters = compression

    with tf.variable_scope(None, default_name="TransitionLayer"):

        net = tf.layers.batch_normalization(net, **batch_norm)
        net = activation(net) if activation else net
        net = tf.layers.dense(net, filters, **kwargs)
        net = tf.layers.dropout(net, **dropout)

        return net


def fc_dense_block(net, growth_rate, n_layers, **kwargs):
    name = kwargs.pop("name", None)
    bottleneck = kwargs.pop("bottleneck", None)
    compression = kwargs.pop("compression", None)
    batch_norm = kwargs.pop("batch_norm", {})
    dropout = kwargs.pop("dropout", {})
    activation = kwargs.pop("activation", None)
    weight_decay = kwargs.pop("weight_decay", None)

    kwargs.setdefault("use_bias", False)

    if weight_decay:
        kwargs.setdefault("kernel_regularizer", tf.contrib.layers.l2_regularizer(weight_decay))
        batch_norm.setdefault("beta_regularizer", tf.contrib.layers.l2_regularizer(weight_decay))
        batch_norm.setdefault("gamma_regularizer", tf.contrib.layers.l2_regularizer(weight_decay))



    with tf.variable_scope(name, default_name="FCDenseNetBlock"):

        for layers in range(n_layers):
            layer = fc_densenet_layer(net, growth_rate, bottleneck, batch_norm, dropout, activation, **kwargs)
            net = tf.concat([net, layer], axis=1)

        if compression:
            net = fc_densenet_transition(net, compression, batch_norm, dropout, activation, **kwargs)

    return net

#####################################
# densefire_block
#####################################

def conv2d_densefire_layer(net, bottleneck, growth_rate_1x1, growth_rate_3x3, batch_norm, dropout, activation, **kwargs):

    with tf.variable_scope(None, default_name="Conv2dDenseFireLayer"):

        net = tf.layers.batch_normalization(net, **batch_norm)
        net = activation(net) if activation else net

# squeeze
        net = tf.layers.conv2d(net, bottleneck, [1, 1], **kwargs)
        net = tf.layers.dropout(net, **dropout)
        net = tf.layers.batch_normalization(net, **batch_norm)
        net = activation(net) if activation else net

# expand
        expand_1x1 = tf.layers.conv2d(net, growth_rate_1x1, [1, 1], **kwargs)
        expand_3x3 = tf.layers.conv2d(net, growth_rate_3x3, [3, 3], **kwargs)

# concat
        net = tf.concat([expand_1x1, expand_3x3], axis=3)
        net = tf.layers.dropout(net, **dropout)

        return net



def conv2d_densefire_block(net, bottleneck, growth_rate_1x1, growth_rate_3x3, n_layers, **kwargs):
    name = kwargs.pop("name", None)
    compression = kwargs.pop("compression", None)
    batch_norm = kwargs.pop("batch_norm", {})
    dropout = kwargs.pop("dropout", {})
    activation = kwargs.pop("activation")

    with tf.variable_scope(name, default_name="Conv2dDenseFireBlock"):

        for layers in range(n_layers):
            layer = conv2d_densefire_layer(net, bottleneck, growth_rate_1x1, growth_rate_3x3, batch_norm, dropout, activation, **kwargs)
            net = tf.concat([net, layer], axis=3)

        net = conv2d_densenet_transition(net, compression, batch_norm, dropout, activation, **kwargs)

    return net




#####################################
# relation
#####################################

def relation_network(net, dense_fn, *args, **kwargs):
    # get kwargs
    name = kwargs.pop("name", None)
    reduce_fn = kwargs.pop("reduce_fn", tf.reduce_sum)

    # get network shape
    shape = [-1] + [int(d) for d in net.get_shape()[1:]]

    with tf.name_scope(name, default_name="RelationNetwork"):

        if len(shape) > 2:
            # get object properties
            n_objects = np.prod(shape[1:-1])
            object_length = shape[-1]

            # get objects tensor
            objects = tf.reshape(net, shape=(-1, n_objects, object_length))

            # extract all pair of objects
            pairs = itertools.product(range(n_objects), range(n_objects))
            pairs = ((objects[:, a, :], objects[:, b, :]) for a, b in pairs)
            pairs = (tf.concat([a, b], axis=1) for a, b in pairs)
            pairs = list(pairs)

            # get pairs properties
            n_pairs = len(pairs)

            # fuse pairs into pairs tensor
            net = tf.concat(pairs, axis=0)

            # construct pairs net
            net = dense_fn(net, *args, **kwargs)

            # count relations
            n_relations = net.get_shape()[-1]
            n_relations = int(n_relations)

            # reshape to fix batch dimension
            net = tf.reshape(net, shape=(-1, n_pairs, n_relations))

            # reduce to sum or average along the pairs dimension
            net = reduce_fn(net, axis=1)

            return net

        else:
            raise NotImplementedError("Tensors with dims <= 2 not supported for now, got {}".format(len(shape)))


#####################################
# add_coordinates
#####################################

def add_coordinates(net, min_value = -1.0, max_value = 1.0):

    assert len(net.shape) > 2

    sample_base_shape = [ dim.value for dim in net.shape[1:-1] ]

    if not hasattr(min_value, "__iter__"):
        min_value = [ min_value ] * len(sample_base_shape)

    if not hasattr(max_value, "__iter__"):
        max_value = [ max_value ] * len(sample_base_shape)

    linspaces = [
        tf.linspace(start, stop, num)
        for start, stop, num in reversed(list(zip(min_value, max_value, sample_base_shape)))
    ]

    multiples = [tf.shape(net)[0]] + [1] * (len(net.shape) - 1)

    coords = tf.meshgrid(*linspaces)
    coords = tf.stack(coords, axis = -1)
    coords = tf.expand_dims(coords, axis = 0)
    coords = tf.tile(coords,multiples)

    return tf.concat([net, coords], axis = -1)



#####################################
# get_relations
#####################################

def get_relations(inputs, num_related, main_shape = "flatten", structure = "product"):

    assert len(inputs.shape) > 2

    obj_shape = inputs.shape[1:-1]
    num_objs = np.prod(obj_shape)
    obj_size = inputs.shape[-1].value

    net = tf.reshape(inputs, [-1, num_objs, obj_size])

    if structure == "product":
        relation_idxs_list = itertools.product(range(num_objs), repeat = num_related)
    elif structure == "permutations":
        relation_idxs_list = itertools.permutations(range(num_objs), num_related)
    else:
        raise ValueError(structure)


    relations = tf.stack([
        tf.concat([
            net[:, idx, :] for idx in relation_idxs
        ], axis = 1)
        for relation_idxs in relation_idxs_list
    ], axis = 1)


    if structure == "product":
        relation_shape = [num_objs] * (num_related - 1)
        
    elif structure == "permutations":
        relation_shape = list(range(num_objs - 1, num_objs - num_related, -1))

    if main_shape == "flatten":
        shape = [num_objs]
    elif main_shape == "same":
        shape = list(obj_shape)
    
    relation_shape = [-1] + shape + relation_shape + [obj_size]

    return tf.reshape(relations, relation_shape)


def associative_module(f, alfa=0.5, training=None):
    def _associative_module(inputs):

        if training is None:
            training_ = tf.keras.backend.learning_phase()
        else:
            training_ = training

        x = tf.stop_gradient(inputs)

        y = f(x)

        if training:
            tf.losses.mean_squared_error(x, y)
            output = inputs
        else:
            output =  alfa * y + (1.0 - alfa) * x

        return output

    return _associative_module

def associative_2d(inputs, depth, dropout_rate=None, alfa=0.5, training=None, activation=None, batch_normalization=False):

    def f(inputs):
        channels = inputs.shape[-1].value
        net = inputs

        if dropout_rate:
            net = tf.layers.dropout(net, rate=dropout_rate, training=training)

        for i in range(depth):
            channels *= 2
            net = tf.keras.layers.Conv2D(channels, [3, 3], strides=2, padding="same")(net)
            
            if batch_normalization:
                net = tf.keras.layers.BatchNormalization()(net)
            if activation:
                net = tf.keras.layers.Activation(activation)(net)
           

        for i in range(depth):
            channels //= 2
            net = tf.keras.layers.Conv2DTranspose(channels, [3, 3], strides=2, padding="same")(net)
            if batch_normalization:
                net = tf.keras.layers.BatchNormalization()(net)
            if activation:
                net = tf.keras.layers.Activation(activation)(net)

        if tuple(net.shape) != tuple(inputs.shape):
            net = tf.image.resize_images(net, [inputs.shape[1], inputs.shape[2]])
        
        return net
            

    return associative_module(f, alfa=alfa, training=training)(inputs)


if __name__ == '__main__':

    tf.enable_eager_execution()

    x = tf.ones([1, 14, 14, 4], dtype=tf.float32)

    print(list(x.shape))

    # x = get_relations(x, 1, main_shape="same", structure="permutations")

    # print(x.shape)

    # net = tf.layers.dense(x, 8)

    # print(net.shape)

    net = associative_2d(x, 1)

    print(net.shape)

