#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x8d2134b7

# Compiled with Coconut version 1.2.3-post_dev5 [Colonel]

# Coconut Header: --------------------------------------------------------------

from __future__ import print_function, absolute_import, unicode_literals, division
import sys as _coconut_sys, os.path as _coconut_os_path
_coconut_file_path = _coconut_os_path.dirname(_coconut_os_path.abspath(__file__))
_coconut_sys.path.insert(0, _coconut_file_path)
from __coconut__ import _coconut, _coconut_MatchError, _coconut_tail_call, _coconut_tco, _coconut_igetitem, _coconut_compose, _coconut_pipe, _coconut_starpipe, _coconut_backpipe, _coconut_backstarpipe, _coconut_bool_and, _coconut_bool_or, _coconut_minus, _coconut_map, _coconut_partial
from __coconut__ import *
_coconut_sys.path.remove(_coconut_file_path)

# Compiled Coconut: ------------------------------------------------------------

import tensorflow as tf
import numpy as np

class Required(object):
    pass
class RequiredTensor(Required):
    pass
class RequiredEnvironment(Required):
    pass

class RequiredExperienceBuffer(Required):
    pass
class RequiredTrainer(Required):
    pass

REQUIRED = Required()
TENSOR = RequiredTensor()
ENV = RequiredEnvironment()
EXPERIENCEBUFFER = RequiredExperienceBuffer()
TRAINER = RequiredTrainer()


def random_batch_generator(*datas, **kwargs):
    batch_size = kwargs.get("batch_size", 32)
    n = len(datas[0])

    while True:
        idx = np.random.random_integers(0, high=n - 1, size=(batch_size,))
        yield tuple([data[idx] for data in datas])

def select_columns(tensor, indexes):
    idx = tf.stack((tf.range(tf.shape(indexes)[0]), indexes), 1)
    return tf.gather_nd(tensor, idx)


def soft_if(cond, then, else_):
    return (cond * then) + (1.0 - cond) * else_


def map_gradients(f, gradients):
    return [(f(g), v) for g, v in gradients]

def get_run():
    try:
        with open("run.txt") as f:
            run = int(f.read().split("/n")[0])
    except:
        run = -1

    with open("run.txt", 'w+') as f:
        run += 1

        f.seek(0)
        f.write(str(run))
        f.truncate()

    return run

def shifted_log_loss(x, alfa=0.05):
    return -tf.log(x + alfa * (1.0 - x))

def huber_loss(x, d=1.0):
    """
See: https://en.wikipedia.org/wiki/Huber_loss
    """
    return tf.where(tf.abs(x) <= d, 0.5 * tf.square(x), d * (tf.abs(x) - 0.5 * d))


def get_global_step():
    global_step = tf.train.get_global_step()

    if not global_step:
        global_step = tf.get_variable("global_step", initializer=0, trainable=False)
        tf.add_to_collection(tf.GraphKeys.GLOBAL_STEP, global_step)

    return global_step




def shuffle_batch_tensor_fns(tensors_dict, **shuffle_batch_kwargs):
    self = shuffle_batch_tensor_fns

    self.tensors_dict = tensors_dict
    self.shuffled_tensors = None

    def shuffle_tensors():
        if self.shuffled_tensors is None:
            self.tensors_dict = (dict(((key), (value)) for key, value in self.tensors_dict.items()))

            self.shuffled_tensors = tf.train.shuffle_batch(self.tensors_dict, **shuffle_batch_kwargs)

    def get_fn(name):
        def tensor_fn():
            shuffle_tensors()
            return self.shuffled_tensors[name]

        return tensor_fn

    return (dict(((name), (get_fn(name))) for name in self.tensors_dict))
