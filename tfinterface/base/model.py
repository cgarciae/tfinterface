#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x8948bfda

# Compiled with Coconut version 1.2.3 [Colonel]

# Coconut Header: --------------------------------------------------------

from __future__ import print_function, absolute_import, unicode_literals, division

import sys as _coconut_sys, os.path as _coconut_os_path
_coconut_file_path = _coconut_os_path.dirname(_coconut_os_path.abspath(__file__))
_coconut_sys.path.insert(0, _coconut_file_path)
from __coconut__ import _coconut, _coconut_MatchError, _coconut_tail_call, _coconut_tco, _coconut_igetitem, _coconut_compose, _coconut_pipe, _coconut_starpipe, _coconut_backpipe, _coconut_backstarpipe, _coconut_bool_and, _coconut_bool_or, _coconut_minus, _coconut_map, _coconut_partial
from __coconut__ import *
_coconut_sys.path.remove(_coconut_file_path)

# Compiled Coconut: ------------------------------------------------------

from .base_class import Base
from .inputs import GeneralInputs
from .inputs import Inputs
import tensorflow as tf
from tfinterface.decorators import return_self
from tfinterface.decorators import with_graph_as_default
from tfinterface.decorators import copy_self
from abc import abstractmethod
from abc import ABCMeta
import os
from tfinterface import utils
import numpy as np
ModeKeys = tf.estimator.ModeKeys

class Model(Base):

    __metaclass__ = ABCMeta

    def __init__(self, name, graph=None, sess=None, model_path=None, logs_path="logs", seed=None):
        super(Model, self).__init__(name, graph=graph, sess=sess)

        with self.graph.as_default():
            self.seed = seed
            self.model_path = model_path if model_path else name
            self.logs_path = logs_path

            if self.seed is not None:
                tf.set_random_seed(self.seed)


    @return_self
    @with_graph_as_default
    def initialize(self, restore=False, model_path=None, var_list=None, only_in_scope=False):
        if not restore:
            self.sess.run(tf.global_variables_initializer())
        else:
            if not var_list and only_in_scope:
                var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)

            model_path = (os.path.abspath)((self.model_path if not model_path else model_path))
            tf.train.Saver(var_list=var_list).restore(self.sess, model_path)

    @return_self
    @with_graph_as_default
    def save(self, model_path=None, var_list=None, only_trainable=False):
        if only_trainable and not var_list:
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        model_path = (os.path.abspath)((self.model_path if not model_path else model_path))
        utils.make_dirs_for_path(model_path)
        tf.train.Saver(var_list=var_list).save(self.sess, model_path)

    @with_graph_as_default
    def get_variables(self, graph_keys=tf.GraphKeys.TRAINABLE_VARIABLES, scope=None):
        scope = scope if scope else self.name
        return tf.get_collection(graph_keys, scope=scope)

    @with_graph_as_default
    def count_weights(self, *args, **kwargs):
        return ((np.sum)((list)((_coconut.functools.partial(map, np.prod))((_coconut.functools.partial(map, _coconut.operator.methodcaller("as_list")))((_coconut.functools.partial(map, _coconut.operator.methodcaller("get_shape")))(self.get_variables(*args, **kwargs)))))))

    @return_self
    def build_tensors(self, *args, **kwargs):
        self.inputs = self.get_inputs(*args, **kwargs)
        super(Model, self).build_tensors(*args, **kwargs)


    def get_inputs(self, inputs, *args, **kwargs):
        return inputs


    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    def run(self, tensors, **kwargs):

        if hasattr(tensors, "__iter__"):
            tensors = [getattr(self, tensor) for tensor in tensors]
        elif isinstance(tensors, dict):
            tensors = dict(((key), (getattr(self, tensor))) for key, tensor in tensors.items())
        else:
            tensors = getattr(self, tensors)

        feed = self.inputs.predict_feed(**kwargs)

        return self.sess.run(tensors, feed_dict=feed)


    def batch_predict(self, generator, print_fn=None, **kwargs):

        preds_list = []

        for batch in generator:
            kwargs = kwargs.copy()
            kwargs.update(batch)

            preds = self.predict(**kwargs)
            preds_list.append(preds)

            if print_fn:
                print_fn(batch)

        return preds_list

    @with_graph_as_default
    @copy_self
    def __call__(self, *args, **kwargs):
        self._mode = kwargs.pop("mode", ModeKeys.TRAIN)
        return super(Model, self).__call__(*args, **kwargs)
