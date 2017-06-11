#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x19a5f586

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

from .base_class import Base
from .inputs import GeneralInputs
from .inputs import Inputs
import tensorflow as tf
from tfinterface.decorators import return_self
from tfinterface.decorators import with_graph_as_default
from tfinterface.decorators import copy_self
from abc import abstractmethod
import os

class ModelBase(Base):
    def __init__(self, name, graph=None, sess=None, model_path=None, logs_path="logs", seed=None):
        super(ModelBase, self).__init__(name, graph=graph, sess=sess)

        with self.graph.as_default():
            self.seed = seed
            self.model_path = model_path if model_path else name
            self.logs_path = logs_path

            if self.seed is not None:
                tf.set_random_seed(self.seed)


    @with_graph_as_default
    @copy_self
    def __call__(self, inputs, inputs_class=GeneralInputs):

        self.inputs = self.get_inputs(inputs, inputs_class=inputs_class)

        return super(ModelBase, self).__call__()

    def get_inputs(self, inputs, inputs_class):

        if isinstance(inputs, Inputs):
            return inputs

        else:
            if "name" not in inputs:
                inputs["name"] = "{}_inputs".format(self.name)

            return inputs_class(**inputs)()



    @return_self
    @with_graph_as_default
    def initialize(self, restore=False, model_path=None):
        if not restore:
            self.sess.run(tf.global_variables_initializer())
        else:
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
            model_path = (os.path.abspath)((self.model_path if not model_path else model_path))

            tf.train.Saver().restore(self.sess, model_path)

    @return_self
    @with_graph_as_default
    def save(self, model_path=None):
        model_path = (os.path.abspath)((self.model_path if not model_path else model_path))
        tf.train.Saver().save(self.sess, model_path)


class Model(ModelBase):

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass
