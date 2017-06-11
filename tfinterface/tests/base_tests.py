#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x51ad9a6

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

from tfinterface.base import ModelBase
from tfinterface.base import Inputs
import tensorflow as tf
import numpy as np


class TestBase(object):

    def test_build(self):
        class VariableInputs(Inputs):

            def _build(self, n_features, n_labels):
                self.n_features = n_features
                self.n_labels = n_labels

                self.features = tf.placeholder(tf.float32, [None, self.n_features])
                self.labels = tf.placeholder(tf.float32, [None, self.n_labels])

            def fit_feed(self):
                return {}
            def predict_feed(self):
                return {}

        class ConstantInputs(Inputs):

            def _build(self, n_features, n_labels):
                self.n_features = n_features
                self.n_labels = n_labels

                self.features = tf.constant(np.random.normal(loc=0.0, scale=1.0, size=(5, n_features)), dtype=tf.float32)
                self.labels = tf.constant(np.random.normal(loc=0.0, scale=1.0, size=(5, n_labels)), dtype=tf.float32)

            def fit_feed(self):
                return {}
            def predict_feed(self):
                return {}

        class TestModel(ModelBase):
            """docstring for TestModel."""
            def _build(self, inputs):
                units = inputs.labels.get_shape()[1].value
                self.predictions = tf.layers.dense(inputs.features, units)


        constant_inputs = ConstantInputs("constant_inputs")(10, 1)
        variable_inputs = VariableInputs("variable_inputs")(10, 1)

        model_template = TestModel("test_model")

        model1 = model_template(constant_inputs)
        model2 = model_template(variable_inputs)

        assert model1.predictions.get_shape()[0].value is 5
        assert model1.predictions.get_shape()[1].value is 1
        assert model2.predictions.get_shape()[0].value is None
        assert model2.predictions.get_shape()[1].value is 1


        vars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=model1.template._name)
        vars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=model2.template._name)

        assert vars1 == vars2
