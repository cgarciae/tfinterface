#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xaca65f20

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

from tfinterface.supervised import SoftmaxClassifier
from tfinterface.supervised import SupervisedInputs
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score

n_classes = 3
n_features = 2

class Model(SoftmaxClassifier):

    def get_labels(self, inputs):
# one hot labels
        return tf.one_hot(inputs.labels, n_classes)

    def get_logits(self, inputs):

        print(inputs.features)
        net = tf.layers.dense(inputs.features, 10, activation=tf.nn.relu)

        return tf.layers.dense(net, n_classes)


def get_templates():
    graph = tf.Graph()
    sess = tf.Session(graph=graph)

# inputs
    inputs_t = SupervisedInputs(name="network_name" + "_inputs", graph=graph, sess=sess, features=dict(shape=(None, n_features)), labels=dict(shape=(None,), dtype=tf.uint8))

    model_t = template = Model(name="network_name", graph=graph, sess=sess)

    return inputs_t, model_t


def get_data():
    x = np.random.uniform(size=(10, n_features))
    y = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]

    return x, y


class TestSupervised(object):

    def test_softmax_score(self):

        x, y = get_data()
        inputs_t, model_t = get_templates()

        inputs = inputs_t()
        model = model_t(inputs)

        model.initialize()

        predictions_probs = model.predict(features=x)
        predictions = np.argmax(predictions_probs, axis=1)

        np.testing.assert_almost_equal(model.score(features=x, labels=y), accuracy_score(predictions, y))
