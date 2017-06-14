#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xd59b449b

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

# import numpy as np
# import tensorflow as tf
# from tensorflow import layers as L
# from tfinterface.supervised import SupervisedModel, LinearRegression
# from phi.api import *
#
#
# class TestSupervised(object):
#
#
#     def test_supervised(self):
#
#         n = 2; std = 30.0;
#
#         x = np.random.uniform(low=0.0, high=100.0, size=(n, 1))
#         y = 3.5 * x + 40.3 + np.random.normal(loc=0.0, scale=std, size=(n, 1))
#
#         class LinearRegression(SupervisedModel):
#
#             def _build(self, learning_rate=0.01):
#                 with self.graph.as_default():
#                     self.x = tf.placeholder(tf.float32, [None, 1])
#                     self.y = tf.placeholder(tf.float32, [None, 1])
#
#                     self.h = self.x |> L.dense$(?, 1)
#
#                     error = self.h - self.y
#
#                     self.loss = error |> tf.nn.l2_loss |> tf.reduce_mean
#
#                     tf.summary.scalar('loss', self.loss)
#
#                     trainer = tf.train.AdamOptimizer(learning_rate)
#                     self.update = trainer.minimize(self.loss)
#
#         model = LinearRegression(learning_rate=0.1)
#         model.fit(x, y, steps=1)
#
#     def test_linear_regression(self):
#
#         n = 2; std = 30.0;
#
#         x = np.random.uniform(low=0.0, high=100.0, size=(n, 1))
#         y = 3.5 * x + 40.3 + np.random.normal(loc=0.0, scale=std, size=(n, 1))
#
#         model = LinearRegression()
#         model.fit(x, y, epochs=1)
