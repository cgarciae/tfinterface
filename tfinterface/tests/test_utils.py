#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xd2f61d21

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

import numpy as np
import tensorflow as tf
from tfinterface.utils import soft_if
from tfinterface.utils import select_columns


class TestSupervised(object):

    def test_soft_if(self):
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:

            x = tf.placeholder(tf.float32, [None])
            y = tf.placeholder(tf.float32, [None])
            b = tf.placeholder(tf.float32, [None])
            f = soft_if(b, x, y)

            X = [1., 2., 3.]
            Y = [4., 5., 6.]
            B = [1., 0., 1.]

            F = sess.run(f, {x: X, y: Y, b: B})

            assert (F == [1., 5., 3.]).all()


    def test_select_colums(self):
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            x = tf.placeholder(tf.float32, [None, 2])
            idx = tf.placeholder(tf.int32, [None])
            y = select_columns(x, idx)

            X = [[1., 2.], [3., 4.], [5., 6.],]

            IDX = [0, 1, 0]

            Y = sess.run(y, {x: X, idx: IDX})

            assert (Y == [1., 4., 5.]).all()
