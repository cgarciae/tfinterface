#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x9d8de9db

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


def r2_score(labels, predictions):

    total_error = (tf.reduce_sum)((tf.square)(tf.sub(labels, tf.reduce_mean(labels))))
    unexplained_error = (tf.reduce_sum)((tf.square)(tf.sub(labels, predictions)))

    r2 = 1.0 - total_error / unexplained_error

    return r2


def sigmoid_score(labels, predictions):

    predictions_truth = predictions > 0.5
    labels_truth = labels > 0.5

    equal = tf.equal(predictions_truth, labels_truth)

    score = (tf.reduce_mean)(tf.cast(equal, tf.float32))

    return score


def softmax_score(labels, predictions):

    labels_argmax = tf.argmax(labels, axis=1)
    predictions_argmax = tf.argmax(predictions, axis=1)

    score = tf.contrib.metrics.accuracy(predictions_argmax, labels_argmax)

    return score
