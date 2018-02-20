#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xf835fc47

# Compiled with Coconut version 1.3.1 [Dead Parrot]

# Coconut Header: -------------------------------------------------------------

from __future__ import print_function, absolute_import, unicode_literals, division
import sys as _coconut_sys, os.path as _coconut_os_path
_coconut_file_path = _coconut_os_path.dirname(_coconut_os_path.abspath(__file__))
_coconut_sys.path.insert(0, _coconut_file_path)
from __coconut__ import _coconut, _coconut_NamedTuple, _coconut_MatchError, _coconut_tail_call, _coconut_tco, _coconut_igetitem, _coconut_base_compose, _coconut_forward_compose, _coconut_back_compose, _coconut_forward_star_compose, _coconut_back_star_compose, _coconut_pipe, _coconut_star_pipe, _coconut_back_pipe, _coconut_back_star_pipe, _coconut_bool_and, _coconut_bool_or, _coconut_none_coalesce, _coconut_minus, _coconut_map, _coconut_partial
from __coconut__ import *
_coconut_sys.path.remove(_coconut_file_path)

# Compiled Coconut: -----------------------------------------------------------

import tensorflow as tf

class Inference(object):
    """ Inference class to abstract evaluation """
    def __init__(self, input_fn, model_fn, model_dir, params, sess=None):
        self.sess = sess
        self.input_fn = input_fn
        self.model_fn = model_fn
        self.model_dir = model_dir
        self.params = params
        self.features = self.input_fn()

        if isinstance(self.features, tuple) and len(self.features) == 2:
            self.features, _ = self.features

        spec = self.model_fn(self.features, None, tf.estimator.ModeKeys.PREDICT, self.params)
        self.predictions = spec.predictions()

    @_coconut_tco
    def predict(self, **kargs):
        if self.sess is None:
            self.sess = tf.Session()
            saver = tf.train.Saver()
            path = tf.train.latest_checkpoint(self.model_dir)
            saver.restore(self.sess, path)
        feed_dict = dict(((key), (kargs[key])) for key in self.features)
        return _coconut_tail_call(self.sess.run, predictions, feed_dict=feed_dict)
