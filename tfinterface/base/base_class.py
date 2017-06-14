#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x26d8aeb5

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
from tfinterface.decorators import return_self
from tfinterface.decorators import with_graph_as_default
from tfinterface.decorators import copy_self
from abc import ABCMeta
from abc import abstractmethod


class Base(object):
    __metaclass__ = ABCMeta


    def __init__(self, name, graph=None, sess=None):
        self.name = name
        self._template = tf.make_template(self.name, self.__class__._build)
        self.graph = graph if graph else tf.get_default_graph()
        self.sess = sess if sess else tf.get_default_session() if tf.get_default_session() else tf.Session(graph=self.graph)



    def _build(self, *args, **kwargs):
        return self.build_tensors(*args, **kwargs)

    @abstractmethod
    def build_tensors(self, *args, **kwargs):
        pass

    @with_graph_as_default
    @copy_self
    def __call__(self, *args, **kwargs):
        return self._template(self, *args, **kwargs)
