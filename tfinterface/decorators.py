#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x7a58328c

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

from copy import copy

def return_self(method):

    def new_method(self, *args, **kwargs):

        method(self, *args, **kwargs)

        return self

    return new_method

def copy_self(method):

    def new_method(self, *args, **kwargs):
        new = copy(self)
        return method(new, *args, **kwargs)

    return new_method


def with_graph_as_default(method):

    def new_method(self, *args, **kwargs):
        with self.graph.as_default():
            return method(self, *args, **kwargs)

    return new_method
