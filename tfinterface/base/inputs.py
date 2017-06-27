#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xdb0a47ac

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

from .base_class import Base
from tfinterface.decorators import return_self
import tensorflow as tf
from abc import abstractmethod

class PlaceholderDefaults(_coconut.collections.namedtuple("PlaceholderDefaults", "tensor predict fit"), _coconut.object):
    __slots__ = ()
    __ne__ = _coconut.object.__ne__
class NoValue(_coconut.collections.namedtuple("NoValue", ""), _coconut.object):
    __slots__ = ()
    __ne__ = _coconut.object.__ne__

class NoValueException(Exception):
    pass

def fit_tuple(*_coconut_match_to_args, **_coconut_match_to_kwargs):
    _coconut_match_check = False
    if (_coconut.len(_coconut_match_to_args) == 1) and (_coconut.isinstance(_coconut_match_to_args[0], PlaceholderDefaults)) and (_coconut.len(_coconut_match_to_args[0]) == 3):
        tensor = _coconut_match_to_args[0][0]
        fit = _coconut_match_to_args[0][2]
        if (not _coconut_match_to_kwargs):
            _coconut_match_check = True
    if not _coconut_match_check:
        _coconut_match_err = _coconut_MatchError("pattern-matching failed for " "'def fit_tuple(PlaceholderDefaults(tensor, _, fit)):'" " in " + _coconut.repr(_coconut.repr(_coconut_match_to_args)))
        _coconut_match_err.pattern = 'def fit_tuple(PlaceholderDefaults(tensor, _, fit)):'
        _coconut_match_err.value = _coconut_match_to_args
        raise _coconut_match_err

    if isinstance(fit, NoValue):
        raise NoValueException("No fit value given for {}".format(tensor))

    return tensor, fit

def predict_tuple(*_coconut_match_to_args, **_coconut_match_to_kwargs):
    _coconut_match_check = False
    if (_coconut.len(_coconut_match_to_args) == 1) and (_coconut.isinstance(_coconut_match_to_args[0], PlaceholderDefaults)) and (_coconut.len(_coconut_match_to_args[0]) == 3):
        tensor = _coconut_match_to_args[0][0]
        predict = _coconut_match_to_args[0][1]
        if (not _coconut_match_to_kwargs):
            _coconut_match_check = True
    if not _coconut_match_check:
        _coconut_match_err = _coconut_MatchError("pattern-matching failed for " "'def predict_tuple(PlaceholderDefaults(tensor, predict, _)):'" " in " + _coconut.repr(_coconut.repr(_coconut_match_to_args)))
        _coconut_match_err.pattern = 'def predict_tuple(PlaceholderDefaults(tensor, predict, _)):'
        _coconut_match_err.value = _coconut_match_to_args
        raise _coconut_match_err

    if isinstance(predict, NoValue):
        raise NoValueException("No predict value given for {}".format(tensor))

    return tensor, predict

class Inputs(Base):

    @abstractmethod
    def fit_feed(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict_feed(self, *args, **kwargs):
        pass

class GeneralInputs(Inputs):
    """docstring for GeneralInputs."""
    def __init__(self, name, graph=None, sess=None, **input_specs):
        super(GeneralInputs, self).__init__(name, graph=graph, sess=sess)
        self._input_specs = input_specs

    @return_self
    def build_tensors(self, **input_overrides):
        input_specs = self._input_specs.copy()
        input_specs.update(input_overrides)
        self._placeholder_defaults = {}

        for name, spec in input_specs.items():

            if type(spec) is not dict:

                if type(spec) is tuple:
                    spec = dict(dtype=tf.float32, shape=spec)

                elif hasattr(spec, "__call__"):
                    spec = dict(tensor_fn=spec)

                else:
                    spec = dict(value=spec)


            if "shape" in spec:
                dtype = spec.get("dtype", tf.float32)
                shape = spec.get("shape")
                tensor = tf.placeholder(dtype=dtype, shape=shape, name=name)

                self._placeholder_defaults[name] = PlaceholderDefaults(tensor, spec.get("predict", NoValue()), spec.get("fit", NoValue()))

            elif "value" in spec:
                value = spec.get("value")
                dtype = spec.get("dtype", None)
                tensor = tf.convert_to_tensor(value, dtype=dtype, name=name)


            elif "tensor_fn" in spec:
                tensor_fn = spec.get("tensor_fn")
                tensor = tensor_fn()


            setattr(self, name, tensor)


    def get_feed(self, **kwargs):
        return (dict(((getattr(self, key)), (value)) for key, value in kwargs.items()))

    def _get_fit_defaults(self):
        feed = {}

        for name, placeholder_defaults in self._placeholder_defaults.items():
            try:
                tensor, value = fit_tuple(placeholder_defaults)
                feed[tensor] = value
            except NoValueException as e:
                pass

        return feed

    def _get_predict_defaults(self):
        feed = {}

        for name, placeholder_defaults in self._placeholder_defaults.items():
            try:
                tensor, value = predict_tuple(placeholder_defaults)
                feed[tensor] = value
            except NoValueException as e:
                pass

        return feed

    def fit_feed(self, *args, **kwargs):
        feed = self._get_fit_defaults()
        feed.update(self.get_feed(*args, **kwargs))

        return feed


    def predict_feed(self, *args, **kwargs):
        feed = self._get_predict_defaults()
        feed.update(self.get_feed(*args, **kwargs))

        return feed
