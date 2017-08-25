#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xca14edb5

# Compiled with Coconut version 1.2.3 [Colonel]

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
from tfinterface.decorators import with_graph_as_default
import tensorflow as tf
from abc import abstractmethod
import threading

##############
# data
##############
class PlaceholderDefaults(_coconut.collections.namedtuple("PlaceholderDefaults", "tensor predict fit"), _coconut.object):
    __slots__ = ()
    __ne__ = _coconut.object.__ne__
class NoValue(_coconut.collections.namedtuple("NoValue", ""), _coconut.object):
    __slots__ = ()
    __ne__ = _coconut.object.__ne__


##############
# exceptions
##############

class NoValueException(Exception):
    pass


##############
# functions
##############

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



##############
# classes
##############
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

        queue_ops = input_specs.pop("queue_ops", {})

        queued = {}
        for name, spec in input_specs.items():

            if type(spec) is not dict:

                if type(spec) is tuple:
                    spec = dict(dtype=tf.float32, shape=spec)

                elif hasattr(spec, "__call__"):
                    spec = dict(tensor_fn=spec)

                else:
                    spec = dict(value=spec)


            if spec.get("queue", False):
                queued[name] = spec
                continue

            elif "shape" in spec:
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

        if queued:
            self.queue_runner = CustomRunner(self, queued, **queue_ops)

            for name, tensor in self.queue_runner.tensors_dict.items():
                setattr(self, name, tensor)

    @with_graph_as_default
    def start_queue(self, *args, **kwargs):
        return self.queue_runner.start_queue(*args, **kwargs)

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





class CustomRunner(object):
    """
    This class manages the the background threads needed to fill
        a queue full of data.
    """
    def __init__(self, inputs, queued, batch_size=64, capacity=2000, min_after_dequeue=1000, **queue_ops):
        self.inputs = inputs

        names = names = queued.keys()
        specs = queued.values()
        placeholder_shapes = [spec.get("shape", [None]) for spec in specs]
        shapes = [shape[1:] for shape in placeholder_shapes]
        dtypes = [spec.get("dtype", tf.float32) for spec in specs]

# placeholders_dict
        self.placeholders_dict = dict(((name), (tf.placeholder(dtype=dtype, shape=shape, name=name + "_placeholder"))) for dtype, shape, name in zip(dtypes, placeholder_shapes, names))


# The actual queue of data. The queue contains a vector for
# the mnist features, and a scalar label.
        self.queue = tf.RandomShuffleQueue(capacity, min_after_dequeue, dtypes, shapes=shapes, names=names, **queue_ops)

# The symbolic operation to add data to the queue
# we could do some preprocessing here or do it in numpy. In this example
# we do the scaling in numpy
        self.enqueue_op = self.queue.enqueue_many(self.placeholders_dict)

# tensors_dict
        self.tensors_dict = self.queue.dequeue_many(batch_size)



    def thread_main(self, data_generator_fn):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        data_generator = data_generator_fn()

        try:
            for data in data_generator:
                feed_dict = dict(((self.placeholders_dict[name]), (value)) for name, value in data.items())
                self.inputs.sess.run(self.enqueue_op, feed_dict=feed_dict)
        except Exception as e:
            import thread

            print(e)
            thread.interrupt_main()

    def start_queue(self, data_generator_fn, n_threads=1):
        """ Start background threads to feed queue """
        threads = []
        for n in range(n_threads):
            t = threading.Thread(target=self.thread_main, args=(data_generator_fn,))
            t.daemon = True  # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads
