#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xf55a6a6f

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
from tfinterface.base import Model
from tfinterface.metrics import r2_score
from tfinterface.metrics import sigmoid_score
from tfinterface.metrics import softmax_score
from tfinterface.utils import huber_loss
import cytoolz as cz
import itertools as it
from tfinterface.decorators import return_self
from tfinterface.decorators import with_graph_as_default
from tfinterface.decorators import copy_self
from .supervised_inputs import SupervisedInputs
from abc import abstractmethod

class SupervisedModel(Model):
    """
# Inteface
* `inputs : SupervisedInputs` -
* `predictions : Tensor` -
* `loss : Tensor` -
* `update : Tensor` -
    """

    def __init__(self, name, loss="mse", optimizer=tf.train.AdamOptimizer, learning_rate=0.001, **kwargs):
        super(SupervisedModel, self).__init__(name, **kwargs)

        self._loss_arg = loss
        self._optimizer = optimizer
        self._learning_rate_arg = learning_rate


    def __call__(self, inputs, inputs_class=SupervisedInputs):
        return super(SupervisedModel, self).__call__(inputs, inputs_class=inputs_class)

    @return_self
    def build_tensors(self):
        super(SupervisedModel, self).build_tensors()

        self.labels = self.get_labels()
        self.learning_rate = self.get_learning_rate()
        self.predictions = self.get_predictions()
        self.loss = self.get_loss()
        self.score_tensor = self.get_score_tensor()
        self.update = self.get_update()
        self.summaries = self.get_all_summaries()


    def get_labels(self):
        return self.inputs.labels

    def get_learning_rate(self):
        if hasattr(self.inputs, "learning_rate"):
            return self.inputs.learning_rate
        else:
            return self._learning_rate_arg

    @abstractmethod
    def get_predictions(self):
        pass

    @abstractmethod
    def get_loss(self):
        pass

    @abstractmethod
    def get_score_tensor(self):
        pass

    def get_update(self):
        return self._optimizer(self.learning_rate).minimize(self.loss, global_step=self.inputs.global_step)

    def get_all_summaries(self):
        standard = self.get_standard_summaries()
        summaries = self.get_summaries()


        return tf.summary.merge(standard + summaries)


    def get_standard_summaries(self):
        return [tf.summary.scalar("loss_summary", self.loss), tf.summary.scalar("score_summary", self.score_tensor)]

    def get_summaries(self):
        return []

    def predict(self, **kwargs):
        predict_feed = self.inputs.predict_feed(**kwargs)
        return self.sess.run(self.predictions, feed_dict=predict_feed)


    def score(self, **kwargs):
        fit_feed = self.inputs.fit_feed(**kwargs)
        score = self.sess.run(self.score_tensor, feed_dict=fit_feed)

        return score


    @with_graph_as_default
    @return_self
    def fit(self, epochs=2000, data_generator=None, log_summaries=False, log_interval=20, print_test_info=False, writer_kwargs={}):
        if log_summaries and not hasattr(self, "writer"):
            self.writer = tf.summary.FileWriter(self.logs_path, graph=self.graph, **writer_kwargs)

        if not hasattr(self, "summaries"):
            self.summaries = tf.no_op()

        if data_generator is None:
#generator of empty dicts
            data_generator = it.repeat({})

        data_generator = (_coconut.functools.partial(cz.take, epochs))(data_generator)

        for i, batch_feed_data in enumerate(data_generator):

            fit_feed = self.inputs.fit_feed(**batch_feed_data)
            _, summaries = self.sess.run([self.update, self.summaries], feed_dict=fit_feed)

            if log_summaries and i % log_interval == 0 and summaries is not None:
                self.writer.add_summary(summaries, global_step=self.sess.run(self.inputs.global_step))


            if print_test_info and i % log_interval == 0:
                loss, score = self.sess.run([self.loss, self.score_tensor], feed_dict=fit_feed)
                print("loss {}, score {}, at {}".format(loss, score, i))


class SoftmaxClassifier(SupervisedModel):
    """docstring for SoftmaxClassifier."""

    @abstractmethod
    def get_logits(self):
        pass

    def get_predictions(self):
        self.logits = self.get_logits()

        return tf.nn.softmax(self.logits)


    def get_loss(self):
        return ((tf.reduce_mean)(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)))


    def get_score_tensor(self):
        return softmax_score(self.labels, self.predictions)


class SigmoidClassifier(SupervisedModel):
    """docstring for SoftmaxClassifier."""

    @abstractmethod
    def get_logits(self):
        pass

    def get_predictions(self):
        self.logits = self.get_logits()

        return tf.nn.sigmoid(self.logits)


    def get_loss(self):
        return ((tf.reduce_mean)(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.labels)))


    def get_score_tensor(self):
        return sigmoid_score(self.labels, self.predictions)


class LinearClassifier(SupervisedModel):
    """docstring for SoftmaxClassifier."""

    def __init__(self, *args, **kwargs):
        loss = kwargs.pop("loss", huber_loss)

        if loss == "mse":
            loss = tf.nn.l2_loss

        elif loss == "huber":
            loss = huber_loss

        self._loss_fn = loss

        super(LinearClassifier, self).__init__(*args, **kwargs)



    def get_loss(self):
        return ((tf.reduce_mean)(self._loss_fn(self.predictions - self.labels)))


    def get_score_tensor(self):
        return r2_score(self.labels, self.predictions)
