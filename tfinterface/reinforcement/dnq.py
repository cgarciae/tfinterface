#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xc5ae263

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

from tfinterface.model_base import ModelBase
from tfinterface.utils import select_columns
from tfinterface.utils import soft_if
from tfinterface.utils import huber_loss
from .experience_buffer import ExperienceReplay
from rl.policy import Policy
from rl.policy import GreedyQPolicy
# from rl.agents import DQNAgent
from tfinterface.reinforcement import ExperienceReplay

import numpy as np
from numpy.random import choice
import random
import tensorflow as tf
from scipy.interpolate import interp1d


class EpsGreedyQPolicy(object):
    def __init__(self, model, eps=0.1, scope="dqn_policy"):
        self.model = model
        with tf.variable_scope(scope):
            if hasattr(eps, '__call__'):
                self.eps = eps(model.inputs.global_step)
            else:
                self.eps = tf.convert_to_tensor(eps)

    def select_action(self, q_values):
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        eps = self.model.sess.run(self.eps)

        if np.random.uniform() < eps:
            action = np.random.random_integers(0, nb_actions - 1)
        else:
            action = np.argmax(q_values)
        return action


class DQNInputs(object):

    def __init__(self, nb_states, batch_size=None, scope="inputs"):

        with tf.name_scope(scope):
            self.s = tf.placeholder(tf.float32, shape=[batch_size, nb_states], name="s")
            self.a = tf.placeholder(tf.int32, shape=[batch_size], name="a")
            self.r = tf.placeholder(tf.float32, shape=[batch_size], name="r")
            self.done = tf.placeholder(tf.bool, shape=[batch_size], name="done")

            self.keep_prob = tf.placeholder(tf.float32, shape=[], name="keep_prob")
            self.training = tf.placeholder(tf.bool, shape=[], name="training")

            self.global_step = tf.Variable(0, trainable=False, name="global_step")


    def predict_feed(self, S):
        return {self.s: S, self.keep_prob: 1.0, self.training: False}

    def train_feed(self, S, A, R, Done, keep_prob=0.5):
        return {self.s: S, self.a: A, self.r: R, self.done: Done, self.keep_prob: keep_prob, self.training: True}


class DQN(ModelBase):

    def _build(self, model_fn, nb_states, memory=None, policy=None, test_policy=None, enable_double_dqn=True, enable_dueling_network=False, dueling_type='avg', gamma=0.99, inputs_batch_size=None, nb_steps_warmup=1000, train_interval=1, memory_interval=1, target_update=10000, delta_range=None, delta_clip=np.inf, scope="dqn", model_scope="model", target_model_scope="target_model", optimizer=tf.train.AdamOptimizer, inputs=None, inputs_class=DQNInputs, memory_max_length=100000, learning_rate=0.001, eps=0.1):

        self.memory = memory if memory else ExperienceReplay(4, max_length=memory_max_length)
        self.gamma = gamma
        self.nb_steps_warmup = nb_steps_warmup
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.target_update = target_update
        self.delta_range = delta_range
        self.delta_clip = delta_clip


        self.inputs = inputs_class(nb_states, inputs_batch_size) if inputs is None else inputs
        self.global_step_update = self.global_step.assign_add(1)

        self.policy = policy(self) if policy else EpsGreedyQPolicy(self, eps=eps)
        self.test_policy = test_policy(self) if test_policy else GreedyQPolicy()

        with tf.variable_scope(model_scope):
            self.model = model_fn(self.inputs)
            self.model_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_scope)

        with tf.variable_scope(target_model_scope):
            self.target_model = model_fn(self.inputs)
            self.target_model_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_model_scope)

        with tf.variable_scope(scope):

            self.target_model_target = tf.where(self.inputs.done, self.inputs.r, self.inputs.r + self.gamma * tf.reduce_max(self.target_model.Qs, axis=1))

            self.model_Qsa = select_columns(self.model.Qs, self.inputs.a) if not hasattr(self.model, "Qsa") else self.model.Qsa
            self.model_error = self.target_model_target - self.model_Qsa if not hasattr(self.model, "error") else self.model.error
            self.model_loss = (tf.reduce_mean)((huber_loss)(self.model_error)) if not hasattr(self.model, "loss") else self.model.loss
            self.model_learning_rate = self.model.learning_rate if hasattr(self.model, 'learning_rate') else learning_rate
            self.update = optimizer(self.model_learning_rate).minimize(self.model_loss, var_list=self.model_variables) if not hasattr(self.model, "update") else self.model.update

            if self.target_update < 1:
                self.update = tf.group(self.update, *[tv.assign_add(self.target_update * (mv - tv)) for mv, tv in zip(self.target_model_variables, self.model_variables)])
                self.update_target_hard = None
            else:
                self.update_target_hard = tf.group(*[tv.assign(mv) for mv, tv in zip(self.target_model_variables, self.model_variables)])


    def predict(self, S, training=False):
        Qs = self.sess.run(self.model.Qs, feed_dict=self.inputs.predict_feed(S))[0]
        policy = self.policy if training else self.test_policy
        return policy.select_action(q_values=Qs)

    def fit(self, env, nb_steps=1000000, keep_prob=0.5, step=0, batch_size=32):

        s = env.reset()

        while step < nb_steps:
            step += 1
            self.sess.run(self.global_step_update)

            a = self.predict([s], training=True)
            s1, r, done, info = env.step(a)

            if step % self.memory_interval == 0:
                self.memory.append(s, a, r, done)

            if step > self.nb_steps_warmup and step % self.train_interval == 0:
                S, A, R, Done = self.memory.random_batch(batch_size).unzip()

                train_feed = self.inputs.train_feed(S, A, R, Done, keep_prob=keep_prob)
                _ = self.sess.run(self.update, feed_dict=train_feed)

                if self.update_target_hard and step % self.target_update == 0:
                    self.sess.run(self.update_target_hard)


            if done:
                s = env.reset()
            else:
                s = s1
