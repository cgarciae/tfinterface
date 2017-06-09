#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x3f7db6a3

# Compiled with Coconut version 1.2.2-post_dev12 [Colonel]

# Coconut Header: --------------------------------------------------------

from __future__ import print_function, absolute_import, unicode_literals, division

import sys as _coconut_sys, os.path as _coconut_os_path
_coconut_file_path = _coconut_os_path.dirname(_coconut_os_path.abspath(__file__))
_coconut_sys.path.insert(0, _coconut_file_path)
from __coconut__ import _coconut, _coconut_MatchError, _coconut_tail_call, _coconut_tco, _coconut_igetitem, _coconut_compose, _coconut_pipe, _coconut_starpipe, _coconut_backpipe, _coconut_backstarpipe, _coconut_bool_and, _coconut_bool_or, _coconut_minus, _coconut_map, _coconut_partial
from __coconut__ import *
_coconut_sys.path.remove(_coconut_file_path)

# Compiled Coconut: ------------------------------------------------------

import tensorflow as tf

import numpy as np
from numpy.random import choice
import random
from phi.api import *
import tensorflow as tf
from tfinterface.reinforcement import OnBatchModel
from tfinterface.reinforcement import ExperienceReplay
from tfinterface.utils import select_columns
from tfinterface.utils import soft_if
from tfinterface.model_base import ModelBase
from tensorflow.python import debug as tf_debug
import os
from scipy.interpolate import interp1d


class Inputs(object):
    def __init__(self, n_states, scope):
        with tf.variable_scope(scope) :
            self.episode_length = tf.placeholder(tf.int64, [], name='episode_length')

            self.s = tf.placeholder(tf.float32, [None, n_states], name='s')
            self.a = tf.placeholder(tf.int32, [None], name='a')
            self.r = tf.placeholder(tf.float32, [None], name='r')
            self.V1 = tf.placeholder(tf.float32, [None], name='V1')
            self.done = tf.placeholder(tf.float32, [None], name='done')
            self.learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
            self.keep_prob = tf.placeholder(tf.float32, [], name='training')

    def fit_feed(self, S, A, R, V1, Done, learning_rate, keep_prob):
        return {self.s: S, self.a: A, self.r: R, self.V1: V1, self.done: Done, self.learning_rate: learning_rate, self.keep_prob: keep_prob}


class Critic(object):
    def __init__(self, base_model, inputs, n_actions, n_states, y, scope):
        with tf.variable_scope(scope) :
            self.V = base_model.define_critic_network(inputs, n_actions, n_states)

            self.target = soft_if(inputs.done, inputs.r, inputs.r + y * inputs.V1)

            self.error = self.target - self.V
            self.loss = (tf.reduce_mean)((tf.nn.l2_loss)(self.error))

            self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

            self.update = tf.train.AdamOptimizer(inputs.learning_rate).minimize(self.loss, var_list=self.variables)

            avg_error, std_error = tf.nn.moments(self.error, [0])
            self.summaries = tf.summary.merge([tf.summary.scalar('loss', self.loss), tf.summary.scalar('avg_target', tf.reduce_mean(self.target)), tf.summary.scalar('variables_sum', sum([tf.reduce_sum(v) for v in self.variables])), tf.summary.scalar('avg_error', avg_error), tf.summary.scalar('std_error', std_error), tf.summary.histogram('avg_action', ((_coconut_partial(tf.reduce_mean, {}, 1, axis=0))((_coconut_partial(tf.one_hot, {1: n_actions}, 2))(inputs.a))))] + [tf.summary.histogram('var{}'.format(i), self.variables[i]) for i in range(len(self.variables))])



class Actor(object):
    def __init__(self, base_model, inputs, target_critic, n_actions, n_states, y, scope):
        with tf.variable_scope(scope) :
            self.P = base_model.define_actor_network(inputs, n_actions, n_states)

            self.Pa = select_columns(self.P, inputs.a)

            self.loss = -tf.log(tf.clip_by_value(self.Pa, 1e-3, 1.0)) * target_critic.error
            self.loss = tf.reduce_mean(self.loss)

            self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

            self.update = tf.train.AdamOptimizer(inputs.learning_rate).minimize(self.loss, var_list=self.variables)

            self.summaries = tf.summary.merge([tf.summary.scalar('loss', self.loss), tf.summary.scalar('Pa0', tf.reduce_mean(self.P[:, 0])), tf.summary.scalar('Pa1', tf.reduce_mean(self.P[:, 1])), tf.summary.scalar('variables_sum', sum([tf.reduce_sum(v) for v in self.variables])), tf.summary.histogram('avg_action', ((_coconut_partial(tf.reduce_mean, {}, 1, axis=0))((_coconut_partial(tf.one_hot, {1: n_actions}, 2))(inputs.a))))] + [tf.summary.histogram('var{}'.format(i), self.variables[i]) for i in range(len(self.variables))])

class DeepActorCritic(ModelBase):

    def define_model(self, n_actions, n_states, y=0.98, buffer_length=50000, pi=0.01):
        self.global_max = float('-inf')

        self.replay_buffer = ExperienceReplay(max_length=buffer_length)


        with self.graph.as_default(), tf.device("cpu:0") :

            self.inputs = Inputs(n_states, "inputs")

            self.critic = Critic(self, self.inputs, n_actions, n_states, y, "critic")
            self.target_critic = Critic(self, self.inputs, n_actions, n_states, y, "target_critic")
            self.actor = Actor(self, self.inputs, self.target_critic, n_actions, n_states, y, "actor")

            self.update = tf.group(self.critic.update, self.actor.update)

            self.episode_length_summary = tf.summary.scalar('episode_length', self.inputs.episode_length)

            self.summaries = tf.summary.merge([self.actor.summaries, self.critic.summaries, self.target_critic.summaries])

            self.update_target = tf.group(*[t.assign_add(pi * (a - t)) for t, a in zip(self.target_critic.variables, self.critic.variables)])

    def define_actor_network(self, inputs, n_actions, n_states):
        ops = dict(trainable=True, kernel_initializer=tf.random_uniform_initializer(minval=0.0, maxval=0.01), use_bias=False, bias_initializer=None)

        return ((_coconut_partial(tf.layers.dense, {1: n_actions}, 2, activation=tf.nn.softmax, name='softmax_layer', **ops))((_coconut_partial(tf.nn.dropout, {1: inputs.keep_prob}, 2))((_coconut_partial(tf.layers.dense, {1: 32}, 2, activation=tf.nn.relu, name='relu_layer', **ops))(inputs.s))))


    def define_critic_network(self, inputs, n_actions, n_states):
        ops = dict(trainable=True, kernel_initializer=tf.random_uniform_initializer(minval=0.0, maxval=0.01), use_bias=False, bias_initializer=None)

        return (((lambda t: t[:, 0]))((_coconut_partial(tf.layers.dense, {1: 1}, 2, name='linear_layer', **ops))((_coconut_partial(tf.nn.dropout, {1: inputs.keep_prob}, 2))((_coconut_partial(tf.layers.dense, {1: 32}, 2, activation=tf.nn.relu, name='relu_layer', **ops))(inputs.s)))))

    @_coconut_tco
    def fit_feed(self, *args, **kwargs):
        raise _coconut_tail_call(self.inputs.fit_feed, *args, **kwargs)

    @_coconut_tco
    def next_action(self, state, keep_prob, e=0.0):
        actions = self.sess.run(self.actor.P, feed_dict={self.inputs.s: [state], self.inputs.keep_prob: keep_prob})[0]
        n = len(actions)

        if random.random() < e:
            raise _coconut_tail_call(random.randint, 0, n - 1)
        else:
            raise _coconut_tail_call(np.random.choice, n, p=actions)




    def fit(self, env, keep_prob=0.5, learning_rate=0.01, print_step=10, update_target=32, episodes=100000, max_episode_length=float('inf'), batch_size=32):
        r_total = 0.

        for episode in range(episodes):
            done = False
            ep_step = 0
            s = env.reset()
            episode_length = 0
            ep_reward = 0.


            while not done and ep_step <= max_episode_length:
                self.global_step += 1
                episode_length += 1
                ep_step += 1

                _learning_rate = learning_rate(self.global_step) if hasattr(learning_rate, '__call__') else learning_rate

                a = self.next_action(s, keep_prob)
                s1, r, done, info = env.step(a)
                r_total += r
                ep_reward += r

                self.replay_buffer.append((s, a, r, s1, float(done)))

                S, A, R, S1, Done = self.replay_buffer.random_batch(batch_size).unzip()
                V1 = self.sess.run(self.target_critic.V, feed_dict={self.inputs.s: S1, self.inputs.keep_prob: 1.0})

                feed_dict = self.fit_feed(S, A, R, V1, Done, _learning_rate, True)

                if self.global_step > 1:
                    _, summaries = self.sess.run([self.update, self.summaries], feed_dict=feed_dict)
                    self.writer.add_summary(summaries)

                if self.global_step % update_target == 0:
                    self.sess.run(self.update_target)

                s = s1



            episode_length_summary = self.sess.run(self.episode_length_summary, feed_dict={self.inputs.episode_length: episode_length})
            self.writer.add_summary(episode_length_summary)


            if ep_reward >= self.global_max:
                print("[MAX] Episode: {}, Length: {}, Reward: {}, buffer_len: {}".format(episode, episode_length, ep_reward, len(self.replay_buffer)))
                self.save(model_path=self.model_path + ".max")
                self.global_max = episode_length


            if episode % print_step == 0 and episode > 0:
                avg_r = r_total / print_step
                actor_loss = self.sess.run(self.actor.loss, feed_dict=feed_dict)
                print("[NOR] Episode: {}, Length: {}, Avg Reward: {}, Learning Rate: {}, buffer_len: {}".format(episode, episode_length, avg_r, _learning_rate, len(self.replay_buffer)))
                print("Loss: {}".format(actor_loss))
                self.save()
                r_total = 0.
