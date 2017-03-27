#!/usr/bin/python

# DQN implementation of https://github.com/matthiasplappert/keras-rl for Keras
# was used with epsilon-greedy per-episode decay policy.

import numpy as np
import gym
from gym import wrappers
from tfinterface.utils import get_run
from tfinterface.reinforcement import DQN
import random
import tensorflow as tf

ENV_NAME = 'LunarLander-v2'

run = get_run()

#env
env = gym.make(ENV_NAME)
env = wrappers.Monitor(env, "monitor/{run}".format(run = run))

# To get repeatable results.
sd = 16
np.random.seed(sd)
random.seed(sd)
env.seed(sd)

#parameters
state_temporal_augmentation = 3
nb_actions = env.action_space.n
nb_states = env.observation_space.shape[0] * state_temporal_augmentation

class Network(object):
    def __init__(self, inputs, nb_actions):

        ops = dict(
            kernel_initializer=tf.random_uniform_initializer(minval=0.0, maxval=0.01),
            bias_initializer=tf.random_uniform_initializer(minval=0.0, maxval=0.01)
        )

        net = inputs.s

        net = tf.layers.dense(net, 64, activation=tf.nn.elu, name="elu_layer")
        net = tf.nn.dropout(net, inputs.keep_prob)

        self.Qs = tf.layers.dense(net, nb_actions)


dqn = DQN(
    lambda inputs: Network(inputs, nb_actions),
    nb_states
)
