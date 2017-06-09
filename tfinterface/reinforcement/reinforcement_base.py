#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x6469fc5c

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

from .experience_buffer import ListEpisodeBuffer
from .experience_buffer import EmptyBuffer
from tfinterface.interfaces import ReinforcementInterface
from tfinterface.interfaces import TrainerInterface
from tfinterface.model_base import ModelBase
from tfinterface.trainer_base import TrainerBase
from abc import abstractmethod

class ReinforcementModel(ModelBase, ReinforcementInterface):
    """docstring for ReinforcementModel."""

    def fit(self, *args, **kwargs):
        self.trainer.fit(*args, **kwargs)

    @abstractmethod
    def experience_feed(self):
        pass

    @abstractmethod
    def batch_feed(self):
        pass


class OnExperienceModel(ReinforcementModel):
    def batch_feed(self, *args, **kwargs):
        return {}

class OnBatchModel(ReinforcementModel):
    def experience_feed(self, *args, **kwargs):
        return {}

class ReinforcementTrainerBase(TrainerBase):
    """docstring for ReinforcementTrainer."""

    def __init__(self, *args, **kwargs):
        self.experience_buffer = kwargs.pop("experience_buffer", self.default_buffer)
        self.batch_size = kwargs.pop("batch_size", 64)

        super(ReinforcementTrainerBase, self).__init__(*args, **kwargs)
        self.global_reward = 0



    def fit(self, env, episodes=5000, max_episodes=float('Inf')):
        self.env = env
        self.fit_step = -1
        self.fit_reward = 0
        self.on_training_start()

        for episode in range(episodes):
            s = env.reset()
            self.model.reset()
            self.experience_buffer.reset()
            self.on_episode_start()

            done = False
            self.episode_step = -1
            self.episode_reward = 0
            self.episode = episode


            while not done and self.episode_step < max_episodes:
                self.global_step += 1
                self.fit_step += 1
                self.episode_step += 1
                self.on_experience_start()
                action_kwargs = self.choose_action_kwargs()
                a = self.model.next_action(s, **action_kwargs)
                s1, r, done, info = env.step(a)

                self.global_reward += r
                self.fit_reward += r
                self.episode_reward += r

                experience = self.get_experience(s, a, r, s1, done, info)
                self.experience_buffer.append(experience)
                self.train_on_experience(*experience)
                self.after_experience(*experience)

                s = s1
                if done:
                    break


            experience_batch = self.process_experience_buffer()
            self.train_on_experience_batch(*experience_batch)
            self.after_episode(*experience_batch)

    def choose_action_kwargs(self):
        return {}

    @abstractmethod
    def get_experience(self, s, a, r, s1, done, info):
        pass

    @abstractmethod
    def on_training_start(self):
        pass

    @abstractmethod
    def on_episode_start(self):
        pass

    @abstractmethod
    def on_experience_start(self):
        pass

    @abstractmethod
    def train_on_experience(self, *args, **kwargs):
        pass

    @abstractmethod
    def train_on_experience_batch(self, *args, **kwargs):
        pass

    @abstractmethod
    def process_experience_buffer(self, *args, **kwargs):
        pass

    @abstractmethod
    def after_experience(self, *args, **kwargs):
        pass

    @abstractmethod
    def after_episode(self, *args, **kwargs):
        pass

    @abstractmethod
    def default_buffer(self):
        pass


class ReinforcementTrainer(ReinforcementTrainerBase):

    def on_episode_start(self):
        pass

    def on_experience_start(self):
        pass

    def on_training_start(self):
        pass

    @_coconut_tco
    def process_experience_buffer(self):
        raise _coconut_tail_call(self.experience_buffer.unzip)

    def after_experience(self, *args):
        pass

    def after_episode(self, *args):
        pass

    @property
    @_coconut_tco
    def default_buffer(self):
        raise _coconut_tail_call(EmptyBuffer)


class OnExperienceTrainer(ReinforcementTrainer):

    def process_experience_buffer(self):
        return ()

    def train_on_experience(self, *experience):
        feed_dict = self.model.experience_feed(*experience)
        self.model.sess.run(self.model.update, feed_dict=feed_dict)

    def train_on_experience_batch(self, *experience_batch):
        pass


class OnEpisodeTrainer(ReinforcementTrainer):
    def train_on_experience(self, *experience):
        pass

    def train_on_experience_batch(self, *experience_batch):
        feed_dict = self.model.batch_feed(*experience_batch)
        self.model.sess.run(self.model.update, feed_dict=feed_dict)