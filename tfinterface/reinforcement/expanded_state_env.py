#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x4f2ce9be

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

from tfinterface.interfaces.environment import EnvironmentInterface
import numpy as np
import six
import gym

class ExpandedStateEnv(EnvironmentInterface):

    def __init__(self, env, expansion):
        self.env = gym.make(env) if isinstance(env, six.string_types) else env
        self.expansion = expansion

    def reset(self):
        s = self.env.reset()
        self.s = np.hstack((s,) * self.expansion)
        return self.s

    def step(self, a):
        s, r, done, info = self.env.step(a)
        n = len(s)
        self.s = np.hstack((self.s[n:], s))

        return self.s, r, done, info

    def __getattr__(self, attr):
        return getattr(self.env, attr)
