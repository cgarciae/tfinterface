#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x58e477fc

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

import numpy as np
import tensorflow as tf
from tfinterface.reinforcement import ExperienceReplay


class TestSupervised(object):

    def test_replay(self):

        xp = ExperienceReplay(3)

        xp.append(1, 2, 3)
        xp.append(4, 5, 6)

        a, b, c = xp.unzip()

        assert a == [1, 4]
        assert b == [2, 5]
        assert c == [3, 6]

        ar, br, cr = xp.random_batch(1).unzip()

        assert 1 in ar or 4 in ar
        assert 2 in br or 5 in br
        assert 3 in cr or 6 in cr
