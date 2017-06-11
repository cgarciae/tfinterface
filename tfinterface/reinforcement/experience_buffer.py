#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x4a5b9aa9

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

from tfinterface.interfaces import ExperienceBufferInterface
from operator import itemgetter


class ExperienceReplay(ExperienceBufferInterface):

    def __init__(self, tuple_len, lst=None, max_length=100000):
        self.list = lst if lst is not None else []
        self.max_length = max_length
        self.tuple_len = tuple_len

    def append(self, *experience):
        self.list.append(experience)

        if len(self.list) > self.max_length:
            self.list.pop(0)


    def random_batch(self, batch_size):
        idx = self.get_random_idx(batch_size)
        batch = [self.list[i] for i in idx]
        return self.__class__(self.__class__, lst=batch)

    def unzip(self):
        if len(self) > 0:
            return ([list(e) for e in zip(*self.list)])
        else:
            return [tuple()] * self.tuple_len

    def __iter__(self):
        return iter(self.list)


    def reset(self):
        pass


    def __len__(self):
        return len(self.list)

    def __getitem__(self, *args, **kwargs):
        return self.list.__getitem__(*args, **kwargs)
