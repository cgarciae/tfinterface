#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xf2ea3106

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

#
# from tfinterface.model_base import ModelBase
# from tfinterface.supervised import SupervisedModel
# from tfinterface.reinforcement import ListEpisodeBuffer
# from pytest import raises
#
#
# class TestBase(object):
#
#     def test_base(self):
#         class SomeModel(ModelBase):
#             def _build(self, y, z = 1):
#                 self.y = y
#                 self.z = z
#
#             @property
#             def default_trainer(self): return lambda x: None
#
#         assert SomeModel(42).y == 42
#         assert SomeModel(42).z == 1
#         assert SomeModel(42, 2).z == 2
#
#     def test_raises_on_not_implemented(self):
#         class SomeModel(ModelBase):
#             pass
#
#         class SomeSupervisedModel(SupervisedModel):
#             pass
#
#         with raises(TypeError):
#             SomeModel()
#             SomeSupervisedModel()
#
#
#     def test_supervised_model_implementation(self):
#
#         class SomeSupervisedModel(SupervisedModel):
#
#             def _build(*args, **kwargs):
#                 self.x = 1
#                 self.y = 2
#                 self.loss = 3
#
#             def predict_feed(): pass
#
#         assert SomeSupervisedModel()
#
#
#     def test_list_buffer(self):
#         buf = ListEpisodeBuffer()
#         buf.append((5,6))
#
#         assert buf.random_batch(10)[0] == (5,6)
#
#
