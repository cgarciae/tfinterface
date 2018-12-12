#!/usr/bin/env python
from __future__ import print_function, absolute_import, unicode_literals, division

from copy import copy

def return_self(method):

    def new_method(self, *args, **kwargs):

        method(self, *args, **kwargs)

        return self

    return new_method

def copy_self(method):

    def new_method(self, *args, **kwargs):
        new = copy(self)
        return method(new, *args, **kwargs)

    return new_method


def with_graph_as_default(method):

    def new_method(self, *args, **kwargs):
        with self.graph.as_default():
            return method(self, *args, **kwargs)

    return new_method
