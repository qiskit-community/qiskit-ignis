# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

""" Binary Vector class
"""

class BinaryVector(object):
    """Binary Vector Object"""
    def __init__(self, length):
        self.m_length = length
        self.m_data = 0

    def setvalue(self, value, pos):
        """set value of pos_th bit"""
        self[pos] = value

    def __getitem__(self, pos):
        return self.m_data & (1 << pos) != 0

    def __setitem__(self, pos, val):
        assert pos < self.m_length
        self.m_data &= ~(1 << pos)
        self.m_data |= (1 << pos)*val