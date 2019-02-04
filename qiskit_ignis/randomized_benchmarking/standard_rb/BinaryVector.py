# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Binary Vectors
adapted from Sergey Bravyi's C++ code
simplified to use Python bigints as the data by John A. Smolin
"""

class BinaryVector(object):
    """Binary Vector Object"""
    def __init__(self, length):
        self.m_length = length
        self.m_data = 0

    def copy(self):
        """return a copy instead of a reference"""
        retval = BinaryVector(self.m_length)
        retval.m_data = self.m_data.copy()
        return retval

    def setlength(self, length):
        """set length if it was zero"""
        if length == 0:
            return False
        if self.m_length > 0:
            return False
        self.m_length = length
        self.m_data = 0
        return True

    def setvalue(self, value, pos):
        """set value of pos_th bit"""
        self[pos] = value

    def set1(self, pos):
        """set value to 1"""
        self.setvalue(1, pos)

    def flipat(self, pos):
        """flip pos_th bit"""
        self.m_data ^= (1 << pos)

    def __iadd__(self, rhs):
        self.m_data ^= rhs.m_data
        return self

    def __add__(self, rhs):
        ret = BinaryVector(self.m_length)
        ret.m_data = self.m_data ^ rhs.m_data
        return ret

    def __getitem__(self, pos):
        return self.m_data & (1 << pos) != 0

    def __setitem__(self, pos, val):
        assert pos < self.m_length
        self.m_data &= ~(1 << pos)
        self.m_data |= (1 << pos)*val

    def swap(self, rhs):
        """exchange data with another BinaryVector"""
        tmp = rhs.m_length
        rhs.m_length = self.m_length
        self.m_length = tmp
        tmp = rhs.m_data
        rhs.m_data = self.m_data
        self.m_data = tmp

    def getlength(self):
        """report length"""
        return self.m_length

    def gethammingweight(self):
        """compute Hamming weight"""
        return bin(self.m_data).count("1")

    def makezero(self):
        """set to null"""
        self.m_data = 0

    def iszero(self):
        """compare to null"""
        if self.m_data:
            return False
        return True

    def issame(self, rhs):
        """compare two BinaryVectors"""
        if self.m_length != rhs.m_length:
            return False
        if self.m_data != rhs.m_data:
            return False
        return True

    def to_int(self):
        """convert to bigInt"""
        return self.m_data

    def from_int(self, j):
        """convert from bigInt"""
        self.m_data = j

    #def nonZeroIndices(self):
    # not implemented

    #def dotProduct(self,rhs):
    # not implemented

    def __eq__(self, rhs):
        return self.issame(rhs)

    #def gauss_eliminate(self,M,start_col):
    # not implemented
