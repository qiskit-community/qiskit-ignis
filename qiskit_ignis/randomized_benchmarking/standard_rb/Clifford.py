# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Clifford Operator class
"""

import copy
import numpy as np
from verification.randomized_benchmarking.standard_rb.BinaryVector import BinaryVector

def invert_circuit(circuit):
    """invert the circuit--reverse and invert non self-inverse gates"""
    invcircuit = circuit.copy()
    invcircuit.reverse()
    # replace r by rinv and rinv by r, s by sinv and sinv by s
    for i, _ in enumerate(invcircuit):
        split = invcircuit[i].split()
        if split[0] == 'r':
            invcircuit[i] = 'rinv ' + split[1]
        elif split[0] == 'rinv':
            invcircuit[i] = 'r ' + split[1]
        elif split[0] == 's':
            invcircuit[i] = 'sinv ' + split[1]
        elif split[0] == 'sinv':
            invcircuit[i] = 's ' + split[1]
    return invcircuit

#----------------------------------------------------------------------------------------

class Clifford(object):
    """Clifford Tableau Object"""
    def __init__(self, nqubits):
        self.n = nqubits
        self.table = []
        # initial state = all-zeros
        # add destabilizers
        for i in range(self.n):
            pauli = {'X': BinaryVector(self.n), 'Z': BinaryVector(self.n), 'phase': 0}
            pauli['X'].setvalue(1, i)
            self.table.append(pauli)
        # add stabilizers
        for i in range(self.n):
            pauli = {'X': BinaryVector(self.n), 'Z': BinaryVector(self.n), 'phase': 0}
            pauli['Z'].setvalue(1, i)
            self.table.append(pauli)
        # add auxiliary row
        pauli = {'X': BinaryVector(self.n), 'Z': BinaryVector(self.n), 'phase': 0}
        self.table.append(pauli)
        self.circuit = []
        self.cannonical = None
        self.matrix = None


    def get_table(self):
        """return the table representation, updating from matrix if needed"""
        if self.table is None:
            return self.to_table()
        return self.table

    def from_matrix(self, matrix):
        """set a clifford from a matrix representation"""
        self.matrix = matrix
        self.table = None

    def set_cannonical(self, cannonical):
        """set internal memory of cannoical order
        --must be set externally, does not decide for itself"""
        self.cannonical = cannonical

    def get_cannonical(self):
        """get cannoical order, if set -- not used"""
        return self.cannonical

    def get_circuit(self):
        """give circuit structure--does not compute the circuit"""
        # if you want to compute the circuit use decompose()
        return self.circuit


    def circuit_append(self, gatelist):
        """add to circuit list"""
        self.circuit = self.circuit + gatelist
    def circuit_prepend(self, gatelist):
        """add to circuit list -- not used"""
        self.circuit = gatelist + self.circuit
    def size(self):
        """report size in number of qubits"""
        return self.n

    def print_table(self):
        """print out the table form of the Clifford -- not used, only for debug"""
        table = self.get_table()
        for i in range(2*self.n):
            print(table[i]['X'].m_data, table[i]['Z'].m_data, table[i]['phase'])
        print("--------------------------------------------")

    def print_matrix(self):
        """Print out the matrix form of the Clifford tableu -- not used, only for debug"""
        matrix = self.get_matrix()
        print(matrix)
        print("--------------------------------------------")

    def get_matrix(self):
        """Return and set 2n+1 X 2n dimensional matrix for of a Clifford tableu"""
        self.matrix = np.zeros([2*self.n, 2*self.n+1], dtype=np.uint8, order='F')
        for i in range(2*self.n):
            for j in range(self.n):
                self.matrix[i, j] = self.table[i]['X'][j]
                self.matrix[i, j+self.n] = self.table[i]['Z'][j]
        for i in range(2*self.n):
            self.matrix[i, 2*self.n] = self.table[i]['phase']
        return self.matrix

    def to_table(self):
        """Return and set tableu in table form -- internal (in get_table)"""
        self.table = []
        for i in range(2*self.n):
            pauli = {'X': BinaryVector(self.n), 'Z': BinaryVector(self.n), 'phase': 0}
            for j in range(self.n):
                pauli['X'][j] = self.matrix[i, j]
                pauli['Z'][j] = self.matrix[i, j+self.n]
            pauli['phase'] = self.matrix[i, 2*self.n]
            self.table.append(pauli)
        # append auxiallary row
        pauli = {'X': BinaryVector(self.n), 'Z': BinaryVector(self.n), 'phase': 0}
        self.table.append(pauli)
        self.matrix = None
        return self.table

    def key(self):
        """Return a big int from the matrix form to use as a key in dictionaries"""
        # the transpose is so that the phase bits will be the lowest order bits in the key
        mat = self.get_matrix().transpose()
        mat = mat.reshape(mat.size)
        ret = int(0)
        for bit in mat:
            ret = (ret << 1) | int(bit)
        return ret

# ----------------------------------------------------------------------------------------
# Quantum gates operations
# ----------------------------------------------------------------------------------------
    def cnot(self, con, tar):
        """controlled-NOT"""
        self.get_table()
        for i in range(2*self.n):
            self.table[i]['phase'] ^= self.table[i]['X'][con] and\
                self.table[i]['Z'][tar] and (self.table[i]['X'][tar] ^ self.table[i]['Z'][con] ^ 1)
        for i in range(2*self.n):
            self.table[i]['X'].setvalue(self.table[i]['X'][tar] ^ self.table[i]['X'][con], tar)
            self.table[i]['Z'].setvalue(self.table[i]['Z'][tar] ^ self.table[i]['Z'][con], con)

    def s(self, q):
        """S-gate"""
        self.get_table()
        for i in range(2*self.n):
            self.table[i]['phase'] ^= (self.table[i]['X'][q] and self.table[i]['Z'][q])
            self.table[i]['Z'].setvalue(self.table[i]['Z'][q] ^ self.table[i]['X'][q], q)

    def z(self, q):
        """Z-Gate"""
        self.get_table()
        for i in range(2*self.n):
            self.table[i]['phase'] ^= self.table[i]['X'][q]

    def x(self, q):
        """X-Gate"""
        self.get_table()
        for i in range(2*self.n):
            self.table[i]['phase'] ^= self.table[i]['Z'][q]

    def y(self, q):
        """Y-Gate"""
        self.get_table()
        for i in range(2*self.n):
            self.table[i]['phase'] ^= (self.table[i]['Z'][q] ^ self.table[i]['X'][q])

    def h(self, q):
        """Hadamard Gate"""
        self.get_table()
        for i in range(2*self.n):
            self.table[i]['phase'] ^= (self.table[i]['X'][q] and self.table[i]['Z'][q])
            # exchange X and Z
            b = self.table[i]['X'][q]
            self.table[i]['X'].setvalue(self.table[i]['Z'][q], q)
            self.table[i]['Z'].setvalue(b, q)

    def r(self, q):
        """R Gate"""
        self.h(q)
        self.s(q)
        self.h(q)
        self.s(q)

    def transpose(self, i, j):
        """interchange qubits i and j"""
        self.get_table()
        for r in range(2*self.n):
            b = self.table[r]['X'][i]
            self.table[r]['X'].setvalue(self.table[r]['X'][j], i)
            self.table[r]['X'].setvalue(b, j)
            b = self.table[r]['Z'][i]
            self.table[r]['Z'].setvalue(self.table[r]['Z'][j], i)
            self.table[r]['Z'].setvalue(b, j)

#----------------------------------------------------------------------------------------
# used in row-reduction Gottesman method (see also R. Koenig)
# Note, I think Gottestman and Koenig use R when we would use S as the pi/8 phase gate
# internal -- used in decompose
# ----------------------------------------------------------------------------------------
    def make1forXkk(self, k):
        """make a 1 in the kth row for qubit k by permuting columns l>k
        or if necessary applying Hadamard"""
        self.get_table()
        if self.table[k]['X'][k] == 1:
            return
#       try to find non-zero element
        for l in range(k+1, self.n):
            if self.table[k]['X'][l] == 1:
                self.transpose(k, l)
                self.circuit_append(['cnot '+str(k)+' '+str(l)])
                self.circuit_append(['cnot '+str(l)+' '+str(k)])
                self.circuit_append(['cnot '+str(k)+' '+str(l)])
                return
#       no non-zero element found: need to apply Hadamard somewhere
        for l in range(k, self.n):
            if self.table[k]['Z'][l] == 1:
                self.h(l)
                self.circuit_append(['h '+str(l)])
                if l != k:
                    self.transpose(l, k)
                    self.circuit_append(['cnot '+str(k)+' '+str(l)])
                    self.circuit_append(['cnot '+str(l)+' '+str(k)])
                    self.circuit_append(['cnot '+str(k)+' '+str(l)])
                return

    def makeXrowzero(self, k):
        """makes sure A  [k][l]=0 for all  l>k by applying CNOTS assumes k<=N and A[k][k]=1"""
        self.get_table()
        for l in range(k+1, self.n): # treat Xs
            if self.table[k]['X'][l] == 1:
                self.cnot(k, l)
                self.circuit_append(['cnot '+str(k)+' '+str(l)])

#       check whether Zs need to be set to zero:
        nonzero = False
        for l in range(k, self.n):
            if self.table[k]['Z'][l] == 1:
                nonzero = True

        if nonzero:  # do so
            if self.table[k]['Z'][k] == 0: # to treat Zs: make sure A[k]['Z'][k]=1
                self.s(k)
                self.circuit_append(['s '+str(k)])
            # reverse CNOTS
            for l in range(k+1, self.n):
                if self.table[k]['Z'][l] == 1:
                    self.cnot(l, k)
                    self.circuit_append(['cnot '+str(l)+' '+str(k)])
            # set A[k][Z][k] to zero
            self.s(k)
            self.circuit_append(['s '+str(k)])

    def makeZrowzero(self, k):
        """makes sure A[k+N]['Z'][l]=0 for all  l>k by applying (reverse)
        CNOTS assumes k<N and makeXrowzero called first"""
        self.get_table()
        # check whether Zs need to be set to zero:
        nonzero = False
        for l in range(k+1, self.n):
            if self.table[k+self.n]['Z'][l] == 1:
                nonzero = True
        if nonzero:  # do so
            for l in range(k+1, self.n):  # treat Zs
                if self.table[k+self.n]['Z'][l] == 1:
                    self.cnot(l, k)
                    self.circuit_append(['cnot '+str(l)+' '+str(k)])
        # check whether Xs need to be set to zero:
        nonzero = False
        for l in range(k, self.n):
            if self.table[k+self.n]['X'][l] == 1:
                nonzero = True
        if nonzero:  # do so
            self.h(k)
            self.circuit_append(['h '+str(k)])
            for l in range(k+1, self.n): # treat Xs
                if self.table[k+self.n]['X'][l] == 1:
                    self.cnot(k, l)
                    self.circuit_append(['cnot '+str(k)+' '+str(l)])
            if self.table[k+self.n]['Z'][k] == 1:
                self.s(k)
                self.circuit_append(['s '+str(k)])
            self.h(k)
            self.circuit_append(['h '+str(k)])

# ----------------------------------------------------------------------------------------
    def decompose(self):
        """extract a circuit to make the Clifford Gottesman-style"""
        self.circuit = []
        self.get_table()
        savedtable = copy.deepcopy(self.table)
        for i in range(self.n):
            self.make1forXkk(i) # put a 1 one into position by permuting and using Hadamards(i,i)
            self.makeXrowzero(i) # make all entries in row i except ith equal to 0
                                 #by using phase gate and CNOTS
            self.makeZrowzero(i) # treat Zs
        self.get_table()
        for i in range(self.n):
            if self.table[i]['phase'] == 1:
                self.z(i)
                self.circuit_append(['z '+str(i)])
            if self.table[i+self.n]['phase'] == 1:
                self.x(i)
                self.circuit_append(['x '+str(i)])
        # now reverse the circuit so that isn't the inverse of what it should be
        self.circuit = invert_circuit(self.circuit)
        self.table = savedtable
        self.matrix = None

#----------------------------------------------------------------------------------------
    def compose_circuit(self, cliff):
        """ compsose circuit """
        circ = cliff.get_circuit()
        for op in circ:
            split = op.split()
            q1 = int(split[1])
            if split[0] == 'r':
                self.h(q1)
                self.s(q1)
                self.h(q1)
                self.s(q1)
            elif split[0] == 'rinv':  # s.z = s^inverse
                self.s(q1)
                self.z(q1)
                self.h(q1)
                self.s(q1)
                self.z(q1)
                self.h(q1)
            elif split[0] == 'x':
                self.x(q1)
            elif split[0] == 'y':
                self.y(q1)
            elif split[0] == 'z':
                self.z(q1)
            elif split[0] == 'cnot':
                self.cnot(q1, int(split[2]))
            elif split[0] == 'h':
                self.h(q1)
            elif split[0] == 's':
                self.s(q1)
            elif split[0] == 'sinv':  # s.z = s^inverse
                self.s(q1)
                self.z(q1)
            else:
                print("error: unknown gate type: ", op)
        self.circuit_append(circ)

