# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
    Clifford Operator class
"""

import numpy as np
from qiskit_ignis.randomized_benchmarking.standard_rb.BinaryVector import BinaryVector


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

# ----------------------------------------------------------------------------------------
# Clifford properties
# ----------------------------------------------------------------------------------------
    def get_table(self):
        """return the table representation, updating from a matrix if needed"""
        if self.table is None:
            return self.to_table()
        return self.table

    def set_cannonical(self, cannonical):
        """set internal memory of cannonical order
        --must be set externally, does not decide for itself"""
        self.cannonical = cannonical

    def get_cannonical(self):
        """get cannoical order, if set"""
        return self.cannonical

    def get_circuit(self):
        """give circuit structure -- does not compute the circuit"""
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
        """print out the table form of the Clifford -- used for debug"""
        table = self.get_table()
        for i in range(2*self.n):
            print(table[i]['X'].m_data, table[i]['Z'].m_data, table[i]['phase'])
        print("--------------------------------------------")

    def print_matrix(self):
        """Print out the matrix form of the Clifford tableau -- used for debug"""
        matrix = self.get_matrix()
        print(matrix)
        print("--------------------------------------------")

    def get_matrix(self):
        """Return and set 2n+1 X 2n dimensional matrix for of a Clifford tableau"""
        self.matrix = np.zeros([2*self.n, 2*self.n+1], dtype=np.uint8, order='F')
        for i in range(2*self.n):
            for j in range(self.n):
                self.matrix[i, j] = self.table[i]['X'][j]
                self.matrix[i, j+self.n] = self.table[i]['Z'][j]
        for i in range(2*self.n):
            self.matrix[i, 2*self.n] = self.table[i]['phase']
        return self.matrix

    def to_table(self):
        """Return and set tableau in table form (used in get_table)"""
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
    def cx(self, con, tar):
        """Controlled-x gate"""
        self.get_table()
        for i in range(2*self.n):
            self.table[i]['phase'] ^= self.table[i]['X'][con] and\
                self.table[i]['Z'][tar] and (self.table[i]['X'][tar] ^ self.table[i]['Z'][con] ^ 1)
        for i in range(2*self.n):
            self.table[i]['X'].setvalue(self.table[i]['X'][tar] ^ self.table[i]['X'][con], tar)
            self.table[i]['Z'].setvalue(self.table[i]['Z'][tar] ^ self.table[i]['Z'][con], con)

    def s(self, q):
        """Phase-gate"""
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

    def sdg(self, q):
        """Inverse phase Gate, sdg=s.z"""
        self.s(q)
        self.z(q)

    def v(self, q):
        """V Gate, V=HSHS"""
        self.h(q)
        self.s(q)
        self.h(q)
        self.s(q)

    def w(self, q):
        """W Gate, the inverse of V-gate"""
        self.sdg(q)
        self.h(q)
        self.sdg(q)
        self.h(q)

# ----------------------------------------------------------------------------------------
# Compose a Clifford circuit from basis gates
# ----------------------------------------------------------------------------------------
    def compose_circuit(self, cliff):
        """ compsose circuit """
        circ = cliff.get_circuit()
        for op in circ:
            split = op.split()
            q1 = int(split[1])
            if split[0] == 'v':
                self.v(q1)
            elif split[0] == 'w':
                self.w(q1)
            elif split[0] == 'x':
                self.x(q1)
            elif split[0] == 'y':
                self.y(q1)
            elif split[0] == 'z':
                self.z(q1)
            elif split[0] == 'cx':
                self.cx(q1, int(split[2]))
            elif split[0] == 'h':
                self.h(q1)
            elif split[0] == 's':
                self.s(q1)
            elif split[0] == 'sdg':
                self.sdg(q1)
            else:
                print("error: unknown gate type: ", op)
        self.circuit_append(circ)
