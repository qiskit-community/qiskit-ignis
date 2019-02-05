# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Advanced Clifford operations needed for randomized benchmarking
"""

import numpy as np
import qiskit_ignis.randomized_benchmarking.standard_rb.Clifford as clf
import qiskit

try:
    import cPickle as pickle
except ImportError:
    import pickle


# --------------------------------------------------------
# Add gates to Cliffords
# --------------------------------------------------------

def pauli_gates(cliff, q, pauli):
    """does pauli gate on qubit q"""
    if pauli == 2:
        cliff.x(q)
        cliff.circuit_append(['x '+str(q)])
    elif pauli == 3:
        cliff.y(q)
        cliff.circuit_append(['y '+str(q)])
    elif pauli == 1:
        cliff.z(q)
        cliff.circuit_append(['z '+str(q)])

def h_gates(cliff, q, h):
    """does hadamard gate or not on qubit q"""
    if h == 1:
        cliff.h(q)
        cliff.circuit_append(['h '+str(q)])

def r_gates(cliff, q, r):
    """does r R-gates on qubit q"""
    #  rotatation is HSHS = [[0,1],[1,1]] tableu
    #    takes Z->X->Y->Z
    #  two R-gates is r2
    if r == 1:
        cliff.h(q)
        cliff.s(q)
        cliff.h(q)
        cliff.s(q)
        cliff.circuit_append(['r '+str(q)])
    elif r == 2:  # s.z = s^inverse
        cliff.s(q)
        cliff.z(q)
        cliff.h(q)
        cliff.s(q)
        cliff.z(q)
        cliff.h(q)
        cliff.circuit_append(['rinv '+str(q)])

def cnot_gates(cliff, ctrl, tgt):
    """does cnot gates"""
    cliff.cnot(ctrl, tgt)
    cliff.circuit_append(['cnot ' + str(ctrl) + ' ' + str(tgt)])


# --------------------------------------------------------
# Create a 1 or 2 Qubit Clifford based on a unique index
# --------------------------------------------------------

def clifford1(idx: int):
    """Make a single qubit Clifford gate.

    Args:
        idx: the index (mod 24) of a single qubit Clifford.

    Return:
        A single qubit Clifford class object.
    """

    # Cannonical Ordering of Cliffords 0,...,23
    cannonicalorder = idx % 24
    pauli = np.mod(cannonicalorder, 4)
    rotation = np.mod(cannonicalorder // 4, 3)
    h_or_not = np.mod(cannonicalorder // 12, 2)

    cliff = clf.Clifford(1)
    cliff.set_cannonical(cannonicalorder)

    h_gates(cliff, 0, h_or_not)

    r_gates(cliff, 0, rotation)  # do the R-gates

    pauli_gates(cliff, 0, pauli)

    return cliff


def clifford2(idx: int):
    """Make a two-qubit Clifford gate.

    Args:
        idx: the index (mod 11520) of a two-qubit Clifford.

    Return:
        A two-qubit Clifford class object.
    """

    cannon = idx % 11520
    cliff = clf.Clifford(2)
    cliff.set_cannonical(cannon)

    pauli = np.mod(cannon, 16)
    symp = cannon//16

    if symp < 36:
        r0 = np.mod(symp, 3)
        r1 = np.mod(symp//3, 3)
        h0 = np.mod(symp//9, 2)
        h1 = np.mod(symp//18, 2)

        h_gates(cliff, 0, h0)
        h_gates(cliff, 1, h1)
        r_gates(cliff, 0, r0)
        r_gates(cliff, 1, r1)

    elif symp < 360:
        symp = symp-36
        r0 = np.mod(symp, 3)
        r1 = np.mod(symp//3, 3)
        r2 = np.mod(symp//9, 3)
        r3 = np.mod(symp//27, 3)
        h0 = np.mod(symp//81, 2)
        h1 = np.mod(symp//162, 2)

        h_gates(cliff, 0, h0)
        h_gates(cliff, 1, h1)
        r_gates(cliff, 0, r0)
        r_gates(cliff, 1, r1)
        cnot_gates(cliff, 0, 1)
        r_gates(cliff, 0, r2)
        r_gates(cliff, 1, r3)

    elif symp < 684:
        symp = symp-360
        r0 = np.mod(symp, 3)
        r1 = np.mod(symp//3, 3)
        r2 = np.mod(symp//9, 3)
        r3 = np.mod(symp//27, 3)
        h0 = np.mod(symp//81, 2)
        h1 = np.mod(symp//162, 2)

        h_gates(cliff, 0, h0)
        h_gates(cliff, 1, h1)
        r_gates(cliff, 0, r0)
        r_gates(cliff, 1, r1)
        cnot_gates(cliff, 0, 1)
        cnot_gates(cliff, 1, 0)
        r_gates(cliff, 0, r2)
        r_gates(cliff, 1, r3)
    else:
        symp = symp-684
        r0 = np.mod(symp, 3)
        r1 = np.mod(symp//3, 3)
        h0 = np.mod(symp//9, 2)
        h1 = np.mod(symp//18, 2)

        h_gates(cliff, 0, h0)
        h_gates(cliff, 1, h1)

        cnot_gates(cliff, 0, 1)
        cnot_gates(cliff, 1, 0)
        cnot_gates(cliff, 0, 1)

        r_gates(cliff, 0, r0)
        r_gates(cliff, 1, r1)

    pauli_gates(cliff, 0, np.mod(pauli, 4))
    pauli_gates(cliff, 1, pauli//4)

    return cliff

# --------------------------------------------------------
# Create a 1 or 2 Qubit Clifford tables
# --------------------------------------------------------
def clifford2_table():
    """ Generate a table of all 2 qubit cliffords """
    cliffords2 = {}
    for i in range(11520):
        cliff = clifford2(i)
        cliffords2[cliff.key()] = cliff.get_circuit()
    return cliffords2

def clifford1_table():
    """ Generate a table of all 1 qubit cliffords """
    cliffords1 = {}
    for i in range(24):
        cliff = clifford1(i)
        cliffords1[cliff.key()] = cliff.get_circuit()
    return cliffords1

def pickle_clifford_table(picklefile='cliffords2.pickle', n=2):
    """ code to create pickled versions of the 1 and 2 qubit data tables """
    if n == 1:
        cliffords = clifford1_table()
    elif n == 2:
        cliffords = clifford2_table()
    else:
        print("n>2 not supported for pickle")

    with open(picklefile, "wb") as pf:
        pickle.dump(cliffords, pf)

def load_clifford_table(picklefile='cliffords2.pickle'):
    """ #code to load any clifford table """
    with open(picklefile, "rb") as pf:
        return pickle.load(pf)


# --------------------------------------------------------
# Main function that generates a random clifford
# --------------------------------------------------------
def random_clifford(n):
    """pick a random Clifford gate

    Args:
        n: dimension of the clifford

    Returns:
        Clifford
    """

    if n == 1:
        return clifford1(np.random.randint(0, 24))
    elif n == 2:
        return clifford2(np.random.randint(0, 11520))
    else:
        print ("Error: the number of qubits should be only 1 or 2 \n")

# --------------------------------------------------------
# Main function that calculates an inverse of a clifford
# --------------------------------------------------------
def find_inverse_clifford_circuit(cliff, clifford_table=None):
    """Find the inverse of the Clifford, and a circuit to make it"""
    n = cliff.size()

    if n in (1, 2):
        invcircuit = clifford_table[cliff.key()].copy()
        # we want to run the circuit backwards
        invcircuit.reverse()
        # replace r by rinv and rinv by r
        for i, _ in enumerate(invcircuit):
            split = invcircuit[i].split()
            if split[0] == 'r':
                invcircuit[i] = 'rinv ' + split[1]
            elif split[0] == 'rinv':
                invcircuit[i] = 'r ' + split[1]

        return invcircuit

    else:
        print ("Error: the number of qubits should be only 1 or 2 \n")


# --------------------------------------------------------
# Returns the Clifford circuit in the form of a QuantumCircuit object
# --------------------------------------------------------
def get_quantum_circuit(cliff):
    """ Returns the Clifford circuit in the form of a QuantumCircuit object """

    qr = qiskit.QuantumRegister(cliff.n)
    qc = qiskit.QuantumCircuit(qr)

    circuit = cliff.get_circuit()
    for op in circuit:
        split = op.split()
        op_names = [split[0]]

        if op_names == ['cnot']:
            op_names = ['cx']
        elif op_names == ['sinv']:
            op_names = ['sdg']
        elif op_names == ['r']:
            op_names = ['h', 's', 'h', 's']
        elif op_names == ['rinv']:
            op_names = ['sdg', 'h', 'sdg', 'h']


        qubits = [qr[int(x)] for x in split[1:]]
        for sub_op in op_names:
            operation = eval('qiskit.QuantumCircuit.' + sub_op)
            operation(qc, *qubits)

    return qc