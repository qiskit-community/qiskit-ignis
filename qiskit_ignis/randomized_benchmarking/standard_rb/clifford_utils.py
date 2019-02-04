# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Advanced Clifford operations needed for randomized benchmarking
"""

import os
import copy
import random
import subprocess as sp
import numpy as np
import verification.randomized_benchmarking.standard_rb.Clifford as clf
import verification.randomized_benchmarking.standard_rb.symplectic as symplectic
import qiskit

try:
    import cPickle as pickle
except ImportError:
    import pickle


# --------------------------------------------------------
# Add gates to cliffords
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
# Generate random clifford gates
# using the pre-generated files
# --------------------------------------------------------

def get_circuits_from_files(cliffs):
    """get the circuit that makes the symplectic part of tableu of cliffords in list cliffs"""
    nbits = cliffs[0].size()
    n2 = nbits*2
    cin = ""

    for i, _ in enumerate(cliffs):
        # this calculation of symp is similar to cliff.key() but not the same.
        #  leaves off pauli frame column and might differ to be compatible with
        # Sergey's compile code
        symp_mat = cliffs[i].matrix[:n2, :n2]#.transpose()
        symp = 0
        symp_list = symp_mat.reshape(n2*n2).tolist()
        symp_list.reverse()
        for bit in symp_list:
            symp = (symp << 1) | int(bit)
        cin = cin + str(symp) + '\n'

    #this will fail if the .dat files do not exist
    cmd = os.path.join(os.path.dirname(__file__), 'src', 'compile') + ' %d'%nbits
    with sp.Popen(cmd.split(),
                  stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE) as proc:
        cout, _ = proc.communicate(cin.encode())
    cout = cout.decode('ascii') # convert from binary string to string string
    cout = cout.split('\n')
    for i, _ in enumerate(cliffs):
        cliffs[i].circuit = []
        circ = cout[i]
        if circ:  # if string is non-empty (could be empty if cliff was the identity)
            circ = circ[:-1] # strip trailing comma
            circ = circ.split(',')
            cliffs[i].circuit_append(circ)

def rand_symplectic_part(symp, nq):
    """ generate the symplectic part of the random clifford """
    symp2 = np.zeros([2*nq, 2*nq], dtype=np.uint8)
    symp3 = np.zeros([2*nq, 2*nq], dtype=np.uint8)
    # these interchange rows and columns because the random symplectic code
    # uses a different convention
    for i in range(nq):
        symp2[i] = symp[2*i]
        symp2[i+nq] = symp[2*i+1]
    for i in range(nq):
        symp3[:, i] = symp2[:, 2*i]
        symp3[:, i+nq] = symp2[:, 2*i+1]

    # add on extra column for phases
    mat = np.zeros([2*nq, 2*nq+1], dtype=np.uint8)
    mat[0:2*nq, 0:2*nq] = symp3
    return(mat)

def rand_clifford_using_files(nq):
    """pick a random 2,3,or 4-qubit Clifford gate using the .dat files"""
    assert 1 < nq < 5
    sizes = [24, 11520, 1451520, 47377612800]

    rint = np.random.randint(0, sizes[nq-1], dtype=np.int64)
    symp = symplectic.symplectic(rint, nq)
    mat = rand_symplectic_part(symp, nq)
    cliff = clf.Clifford(nq)
    cliff.from_matrix(mat)
    get_circuits_from_files([cliff])

    for i in range(nq):
        pauli_gates(cliff, i, np.random.randint(0, 4))
    return cliff

# --------------------------------------------------------
# Generate random clifford gates "on the fly"
# using the code in symplectic.py
# --------------------------------------------------------

def random_clifford_advanced(cliff):
    """pick a random Clifford gate"""
    cliff.circuit = []
    nq = cliff.n
    # compute size of nq-qubit sympletic group
    size = 1
    for i in range(1, nq+1):
        size = size*(pow(4, i)-1)
    size = size*pow(2, nq*nq)
    rint = random.randrange(size)
    symp = symplectic.symplectic(rint, nq)
    symp = symp.transpose()  # transpose to make cannonical ordering agree
                             # with Koenig-Smolin paper used in symplectic.py
    mat = rand_symplectic_part(symp, nq)
    for i in range(2*nq):
        mat[i, 2*nq] = np.random.randint(2)  # random phases
    cliff.from_matrix(mat)
    return(cliff)

# --------------------------------------------------------
# Main function that generates a random clifford
# --------------------------------------------------------
def random_clifford(n, on_the_fly=False):
    """pick a random Clifford gate

    Args:
        n: dimension of the clifford
        on_the_fly: use the "on the fly method" (this is automatically used for n>4)

    Returns:
        Clifford
    """

    if n > 4 or on_the_fly:
        #use on the fly method
        cliff2 = clf.Clifford(n)
        cliff2 = random_clifford_advanced(cliff2)
        cliff2.decompose()
        return cliff2

    elif n == 1:
        return clifford1(np.random.randint(0, 24))
    elif n == 2:
        return clifford2(np.random.randint(0, 11520))
    elif n == 3 or n == 4:
        return rand_clifford_using_files(n)

# --------------------------------------------------------
# Invert a random clifford
# using the pre-generated files
# --------------------------------------------------------

def find_inverse_clifford_using_files(cliffs):
    """
    Find the inverse Clifford
    """
    nq = cliffs[0].size()
    assert 1 < nq < 5

    l = len(cliffs)
    cliffs2 = copy.deepcopy(cliffs)
    invs = []
    for j in range(l):  # invs is a list of fresh cliffords
        invs = invs + [clf.Clifford(nq)]
    # find the circuits for the (original) cliffords
    get_circuits_from_files(cliffs2)  # using cliffs2 to avoid stomping on original circuits in cliffs
    # we want to run the circuit backwards
    for j in range(l):
        inv = invs[j]
        cliff = cliffs2[j]
        circuit = cliff.get_circuit()
        invcircuit = clf.invert_circuit(circuit)

        # actually run it
        for i, _ in enumerate(invcircuit):
            split = invcircuit[i].split()
            q1 = int(split[1])
            if split[0] == 'r':
                inv.h(q1)
                inv.s(q1)
                inv.h(q1)
                inv.s(q1)
            elif split[0] == 'rinv':  # s.z = s^inverse
                inv.s(q1)
                inv.z(q1)
                inv.h(q1)
                inv.s(q1)
                inv.z(q1)
                inv.h(q1)
            elif split[0] == 'cnot':
                inv.cnot(q1, int(split[2]))
            elif split[0] == 'h':
                inv.h(q1)
            elif split[0] == 's':
                inv.s(q1)
            else:
                print("error: inverse circuit line is: ", invcircuit[i])

        inv.circuit_append(invcircuit)

        # now read off the phases to do the Pauli part of the inverse
        cliff.compose_circuit(inv)
        table = cliff.get_table()
        for i in range(nq):
            p0 = table[i]['phase']
            p1 = table[i+nq]['phase']
            if p0 == 1 and p1 == 1:
                inv.circuit_append(['y '+ str(i)])
                inv.y(i)
            elif p0 == 1 and p1 == 0:
                inv.circuit_append(['z '+ str(i)])
                inv.z(i)
            elif p0 == 0 and p1 == 1:
                inv.circuit_append(['x '+ str(i)])
                inv.x(i)
    return invs


# --------------------------------------------------------
# Main function that calculates an inverse of a clifford
# --------------------------------------------------------
def find_inverse_clifford_circuit(cliff, clifford_table=None, on_the_fly=False):
    """Find the inverse of the Clifford, and a circuit to make it"""
    n = cliff.size()

    if n > 4 or on_the_fly:
        #find the inverse for the "on the fly method"
        cliff_cpy = copy.deepcopy(cliff)
        cliff_cpy.decompose()
        ret = clf.Clifford(cliff_cpy.n)
        cliff_cpy.circuit = clf.invert_circuit(cliff_cpy.circuit)
        ret.compose_circuit(cliff_cpy)
        return ret.get_circuit()

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

    cliff.get_matrix()
    inv = find_inverse_clifford_using_files([cliff])
    return inv[0].get_circuit()

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