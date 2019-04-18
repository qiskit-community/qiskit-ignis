# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Generates quantum volume circuits
"""

import copy
import numpy as np
import qiskit
from qiskit.quantum_info.random import random_unitary


def qv_circuits(qubit_list=None, depth_list=None, ntrials=1,
                qr=None, cr=None):
    """
    Return a list of quantum volume circuits

    For the set of qubits listed (number of qubits here is the width)
    circuits will be returned for the each of the depths in depth list

    Args:
        qubit_list: list of qubits to apply qv circuits to
        depth_list: list of depths (typically [0,1,2,..,d])
        ntrials: number of random iterations
        qr: quantum register to act on (if None one is created)
        cr: classical register to measure to (if None one is created)

    Returns:
        qv_circs: list of lists of circuits for the qv sequences
        (separate list for each trial)
        qv_circs_nomeas: same as above with no measurements for the ideal
        simulation
    """

    circuits = [[] for e in range(ntrials)]
    circuits_nomeas = [[] for e in range(ntrials)]

    n_q_max = np.max(qubit_list)
    width = len(qubit_list)

    # go through for each trial
    for trial in range(ntrials):
        qr = qiskit.QuantumRegister(n_q_max+1, 'qr')
        cr = qiskit.ClassicalRegister(width, 'cr')

        #go through for each depth in the depth list
        for depth in depth_list:

            qc = qiskit.QuantumCircuit(qr, cr)

            qc.name = 'qv_depth_%d_trial_%d'%(depth,trial)

            #build the circuit
            for j in range(depth):
                # Generate uniformly random permutation Pj of [0...n-1]
                perm = np.random.permutation(width)
                # For each pair p in Pj, generate Haar random SU(4)
                for k in range(int(np.floor(width/2))):
                    U = random_unitary(4)
                    pair = int(perm[2*k]), int(perm[2*k+1])
                    qc.append(U, [qr[qubit_list[pair[0]]],
                                  qr[qubit_list[pair[1]]]])

            circuits_nomeas[trial].append(copy.deepcopy(qc))

            #add measurement
            for qind, qubit in enumerate(qubit_list):
                qc.measure(qr[qubit], cr[qind])

            circuits[trial].append(qc)

    return circuits, circuits_nomeas
