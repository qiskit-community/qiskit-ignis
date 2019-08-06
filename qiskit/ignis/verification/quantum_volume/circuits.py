# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Generates quantum volume circuits
"""

import numpy as np
import qiskit
from qiskit.quantum_info.random import random_unitary


def qv_circuits(qubit_lists=None, ntrials=1,
                qr=None, cr=None):
    """
    Return a list of square quantum volume circuits (depth=width)

    The qubit_lists is specified as a list of qubit lists. For each
    set of qubits, circuits the depth as the number of qubits in the list
    are generated

    Args:
        qubit_lists: list of list of qubits to apply qv circuits to. Assume
        the list is ordered in increasing number of qubits
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

    # get the largest qubit number out of all the lists (for setting the
    # register)

    depth_list = [len(l) for l in qubit_lists]

    # go through for each trial
    for trial in range(ntrials):

        # go through for each depth in the depth list
        for depthidx, depth in enumerate(depth_list):

            n_q_max = np.max(qubit_lists[depthidx])

            qr = qiskit.QuantumRegister(int(n_q_max+1), 'qr')
            qr2 = qiskit.QuantumRegister(int(depth), 'qr')
            cr = qiskit.ClassicalRegister(int(depth), 'cr')

            qc = qiskit.QuantumCircuit(qr, cr)
            qc2 = qiskit.QuantumCircuit(qr2, cr)

            qc.name = 'qv_depth_%d_trial_%d' % (depth, trial)
            qc2.name = qc.name

            # build the circuit
            for _ in range(depth):
                # Generate uniformly random permutation Pj of [0...n-1]
                perm = np.random.permutation(depth)
                # For each pair p in Pj, generate Haar random SU(4)
                for k in range(int(np.floor(depth/2))):
                    U = random_unitary(4)
                    pair = int(perm[2*k]), int(perm[2*k+1])
                    qc.append(U, [qr[qubit_lists[depthidx][pair[0]]],
                                  qr[qubit_lists[depthidx][pair[1]]]])
                    qc2.append(U, [qr2[pair[0]],
                                   qr2[pair[1]]])

            # append an id to all the qubits in the ideal circuits
            # to prevent a truncation error in the statevector
            # simulators
            qc2.u1(0, qr2)

            circuits_nomeas[trial].append(qc2)

            # add measurement
            for qind, qubit in enumerate(qubit_lists[depthidx]):
                qc.measure(qr[qubit], cr[qind])

            circuits[trial].append(qc)

    return circuits, circuits_nomeas
