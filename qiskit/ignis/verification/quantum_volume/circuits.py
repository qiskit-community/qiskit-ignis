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
from qiskit import QiskitError
from qiskit.quantum_info.random import random_unitary


def qv_circuits(qubit_lists=None, width_list=None, ntrials=1,
                qr=None, cr=None):
    """
    Return a list of square quantum volume circuits (depth=width)

    Either a set of qubit_lists or width_list is specified.
    The qubit_lists is specified as a list of qubit lists. For each
    set of qubits, circuits the depth as the number of qubits in the list
    are generated. If width_list is specified then circuits equal
    to those widths are created (started at Q0)

    Args:
        qubit_lists: list of list of qubits to apply qv circuits to. Assume
        the list is ordered in increasing number of qubits
        width_list: list of widths (either qubit_lists or width_list is
        specified)
        ntrials: number of random iterations
        qr: quantum register to act on (if None one is created)
        cr: classical register to measure to (if None one is created)

    Returns:
        qv_circs: list of lists of circuits for the qv sequences
        (separate list for each trial)
        qv_circs_nomeas: same as above with no measurements for the ideal
        simulation
    """

    if qubit_lists is None and width_list is None:
        raise QiskitError("Must specified qubit_lists or width_list")

    if (qubit_lists is not None) and (width_list is not None):
        raise QiskitError("Only specified one of qubit_lists or width_list")

    if qubit_lists is not None:
        # popuplate the width_list from the specified
        # qubit_lists
        width_list = [len(l) for l in qubit_lists]

    else:
        # if the width list is specified populate the
        # qubit lists
        qubit_lists = [list(range(l)) for l in width_list]

    circuits = [[] for e in range(ntrials)]
    circuits_nomeas = [[] for e in range(ntrials)]

    # go through for each trial
    for trial in range(ntrials):

        # go through for each depth in the depth list
        for depthidx, depth in enumerate(width_list):

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

            circuits_nomeas[trial].append(qc2)

            # add measurement
            for qind, qubit in enumerate(qubit_lists[depthidx]):
                qc.measure(qr[qubit], cr[qind])

            circuits[trial].append(qc)

    return circuits, circuits_nomeas


def rotate_qv_circ_list(qv_circs):
    """
    rotate the list of circuits so that each sublist is a single
    qubit set vs the trials

    Args:
        qv_circs: list of lists of circuits for the qv sequences
        (separate list for each trial)

    Returns:
        qv_circs2: list of lists of circuits for the qv sequences
        (separate list for each subset)
    """

    qv_circs2 = []
    for jj, _ in enumerate(qv_circs[0]):
        qv_circs2.append([])
        for ii, _ in enumerate(qv_circs):
            qv_circs2[-1].append(qv_circs[ii][jj])

    return qv_circs2
