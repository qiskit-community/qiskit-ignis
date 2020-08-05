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

import copy
import itertools
import warnings

import numpy as np

from qiskit.circuit.library import QuantumVolume
from qiskit.circuit.quantumcircuit import QuantumCircuit


def qv_circuits(qubit_lists, ntrials=1,
                qr=None, cr=None, seed=None):
    """
    Return a list of square quantum volume circuits (depth=width)

    The qubit_lists is specified as a list of qubit lists. For each
    set of qubits, circuits the depth as the number of qubits in the list
    are generated

    Args:
        qubit_lists (list): list of list of qubits to apply qv circuits to. Assume
            the list is ordered in increasing number of qubits
        ntrials (int): number of random iterations
        qr (QuantumRegister): quantum register to act on (if None one is created)
        cr (ClassicalRegister): classical register to measure to (if None one is created)
        seed (int): An optional RNG seed to use for the generated circuit

    Returns:
        tuple: A tuple of the type (``circuits``, ``circuits_nomeas``) wheere:
            ``circuits`` is a list of lists of circuits for the qv sequences
            (separate list for each trial) and `` circuitss_nomeas`` is the
            same circuits but with no measurements for the ideal simulation
    """
    if qr is not None:
        warnings.warn("Passing in a custom quantum register is deprecated and "
                      "will be removed in a future release. This argument "
                      "never had any effect.",
                      DeprecationWarning)

    if cr is not None:
        warnings.warn("Passing in a custom classical register is deprecated "
                      "and will be removed in a future release. This argument "
                      "never had any effect.",
                      DeprecationWarning)
    for qubit_list in qubit_lists:
        count = itertools.count(qubit_list[0])
        for qubit in qubit_list:
            if qubit != next(count):
                warnings.warn("Using a qubit list to map a virtual circuit to "
                              "a physical layout is deprecated and will be "
                              "removed in a future release. Instead use "
                              "''qiskit.transpile' with the "
                              "'initial_layout' parameter",
                              DeprecationWarning)
    depth_list = [len(qubit_list) for qubit_list in qubit_lists]

    if seed:
        rng = np.random.default_rng(seed)
    else:
        _seed = None

    circuits = [[] for e in range(ntrials)]
    circuits_nomeas = [[] for e in range(ntrials)]

    for trial in range(ntrials):
        for depthidx, depth in enumerate(depth_list):
            n_q_max = np.max(qubit_lists[depthidx])
            if seed:
                _seed = rng.integers(1000)
            qv_circ = QuantumVolume(depth, depth, seed=_seed)
            qc2 = copy.deepcopy(qv_circ)
            # TODO: Remove this when we remove support for doing pseudo-layout
            # via qubit lists
            if n_q_max != depth:
                qc = QuantumCircuit(int(n_q_max + 1))
                qc.compose(qv_circ, qubit_lists[depthidx], inplace=True)
            else:
                qc = qv_circ
            qc.measure_active()
            qc.name = 'qv_depth_%d_trial_%d' % (depth, trial)
            qc2.name = qc.name

            circuits_nomeas[trial].append(qc2)
            circuits[trial].append(qc)

    return circuits, circuits_nomeas
