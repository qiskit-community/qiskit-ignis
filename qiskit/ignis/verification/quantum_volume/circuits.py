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
from qiskit.compiler.transpile import transpile
from qiskit.test.mock import FakeMelbourne
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroll3qOrMore, NoiseAdaptiveLayout


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


def qv_circuits_opt(qubit_lists=None, ntrials=1, max_qubits=2,
                backend=None, qr=None, cr=None, seed=None):
    """
    Return a list of quantum volume circuits transpiled on
     for specific backend. The circuit will be square (depth=width)
     as long as the user's specified layout doesn't require extra
     qubits to perform any swaps needed. If  no user-specified layout is
     provided, the function will try to find the best suitable qubtis.

    Args:
        qubit_lists (list): list of list of qubits to apply qv circuits to. Assume
            the list is ordered in increasing number of qubits
        ntrials (int): number of random iterations
        qr (QuantumRegister): quantum register to act on (if None one is created)
        cr (ClassicalRegister): classical register to measure to (if None one is created)
        seed (int): An optional RNG seed to use for the generated circuit
        max_qubits (int): Will be used if the user doesn't specify their desired layout. Minimum value is 2
        backend (IBMQBackend): A backend for the quantum volume circuits to be transpiled
        for.

    Returns:
        tuple: A tuple of the type (``circuits``, ``circuits_nomeas``) wheere:
            ``circuits`` is a list of lists of circuits for the qv sequences
            (separate list for each trial) and `` circuitss_nomeas`` is the
            same circuits but with no measurements for the ideal simulation
    """

    if seed:
        rng = np.random.default_rng(seed)
    else:
        _seed = None

    
    qv_circs = [[] for e in range(ntrials)]
    circuits_nomeas = [[] for e in range(ntrials)]

    if qubit_lists == None:
        depth_list = [i for i in range(2,max_qubits+1)]

    else:
        depth_list = [len(qubit_list) for qubit_list in qubit_lists]


    qv_circ = [[] for e in range(ntrials)]

    for trial in range(ntrials):
        for depthidx, depth in enumerate(depth_list):
            if seed:
                _seed = rng.integers(1000)

            qv_circ = QuantumVolume(depth, depth, seed=_seed)

            qc2 = copy.deepcopy(qv_circ)

            qv_circ.measure_active()
            qv_circ.name = 'qv_depth_%d_trial_%d' % (depth, trial)
            qc2.name = qv_circ.name
            circuits_nomeas[trial].append(qc2)
            qv_circs[trial].append(qv_circ)

    if qubit_lists == None:
        # find layouts for a range of qubits from 2 up to max_qubits
        best_layouts_list = [[] for tmp in range(max_qubits-1)]
        for n_qubits in range(2, max_qubits+1, 1):
            best_layouts_list[n_qubits-2] = get_layout(qv_circs, n_qubits, ntrials, backend,
                                                          transpile_trials=None, n_desired_layouts=1)
        # [[n_desired_layouts * Layouts], [n_desired_layouts * Layouts], [], [], [], []]
        qubit_lists = []

        for good_layout in best_layouts_list:
            qubit_lists.append(good_layout[0])

    else:
        warnings.warn("The choice of the qubit list may result in extra swaps and extra qubits"
                      "when running on the actual machine. Please check the backend's"
                      "coupling map to choose the right set of qubits.")

    
    depth_list = [len(qubit_list) for qubit_list in qubit_lists]

    circuits = [[] for e in range(ntrials)]
    for trial in range(ntrials):
        for depthidx, depth in enumerate(depth_list):

            qc = transpile(qv_circs[trial][depthidx], backend, initial_layout=qubit_lists[depthidx])
            qc.name = 'qv_depth_%d_trial_%d' % (depth, trial)

            circuits[trial].append(qc)

    return circuits, circuits_nomeas


def get_layout(qv_circs, n_qubits, n_trials, backend, transpile_trials=None, n_desired_layouts=1):
    """
    Multiple runs of transpiler level 3
    Counting occurrences of different layouts
    Return a list of layouts, ordered by occurrence/probability for good QV

    qv_circs(int): qv circuits
    n_qubits(int): number qubits for which to find a layout
    backend(BaseBackend): the backend onto which the QV measurement is done
    n_trials(int): total number of trials for QV measurement
    transpile_trials(int): number of transpiler trials to search for a layout, less or equal to n_trials
    """

    n_qubit_idx = 0
    if not transpile_trials:
        transpile_trials = n_trials

    for idx, qv in enumerate(qv_circs[0]):
        if qv.n_qubits >= n_qubits:
            n_qubit_idx = idx
            break

    layouts_list = []
    layouts_counts = []
    for trial in range(transpile_trials):
        pm = PassManager()
        pm.append(Unroll3qOrMore())
        pm.append(NoiseAdaptiveLayout(backend.properties()))
        pm.run(qv_circs[trial][n_qubit_idx])
        layout = list(pm.property_set['layout'].get_physical_bits().keys())

        if layout in layouts_list:
            idx = layouts_list.index(layout)
            layouts_counts[idx] += 1
        else:
            layouts_list.append(layout)
            layouts_counts.append(1)

    # Sort the layout list based on max occurrences
    sorted_layouts = sorted(layouts_list, key=lambda x: layouts_counts[layouts_list.index(x)], reverse=True)

    return sorted_layouts[:n_desired_layouts]


    def get_layout_low_cost(num_qubits, backend, layout_method='noise_adaptive'):     
    """     
    Creates a mock circuit similar to the SU(4) circuit
    to be used with the transpiler allocation passes.

    returns the initial_layour
    """     
    if (num_qubits > backend.configuration().n_qubits): 
        return None     
    
    cct = QuantumCircuit(num_qubits)     
    comb = combinations(range(num_qubits), 2)    

    for i in comb:
        for j in range(3):
            cct.cx(i[0],i[1]) 
            cct.u3(np.pi/2, np.pi/2,np.pi/2, i[0])
            cct.u3(np.pi/2, np.pi/2,np.pi/2, i[1])         
 
    cct_meas = copy.deepcopy(cct)
    cct_meas.measure_active()


    pm = PassManager()
    pm.append(Unroll3qOrMore())
    
    if layout_method == 'noise_adaptive':
        pm.append(NoiseAdaptiveLayout(backend.properties()))
    elif layout_method == 'dense':
        pm.append(DenseLayout(CouplingMap(backend.configuration().coupling_map), backend.properties()))
    elif layout_method == 'sabre':
        pm.append(SabreLayout(CouplingMap(backend.configuration().coupling_map)))

    
    pm.run(cct_meas)
    layout = list(pm.property_set['layout'].get_physical_bits().keys())

    return layout

if __name__ == "__main__":
    qubit_lists = [[0,1,3], [0,1,3,5], [0,1,3,5,7], [0,1,3,5,7,10]]
    # ntrials: Number of random circuits to create for each subset
    ntrials = 50

    fake_backend = FakeMelbourne()

    qv_circs, qv_circs_nomeas = qv_circuits_opt(ntrials=ntrials, max_qubits=4,
                    backend=fake_backend)
    print("qv_circ[0]: ", qv_circs[0])
    print(qv_circs[0][0])
