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

# TODO(mtreinish): Remove these disables when implementation is finished
# pylint: disable=unused-argument,unnecessary-pass

"""
Generates randomized benchmarking sequences
"""

import copy
import numpy as np
import qiskit

from .Clifford import Clifford
from .clifford_utils import CliffordUtils as clutils


def handle_length_multiplier(length_multiplier, len_pattern,
                             is_purity=False):
    """
    Check validity of length_multiplier.
    In addition, transform it into a vector if it is a constant.
    In case of purity rb the length multiplier should be None.

    Args:
        length_multiplier: length of the multiplier
        len_pattern: length of the RB pattern
        is_purity: True only for purity rb (default is False)

    Returns:
        length_multiplier
    """

    if hasattr(length_multiplier, "__len__"):
        if is_purity:
            raise ValueError(
                "In case of Purity RB the length multiplier should be None")
        if len(length_multiplier) != len_pattern:
            raise ValueError(
                "Length mulitiplier must be the same length as the pattern")
        length_multiplier = np.array(length_multiplier)
        if length_multiplier.dtype != 'int' or (length_multiplier < 1).any():
            raise ValueError("Invalid length multiplier")
    else:
        length_multiplier = np.ones(len_pattern, dtype='int')*length_multiplier

    return length_multiplier


def check_pattern(pattern, is_purity=False):
    """
    Verifies that the input pattern is valid
    i.e., that each qubit appears at most once

    In case of purity rb, checkes that all
    simultaneous sequences have the same dimension
    (e.g. only 1-qubit squences, or only 2-qubits
    sequences etc.)

    Args:
        pattern: RB pattern
        n_qubits: number of qubits
        is_purity: True only for purity rb (default is False)

    Raises:
        ValueError: if the pattern is not valid

    Return:
        qlist: flat list of all the qubits in the pattern
        maxqubit: the maximum qubit number
        maxdim: the maximal dimension (maximal number of qubits
        in all sequences)
    """

    pattern_flat = []
    pattern_dim = []
    for pat in pattern:
        pattern_flat.extend(pat)
        pattern_dim.append(len(pat))

    _, uni_counts = np.unique(np.array(pattern_flat), return_counts=True)
    if (uni_counts > 1).any():
        raise ValueError("Invalid pattern. Duplicate qubit index.")

    dim_distinct = np.unique(pattern_dim)
    if is_purity:
        if len(dim_distinct) > 1:
            raise ValueError("Invalid pattern for purity RB. \
            All simultaneous sequences should have the \
            same dimension.")

    return pattern_flat, np.max(pattern_flat).item(), np.max(pattern_dim)


def calc_xdata(length_vector, length_multiplier):
    """
    Calculate the set of sequences lengths

    Args:
        length_vector: vector length
        length_multiplier: length of the multiplier of the vector length

    Returns:
        An array of sequences lengths
    """

    xdata = []
    for mult in length_multiplier:
        xdata.append(np.array(length_vector)*mult)

    return np.array(xdata)


def randomized_benchmarking_seq(nseeds=1, length_vector=None,
                                rb_pattern=None,
                                length_multiplier=1, seed_offset=0,
                                align_cliffs=False,
                                interleaved_gates=None,
                                is_purity=False,
                                group_gates=None):
    """
    Get a generic randomized benchmarking sequence

    Args:
        nseeds: number of seeds
        length_vector: 'm' length vector of sequence lengths. Must be in
        ascending order. RB sequences of increasing length grow on top of the
        previous sequences.
        rb_pattern: A list of the form [[i,j],[k],...] which will make
        simultaneous RB sequences where
        Qi,Qj are a 2Q RB sequence and Qk is a 1Q sequence, etc.
        E.g. [[0,3],[2],[1]] would create RB sequences that are 2Q for Q0/Q3,
        1Q for Q1+Q2
        The number of qubits is the sum of the entries.
        For 'regular' RB the qubit_pattern is just [[0]],[[0,1]].
        length_multiplier: if this is an array it scales each rb_sequence by
        the multiplier
        seed_offset: What to start the seeds at (e.g. if we
        want to add more seeds later)
        align_cliffs: If true adds a barrier across all qubits in rb_pattern
        after each set of elements, not necessarily Cliffords
        (note: aligns after each increment of elements including the
        length multiplier so if the multiplier is [1,3] it will barrier
        after 1 element for the first pattern and 3 for the second).
        interleaved_gates: A list of gates of elements that
        will be interleaved (for interleaved randomized benchmarking)
        The length of the list would equal the length of the rb_pattern.
        is_purity: True only for purity rb (default is False)
        group_gates: On which group (or gate set) we perform RB
        (default is the Clifford group)

    Returns:
        circuits: list of lists of circuits for the rb sequences
        (separate list for each seed)
        xdata: the sequences lengths (with multiplier if applicable)
        circuits_interleaved (only if interleaved_gates is not None):
        list of lists of circuits for the interleaved rb sequences
        (separate list for each seed)
        circuits_purity (only if is_purity=True):
        list of lists of lists of circuits for purity rb
        (separate list for each seed and each of the 3^n circuits)
        npurity (only if is_purity=True):
        the number of purity rb circuits (per seed)
        which equals to 3^n, where n is the dimension
    """
    # Set modules (default is Clifford)
    if group_gates is None or 'Clifford' or 'clifford':
        Gutils = clutils()
        Ggroup = Clifford
    else:
        raise ValueError("Unknown group or set of gates.")

    if rb_pattern is None:
        rb_pattern = [[0]]
    if length_vector is None:
        length_vector = [1, 10, 20]

    qlist_flat, n_q_max, max_dim = check_pattern(rb_pattern, is_purity)
    length_multiplier = handle_length_multiplier(length_multiplier,
                                                 len(rb_pattern),
                                                 is_purity)
    # number of purity rb circuits per seed
    npurity = 3**max_dim

    xdata = calc_xdata(length_vector, length_multiplier)

    pattern_sizes = [len(pat) for pat in rb_pattern]
    max_nrb = np.max(pattern_sizes)

    # load group tables
    group_tables = [[] for _ in range(max_nrb)]
    for rb_num in range(max_nrb):
        group_tables[rb_num] = Gutils.load_tables(rb_num+1)

    # initialization: rb sequences
    circuits = [[] for e in range(nseeds)]
    # initialization: interleaved rb sequences
    circuits_interleaved = [[] for e in range(nseeds)]
    # initialization: purity rb sequences
    circuits_purity = [[[] for d in range(npurity)]
                       for e in range(nseeds)]

    # go through for each seed
    for seed in range(nseeds):
        qr = qiskit.QuantumRegister(n_q_max+1, 'qr')
        cr = qiskit.ClassicalRegister(len(qlist_flat), 'cr')
        general_circ = qiskit.QuantumCircuit(qr, cr)
        interleaved_circ = qiskit.QuantumCircuit(qr, cr)

        # make sequences for each of the separate sequences in
        # rb_pattern
        Elmnts = []
        for rb_q_num in pattern_sizes:
            Elmnts.append(Ggroup(rb_q_num))
        # Sequences for interleaved rb sequences
            Elmnts_interleaved = []
        for rb_q_num in pattern_sizes:
            Elmnts_interleaved.append(Ggroup(rb_q_num))

        # go through and add elements to RB sequences
        length_index = 0
        for elmnts_index in range(length_vector[-1]):
            for (rb_pattern_index, rb_q_num) in enumerate(pattern_sizes):
                for _ in range(length_multiplier[rb_pattern_index]):

                    new_elmnt_gatelist = Gutils.random_gates(
                        rb_q_num)
                    Elmnts[rb_pattern_index] = Gutils.compose_gates(
                        Elmnts[rb_pattern_index], new_elmnt_gatelist)
                    general_circ += replace_q_indices(
                        get_quantum_circuit(new_elmnt_gatelist,
                                            rb_q_num),
                        rb_pattern[rb_pattern_index], qr)

                    # add a barrier
                    general_circ.barrier(
                        *[qr[x] for x in rb_pattern[rb_pattern_index]])

                    # interleaved rb sequences
                    if interleaved_gates is not None:
                        Elmnts_interleaved[rb_pattern_index] = \
                            Gutils.compose_gates(
                                Elmnts_interleaved[rb_pattern_index],
                                new_elmnt_gatelist)
                        Elmnts_interleaved[rb_pattern_index] = \
                            Gutils.compose_gates(
                                Elmnts_interleaved[rb_pattern_index],
                                interleaved_gates[rb_pattern_index])
                        interleaved_circ += replace_q_indices(
                            get_quantum_circuit(new_elmnt_gatelist,
                                                rb_q_num),
                            rb_pattern[rb_pattern_index], qr)
                        # add a barrier - interleaved rb
                        interleaved_circ.barrier(
                            *[qr[x] for x in rb_pattern[rb_pattern_index]])
                        interleaved_circ += replace_q_indices(
                            get_quantum_circuit(interleaved_gates
                                                [rb_pattern_index],
                                                rb_q_num),
                            rb_pattern[rb_pattern_index], qr)
                        # add a barrier - interleaved rb
                        interleaved_circ.barrier(
                            *[qr[x] for x in rb_pattern[rb_pattern_index]])

            if align_cliffs:
                # if align at a barrier across all patterns
                general_circ.barrier(
                    *[qr[x] for x in qlist_flat])
                # align for interleaved rb
                if interleaved_gates is not None:
                    interleaved_circ.barrier(
                        *[qr[x] for x in qlist_flat])

            # if the number of elements matches one of the sequence lengths
            # then calculate the inverse and produce the circuit
            if (elmnts_index+1) == length_vector[length_index]:
                # circ for rb:
                circ = qiskit.QuantumCircuit(qr, cr)
                circ += general_circ
                # circ_interleaved for interleaved rb:
                circ_interleaved = qiskit.QuantumCircuit(qr, cr)
                circ_interleaved += interleaved_circ

                for (rb_pattern_index, rb_q_num) in enumerate(pattern_sizes):
                    inv_key = Gutils.find_key(Elmnts[rb_pattern_index])
                    inv_circuit = Gutils.find_inverse_gates(
                        rb_q_num,
                        group_tables[rb_q_num-1][inv_key])
                    circ += replace_q_indices(
                        get_quantum_circuit(inv_circuit, rb_q_num),
                        rb_pattern[rb_pattern_index], qr)
                    # calculate the inverse and produce the circuit
                    # for interleaved rb
                    if interleaved_gates is not None:
                        inv_key = Gutils.find_key(Elmnts_interleaved
                                                  [rb_pattern_index])
                        inv_circuit = Gutils.find_inverse_gates(
                            rb_q_num,
                            group_tables[rb_q_num - 1][inv_key])
                        circ_interleaved += replace_q_indices(
                            get_quantum_circuit(inv_circuit, rb_q_num),
                            rb_pattern[rb_pattern_index], qr)

                # Circuits for purity rb
                if is_purity:
                    circ_purity = [[] for d in range(npurity)]
                    for d in range(npurity):
                        circ_purity[d] = qiskit.QuantumCircuit(qr, cr)
                        circ_purity[d] += circ
                        circ_purity[d].name = 'rb_purity_'
                        ind_d = d
                        purity_qubit_num = 0
                        while True:
                            # Per each qubit:
                            # do nothing or rx(pi/2) or ry(pi/2)
                            purity_qubit_rot = np.mod(ind_d, 3)
                            ind_d = np.floor_divide(ind_d, 3)
                            if purity_qubit_rot == 0:  # do nothing
                                circ_purity[d].name += 'Z'
                            if purity_qubit_rot == 1:  # add rx(pi/2)
                                for pat in rb_pattern:
                                    circ_purity[d].rx(np.pi / 2,
                                                      qr[pat[
                                                          purity_qubit_num]])
                                circ_purity[d].name += 'X'
                            if purity_qubit_rot == 2:  # add ry(pi/2)
                                for pat in rb_pattern:
                                    circ_purity[d].ry(np.pi / 2,
                                                      qr[pat[
                                                          purity_qubit_num]])
                                circ_purity[d].name += 'Y'
                            purity_qubit_num = purity_qubit_num + 1
                            if ind_d == 0:
                                break
                        # padding the circuit name with Z's so that
                        # all circuits will have names of the same length
                        for _ in range(max_dim - purity_qubit_num):
                            circ_purity[d].name += 'Z'
                        # add measurement for purity rb
                        for qind, qb in enumerate(qlist_flat):
                            circ_purity[d].measure(qr[qb], cr[qind])
                        circ_purity[d].name += '_length_%d_seed_%d' \
                                               % (length_index,
                                                  seed + seed_offset)

                # add measurement for standard rb
                # qubits measure to the c registers as
                # they appear in the pattern
                for qind, qb in enumerate(qlist_flat):
                    circ.measure(qr[qb], cr[qind])
                    # add measurement for interleaved rb
                    circ_interleaved.measure(qr[qb], cr[qind])

                circ.name = 'rb_length_%d_seed_%d' % (length_index,
                                                      seed + seed_offset)
                circ_interleaved.name = 'rb_interleaved_length_%d_seed_%d' \
                                        % (length_index, seed + seed_offset)

                circuits[seed].append(circ)
                circuits_interleaved[seed].append(circ_interleaved)
                if is_purity:
                    for d in range(npurity):
                        circuits_purity[seed][d].append(circ_purity[d])
                length_index += 1

    # output of interleaved rb
    if interleaved_gates is not None:
        return circuits, xdata, circuits_interleaved
    # output of purity rb
    if is_purity:
        return circuits_purity, xdata, npurity
    # output of standard (simultaneous) rb
    return circuits, xdata


def replace_q_indices(circuit, q_nums, qr):
    """
    Take a circuit that is ordered from 0,1,2 qubits and replace 0 with the
    qubit label in the first index of q_nums, 1 with the second index...

    Args:
        circuit: circuit to operate on
        q_nums: list of qubit indices

    Returns:
        updated circuit
    """

    new_circuit = qiskit.QuantumCircuit(qr)
    for instr, qargs, cargs in circuit.data:
        new_qargs = [
            qr[q_nums[x]] for x in [arg.index for arg in qargs]]
        new_op = copy.deepcopy((instr, new_qargs, cargs))
        new_circuit.data.append(new_op)

    return new_circuit


def get_quantum_circuit(gatelist, num_qubits):
    """
    Returns the circuit in the form of a QuantumCircuit object.

    Args:
        num_qubits: the number of qubits (dimension).
        gatelist: a list of gates.

    Returns:
        A QuantumCircuit object.
    """
    qr = qiskit.QuantumRegister(num_qubits)
    qc = qiskit.QuantumCircuit(qr)

    for op in gatelist:
        split = op.split()
        op_names = [split[0]]

        # temporary correcting the ops name since QuantumCircuit has no
        # attributes 'v' or 'w' yet:
        if op_names == ['v']:
            op_names = ['sdg', 'h']
        elif op_names == ['w']:
            op_names = ['h', 's']

        qubits = [qr[int(x)] for x in split[1:]]
        for sub_op in op_names:
            operation = eval('qiskit.QuantumCircuit.' + sub_op)
            operation(qc, *qubits)

    return qc
