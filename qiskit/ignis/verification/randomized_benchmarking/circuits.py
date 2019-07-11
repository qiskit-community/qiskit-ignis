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

from . import Clifford
from . import clifford_utils as clutils


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


def load_tables(max_nrb=2):
    """
    Returns the needed Clifford tables

    Args:
        max_nrb: maximal number of qubits for the largest required table

    Returns:
        A table of Clifford objects
    """

    clifford_tables = [[] for i in range(max_nrb)]
    for rb_num in range(max_nrb):
        # load the clifford tables, but only if we're using that particular rb
        # number
        if rb_num == 0:
            # 1Q Cliffords, load table programmatically
            clifford_tables[0] = clutils.clifford1_gates_table()
        elif rb_num == 1:
            # 2Q Cliffords
            # Try to load the table in from file. If it doesn't exist then make
            # the file
            try:
                clifford_tables[rb_num] = clutils.load_clifford_table(
                    picklefile='cliffords%d.pickle' % (rb_num + 1))
            except OSError:
                # table doesn't exist, so save it
                # this will save time next run
                print('Making the n=%d Clifford Table' % (rb_num + 1))
                clutils.pickle_clifford_table(
                    picklefile='cliffords%d.pickle' % (rb_num + 1),
                    num_qubits=(rb_num+1))
                clifford_tables[rb_num] = clutils.load_clifford_table(
                    picklefile='cliffords%d.pickle' % (rb_num + 1))
        else:
            raise ValueError("The number of qubits should be only 1 or 2")

    return clifford_tables


def randomized_benchmarking_seq(nseeds=1, length_vector=None,
                                rb_pattern=None,
                                length_multiplier=1, seed_offset=0,
                                align_cliffs=False,
                                interleaved_gates=None,
                                is_purity=False):
    """
    Get a generic randomized benchmarking sequence

    Args:
        nseeds: number of seeds
        length_vector: 'm' length vector of Clifford lengths. Must be in
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
        after each set of cliffords (note: aligns after each increment
        of cliffords including the length multiplier so if the multiplier
        is [1,3] it will barrier after 1 clifford for the first pattern
        and 3 for the second)
        interleaved_gates: A list of gates of Clifford elements that
        will be interleaved (for interleaved randomized benchmarking)
        The length of the list would equal the length of the rb_pattern.
        is_purity: True only for purity rb (default is False)

    Returns:
        circuits: list of lists of circuits for the rb sequences (separate list
        for each seed)
        xdata: the Clifford lengths (with multiplier if applicable)
        rb_opts_dict: option dictionary back out with default options appended
        circuits_interleaved: list of lists of circuits for the interleaved
        rb sequences (separate list for each seed)
        circuits_purity: list of lists of lists of circuits for purity rb
        (separate list for each seed and each of the 3^n circuits)
        npurity: the number of purity rb circuits (per seed)
        which equals to 3^n, where n is the dimension
    """
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
    clifford_tables = load_tables(np.max(pattern_sizes))

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

        # make Clifford sequences for each of the separate sequences in
        # rb_pattern
        Cliffs = []
        for rb_q_num in pattern_sizes:
            Cliffs.append(Clifford(rb_q_num))
        # Clifford sequences for interleaved rb sequences
        Cliffs_interleaved = []
        for rb_q_num in pattern_sizes:
            Cliffs_interleaved.append(Clifford(rb_q_num))

        # go through and add Cliffords
        length_index = 0
        for cliff_index in range(length_vector[-1]):
            for (rb_pattern_index, rb_q_num) in enumerate(pattern_sizes):
                for _ in range(length_multiplier[rb_pattern_index]):

                    new_cliff_gatelist = clutils.random_clifford_gates(
                        rb_q_num)
                    Cliffs[rb_pattern_index] = clutils.compose_gates(
                        Cliffs[rb_pattern_index], new_cliff_gatelist)
                    general_circ += replace_q_indices(
                        clutils.get_quantum_circuit(new_cliff_gatelist,
                                                    rb_q_num),
                        rb_pattern[rb_pattern_index], qr)

                    # add a barrier
                    general_circ.barrier(
                        *[qr[x] for x in rb_pattern[rb_pattern_index]])

                    # interleaved rb sequences
                    if interleaved_gates is not None:
                        Cliffs_interleaved[rb_pattern_index] = \
                            clutils.compose_gates(
                                Cliffs_interleaved[rb_pattern_index],
                                new_cliff_gatelist)
                        Cliffs_interleaved[rb_pattern_index] = \
                            clutils.compose_gates(
                                Cliffs_interleaved[rb_pattern_index],
                                interleaved_gates[rb_pattern_index])
                        interleaved_circ += replace_q_indices(
                            clutils.get_quantum_circuit(new_cliff_gatelist,
                                                        rb_q_num),
                            rb_pattern[rb_pattern_index], qr)
                        # add a barrier - interleaved rb
                        interleaved_circ.barrier(
                            *[qr[x] for x in rb_pattern[rb_pattern_index]])
                        interleaved_circ += replace_q_indices(
                            clutils.get_quantum_circuit(interleaved_gates
                                                        [rb_pattern_index],
                                                        rb_q_num),
                            rb_pattern[rb_pattern_index], qr)
                        # add a barrier - interleaved rb
                        interleaved_circ.barrier(
                            *[qr[x] for x in rb_pattern[rb_pattern_index]])

            if align_cliffs:
                # if align cliffords at a barrier across all patterns
                general_circ.barrier(
                    *[qr[x] for x in qlist_flat])
                # align for interleaved rb
                if interleaved_gates is not None:
                    interleaved_circ.barrier(
                        *[qr[x] for x in qlist_flat])

            # if the number of cliffords matches one of the sequence lengths
            # then calculate the inverse and produce the circuit
            if (cliff_index+1) == length_vector[length_index]:
                # circ for rb:
                circ = qiskit.QuantumCircuit(qr, cr)
                circ += general_circ
                # circ_interleaved for interleaved rb:
                circ_interleaved = qiskit.QuantumCircuit(qr, cr)
                circ_interleaved += interleaved_circ

                for (rb_pattern_index, rb_q_num) in enumerate(pattern_sizes):
                    inv_key = Cliffs[rb_pattern_index].index()
                    inv_circuit = clutils.find_inverse_clifford_gates(
                        rb_q_num,
                        clifford_tables[rb_q_num-1][inv_key])
                    circ += replace_q_indices(
                        clutils.get_quantum_circuit(inv_circuit, rb_q_num),
                        rb_pattern[rb_pattern_index], qr)
                    # calculate the inverse and produce the circuit
                    # for interleaved rb
                    if interleaved_gates is not None:
                        inv_key = Cliffs_interleaved[rb_pattern_index].index()
                        inv_circuit = clutils.find_inverse_clifford_gates(
                            rb_q_num,
                            clifford_tables[rb_q_num - 1][inv_key])
                        circ_interleaved += replace_q_indices(
                            clutils.get_quantum_circuit(inv_circuit, rb_q_num),
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
