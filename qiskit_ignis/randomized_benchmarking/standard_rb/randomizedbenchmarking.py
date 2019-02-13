# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Generates randomized benchmarking sequences
"""

import copy
import numpy as np
import qiskit_ignis.randomized_benchmarking.standard_rb.Clifford as clf
import qiskit_ignis.randomized_benchmarking.standard_rb.clifford_utils as clutils
import qiskit

def handle_length_multiplier(length_multiplier, len_pattern):
    """
    Check validity of length_multiplier.
    In addition, transform it into a vector if it is a constant.

    Args:
        length_multiplier: length of the multiplier
        len_pattern: length of the RB pattern

    Returns:
        length_multiplier
    """

    if hasattr(length_multiplier, "__len__"):
        if len(length_multiplier) != len_pattern:
            raise ValueError("Length mulitiplier must be the same length as the pattern")
        length_multiplier = np.array(length_multiplier)
        if length_multiplier.dtype != 'int' or (length_multiplier < 1).any():
            raise ValueError("Invalid length multiplier")
    else:
        length_multiplier = np.ones(len_pattern, dtype='int')*length_multiplier

    return length_multiplier


def check_pattern(pattern, n_qubits):
    """
    Verifies that the input pattern is valid, i.e., that each qubit appears at most once

    Args:
        pattern: RB pattern
        n_qubits: number of qubits

    Raises:
        ValueError: if the pattern is not valid
    """

    pattern_flat = []
    for pat in pattern:
        pattern_flat.extend(pat)

    if np.max(pattern_flat) >= n_qubits:
        print("Invalid pattern. Qubit index in the pattern exceeds the number of qubits.")

    _, uni_counts = np.unique(np.array(pattern_flat), return_counts=True)
    if (uni_counts > 1).any():
        raise ValueError("Invalid pattern. Duplicate qubit index.")


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
        #load the clifford tables, but only if we're using that particular rb number
        if rb_num == 0:
            #1Q Cliffords, load table programmatically
            clifford_tables[0] = clutils.clifford1_gates_table()
        elif rb_num == 1:
            #2Q Cliffords
            #Try to load the table in from file. If it doesn't exist then make the file
            try:
                clifford_tables[rb_num] = clutils.load_clifford_table(
                    picklefile='cliffords%d.pickle'%(rb_num+1))
            except OSError:
                #table doesn't exist, so save it
                #this will save time next run
                print('Making the n=%d Clifford Table'%(rb_num+1))
                clutils.pickle_clifford_table(picklefile='cliffords%d.pickle'%(rb_num+1),
                                              num_qubits=(rb_num+1))
                clifford_tables[rb_num] = \
                    clutils.load_clifford_table(\
                    picklefile='cliffords%d.pickle'%(rb_num+1))
        else:
            raise ValueError("The number of qubits should be only 1 or 2")

    return clifford_tables


def randomized_benchmarking_seq(nseeds=1,length_vector=[1,10,20],\
                                n_qubits=1,rb_pattern=[[0]],length_multiplier=1):
    """
    Get a generic randomized benchmarking sequence

    Args:
        nseeds: number of seeds
        length_vector: 'm' length vector of Clifford lengths. Must be in ascending order.
        RB sequences of increasing length grow on top of the previous sequences.
        n_qubits: total number of qubits
        rb_pattern: A list of the form [[i,j],[k],...] which will make
        simultaneous RB sequences where
        Qi,Qj are a 2Q RB sequence and Qk is a 1Q sequence, etc.
        E.g. [[0,3],[2],[1]] would create RB sequences that are 2Q for Q0/Q3, 1Q for Q1+Q2
        The number of qubits is the sum of the entries.
        For 'regular' RB the qubit_pattern is just [[0]],[[0,1]].
        length_multiplier: if this is an array it scales each rb_sequence by the multiplier

    Returns:
        rb_circs: list of lists of circuits for the rb sequences (separate list for each seed)
        xdata: the Clifford lengths (with multiplier if applicable)
        rb_opts_dict: option dictionary back out with default options appended
    """

    check_pattern(rb_pattern, n_qubits)
    length_multiplier = handle_length_multiplier(length_multiplier,len(rb_pattern))

    xdata = calc_xdata(length_vector, length_multiplier)

    pattern_sizes = [len(pat) for pat in rb_pattern]
    clifford_tables = load_tables(np.max(pattern_sizes))

    circuits = [[] for e in range(nseeds)]
    #go through for each seed
    for seed in range(nseeds):

        qr = qiskit.QuantumRegister(n_qubits, 'qr')
        cr = qiskit.ClassicalRegister(n_qubits, 'cr')
        general_circ = qiskit.QuantumCircuit(qr, cr)

        #make Clifford sequences for each of the separate sequences in rb_pattern
        Cliffs = []

        for rb_q_num in pattern_sizes:
            Cliffs.append(clf.Clifford(rb_q_num))

        #go through and add Cliffords
        length_index = 0
        for cliff_index in range(length_vector[-1]):
            for (rb_pattern_index, rb_q_num) in enumerate(pattern_sizes):

                for _ in range(length_multiplier[rb_pattern_index]):

                    new_cliff_gatelist = clutils.random_clifford_gates(rb_q_num)
                    Cliffs[rb_pattern_index] = clutils.compose_gates(Cliffs[rb_pattern_index], \
                                                                     new_cliff_gatelist)
                    general_circ += replace_q_indices(clutils.get_quantum_circuit(\
                        new_cliff_gatelist, rb_q_num), rb_pattern[rb_pattern_index], qr)
                    #add a barrier
                    general_circ.barrier(*[qr[x] for x in rb_pattern[rb_pattern_index]])

            #if the number of cliffords matches one of the sequence lengths then
            #calculate the inverse and produce the circuit
            if (cliff_index+1) == length_vector[length_index]:

                circ = qiskit.QuantumCircuit(qr, cr)
                circ += general_circ

                for (rb_pattern_index, rb_q_num) in enumerate(pattern_sizes):
                    inv_key = Cliffs[rb_pattern_index].index()
                    inv_circuit = clutils.find_inverse_clifford_gates(\
                        rb_q_num, clifford_tables[rb_q_num-1][inv_key])
                    circ += replace_q_indices(clutils.get_quantum_circuit(inv_circuit, rb_q_num),
                                              rb_pattern[rb_pattern_index], qr)

                circ.measure(qr, cr)
                circ.name = 'rb_seed_' + str(seed) + '_length_' + str(length_vector[length_index])
                circuits[seed].append(circ)
                length_index += 1

    return circuits, xdata


def replace_q_indices(circuit, q_nums, qr):
    """
    Take a circuit that is ordered from 0,1,2 qubits and replace 0 with the qubit
    label in the first index of q_nums, 1 with the second index...

    Args:
        circuit: circuit to operate on
        q_nums: list of qubit indices

    Returns:
        updated circuit
    """

    new_circuit = qiskit.QuantumCircuit(qr)
    for op in circuit.data:
        original_qubits = op.qargs
        new_op = copy.deepcopy(op)
        new_op.qargs = [(qr, q_nums[x]) for x in [arg[1] for arg in original_qubits]]
        new_circuit.data.append(new_op)

    return new_circuit
