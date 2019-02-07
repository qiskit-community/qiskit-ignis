# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Generates randomized benchmarking sequences
"""

import os
import copy
import numpy as np
import qiskit_ignis.randomized_benchmarking.standard_rb.Clifford as clf
import qiskit_ignis.randomized_benchmarking.standard_rb.clifford_utils as clutils
import qiskit

def handle_length_multiplier(length_multiplier, len_pattern):
    '''
    Check validity of length_multiplier.
    In addition, transform it into a vector if it is a constant.
    '''

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
    '''
    Verifies that the input pattern is valid, i.e.,
    that each qubit appears at most once
    '''

    pattern_flat = []
    for pat in pattern:
        pattern_flat.extend(pat)

    if np.max(pattern_flat) >= n_qubits:
        print("Invalid pattern. Qubit index in the pattern exceeds the number of qubits.")

    _, uni_counts = np.unique(np.array(pattern_flat), return_counts=True)
    if (uni_counts > 1).any():
        raise ValueError("Invalid pattern. Duplicate qubit index.")


def set_defaults_if_needed(rb_opts_dict):
    '''
    Set default values to fields of rb_opts_dict
    and perform validity checks
    '''

    if rb_opts_dict is None:
        rb_opts_dict = {}

    rb_opts_dict.setdefault('nseeds', 1)
    rb_opts_dict.setdefault('length_vector', [1, 10, 20])
    rb_opts_dict.setdefault('n_qubits', 1)
    rb_opts_dict.setdefault('rb_pattern', [[0]])
    check_pattern(rb_opts_dict['rb_pattern'], rb_opts_dict['n_qubits'])
    rb_opts_dict.setdefault('length_multiplier', 1)
    rb_opts_dict['length_multiplier'] = handle_length_multiplier(
        rb_opts_dict['length_multiplier'],
        len(rb_opts_dict['rb_pattern']))

    valid_keys = ['nseeds', 'length_vector', 'n_qubits', 'rb_pattern', 'length_multiplier']
    if not set(rb_opts_dict.keys()).issubset(valid_keys):
        raise ValueError("Input dictionary contains an invalid field")


def calc_xdata(length_vector, length_multiplier):
    '''
    Calculate the set of sequences lengths
    '''

    xdata = []
    for mult in length_multiplier:
        xdata.append(np.array(length_vector)*mult)

    return np.array(xdata)


def load_tables(max_nrb):
    '''
    Returns the needed Clifford tables
    max_nrb is the number of qubits for the largest required table
    '''

    clifford_tables = [[] for i in range(max_nrb)]
    for rb_num in range(max_nrb):
        #load the clifford tables, but only if we're using that particular rb number
        if rb_num == 0:
            #1Q Cliffords, load table programmatically
            clifford_tables[0] = clutils.clifford1_table()
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
                                              n=(rb_num+1))
                clifford_tables[rb_num] = \
                    clutils.load_clifford_table(\
                    picklefile='cliffords%d.pickle'%(rb_num+1))
        else:
            raise ValueError("Error: the number of qubits should be only 1 or 2 \n")

    return clifford_tables


#get generic randomized benchmarking sequences
def randomized_benchmarking_seq(rb_opts_dict=None):
    """Get a generic randomized benchmarking sequence
    rb_opts_dict: A dictionary of RB options
        nseeds: number of seeds
        length_vector: 'm' length vector of Clifford lengths. Must be in ascending order.
        RB sequences of increasing length grow on top of the previous sequences.
        n_qubits: total number of qubits
        rb_pattern: A list of the form [[i,j],[k],...] which will make
        simultaneous RB sequences where
        Qi,Qj are a 2Q RB sequence and Qk is a 1Q sequence, etc.
        E.g. [[0,3],[2],[1]] would create RB sequences that are 2Q for Q0/Q3, 1Q for Q1+Q2
        The number of qubits is the sum of the entries. For 'regular' RB the qubit_pattern is just
        [[0]],[[0,1]]
        length_multiplier: if this is an array it scales each rb_sequence by the multiplier

    Returns:
        rb_circs: list of circuits for the rb sequences
        xdata: the Clifford lengths (with multiplier if applicable)
        rb_opts_dict: option dictionary back out with default options appended
    """

    config = copy.deepcopy(rb_opts_dict)
    set_defaults_if_needed(config)

    rb_pattern = config['rb_pattern']
    length_vector = config['length_vector']
    nseeds = config['nseeds']
    n_qubits = config['n_qubits']
    length_multiplier = config['length_multiplier']

    xdata = calc_xdata(length_vector, length_multiplier)
    pattern_sizes = [len(pat) for pat in rb_pattern]
    clifford_tables = load_tables(np.max(pattern_sizes))

    circuits = []
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
                    new_cliff = clutils.random_clifford(rb_q_num)
                    Cliffs[rb_pattern_index].compose_circuit(new_cliff)
                    general_circ += replace_q_indices(clutils.get_quantum_circuit(new_cliff),
                                                      rb_pattern[rb_pattern_index], qr)
                    #add a barrier
                    general_circ.barrier(*[qr[x] for x in rb_pattern[rb_pattern_index]])

            #if the number of cliffords matches one of the sequence lengths then
            #calculate the inverse and produce the circuit
            if (cliff_index+1) == length_vector[length_index]:

                circ = qiskit.QuantumCircuit(qr, cr)
                circ += general_circ

                for (rb_pattern_index, rb_q_num) in enumerate(pattern_sizes):
                    inv_circuit = clutils.find_inverse_clifford_circuit(Cliffs[rb_pattern_index],
                                                                        clifford_tables[rb_q_num-1])
                    inv = clf.Clifford(rb_q_num)
                    inv.circuit = inv_circuit
                    # inv is now a Clifford object whose circuit is the correct
                    # inverse but the tableau is just identity
                    circ += replace_q_indices(clutils.get_quantum_circuit(inv),
                                              rb_pattern[rb_pattern_index], qr)

                circ.measure(qr, cr)
                circ.name = 'rb_seed_' + str(seed) + '_length_' + str(length_vector[length_index])
                circuits.append(circ)
                length_index += 1

    return circuits, xdata, config


def replace_q_indices(circuit, q_nums, qr):
    """Take a circuit that is ordered from 0,1,2 qubits and replace 0 with the qubit
    label in the first index of q_nums, 1 with the second index...

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
