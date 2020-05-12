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
# pylint: disable=unused-argument,unnecessary-pass,invalid-name

"""
Generates randomized benchmarking sequences
"""

import copy
from typing import List, Optional
import numpy as np
import qiskit

from .Clifford import Clifford
from .clifford_utils import CliffordUtils as clutils
from .dihedral import CNOTDihedral
from .dihedral_utils import DihedralUtils as dutils


def handle_length_multiplier(length_multiplier, len_pattern,
                             is_purity=False):
    """
    Check validity of length_multiplier.
    In addition, transform it into a vector if it is a constant.
    In case of purity rb the length multiplier should be None.

    Args:
        length_multiplier (list): length of the multiplier
        len_pattern (int): length of the RB pattern
        is_purity (bool): True only for purity rb (default is False)

    Returns:
        list: length_multiplier

    Raises:
        ValueError: if the input is invalid
    """

    if hasattr(length_multiplier, "__len__"):
        if is_purity:
            raise ValueError(
                "In case of Purity RB the length multiplier should be None")
        if len(length_multiplier) != len_pattern:
            raise ValueError(
                "Length multiplier must be the same length as the pattern")
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

    In case of purity rb, checks that all simultaneous sequences have the same
    dimension (e.g. only 1-qubit sequences, or only 2-qubit sequences etc.)

    Args:
        pattern (list): RB pattern
        is_purity (bool): True only for purity rb (default is False)

    Raises:
        ValueError: if the pattern is not valid

    Return:
        tuple: of the form (``qlist``, ``maxqubit``, ``maxdim``) where:
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
        length_vector (list): vector length
        length_multiplier (list): length of the multiplier of the vector length

    Returns:
        ndarray: An array of sequences lengths
    """

    xdata = []
    for mult in length_multiplier:
        xdata.append(np.array(length_vector)*mult)

    return np.array(xdata)


def randomized_benchmarking_seq(nseeds: int = 1,
                                length_vector: Optional[List[int]] = None,
                                rb_pattern: Optional[List[List[int]]] = None,
                                length_multiplier: Optional[List[int]] = 1,
                                seed_offset: int = 0,
                                align_cliffs: bool = False,
                                interleaved_gates: Optional[List[List[str]]] = None,
                                is_purity: bool = False,
                                group_gates: Optional[str] = None) -> \
        (List[List[qiskit.QuantumCircuit]], List[List[int]],
         Optional[List[List[qiskit.QuantumCircuit]]],
         Optional[List[List[List[qiskit.QuantumCircuit]]]],
         Optional[int]):
    """Generate generic randomized benchmarking (RB) sequences.

    Args:
        nseeds: The number of seeds. For each seed the function generates
            a separate list of output RB circuits.

        length_vector: Length vector of the RB sequence lengths. Must be in
            ascending order. RB sequences of increasing length grow on top of
            the previous sequences.

            For example:

            * ``length_vector = [1, 10, 20, 50, 75, 100, 125, 150, 175]``

            * ``length_vector = None`` is the same as ``length_vector = [1, 10, 20]``

        rb_pattern: A list of the lists of integers representing the
            qubits indexes. For example, ``[[i,j],[k],...]`` will make
            simultaneous RB sequences, where
            there is a 2-qubit RB sequence on qbits Qi and Qj,
            and a 1-qubit RB sequence on qubit Qk, etc.
            Each qubit appers at most once.
            The number of qubits on which RB is done is the sum of the lists
            sizes.

            For example:

            * ``rb_pattern = [[0]]`` or ``rb_pattern = None`` -- \
            create a 1-qubit RB sequence on qubit Q0.

            * ``rb_pattern = [[0,1]]`` -- \
            create a 2-qubit RB sequence on qubits Q0 and Q1.

            * ``rb_pattern = [[2],[6,4]]`` -- \
            create RB sequences that are 2-qubit RB for qubits Q6 and Q4, \
            and 1-qubit RB for qubit Q2.

        length_multiplier: An array that scales each RB sequence by
            the multiplier.

        seed_offset: What to start the seeds at, if we
            want to add more seeds later.

        align_cliffs: If ``True`` adds a barrier across all qubits in
            the pattern after each set of group elements
            (not necessarily Cliffords).

            **Note:** the alignment considers the group multiplier.

        interleaved_gates: A list of lists of gates that
            will be interleaved. It is not ``None`` only for interleaved
            randomized benchmarking.
            The lengths of the lists should be equal to the length of the
            lists in ``rb_pattern``.

        is_purity: ``True`` only for purity randomized benchmarking
            (default is ``False``).

            **Note:** if ``is_purity = True`` then all patterns in
            ``rb_pattern`` should have the same dimension
            (e.g. only 1-qubit sequences, or only 2-qubit sequences),
            and ``length_multiplier = None``.

        group_gates: On which group (or set of gates) we perform RB
            (the default is the Clifford group).

            * ``group_gates='0'`` or ``group_gates=None`` or \
            ``group_gates='Clifford'`` -- Clifford group.

            * ``group_gates='1'`` or ``group_gates='CNOT-Dihedral'`` \
            or ``group_gates='Non-Clifford'`` -- CNOT-Dihedral group.

    Returns:
        A tuple of different fields depending on the inputs.
        The different fields are:

         * ``circuits``: list of lists of circuits for the RB sequences \
            (a separate list for each seed).

         * ``xdata``: the sequences lengths (with multiplier if applicable).

         * ``circuits_interleaved``: \
           (only if ``interleaved_gates`` is not ``None``): \
           list of lists of circuits for the interleaved RB sequences \
           (a separate list for each seed).

         * ``circuits_purity``: (only if ``is_purity=True``): \
           list of lists of lists of circuits for purity RB \
           (a separate list for each seed and each of the :math:`3^n` circuits).

         * ``npurity``: (only if ``is_purity=True``): \
            the number of purity RB circuits (per seed) \
            which equals to :math:`3^n`, where n is the dimension.

    Raises:

        ValueError: if ``group_gates`` is unknown.
        ValueError: if ``rb_pattern`` is not valid.
        ValueError: if ``length_multiplier`` is not valid.


    Examples:

        1) Generate simultaneous standard RB sequences.

        .. code-block::

            length_vector = [1,10,20]
            rb_pattern = [[0,3],[2],[1]]
            length_multiplier = [1,3,3]
            align_cliffs = True

        Create RB sequences that are 2-qubit RB for qubits Q0 and Q3,
        1-qubit RB for qubit Q1, and 1-qubit RB for qubit Q2.
        Generate three times as many 1-qubit RB sequence elements,
        than 2-qubit elements.
        Place a barrier after 1 group element for the first pattern
        and after 3 group elements for the second and third patterns.
        The output ``xdata`` in this case is

        .. code-block::

            xdata=[[1,10,20],[3,30,60],[3,30,60]]

        2) Generate simultaneous interleaved RB sequences.

        .. code-block::

            rb_pattern = [[0,3],[2],[1]]
            interleaved_gates = [['cx 0 1'], ['x 0'], ['h 0']]

        Interleave the 2-qubit gate ``cx`` on qubits Q0 and Q3,
        a 1-qubit gate ``x`` on qubit Q2,
        and a 1-qubit gate ``h`` on qubit Q1.

        3) Generated purity RB sequences.

        .. code-block::

            rb_pattern = [[0,3],[1,2]]
            npurity = True

        Create purity 2-qubit RB circuits separately on qubits
        Q0 and Q3 and on qubtis Q1 and Q2.
        The output is ``npurity = 9`` in this case.

    """
    # Set modules (default is Clifford)
    if group_gates is None or group_gates in ('0',
                                              'Clifford',
                                              'clifford'):
        g_utils = clutils()
        g_group = Clifford
        rb_circ_type = 'rb'
        group_gates_type = 0
    elif group_gates in ('1', 'Non-Clifford',
                         'NonClifford'
                         'CNOTDihedral',
                         'CNOT-Dihedral'):
        g_utils = dutils()
        g_group = CNOTDihedral
        rb_circ_type = 'rb_cnotdihedral'
        group_gates_type = 1
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
        group_tables[rb_num] = g_utils.load_tables(rb_num+1)

    # initialization: rb sequences
    circuits = [[] for e in range(nseeds)]
    # initialization: interleaved rb sequences
    circuits_interleaved = [[] for e in range(nseeds)]
    # initialization: non-clifford cnot-dihedral
    # rb sequences
    circuits_cnotdihedral = [[] for e in range(nseeds)]
    # initialization: non-clifford cnot-dihedral
    # interleaved rb sequences
    circuits_cnotdihedral_interleaved = [[] for e in range(nseeds)]
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
            Elmnts.append(g_group(rb_q_num))
        # Sequences for interleaved rb sequences
            Elmnts_interleaved = []
        for rb_q_num in pattern_sizes:
            Elmnts_interleaved.append(g_group(rb_q_num))

        # go through and add elements to RB sequences
        length_index = 0
        for elmnts_index in range(length_vector[-1]):
            for (rb_pattern_index, rb_q_num) in enumerate(pattern_sizes):

                for _ in range(length_multiplier[rb_pattern_index]):
                    new_elmnt_gatelist = g_utils.random_gates(
                        rb_q_num)
                    Elmnts[rb_pattern_index] = g_utils.compose_gates(
                        Elmnts[rb_pattern_index], new_elmnt_gatelist)
                    general_circ += replace_q_indices(
                        get_quantum_circuit(g_utils.gatelist(),
                                            rb_q_num),
                        rb_pattern[rb_pattern_index], qr)

                    # add a barrier
                    general_circ.barrier(
                        *[qr[x] for x in rb_pattern[rb_pattern_index]])

                    # interleaved rb sequences
                    if interleaved_gates is not None:
                        Elmnts_interleaved[rb_pattern_index] = \
                            g_utils.compose_gates(
                                Elmnts_interleaved[rb_pattern_index],
                                new_elmnt_gatelist)
                        interleaved_circ += replace_q_indices(
                            get_quantum_circuit(g_utils.gatelist(),
                                                rb_q_num),
                            rb_pattern[rb_pattern_index], qr)
                        Elmnts_interleaved[rb_pattern_index] = \
                            g_utils.compose_gates(
                                Elmnts_interleaved[rb_pattern_index],
                                interleaved_gates[rb_pattern_index])
                        # add a barrier - interleaved rb
                        interleaved_circ.barrier(
                            *[qr[x] for x in rb_pattern[rb_pattern_index]])
                        interleaved_circ += replace_q_indices(
                            get_quantum_circuit(g_utils.gatelist(),
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
                    inv_key = g_utils.find_key(Elmnts[rb_pattern_index],
                                               rb_q_num)
                    inv_circuit = g_utils.find_inverse_gates(
                        rb_q_num,
                        group_tables[rb_q_num-1][inv_key])
                    circ += replace_q_indices(
                        get_quantum_circuit(inv_circuit, rb_q_num),
                        rb_pattern[rb_pattern_index], qr)
                    # calculate the inverse and produce the circuit
                    # for interleaved rb
                    if interleaved_gates is not None:
                        inv_key = g_utils.find_key(Elmnts_interleaved
                                                   [rb_pattern_index],
                                                   rb_q_num)
                        inv_circuit = g_utils.find_inverse_gates(
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
                        circ_purity[d].name = rb_circ_type + '_purity_'
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

                # add measurement for Non-Clifford cnot-dihedral rb
                # measure both the ground state |0...0> (circ)
                # and the |+...+> state (cnot-dihedral_circ)
                cnotdihedral_circ = qiskit.QuantumCircuit(qr, cr)
                cnotdihedral_interleaved_circ = qiskit.QuantumCircuit(qr, cr)
                if group_gates_type == 1:
                    for _, qb in enumerate(qlist_flat):
                        cnotdihedral_circ.h(qr[qb])
                        cnotdihedral_circ.barrier(qr[qb])
                        cnotdihedral_interleaved_circ.h(qr[qb])
                        cnotdihedral_interleaved_circ.barrier(qr[qb])
                    cnotdihedral_circ += circ
                    cnotdihedral_interleaved_circ += circ_interleaved
                    for _, qb in enumerate(qlist_flat):
                        cnotdihedral_circ.barrier(qr[qb])
                        cnotdihedral_circ.h(qr[qb])
                        cnotdihedral_interleaved_circ.barrier(qr[qb])
                        cnotdihedral_interleaved_circ.h(qr[qb])
                    for qind, qb in enumerate(qlist_flat):
                        cnotdihedral_circ.measure(qr[qb], cr[qind])
                        cnotdihedral_interleaved_circ.measure(qr[qb], cr[qind])

                # add measurement for standard rb
                # qubits measure to the c registers as
                # they appear in the pattern
                for qind, qb in enumerate(qlist_flat):
                    circ.measure(qr[qb], cr[qind])
                    # add measurement for interleaved rb
                    circ_interleaved.measure(qr[qb], cr[qind])

                circ.name = \
                    rb_circ_type + '_length_%d_seed_%d' % \
                    (length_index, seed + seed_offset)
                circ_interleaved.name = \
                    rb_circ_type + '_interleaved_length_%d_seed_%d' % \
                    (length_index, seed + seed_offset)

                if group_gates_type == 1:
                    circ.name = rb_circ_type + '_Z_length_%d_seed_%d' % \
                                (length_index, seed + seed_offset)
                    circ_interleaved.name = \
                        rb_circ_type + '_interleaved_Z_length_%d_seed_%d' % \
                        (length_index, seed + seed_offset)
                    cnotdihedral_circ.name = \
                        rb_circ_type + '_X_length_%d_seed_%d' % \
                        (length_index, seed + seed_offset)
                    cnotdihedral_interleaved_circ.name = \
                        rb_circ_type + '_interleaved_X_length_%d_seed_%d' % \
                        (length_index, seed + seed_offset)

                circuits[seed].append(circ)
                circuits_interleaved[seed].append(circ_interleaved)
                circuits_cnotdihedral[seed].append(cnotdihedral_circ)
                circuits_cnotdihedral_interleaved[seed].append(
                    cnotdihedral_interleaved_circ)

                if is_purity:
                    for d in range(npurity):
                        circuits_purity[seed][d].append(circ_purity[d])
                length_index += 1

    # output of purity rb
    if is_purity:
        return circuits_purity, xdata, npurity
    # output of non-clifford cnot-dihedral interleaved rb
    if interleaved_gates is not None and group_gates_type == 1:
        return circuits, xdata, circuits_cnotdihedral, circuits_interleaved, \
               circuits_cnotdihedral_interleaved
    # output of interleaved rb
    if interleaved_gates is not None:
        return circuits, xdata, circuits_interleaved
    # output of Non-Clifford cnot-dihedral rb
    if group_gates_type == 1:
        return circuits, xdata, circuits_cnotdihedral
    # output of standard (simultaneous) rb
    return circuits, xdata


def replace_q_indices(circuit, q_nums, qr):
    """
    Take a circuit that is ordered from 0,1,2 qubits and replace 0 with the
    qubit label in the first index of q_nums, 1 with the second index...

    Args:
        circuit (QuantumCircuit): circuit to operate on
        q_nums (list): list of qubit indices
        qr (QuantumRegister): A quantum register to use for the output circuit

    Returns:
        QuantumCircuit: updated circuit
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
        num_qubits (int): the number of qubits (dimension).
        gatelist (list): a list of gates.

    Returns:
        QuantumCircuit: A QuantumCircuit object.
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

        if op_names == ['u1']:
            qubits = [qr[int(x)] for x in split[2:]]
            theta = float(split[1])
        else:
            qubits = [qr[int(x)] for x in split[1:]]

        for sub_op in op_names:
            operation = getattr(qiskit.QuantumCircuit, sub_op)
            if sub_op == 'u1':
                operation(qc, theta, *qubits)
            else:
                operation(qc, *qubits)

    return qc
