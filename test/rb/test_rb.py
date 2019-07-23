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

# pylint: disable=undefined-loop-variable

"""
Run through RB for different qubit numbers to check that it's working
and that it returns the identity
"""

import unittest
import random
import numpy as np
import qiskit
import qiskit.ignis.verification.randomized_benchmarking as rb


class TestRB(unittest.TestCase):
    """ The test class """

    @staticmethod
    def choose_pattern(pattern_type, nq):
        '''
        Choose a valid field for rb_opts['rb_pattern']
        :param pattern_type: a number between 0 and 2.
                             0 - a list of all qubits, for nq=5 it is
                                 [1, 2, 3, 4, 5]
                             1 - a list of lists of single qubits, for nq=5
                                 it is [[1], [2], [3], [4], [5]]
                             2 - randomly choose a pattern which is a list of
                                 two lists where the first one has 2 elements,
                                 for example for nq=5 it can be
                                 [[4, 1], [2, 5, 3]]
        :param nq: number of qubits
        :return: the pattern or None
                 Returns None if the pattern type is not relevant to the
                 number of qubits, i.e,, one of two cases:
                 pattern_type = 1 and nq = 1, which implies [[1]]
                 pattern_type = 2 and nq <= 2: - for nq=1 this is impossible
                                               - for nq=2 this implies
                                                 [[1], [2]], which is already
                                                 tested when pattern_type = 1
                 is_purity = True if the pattern fits for purity rb
                 (namely, all the patterns have the same dimension:
                 only 1-qubit, only 2-qubits etc.).
        '''

        is_purity = True
        if pattern_type == 0:
            res = [list(range(nq))]
            if nq > 2:  # since we only have 1-qubit and 2-qubit Cliffords
                return None, None
        elif pattern_type == 1:
            if nq == 1:
                return None, None
            res = [[x] for x in range(nq)]
        else:
            if nq <= 2:
                return None, None
            shuffled_bits = list(range(nq))
            random.shuffle(shuffled_bits)
            # split_loc = random.randint(1, nq-1)
            split_loc = 2  # deterministic test
            res = [shuffled_bits[:split_loc], shuffled_bits[split_loc:]]
            # since we only have 1-qubit and 2-qubit Cliffords
            if (split_loc > 2) | (nq-split_loc > 2):
                return None, None
            if 2*split_loc != nq:
                is_purity = False

        return res, is_purity

    @staticmethod
    def choose_multiplier(mult_opt, len_pattern):
        '''
        :param multi_opt:
            0: fixed length
            1: vector of lengths
        :param len_pattern: number of patterns
        :return: the length multiplier
        '''
        if mult_opt == 0:
            res = 1
        else:
            res = [i + 1 for i in range(len_pattern)]

        return res

    @staticmethod
    def choose_interleaved_gates(rb_pattern):
        '''
        :param rb_pattern: pattern for randomized benchmarking
        :return: interleaved_gates:
        A list of gates of Clifford elements that
        will be interleaved (for interleaved randomized benchmarking)
        The length of the list would equal the length of the rb_pattern.
        '''
        pattern_sizes = [len(pat) for pat in rb_pattern]
        interleaved_gates = []
        for (_, nq) in enumerate(pattern_sizes):
            gatelist = []
            # The interleaved gates contain x gate on each qubit
            # and cx gate on each pair of consecutive qubits
            for qubit in range(nq):
                gatelist.append('x ' + str(qubit))
            for qubit_i in range(nq):
                for qubit_j in range(qubit_i+1, nq):
                    gatelist.append('cx ' + str(qubit_i) + ' ' + str(qubit_j))
            interleaved_gates.append(gatelist)
        return interleaved_gates

    @staticmethod
    def update_interleaved_gates(gatelist, pattern):
        '''
        :param gatelist: list of Clifford gates
        :param pattern: pattern of indexes (from rb_pattern)
        :return: updated_gatelist: list of Clifford gates
        after the following updates:
        - change the indexes from [0,1,...]
        according to the pattern
        '''
        updated_gatelist = []
        for op in gatelist:
            split = op.split()
            op_names = split[0]
            # updating the qubit indexes according to the pattern
            # given in rb_pattern
            op_qubits = [str(pattern[int(x)]) for x in split[1:]]
            updated_gatelist += [op_names + ' ' +
                                 (' '.join(op_qubits))]
        return updated_gatelist

    @staticmethod
    def update_purity_gates(npurity, purity_ind, rb_pattern):
        '''
        :param npurity: equals to 3^n
        :param purity_ind: purity index in [0,3^n-1]
        :param rb_pattern: rb pattern
        :return: name_type: type of name for rb_circuit
        (e.g. XY, ZZ etc.)
        :return: gate_list: list of purity gates
        (e.g 'rx 0', 'ry 1' etc.) according to rb_pattern
        '''
        name_type = ''
        ind_d = purity_ind
        purity_qubit_num = 0
        gate_list = []
        while True:
            purity_qubit_rot = np.mod(ind_d, 3)
            ind_d = np.floor_divide(ind_d, 3)
            if purity_qubit_rot == 0:
                name_type += 'Z'
            if purity_qubit_rot == 1:
                name_type += 'X'
                for pat in rb_pattern:
                    gate_list.append('rx ' + str(pat[purity_qubit_num]))
            if purity_qubit_rot == 2:
                name_type += 'Y'
                for pat in rb_pattern:
                    gate_list.append('ry ' + str(pat[purity_qubit_num]))

            purity_qubit_num = purity_qubit_num + 1
            if ind_d == 0:
                break
        # padding the circuit name with Z's so that
        # all circuits will have names of the same length
        for _ in range(int(np.log(npurity)/np.log(3)) -
                       purity_qubit_num):
            name_type += 'Z'

        return name_type, gate_list

    @staticmethod
    def ops_to_gates(ops, op_index, stop_gate='barrier'):
        '''
        :param ops: of the form circ.data
        :param op_index: int, the operation index
        :param stop_gate: the gate to stop
        (e.g. barrier or measure)
        :return: gatelist: a list of gates
        :return: op_index: int, updated index
        '''
        gatelist = []
        while ops[op_index][0].name != stop_gate:
            gate = ops[op_index][0].name
            for x in ops[op_index][1]:
                gate += ' ' + str(x.index)
            gatelist.append(gate)
            op_index += 1
        # increment because of the barrier gate
        op_index += 1
        return gatelist, op_index

    def verify_circuit(self, circ, nq, rb_opts, vec_len, result, shots,
                       is_interleaved=False):
        '''
        For a single sequence, verifies that it meets the requirements:
        - Executing it on the ground state ends up in the ground state
        - It has the correct number of Cliffords
        - It fulfills the pattern, as specified by rb_patterns and
          length_multiplier
        :param circ: the sequence to check
        :param nq: number of qubits
        :param rb_opts: the specification that generated the set of sequences
                        which includes circ
        :param vec_len: the expected length vector of circ (one of
                        rb_opts['length_vector'])
        :param result: the output of the simulator
                       when executing all the sequences on the ground state
        :param shots: the number of shots in the simulator execution
        :param is_interleaved: True if this is an interleaved circuit
        '''

        if not hasattr(rb_opts['length_multiplier'], "__len__"):
            rb_opts['length_multiplier'] = [
                rb_opts['length_multiplier'] for i in range(
                    len(rb_opts['rb_pattern']))]

        ops = circ.data
        op_index = 0
        # for each cycle (the sequence should consist of vec_len cycles)
        for _ in range(vec_len):
            # for each component of the pattern...
            for pat_index in range(len(rb_opts['rb_pattern'])):
                # for each Clifford...
                for _ in range(rb_opts['length_multiplier'][pat_index]):
                    # if it is an interleaved RB circuit,
                    # then it has twice as many Cliffords
                    for _ in range(is_interleaved+1):
                        # for each basis gate...
                        # in case of align_cliffs we may have extra barriers
                        # (after another barrier)
                        if ops[op_index][0].name != 'barrier':
                            while ops[op_index][0].name != 'barrier':
                                # Verify that the gate acts
                                # on the correct qubits.
                                # This happens if the sequence is composed
                                # of the correct sub-sequences,
                                # as specified by vec_len and rb_opts
                                self.assertTrue(
                                    all(x.index in rb_opts['rb_pattern'][
                                        pat_index]
                                        for x in ops[op_index][1]),
                                    "Error: operation acts on \
                                    incorrect qubits")
                                op_index += 1
                            # increment because of the barrier gate
                            op_index += 1
        # check if the ground state returns
        self.assertEqual(result.
                         get_counts(circ)['{0:b}'.format(0).zfill(nq)], shots,
                         "Error: %d qubit RB does not return the \
                         ground state back to the ground state" % nq)

    def compare_interleaved_circuit(self, original_circ, interleaved_circ,
                                    nq, rb_opts_interleaved, vec_len):
        '''
        Verifies that interleaved RB circuits meet the requirements:
        - The non-interleaved Clifford gates are the same as the
        original Clifford gates.
        - The interleaved Clifford gates are the same as the ones
        given in: rb_opts_interleaved['interleaved_gates'].
        :param original_circ: original rb circuits
        :param interleaved_circ: interleaved rb circuits
        :param nq: number of qubits
        :param rb_opts_interleaved: the specification that
        generated the set of sequences which includes circ
        :param vec_len: the expected length vector of circ
        (one of rb_opts['length_vector'])
        '''

        if not hasattr(rb_opts_interleaved['length_multiplier'], "__len__"):
            rb_opts_interleaved['length_multiplier'] = [
                rb_opts_interleaved['length_multiplier'] for i in range(
                    len(rb_opts_interleaved['rb_pattern']))]

        original_ops = original_circ.data
        interleaved_ops = interleaved_circ.data

        original_op_index = 0
        interleaved_op_index = 0
        # for each cycle (the sequence should consist of vec_len cycles)
        for _ in range(vec_len):
            # for each component of the pattern...
            for pat_index in range(len(rb_opts_interleaved['rb_pattern'])):
                # updating the gates in:
                # rb_opts_interleaved['interleaved_gates']
                updated_gatelist = self.update_interleaved_gates(
                    rb_opts_interleaved['interleaved_gates']
                    [pat_index], rb_opts_interleaved['rb_pattern'][pat_index])
                # for each Clifford...
                for _ in range(rb_opts_interleaved['length_multiplier']
                               [pat_index]):
                    # original RB sequence
                    original_gatelist, original_op_index = \
                        self.ops_to_gates(original_ops,
                                          original_op_index)
                    # interleaved RB sequence
                    compared_gatelist, interleaved_op_index = \
                        self.ops_to_gates(interleaved_ops,
                                          interleaved_op_index)

                    # Clifford gates in the interleaved RB sequence
                    # should be equal to original gates
                    self.assertEqual(original_gatelist, compared_gatelist,
                                     "Error: The gates in the %d qubit  \
                                     interleaved RB are not the same as \
                                     in the original RB circuits" % nq)
                    # Clifford gates in the interleaved RB sequence
                    # should be equal to the given gates in
                    # rb_opts_interleaved['interleaved_gates']
                    # (after updating them)
                    interleaved_gatelist, interleaved_op_index = \
                        self.ops_to_gates(interleaved_ops,
                                          interleaved_op_index)

                    self.assertEqual(interleaved_gatelist, updated_gatelist,
                                     "Error: The interleaved gates in the \
                                     %d qubit interleaved RB are not the same \
                                     as given in interleaved_gates input" % nq)

    def compare_purity_circuits(self, original_circ, purity_circ, nq,
                                purity_ind, npurity, rb_opts_purity, vec_len):
        '''
        Verifies that purity RB circuits meet the requirements:
        - The Clifford gates are the same as the original Clifford gates.
        - The last gates are either Rx or Ry or nothing
        (depend on d)
        :param original_circ: original rb circuits
        :param purity_circ: purity rb circuits
        :param nq: number of qubits
        :param purity_ind: purity index in [0,3^n-1]
        :param npurity: equal to 3^n
        :param rb_opts_purity: the specification that
        generated the set of sequences which includes circ
        :param vec_len: the expected length vector of circ
        (one of rb_opts['length_vector'])
        '''

        original_ops = original_circ.data
        purity_ops = purity_circ.data
        op_index = 0
        pur_index = 0

        # for each cycle (the sequence should consist of vec_len cycles)
        for _ in range(vec_len):
            # for each component of the pattern...
            for pat_index in range(len(rb_opts_purity['rb_pattern'])):
                # for each Clifford...
                for _ in range(rb_opts_purity['length_multiplier'][pat_index]):
                    # original RB sequence
                    original_gatelist, op_index = \
                        self.ops_to_gates(original_ops, op_index)
                    # purity RB sequence
                    purity_gatelist, pur_index = \
                        self.ops_to_gates(purity_ops, pur_index)
                    # Clifford gates in the purity RB sequence
                    # should be equal to original gates
                    self.assertEqual(original_gatelist, purity_gatelist,
                                     "Error: The purity gates in the \
                                     %d qubit purity RB are not the same \
                                     as in the original RB circuits" % nq)

        # The last gate in the purity RB sequence
        # should be equal to the inverse clifford
        # with either Rx or Ry or nothing (depend on d)
        # original last gate
        original_gatelist, op_index = \
            self.ops_to_gates(original_ops, op_index, 'measure')
        _, purity_gates = self.update_purity_gates(
            npurity, purity_ind, rb_opts_purity['rb_pattern'])
        original_gatelist = original_gatelist + purity_gates
        # purity last gate
        purity_gatelist, pur_index = \
            self.ops_to_gates(purity_ops, pur_index, 'measure')
        self.assertEqual(original_gatelist, purity_gatelist,
                         "Error: The last purity gates in the \
                         %d qubit purity RB are wrong" % nq)

    def test_rb(self):
        """ Main function of the test """

        # Load simulator
        backend = qiskit.Aer.get_backend('qasm_simulator')

        # Test up to 4 qubits (only 1-qubit and 2-qubit RB)
        nq_list = [1, 2, 3, 4]

        for nq in nq_list:

            print("Testing %d qubit RB" % nq)

            for pattern_type in range(3):
                for multiplier_type in range(2):
                    # See documentation of choose_pattern for the meaning of
                    # the different pattern types
                    # Choose options for standard (simultaneous) RB:
                    rb_opts = {}
                    rb_opts['nseeds'] = 3
                    rb_opts['length_vector'] = [1, 3, 4, 7]
                    rb_opts['rb_pattern'], is_purity = \
                        self.choose_pattern(pattern_type, nq)
                    # if the pattern type is not relevant for nq
                    if rb_opts['rb_pattern'] is None:
                        continue
                    rb_opts_purity = rb_opts.copy()
                    rb_opts['length_multiplier'] = self.choose_multiplier(
                        multiplier_type, len(rb_opts['rb_pattern']))
                    # Choose options for interleaved RB:
                    rb_opts_interleaved = rb_opts.copy()
                    rb_opts_interleaved['interleaved_gates'] = \
                        self.choose_interleaved_gates(rb_opts['rb_pattern'])
                    print('rb_opts:', rb_opts_interleaved)
                    # Choose options for purity rb
                    # no length_multiplier
                    rb_opts_purity['length_multiplier'] = 1
                    rb_opts_purity['is_purity'] = is_purity
                    if multiplier_type > 0:
                        is_purity = False
                    if is_purity:
                        print('Testing purity RB')
                    # Adding seed_offset and align_cliffs
                    rb_opts['seed_offset'] = 10
                    rb_opts['align_cliffs'] = True

                    # Generate the sequences
                    try:
                        # Standard (simultaneous) RB sequences:
                        rb_circs, _ = rb.randomized_benchmarking_seq(**rb_opts)
                        # Interleaved RB sequences:
                        rb_original_circs, _, rb_interleaved_circs = \
                            rb.randomized_benchmarking_seq(
                                **rb_opts_interleaved)
                        # Purity RB sequences:
                        if is_purity:
                            rb_purity_circs, _, npurity = \
                                rb.randomized_benchmarking_seq(
                                    **rb_opts_purity)
                            # verify: npurity = 3^n
                            self.assertEqual(
                                npurity, 3 ** len(rb_opts['rb_pattern'][0]),
                                'Error: npurity does not equal to 3^n')

                    except OSError:
                        skip_msg = ('Skipping tests for %s qubits because '
                                    'tables are missing' % str(nq))
                        print(skip_msg)
                        continue

                    # Perform an ideal execution on the generated sequences
                    basis_gates = ['u1', 'u2', 'u3', 'cx']
                    shots = 100
                    result = []
                    result_original = []
                    result_interleaved = []
                    result_purity = [[] for d in range(npurity)]
                    for seed in range(rb_opts['nseeds']):
                        result.append(
                            qiskit.execute(rb_circs[seed], backend=backend,
                                           basis_gates=basis_gates,
                                           shots=shots).result())
                        result_original.append(
                            qiskit.execute(rb_original_circs[seed],
                                           backend=backend,
                                           basis_gates=basis_gates,
                                           shots=shots).result())
                        result_interleaved.append(
                            qiskit.execute(rb_interleaved_circs[seed],
                                           backend=backend,
                                           basis_gates=basis_gates,
                                           shots=shots).result())
                        if is_purity:
                            for d in range(npurity):
                                result_purity[d].append(qiskit.execute(
                                    rb_purity_circs[seed][d],
                                    backend=backend,
                                    basis_gates=basis_gates,
                                    shots=shots).result())

                    # Verify the generated sequences
                    for seed in range(rb_opts['nseeds']):
                        length_vec = rb_opts['length_vector']
                        for circ_index, vec_len in enumerate(length_vec):
                            # Verify circuits names
                            self.assertEqual(
                                rb_circs[seed][circ_index].name,
                                'rb_length_%d_seed_%d' % (
                                    circ_index, seed +
                                    rb_opts['seed_offset']),
                                'Error: incorrect circuit name')
                            self.assertEqual(
                                rb_original_circs[seed][circ_index].name,
                                'rb_length_%d_seed_%d' % (
                                    circ_index, seed),
                                'Error: incorrect circuit name')
                            self.assertEqual(
                                rb_interleaved_circs[seed][circ_index].name,
                                'rb_interleaved_length_%d_seed_%d' % (
                                    circ_index, seed),
                                'Error: incorrect interleaved circuit name')
                            if is_purity:
                                for d in range(npurity):
                                    name_type, _ = self.update_purity_gates(
                                        npurity, d, rb_opts_purity
                                        ['rb_pattern'])
                                    self.assertEqual(
                                        rb_purity_circs[seed][d]
                                        [circ_index].name,
                                        'rb_purity_%s_length_%d_seed_%d' % (
                                            name_type, circ_index, seed),
                                        'Error: incorrect purity circuit name')

                            self.verify_circuit(rb_circs[seed][circ_index],
                                                nq, rb_opts,
                                                vec_len, result[seed], shots)
                            self.verify_circuit(rb_original_circs[seed]
                                                [circ_index],
                                                nq, rb_opts,
                                                vec_len,
                                                result_original[seed], shots)
                            self.verify_circuit(rb_interleaved_circs[seed]
                                                [circ_index],
                                                nq, rb_opts_interleaved,
                                                vec_len,
                                                result_interleaved[seed],
                                                shots,
                                                is_interleaved=True)
                            if is_purity:
                                self.verify_circuit(rb_purity_circs[seed][0]
                                                    [circ_index],
                                                    nq, rb_opts_purity,
                                                    vec_len, result_purity
                                                    [0][seed], shots)
                                # compare the purity RB circuits
                                # with the original circuit
                                for d in range(1, npurity):
                                    self.compare_purity_circuits(
                                        rb_purity_circs[seed][0][circ_index],
                                        rb_purity_circs[seed][d][circ_index],
                                        nq, d, npurity, rb_opts_purity,
                                        vec_len)
                            # compare the interleaved RB circuits with
                            # the original RB circuits
                            self.compare_interleaved_circuit(
                                rb_original_circs[seed][circ_index],
                                rb_interleaved_circs[seed][circ_index],
                                nq, rb_opts_interleaved, vec_len)

                    self.assertEqual(circ_index, len(rb_circs),
                                     "Error: additional circuits exist")

    def test_rb_utils(self):

        """ Test some of the utility calculations, e.g.
        coherence limit"""

        t1 = 100.
        t2 = 100.
        gate2Q = 0.5
        gate1Q = 0.1
        twoq_coherence_err = rb.rb_utils.coherence_limit(2, [t1, t1],
                                                         [t2, t2], gate2Q)

        oneq_coherence_err = rb.rb_utils.coherence_limit(1, [t1],
                                                         [t2], gate1Q)

        self.assertAlmostEqual(oneq_coherence_err, 0.00049975, 6,
                               "Error: 1Q Coherence Limit")

        self.assertAlmostEqual(twoq_coherence_err, 0.00597, 5,
                               "Error: 2Q Coherence Limit")

        twoq_epc = rb.rb_utils.twoQ_clifford_error([5.2, 5.2, 1.5],
                                                   [0, 1, -1],
                                                   [0.001, 0.0015, 0.02])

        self.assertAlmostEqual(twoq_epc, 0.0446283, 6,
                               "Error: 2Q EPC Calculation")


if __name__ == '__main__':
    unittest.main()
