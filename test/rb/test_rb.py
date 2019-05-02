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
                                 two lists, for example for nq=5 it can be
                                 [[4, 1, 2], [5, 3]]
        :param nq: number of qubits
        :return: the pattern or None
                 Returns None if the pattern type is not relevant to the
                 number of qubits, i.e,, one of two cases:
                 pattern_type = 1 and nq = 1, which implies [[1]]
                 pattern_type = 2 and nq <= 2: - for nq=1 this is impossible
                                               - for nq=2 this implies
                                                 [[1], [2]], which is already
                                                 tested when pattern_type = 1
        '''

        if pattern_type == 0:
            res = [list(range(nq))]
        elif pattern_type == 1:
            if nq == 1:
                return None
            res = [[x] for x in range(nq)]
        else:
            if nq <= 2:
                return None
            shuffled_bits = list(range(nq))
            random.shuffle(shuffled_bits)
            split_loc = random.randint(1, nq-1)
            res = [shuffled_bits[:split_loc], shuffled_bits[split_loc:]]

        return res

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
        :param rb_pattern
        :return: interleaved_gates:
        A list of gates of Clifford elements that
        will be interleaved (for interleaved randomized benchmarking)
        The length of the list would equal the length of the rb_pattern).
        '''
        pattern_sizes = [len(pat) for pat in rb_pattern]
        interleaved_gates = []
        for (rb_pattern_index, nq) in enumerate(pattern_sizes):
            interleaved_gates.append(
                rb.clifford_utils.random_clifford_gates(nq))
        return interleaved_gates

    def verify_circuit(self, circ, nq, rb_opts, vec_len, result, shots):
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
                    # for each basis gate...
                    while ops[op_index][0].name != 'barrier':
                        # Verify that the gate acts on the correct qubits
                        # This happens if the sequence is composed of the
                        # correct sub-sequences, as specified by vec_len and
                        # rb_opts
                        self.assertTrue(
                            all(x[1] in rb_opts['rb_pattern'][pat_index]
                                for x in ops[op_index][1]),
                            "Error: operation acts on incorrect qubits")
                        op_index += 1
                    # increment because of the barrier gate
                    op_index += 1
        # check if the ground state returns
        self.assertEqual(result.
                         get_counts(circ)['{0:b}'.format(0).zfill(nq)], shots,
                         "Error: %d qubit RB does not return the \
                         ground state back to the ground state" % nq)

    def test_rb(self):
        """ Main function of the test """

        # Load simulator
        backend = qiskit.Aer.get_backend('qasm_simulator')

        # Test up to 2 qubits
        nq_list = [1, 2]

        for nq in nq_list:

            print("Testing %d qubit RB" % nq)

            for pattern_type in range(2):
                for multiplier_type in range(2):
                    # See documentation of choose_pattern for the meaning of
                    # the different pattern types
                    # Choose options for standard (simultaneous) RB:
                    rb_opts = {}
                    rb_opts['nseeds'] = 3
                    rb_opts['length_vector'] = [1, 3, 4, 7]
                    rb_opts['rb_pattern'] = self.choose_pattern(
                        pattern_type, nq)
                    # if the pattern type is not relevant for nq
                    if rb_opts['rb_pattern'] is None:
                        continue
                    rb_opts['length_multiplier'] = self.choose_multiplier(
                        multiplier_type, len(rb_opts['rb_pattern']))
                    # Choose options for interleaved RB:
                    rb_opts_interleaved = rb_opts.copy()
                    rb_opts_interleaved['interleaved_gates'] = \
                        self.choose_interleaved_gates(rb_opts['rb_pattern'])
                    # print (rb_opts_interleaved)

                    # Generate the sequences
                    try:
                        # Standard (simultaneous) RB sequences:
                        rb_circs, _ = rb.randomized_benchmarking_seq(**rb_opts)
                        # Interleaved RB sequences:
                        rb_original_circs, _, rb_interleaved_circs = \
                            rb.randomized_benchmarking_seq(**rb_opts_interleaved)
                        #print(rb_original_circs[0][0])
                        #print(rb_interleaved_circs[0][0])
                        #print(rb_original_circs[0][1])
                        #print(rb_interleaved_circs[0][1])
                        #print(rb_original_circs[0][2])
                        #print(rb_interleaved_circs[0][2])
                        #print(rb_original_circs[0][3])
                        #print(rb_interleaved_circs[0][3])

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
                    for seed in range(rb_opts['nseeds']):
                        result.append(
                            qiskit.execute(rb_circs[seed], backend=backend,
                                           basis_gates=basis_gates,
                                           shots=shots).result())
                        result_original.append(
                            qiskit.execute(rb_original_circs[seed], backend=backend,
                                           basis_gates=basis_gates,
                                           shots=shots).result())
                        result_interleaved.append(
                            qiskit.execute(rb_interleaved_circs[seed], backend=backend,
                                           basis_gates=basis_gates,
                                           shots=shots).result())

                    # Verify the generated sequences
                    for seed in range(rb_opts['nseeds']):
                        length_vec = rb_opts['length_vector']
                        for circ_index, vec_len in enumerate(length_vec):

                            self.assertEqual(
                                rb_circs[seed][circ_index].name,
                                'rb_length_%d_seed_%d' % (
                                    circ_index, seed),
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
                                'Error: incorrect circuit name')

                            self.verify_circuit(rb_circs[seed][circ_index],
                                                nq, rb_opts,
                                                vec_len, result[seed], shots)
                            self.verify_circuit(rb_original_circs[seed][circ_index],
                                                nq, rb_opts,
                                                vec_len, result_original[seed], shots)
                            #self.verify_circuit(rb_interleaved_circs[seed][circ_index],
                            #                    nq, rb_opts,
                            #                    2*vec_len, result_interleaved[seed], shots)

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
