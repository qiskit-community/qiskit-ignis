# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Run through RB for different qubit numbers to check that it's working
and that it returns the identity
"""

import unittest
import random
import qiskit
import qiskit_ignis.randomized_benchmarking.standard_rb.randomizedbenchmarking as rb

class TestRB(unittest.TestCase):
    """ The test class """

    @staticmethod
    def choose_length_vector():
        '''
        Randomize a valid field for rb_opts['length_vector']
        This function returns a vector of length between 1 to 4
        The first element is between 1 to 3,
        and every subsequent element is greater than the previous one
        with difference between 1 to 3.
        :return: the length vector
        '''

        res = []
        last_elem = 0
        for _ in range(random.randint(1, 4)):
            new_elem = last_elem + random.randint(1, 3)
            res.append(new_elem)
            last_elem = new_elem

        return res

    @staticmethod
    def choose_pattern(pattern_type, nq):
        '''
        Choose a valid field for rb_opts['rb_pattern']
        :param pattern_type: a number between 0 and 2.
                             0 - a list of all qubits, for nq=5 it is [1, 2, 3, 4, 5]
                             1 - a list of lists of single qubits, for nq=5
                                 it is [[1], [2], [3], [4], [5]]
                             2 - randomly choose a pattern which is a list of two lists,
                                for example for nq=5 it can be [[4, 1, 2], [5, 3]]
        :param nq: number of qubits
        :return: the pattern or None
                 Returns None if the pattern type is not relevant to the number of qubits,
                 i.e,, one of two cases:
                 pattern_type = 1 and nq = 1, which implies [[1]]
                 pattern_type = 2 and nq <= 2: - for nq=1 this is impossible
                                               - for nq=2 this implies [[1], [2]],
                                                 which is already tested when pattern_type = 1
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
    def choose_multiplier(len_pattern):
        '''
        Randomize a valid field for rb_opts['length_multiplier']
        The result can be either an integer or a vector of integers
        If it is a vector then its length must be equal to the number of patterns
        :param len_pattern: number of patterns
        :return: the length multiplier
        '''
        is_length_multipler_constant = random.randint(0, 1)
        if is_length_multipler_constant == 0:
            res = random.randint(1, 3)
        else:
            res = []
            for _ in range(len_pattern):
                res.append(random.randint(1, 3))

        return res

    def verify_circuit(self, circ, rb_opts, vec_len, result, shots):
        '''
        For a single sequence, verifies that it meets the requirements:
        - Executing it on the ground state ends up in the ground state
        - It has the correct number of Cliffords
        - It fulfills the pattern, as specified by rb_patterns and length_multiplier
        :param circ: the sequence to check
        :param rb_opts: the specification that generated the set of sequences which includes circ
        :param vec_len: the expected length vector of circ (one of rb_opts['length_vector']
        :param result: the output of the simulator
                       when executing all the sequences on the ground state
        :param shots: the number of shots in the simulator execution
        '''

        ops = circ.data
        op_index = 0
        for _ in range(vec_len):   # for each cycle (the sequence should consist of vec_len cycles)
            for pat_index in range(len(rb_opts['rb_pattern'])):
                # for each component of the pattern...
                for _ in range(rb_opts['length_multiplier'][pat_index]):  # for each Clifford...
                    while ops[op_index].name != 'barrier':  # for each basis gate...
                        # Verify that the gate acts on the correct qubits
                        # This happens if the sequence is composed of the correct sub-sequences,
                        # as specified by vec_len and rb_opts
                        self.assertTrue(all(x[1] in rb_opts['rb_pattern'][pat_index] \
                                            for x in ops[op_index].qargs),
                                        "Error: operation acts on incorrect qubits")
                        op_index += 1
                    op_index += 1  # increment because of the barrier gate

        # check if the ground state returns
        self.assertEqual(result.
                         get_counts(circ)['{0:b}'.format(0).zfill(rb_opts['n_qubits'])], shots,
                         "Error: %d qubit RB does not return the \
                         ground state back to the ground state" % rb_opts['n_qubits'])

    def test_rb(self):
        """ Main function of the test """

        # Load simulator
        backend = qiskit.Aer.get_backend('qasm_simulator')

        # Test up to 2 qubits
        nq_list = [1, 2]

        for nq in nq_list:

            print("Testing %d qubit RB"%nq)

            for pattern_type in range(3):
                # See documentation of choose_pattern for the meaning of the different pattern types

                rb_opts = {}
                rb_opts['nseeds'] = random.randint(1, 3)
                rb_opts['n_qubits'] = nq

                rb_opts['length_vector'] = self.choose_length_vector()
                rb_opts['rb_pattern'] = self.choose_pattern(pattern_type, nq)
                if rb_opts['rb_pattern'] is None:   # if the pattern type is not relevant for nq
                    continue
                rb_opts['length_multiplier'] = self.choose_multiplier(len(rb_opts['rb_pattern']))

                # Generate the sequences
                try:
                    rb_circs, _, rb_opts = rb.randomized_benchmarking_seq(rb_opts)
                except OSError:
                    print('Skipping tests for ' + str(nq) + ' qubits because tables are missing')
                    continue

                # Perform an ideal execution on the generated sequences
                basis_gates = 'u1,u2,u3,cx' # use U, CX for now
                shots = 100
                result = qiskit.execute(rb_circs, backend=backend,
                                        basis_gates=basis_gates, shots=shots).result()

                # Verify the generated sequences
                circ_index = 0
                for seed in range(rb_opts['nseeds']):
                    for vec_len in rb_opts['length_vector']:
                        self.assertEqual(rb_circs[circ_index].name,
                                         'rb_seed_' + str(seed) + '_length_' + str(vec_len),
                                         'Error: incorrect circuit name')
                        self.verify_circuit(rb_circs[circ_index], rb_opts, vec_len, result, shots)
                        circ_index += 1

                self.assertEqual(circ_index, len(rb_circs), "Error: additional circuits exist")


if __name__ == '__main__':
    unittest.main()
