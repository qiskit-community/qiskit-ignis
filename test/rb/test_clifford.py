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
Test Clifford functions:
- Generating Clifford tables on 1 and 2 qubits: clifford_utils.clifford1_table
  and clifford_utils.clifford2_table
- Generating a pseudo-random Clifford (using the tables):
  clifford_utils.random_clifford
- Inverting a Clifford: clifford_utils.find_inverse_clifford_circuit
"""

import unittest
import random
import os
import numpy as np

# Import the clifford_utils functions
from qiskit.ignis.verification.randomized_benchmarking \
    import CliffordUtils as clutils


class TestClifford(unittest.TestCase):
    """
        Test Clifford functions
    """
    def setUp(self):
        """
            setUp and global parameters
        """
        self.number_of_tests = 20  # number of pseudo-random seeds
        self.max_nq = 2  # maximal number of qubits to check
        self.clutils = clutils()

    def test_tables(self):
        """
            test: generating the tables for 1 and 2 qubits
        """
        test_tables_content = []
        test_tables_content.append(
            "test: generating the clifford group table for 1 qubit:\n")
        clifford1 = self.clutils.clifford1_gates_table()
        test_tables_content.append(str(len(clifford1)) + '\n')
        test_tables_content.append(str(sorted(clifford1.values())) + '\n')
        test_tables_content.append(
            "-------------------------------------------------------\n")

        test_tables_content.append(
            "test: generating the clifford group table for 2 qubits:\n")
        clifford2 = self.clutils.clifford2_gates_table()
        test_tables_content.append(str(len(clifford2)) + '\n')
        test_tables_content.append(str(sorted(clifford2.values())) + '\n')
        expected_file_path = os.path.join(
            os.path.dirname(__file__),
            'test_tables_expected.txt')
        with open(expected_file_path, 'r') as fd:
            expected_file_content = fd.readlines()
        self.assertEqual(expected_file_content, test_tables_content,
                         "Error: tables on 1 and 2 qubits are not the same")

    def test_random_and_inverse(self):
        """
            test: generating a pseudo-random Clifford using tables
            and computing its inverse
        """
        clifford_tables = [[]]*self.max_nq
        clifford_tables[0] = self.clutils.clifford1_gates_table()
        clifford_tables[1] = self.clutils.clifford2_gates_table()
        test_random_file_content = []
        # test: generating a pseudo-random Clifford using tables -
        # 1&2 qubits and computing its inverse
        for nq in range(1, 1+self.max_nq):
            for i in range(0, self.number_of_tests):
                my_seed = i
                np.random.seed(my_seed)
                random.seed(my_seed)
                test_random_file_content.append(
                    "test: generating a pseudo-random clifford using the "
                    "tables - %d qubit - seed=%d:\n" % (nq, my_seed))
                cliff_nq = self.clutils.random_gates(nq)
                test_random_file_content.append(str(cliff_nq) + '\n')
                test_random_file_content.append(
                    "test: inverting a pseudo-random clifford using the "
                    "tables - %d qubit - seed=%d:\n" % (nq, my_seed))
                inv_cliff_nq = self.clutils.find_inverse_gates(
                    nq, cliff_nq)
                test_random_file_content.append(str(inv_cliff_nq) + '\n')
                test_random_file_content.append(
                    "-----------------------------------------------------"
                    "--\n")

        expected_file_path = os.path.join(
            os.path.dirname(__file__),
            'test_random_expected.txt')
        with open(expected_file_path, 'r') as fd:
            expected_file_content = fd.readlines()
        self.assertEqual(expected_file_content, test_random_file_content,
                         "Error: random and/or inverse cliffords are not "
                         "the same")


if __name__ == '__main__':
    unittest.main()
