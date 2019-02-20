# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Test Clifford functions:
- Generating Clifford tables on 1 and 2 qubits: clifford_utils.clifford1_table and clifford_utils.clifford2_table
- Generating a pseudo-random Clifford (using the tables): clifford_utils.random_clifford
- Inverting a Clifford: clifford_utils.find_inverse_clifford_circuit
"""

import unittest
import filecmp
import random
import os
import tempfile
import numpy as np

# Import the clifford_utils functions
from  qiskit.ignis.randomized_benchmarking import clifford_utils as clutils

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

    def test_tables(self):
        """
            test: generating the tables for 1 and 2 qubits
        """
        test_tables_fd, test_tables_file_path = tempfile.mkstemp()
        self.addCleanup(os.remove, test_tables_file_path)
        with os.fdopen(test_tables_fd, mode='w') as test_tables_file:
            test_tables_file.write(
                "test: generating the clifford group table for 1 qubit:\n")
            clifford1 = clutils.clifford1_gates_table()
            test_tables_file.write(str(len(clifford1)))
            test_tables_file.write("\n")
            test_tables_file.write(str(sorted(clifford1.values())))
            test_tables_file.write("\n")
            test_tables_file.write(
                "-------------------------------------------------------\n")

            test_tables_file.write(
                "test: generating the clifford group table for 2 qubits:\n")
            clifford2 = clutils.clifford2_gates_table()
            test_tables_file.write(str(len(clifford2)))
            test_tables_file.write("\n")
            test_tables_file.write(str(sorted(clifford2.values())))
            test_tables_file.write("\n")
        expected_file_path = os.path.join(
            os.path.dirname(__file__),
            'test_tables_expected.txt')
        self.assertTrue(
            filecmp.cmp(test_tables_file_path, expected_file_path),
                        "Error: tables on 1 and 2 qubits are not the same")

    def test_random_and_inverse(self):
        """
            test: generating a pseudo-random Clifford using tables
            and computing its inverse
        """
        clifford_tables = [[]]*self.max_nq
        clifford_tables[0] = clutils.clifford1_gates_table()
        clifford_tables[1] = clutils.clifford2_gates_table()
        test_random_fd, test_random_file_path = tempfile.mkstemp()
        self.addCleanup(os.remove, test_random_file_path)
        with os.fdopen(test_random_fd, mode='w') as test_random_file:

            # test: generating a pseudo-random Clifford using tables - 1&2 qubits
            # and computing its inverse
            for nq in range(1, 1+self.max_nq):
                for i in range(0, self.number_of_tests):
                    my_seed = i
                    np.random.seed(my_seed)
                    random.seed(my_seed)
                    test_random_file.write(
                        "test: generating a pseudo-random clifford using the tables "
                        "- %d qubit - seed=%d:\n" %(nq, my_seed))
                    cliff_nq = clutils.random_clifford_gates(nq)
                    test_random_file.write(str(cliff_nq))
                    test_random_file.write("\n")
                    test_random_file.write(
                        "test: inverting a pseudo-random clifford using the tables "
                        "- %d qubit - seed=%d:\n" %(nq, my_seed))
                    inv_cliff_nq = clutils.find_inverse_clifford_gates(nq,cliff_nq)

                    test_random_file.write(str(inv_cliff_nq))
                    test_random_file.write("\n")
                    test_random_file.write(
                        "-----------------------------------------------------"
                        "--\n")

        expected_file_path = os.path.join(
            os.path.dirname(__file__),
            'test_random_expected.txt')
        self.assertTrue(
            filecmp.cmp(test_random_file_path, expected_file_path),
                        "Error: random and/or inverse cliffords are not the same")


if __name__ == '__main__':
    unittest.main()
