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
Test CNOT-dihedral functions:
- Generating CNOT-dihedral group tables on 1 and 2 qubits: ???
- Generating a pseudo-random group element (using the tables):
  dihedral_utils.random_gates
- Inverting an element: dihedral_utils.find_inverse_gates
"""

import unittest
import numpy as np

# Import the dihedral_utils functions
from qiskit.ignis.verification.randomized_benchmarking \
    import DihedralUtils as dutils

try:
    import cPickle as pickle
except ImportError:
    import pickle


class TestCNOTDihedral(unittest.TestCase):
    """
        Test CNOT-dihedral functions
    """
    def setUp(self):
        """
            setUp and global parameters
        """
        self.number_of_tests = 20  # number of pseudo-random seeds
        self.max_nq = 2  # maximal number of qubits to check
        self.dutils = dutils()

    def test_dihedral_tables(self):
        """
            test: generating the tables for 1 and 2 qubits
        """
        for nq in range(1, 1+self.max_nq):
            print('test: generating the cnot-dihedral group table - %d qubit'
                  % nq)
            test_dihedral_tables = self.dutils.cnot_dihedral_tables(nq)
            print("length:", len(test_dihedral_tables))

            picklefile = 'expect_cnot_dihedral_%d.pickle' % nq
            fo = open(picklefile, 'rb')
            expected_dihedral_tables = pickle.load(fo)
            fo.close()

            self.assertEqual(expected_dihedral_tables, test_dihedral_tables,
                             'Error: tables on %d qubit are not the same'
                             % nq)

    def test_random_and_inverse_dihedral(self):
        """
            test: generating a pseudo-random cnot-dihedral element
            using tables and computing its inverse
        """
        dihedral_tables = [[]]*self.max_nq
        dihedral_tables[0] = self.dutils.load_dihedral_table(
            'expect_cnot_dihedral_1.pickle')
        dihedral_tables[1] = self.dutils.load_dihedral_table(
            'expect_cnot_dihedral_2.pickle')

        test_random_dihedral = []
        # test: generating a pseudo-random cnot-dihedral element
        # using tables - 1&2 qubits and computing its inverse
        for nq in range(1, 1+self.max_nq):
            print("test: generating pseudo-random cnot-dihedral "
                  "elements and inverse using the tables - %d qubit" % nq)
            for i in range(0, self.number_of_tests):
                my_seed = i
                np.random.seed(my_seed)
                elem_nq = self.dutils.random_gates(nq)
                test_random_dihedral.append(elem_nq)
                inv_key = self.dutils.find_key(elem_nq, nq)
                inv_elem_nq = self.dutils.find_inverse_gates(
                    nq, dihedral_tables[nq - 1][inv_key])
                test_random_dihedral.append(inv_elem_nq)

        picklefile = 'expect_cnot_dihedral_random.pickle'
        fo = open(picklefile, 'rb')
        expected_random_dihedral = pickle.load(fo)
        fo.close()

        self.assertEqual(expected_random_dihedral, expected_random_dihedral,
                         "Error: random and/or inverse cliffords are not "
                         "the same")


if __name__ == '__main__':
    unittest.main()
