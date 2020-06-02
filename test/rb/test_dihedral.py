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
- Generating CNOT-dihedral group tables on 1 and 2 qubits:
  dihedral_utils.cnot_dihedral_tables
"""

import unittest

# Import the dihedral_utils functions
from qiskit.ignis.verification.randomized_benchmarking \
    import DihedralUtils as dutils


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
        self.table_size = [0, 16, 6144]

    def test_dihedral_tables(self):
        """
            test: generating the tables for 1 and 2 qubits
        """
        for qubit_num in range(1, 1 + self.max_nq):
            print('test: generating the cnot-dihedral group table - %d qubit'
                  % qubit_num)
            test_dihedral_tables = self.dutils.cnot_dihedral_tables(qubit_num)
            test_dihedral_tables = dict(sorted(test_dihedral_tables.
                                               items()))
            len_table = len(test_dihedral_tables)
            print("length:", len(test_dihedral_tables))

            self.assertEqual(len_table, self.table_size[qubit_num],
                             'Error: table on %d qubit does not contain '
                             'the expected number of elements' % qubit_num)


if __name__ == '__main__':
    unittest.main()
