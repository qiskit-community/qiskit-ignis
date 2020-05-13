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
from qiskit.ignis.verification.randomized_benchmarking \
    import CNOTDihedral, append_circuit, decompose_cnotdihedral, \
    random_cnotdihedral


class TestCNOTDihedral(unittest.TestCase):
    """
        Test CNOT-dihedral functions
    """
    def setUp(self):
        """
            setUp and global parameters
        """
        self.number_of_tests = 20  # number of pseudo-random seeds
        self.max_qubit_num = 2  # maximal number of qubits to check
        self.dutils = dutils()
        self.table_size = [0, 16, 6144]

    def test_dihedral_tables(self):
        """
            test: generating the tables for 1 and 2 qubits
        """
        for qubit_num in range(1, 1 + self.max_qubit_num):
            print('test: generating the cnot-dihedral group table - %d qubit'
                  % qubit_num)
            test_dihedral_tables = self.dutils.cnot_dihedral_tables(qubit_num)
            test_dihedral_tables_items = dict(sorted(test_dihedral_tables.
                                                     items()))
            len_table = len(test_dihedral_tables_items)
            print("length:", len(test_dihedral_tables_items))

            self.assertEqual(len_table, self.table_size[qubit_num],
                             'Error: table on %d qubit does not contain '
                             'the expected number of elements' % qubit_num)

            # Test of CNOT-Dihedral circuit decomposition
            for _, elem in test_dihedral_tables.items():
                test_circ = decompose_cnotdihedral(elem[0])
                test_elem = CNOTDihedral(qubit_num)
                append_circuit(test_elem, test_circ)
                self.assertEqual(elem[0], test_elem,
                                 'Error: decomposed circuit is not equal '
                                 'to the original circuit')

        # Test that random elements are CNOTDihedral
        for qubit_num in range(1, 5):
            for nseed in range(20):
                elem = random_cnotdihedral(qubit_num, seed=nseed)
                self.assertTrue(elem,
                                'Error: random element is '
                                'not CNOTDihedral')
                if qubit_num < 3:
                    test_circ = elem.to_circuit()
                    self.assertTrue(test_circ,
                                    'Error: cannot decompose a random '
                                    'CNOTDihedral element to a circuit')
                    new_elem = CNOTDihedral(qubit_num)
                    test_elem = new_elem.from_circuit(test_circ)
                    # Test of to_circuit and from_circuit methods
                    self.assertEqual(elem, test_elem,
                                     'Error: decomposed circuit is not equal '
                                     'to the original circuit')


if __name__ == '__main__':
    unittest.main()
