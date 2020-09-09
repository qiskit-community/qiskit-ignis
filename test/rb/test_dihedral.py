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
Tests for CNOT-dihedral functions
"""

import unittest

import qiskit
# Import the dihedral_utils functions
from qiskit.ignis.verification.randomized_benchmarking \
    import CNOTDihedral, random_cnotdihedral


class TestCNOTDihedral(unittest.TestCase):
    """
        Test CNOT-dihedral functions
    """
    def test_1_qubit_identities(self):
        """Tests identities for 1-qubit gates"""
        # T*X*T = X
        elem1 = CNOTDihedral(1)
        elem1.phase(1, 0)
        elem1.flip(0)
        elem1.phase(1, 0)
        elem2 = CNOTDihedral(1)
        elem2.flip(0)
        self.assertEqual(elem1, elem2,
                         'Error: 1-qubit identity does not hold')

        # X*T*X = Tdg
        elem1 = CNOTDihedral(1)
        elem1.flip(0)
        elem1.phase(1, 0)
        elem1.flip(0)
        elem2 = CNOTDihedral(1)
        elem2.phase(7, 0)
        self.assertEqual(elem1, elem2,
                         'Error: 1-qubit identity does not hold')

        # T*X*Tdg = S*X
        elem1 = CNOTDihedral(1)
        elem1.phase(1, 0)
        elem1.flip(0)
        elem1.phase(7, 0)
        elem2 = CNOTDihedral(1)
        elem2.phase(2, 0)
        elem2.flip(0)
        self.assertEqual(elem1, elem2,
                         'Error: 1-qubit identity does not hold')

    def test_2_qubit_identities(self):
        """Tests identities for 2-qubit gates"""
        # CX01 * CX10 * CX01 = CX10 * CX01 * CX10
        elem1 = CNOTDihedral(2)
        elem1.cnot(0, 1)
        elem1.cnot(1, 0)
        elem1.cnot(0, 1)
        elem2 = CNOTDihedral(2)
        elem2.cnot(1, 0)
        elem2.cnot(0, 1)
        elem2.cnot(1, 0)
        self.assertEqual(elem1, elem2,
                         'Error: 2-qubit SWAP identity does not hold')

        # CS01 = CS10 (symmetric)
        elem1 = CNOTDihedral(2)
        elem1.phase(1, 0)
        elem1.phase(1, 1)
        elem1.cnot(0, 1)
        elem1.phase(7, 1)
        elem1.cnot(0, 1)
        elem2 = CNOTDihedral(2)
        elem2.phase(1, 1)
        elem2.phase(1, 0)
        elem2.cnot(1, 0)
        elem2.phase(7, 0)
        elem2.cnot(1, 0)
        self.assertEqual(elem1, elem2,
                         'Error: 2-qubit CS identity does not hold')

        # TI*CS*TdgI = CS"
        elem3 = CNOTDihedral(2)
        elem3.phase(1, 0)
        elem3.phase(1, 0)
        elem3.phase(1, 1)
        elem3.cnot(0, 1)
        elem3.phase(7, 1)
        elem3.cnot(0, 1)
        elem3.phase(7, 0)
        self.assertEqual(elem1, elem3,
                         'Error: 2-qubit CS identity does not hold')

        # IT*CS*ITdg = CS
        elem4 = CNOTDihedral(2)
        elem4.phase(1, 1)
        elem4.phase(1, 0)
        elem4.phase(1, 1)
        elem4.cnot(0, 1)
        elem4.phase(7, 1)
        elem4.cnot(0, 1)
        elem4.phase(7, 1)
        self.assertEqual(elem1, elem4,
                         'Error: 2-qubit CS identity does not hold')

        # XX*CS*XX*SS = CS
        elem5 = CNOTDihedral(2)
        elem5.flip(0)
        elem5.flip(1)
        elem5.phase(1, 0)
        elem5.phase(1, 1)
        elem5.cnot(0, 1)
        elem5.phase(7, 1)
        elem5.cnot(0, 1)
        elem5.flip(0)
        elem5.flip(1)
        elem5.phase(2, 0)
        elem5.phase(2, 1)
        self.assertEqual(elem1, elem5,
                         'Error: 2-qubit CS identity does not hold')

        # CSdg01 = CSdg10 (symmetric)
        elem1 = CNOTDihedral(2)
        elem1.phase(7, 0)
        elem1.phase(7, 1)
        elem1.cnot(0, 1)
        elem1.phase(1, 1)
        elem1.cnot(0, 1)
        elem2 = CNOTDihedral(2)
        elem2.phase(7, 1)
        elem2.phase(7, 0)
        elem2.cnot(1, 0)
        elem2.phase(1, 0)
        elem2.cnot(1, 0)
        self.assertEqual(elem1, elem2,
                         'Error: 2-qubit CSdg identity does not hold')

        # XI*CS*XI*ISdg = CSdg
        elem3 = CNOTDihedral(2)
        elem3.flip(0)
        elem3.phase(1, 0)
        elem3.phase(1, 1)
        elem3.cnot(0, 1)
        elem3.phase(7, 1)
        elem3.cnot(0, 1)
        elem3.flip(0)
        elem3.phase(6, 1)
        self.assertEqual(elem1, elem3,
                         'Error: 2-qubit CSdg identity does not hold')

        # IX*CS*IX*SdgI = CSdg
        elem4 = CNOTDihedral(2)
        elem4.flip(1)
        elem4.phase(1, 0)
        elem4.phase(1, 1)
        elem4.cnot(0, 1)
        elem4.phase(7, 1)
        elem4.cnot(0, 1)
        elem4.flip(1)
        elem4.phase(6, 0)
        self.assertEqual(elem1, elem4,
                         'Error: 2-qubit CSdg identity does not hold')

        # relations for CZ
        elem1 = CNOTDihedral(2)
        elem1.phase(1, 0)
        elem1.phase(1, 1)
        elem1.cnot(0, 1)
        elem1.phase(7, 1)
        elem1.cnot(0, 1)
        elem1.phase(1, 0)
        elem1.phase(1, 1)
        elem1.cnot(0, 1)
        elem1.phase(7, 1)
        elem1.cnot(0, 1)

        elem2 = CNOTDihedral(2)
        elem2.phase(7, 0)
        elem2.phase(7, 1)
        elem2.cnot(0, 1)
        elem2.phase(1, 1)
        elem2.cnot(0, 1)
        elem2.phase(7, 0)
        elem2.phase(7, 1)
        elem2.cnot(0, 1)
        elem2.phase(1, 1)
        elem2.cnot(0, 1)

        elem3 = CNOTDihedral(2)
        elem3.phase(1, 1)
        elem3.phase(1, 0)
        elem3.cnot(1, 0)
        elem3.phase(7, 0)
        elem3.cnot(1, 0)
        elem3.phase(1, 1)
        elem3.phase(1, 0)
        elem3.cnot(1, 0)
        elem3.phase(7, 0)
        elem3.cnot(1, 0)

        elem4 = CNOTDihedral(2)
        elem4.phase(7, 1)
        elem4.phase(7, 0)
        elem4.cnot(1, 0)
        elem4.phase(1, 0)
        elem4.cnot(1, 0)
        elem4.phase(7, 1)
        elem4.phase(7, 0)
        elem4.cnot(1, 0)
        elem4.phase(1, 0)
        elem4.cnot(1, 0)

        # CZ = TdgTdg * CX * T^2I * CX * TdgTdg
        elem5 = CNOTDihedral(2)
        elem5.phase(7, 1)
        elem5.phase(7, 0)
        elem5.cnot(1, 0)
        elem5.phase(2, 0)
        elem5.cnot(1, 0)
        elem5.phase(7, 1)
        elem5.phase(7, 0)

        self.assertEqual(elem1, elem2,
                         'Error: 2-qubit CZ identity does not hold')
        self.assertEqual(elem1, elem3,
                         'Error: 2-qubit CZ identity does not hold')
        self.assertEqual(elem1, elem4,
                         'Error: 2-qubit CZ identity does not hold')
        self.assertEqual(elem1, elem5,
                         'Error: 2-qubit CZ identity does not hold')

        # relations for CX
        elem1 = CNOTDihedral(2)
        elem1.cnot(0, 1)

        # TI*CX*TdgI = CX
        elem2 = CNOTDihedral(2)
        elem2.phase(1, 0)
        elem2.cnot(0, 1)
        elem2.phase(7, 0)

        # IZ*CX*ZZ = CX
        elem3 = CNOTDihedral(2)
        elem3.phase(4, 1)
        elem3.cnot(0, 1)
        elem3.phase(4, 0)
        elem3.phase(4, 1)

        # IX*CX*IX = CX
        elem4 = CNOTDihedral(2)
        elem4.flip(1)
        elem4.cnot(0, 1)
        elem4.flip(1)

        # XI*CX*XX = CX
        elem5 = CNOTDihedral(2)
        elem5.flip(0)
        elem5.cnot(0, 1)
        elem5.flip(0)
        elem5.flip(1)

        self.assertEqual(elem1, elem2,
                         'Error: 2-qubit CX identity does not hold')
        self.assertEqual(elem1, elem3,
                         'Error: 2-qubit CX identity does not hold')
        self.assertEqual(elem1, elem4,
                         'Error: 2-qubit CX identity does not hold')
        self.assertEqual(elem1, elem5,
                         'Error: 2-qubit CX identity does not hold')

        # IT*CX01*CX10*TdgI = CX01*CX10
        elem1 = CNOTDihedral(2)
        elem1.cnot(0, 1)
        elem1.cnot(1, 0)

        elem2 = CNOTDihedral(2)
        elem2.phase(1, 1)
        elem2.cnot(0, 1)
        elem2.cnot(1, 0)
        elem2.phase(7, 0)
        self.assertEqual(elem1, elem2,
                         'Error: 2-qubit CX01*CX10 identity does not hold')

    def test_dihedral_random_decompose(self):
        """
        Test that random elements are CNOTDihedral
        and to_circuit and from_circuit methods
        (where num_qubits < 3)
        """
        for qubit_num in range(1, 5):
            for nseed in range(20):
                elem = random_cnotdihedral(qubit_num, seed=nseed)
                self.assertTrue(elem,
                                'Error: random element is not CNOTDihedral')
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

                    test_gates = elem.to_instruction()
                    self.assertIsInstance(test_gates, qiskit.circuit.Gate,
                                          'Error: cannot decompose a random '
                                          'CNOTDihedral element to a Gate')
                    self.assertEqual(test_gates.num_qubits, test_circ.num_qubits,
                                     'Error: wrong num_qubits in decomposed gates')
                    new_elem1 = CNOTDihedral(qubit_num)
                    test_elem1 = new_elem1.from_circuit(test_gates)
                    # Test of to_instruction and from_circuit methods
                    self.assertEqual(elem, test_elem1,
                                     'Error: decomposed gates are not equal '
                                     'to the original gates')

    def test_compose_method(self):
        """Test compose method"""
        samples = 10
        nseed = 111
        for qubit_num in range(1, 3):
            for i in range(samples):
                elem1 = random_cnotdihedral(qubit_num, seed=nseed + i)
                elem2 = random_cnotdihedral(qubit_num, seed=nseed + samples + i)
                circ1 = elem1.to_circuit()
                circ2 = elem2.to_circuit()
                value = elem1.compose(elem2)
                target = CNOTDihedral(qubit_num)
                target = target.from_circuit(circ1.extend(circ2))
                self.assertEqual(target, value,
                                 'Error: composed circuit is not the same')

    def test_dot_method(self):
        """Test dot method"""
        samples = 10
        nseed = 222
        for qubit_num in range(1, 3):
            for i in range(samples):
                elem1 = random_cnotdihedral(qubit_num, seed=nseed + i)
                elem2 = random_cnotdihedral(qubit_num, seed=nseed + samples + i)
                circ1 = elem1.to_circuit()
                circ2 = elem2.to_circuit()
                value = elem1.dot(elem2)
                target = CNOTDihedral(qubit_num)
                target = target.from_circuit(circ2.extend(circ1))
                self.assertEqual(target, value,
                                 'Error: composed circuit is not the same')


if __name__ == '__main__':
    unittest.main()
