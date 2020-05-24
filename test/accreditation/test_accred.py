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

# pylint: disable=no-member,invalid-name

"""
Run through Accreditation
"""

import unittest
import os
import json
from qiskit.result.result import Result
# Import Qiskit classes
import qiskit.ignis.verification.accreditation as accred
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


class TestAccred(unittest.TestCase):
    """ The test class """

    def test_accred_circuits(self):
        """ Test circuit generation """
        seed_accreditation = 208723512
        n_qb = 4
        v = 10
        # Create a Quantum Register with n_qb qubits.
        q_reg = QuantumRegister(n_qb, 'q')
        # Create a Classical Register with n_qb bits.
        c_reg = ClassicalRegister(n_qb, 's')
        # Create a Quantum Circuit acting on the q register
        target_circuit = QuantumCircuit(q_reg, c_reg)

        # dummy circuit
        target_circuit.h(0)
        target_circuit.h(1)
        target_circuit.h(2)
        target_circuit.h(3)
        target_circuit.cz(0, 1)
        target_circuit.cz(0, 2)
        target_circuit.cz(0, 3)
        target_circuit.h(1)
        target_circuit.h(2)
        target_circuit.h(3)
        target_circuit.measure(q_reg, c_reg)
        # make trap circuits
        accredsys = accred.AccreditationCircuits(target_circuit, seed=seed_accreditation)
        circ_list, postp_list, v_zero = accredsys.generate_circuits(v)
        self.assertEqual(len(circ_list), v+1,
                         "Error: Not correct number of trap circuits")
        self.assertEqual(len(postp_list)*len(postp_list[0]), (v+1)*n_qb,
                         "Error: Not correct number of outcome bits")
        self.assertTrue((v+1) > v_zero > -1,
                        "Error: marked element outside of list of circuits")

    def test_accred_fitter(self):

        """ Test the fitter with some saved result data"""
        # ideal results
        with open(os.path.join(os.path.dirname(__file__),
                               'accred_ideal_results.json'), "r") as saved_file:
            ideal_results = json.load(saved_file)
        all_results = [Result.from_dict(result) for result in ideal_results['all_results']]
        all_postp_list = ideal_results['all_postp_list']
        all_v_zero = ideal_results['all_v_zero']
        test_1 = accred.AccreditationFitter()
        for a, b, c in zip(all_results, all_postp_list, all_v_zero):
            test_1.single_protocol_run(a, b, c)
            self.assertEqual(test_1.flag,
                             'accepted',
                             "Error: Ideal outcomes not passing accred")
        # noisy results
        with open(os.path.join(os.path.dirname(__file__),
                               'accred_noisy_results.json'), "r") as saved_file:
            noisy_results = json.load(saved_file)
        all_results = [Result.from_dict(result) for result in noisy_results['all_results']]
        all_postp_list = noisy_results['all_postp_list']
        all_v_zero = noisy_results['all_v_zero']
        all_acc = noisy_results['all_acc']
        test_1 = accred.AccreditationFitter()
        for a, b, c, d in zip(all_results, all_postp_list, all_v_zero, all_acc):
            test_1.single_protocol_run(a, b, c)
            self.assertEqual(test_1.flag,
                             d,
                             "Error: Noisy outcomes not correct accred")
        test_1.bound_variation_distance(noisy_results['theta'])
        bound = test_1.bound
        self.assertEqual(bound,
                         noisy_results['bound'],
                         "Error: Incorrect bound for noisy outcomes")


if __name__ == '__main__':
    unittest.main()
