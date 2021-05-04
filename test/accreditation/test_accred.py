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
import numpy as np
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
            test_1.AppendResults(a, b, c)
            
        (counts, bnd, conf) = test_1.FullAccreditation(0.95)
            
        self.assertEqual(test_1._Nruns,
                         test_1._Nacc,
                         "Error: Ideal outcomes not passing accred")
        
        theta = np.sqrt(np.log(2/(1-conf))/(2*len(all_postp_list)))     
        bound =1.7/len(all_postp_list[0])
        bound = bound/(1.0-theta)
        
        self.assertAlmostEqual(bound,
                               bnd,
                               msg="Error: Ideal outcomes not giving correct bound")
        
        # noisy results
        with open(os.path.join(os.path.dirname(__file__),
                               'accred_noisy_results.json'), "r") as saved_file:
            noisy_results = json.load(saved_file)
        all_strings = noisy_results['all_strings']
        all_postp_list = noisy_results['all_postp_list']
        all_v_zero = noisy_results['all_v_zero']
        confidence = noisy_results['confidence']
        accred_full = noisy_results['accred_full']
        accred_mean = noisy_results['accred_mean']
        
        test_1 = accred.AccreditationFitter()
        for a, b, c in zip(all_strings, all_postp_list, all_v_zero):
            test_1.AppendStrings(a, b, c)
            
        accred_full_test = test_1.FullAccreditation(confidence) 
        accred_mean_test = test_1.MeanAccreditation(confidence)
        self.assertEqual(accred_full_test[1],
                         accred_full[1],
                         "Error: Noisy outcomes fail full accred")
        
        self.assertEqual(accred_mean[1],
                         accred_mean_test[1],
                         "Error: Noisy outcomes fail mean accred")
        

if __name__ == '__main__':
    unittest.main()
