# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Run through Quantum volume
"""
# uncomment after the base experiment class is merged

# import unittest
# from qiskit import Aer
#
# from qiskit.ignis.experiments.quantum_volume import QuantumVolumeExperiment
# from qiskit.providers.aer.noise import NoiseModel
# from qiskit.providers.aer.noise.errors.standard_errors import depolarizing_error
#
# SEED = 42
# BACKEND = Aer.get_backend('qasm_simulator')
#
#
# class TestQV(unittest.TestCase):
#     """ The test class """
#
#     def test_qv_circuits(self):
#         """ Test circuit generation """
#
#         # Qubit list
#         qubit_lists = [[0, 1, 2], [0, 1, 2, 4], [0, 1, 2, 4, 7]]
#         ntrials = 5
#
#         qv_exp = QuantumVolumeExperiment(qubits=qubit_lists, trials=ntrials)
#
#         self.assertEqual(len(qv_exp.circuits), ntrials * len(qubit_lists),
#                          "Error: Not enough circuits generated")
#         for i, _ in enumerate(qubit_lists):
#             self.assertEqual(qv_exp.circuits[i].data[0][0].num_qubits, len(qubit_lists[i]),
#                              "Error: number of qubits do not match the "
#                              "number specified in qubit lists")
#
#     def test_qv_fitter(self):
#         """ Test the fitter"""
#         qubit_lists = [[0, 1, 3], [0, 1, 3, 5], [0, 1, 3, 5, 7],
#                        [0, 1, 3, 5, 7, 10]]
#         ntrials = 5
#         shots = 2048
#
#         # define noise_model
#         noise_model = NoiseModel()
#         p1q = 0.002
#         p2q = 0.02
#         noise_model.add_all_qubit_quantum_error(depolarizing_error(p1q, 1), 'u2')
#         noise_model.add_all_qubit_quantum_error(depolarizing_error(2 * p1q, 1), 'u3')
#         noise_model.add_all_qubit_quantum_error(depolarizing_error(p2q, 2), 'cx')
#
#         qv_exp = QuantumVolumeExperiment(qubits=qubit_lists, trials=ntrials)
#         qv_exp.execute(BACKEND, shots=shots, noise_model=noise_model)
#         res = qv_exp.run_analysis()
#
#         qv_success_list = res._qv_success()
#         self.assertFalse(qv_success_list[0][0])
#
#
# if __name__ == '__main__':
#     unittest.main()
