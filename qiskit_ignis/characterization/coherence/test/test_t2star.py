# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Test measurement of T2*
"""

import copy
import unittest
import random
import numpy as np
from matplotlib import pyplot as plt
import qiskit
from qiskit.providers.aer.noise.errors.standard_errors import phase_damping_error, mixed_unitary_error
from qiskit.providers.aer.noise import NoiseModel
from qiskit_ignis.characterization.coherence.generators.t2star import t2starexp_generate_circuits_bygates as t2expgen
from qiskit_ignis.characterization.coherence.generators.t2star import t2starosc_generate_circuits_bygates as t2oscgen
from qiskit_ignis.characterization.coherence.fitters.t2starfitter import T2StarExpFitter, T2StarOscFitter

class TestT2Star(unittest.TestCase):
    """
    Test measurement of T2*
    """

    def test_t2star(self):
        """
        Run the simulator with phase damping noise.
        Then verify that the calculated T2star matches the phase damping parameter.
        """

        # Setting parameters

        # 25 numbers ranging from 100 to 1000, linearly spaced
        num_of_gates = (np.linspace(100, 1000, 25)).astype(int)
        gate_time = 0.1
        num_of_qubits = 1
        qubit = 0

        expected_t2 = random.randint(20, 80)
        p = 1 - np.exp(-2*gate_time/expected_t2)
        error = phase_damping_error(p)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, 'id')

        backend = qiskit.Aer.get_backend('qasm_simulator')
        shots = 1500


        # Estimating T2* via an exponential function

        circs, xdata = t2expgen(num_of_gates, gate_time, num_of_qubits, qubit)
        backend_result = qiskit.execute(circs, backend,
                                        shots=shots, noise_model=noise_model).result()

        initial_t2 = expected_t2 + 20*(-1)**random.randint(0, 1)
        initial_a = 0.5 + 0.5*(-1)**random.randint(0, 1)
        initial_c = 0.5 + 0.5*(-1)**random.randint(0, 1)

        fit = T2StarExpFitter(backend_result, shots, xdata, num_of_qubits, qubit,
                              fit_p0=[initial_a, initial_t2, initial_c],
                              fit_bounds=([-0.5, expected_t2-30, -0.5], [1.5, expected_t2+30, 1.5]))

        fit.plot_coherence()

        self.assertAlmostEqual(fit.time, expected_t2, delta=20,
                               msg='Calculated T2 is inaccurate')
        self.assertTrue(fit.time_err < 30,
                       'Confidence in T2 calculation is too low: ' + str(fit.time_err))


        # Estimate T2* via an oscilliator function

        circs_osc, xdata, omega = t2oscgen(num_of_gates, gate_time, num_of_qubits, qubit)

        backend_result = qiskit.execute(circs_osc, backend,
                                        shots=shots, noise_model=noise_model).result()

        initial_a = 0.5 + 0.5*(-1)**random.randint(0, 1)
        initial_c = 0.5 + 0.5*(-1)**random.randint(0, 1)
        initial_f = omega + 0.01*(-1)**random.randint(0, 1)
        initial_phi = (np.pi/20)*(-1)**random.randint(0, 1)

        fit = T2StarOscFitter(backend_result, shots, xdata, num_of_qubits, qubit,
                              fit_p0=[initial_a, initial_t2, initial_f, initial_phi, initial_c],
                              fit_bounds=([-0.5, expected_t2-30, omega-0.02, -np.pi/10, -0.5],
                                          [1.5, expected_t2+30, omega+0.02, np.pi/10, 1.5]))

        fit.plot_coherence()

        self.assertAlmostEqual(fit.time, expected_t2, delta=20,
                               msg='Calculated T2 is inaccurate')
        self.assertTrue(fit.time_err < 30,
                       'Confidence in T2 calculation is too low: ' + str(fit.time_err))


        # TODO: add SPAM


if __name__ == '__main__':
    unittest.main()
