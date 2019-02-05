# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Test measurement of T1
"""

import unittest
import random
import numpy as np
import qiskit
from qiskit.providers.aer.noise.errors.standard_errors import amplitude_damping_error
from qiskit.providers.aer.noise import NoiseModel
from characterization.coherence.basis.t1 import t1_generate_circuits as t1gen
from characterization.coherence.fitters.t1fitter import T1Fitter

class TestT1(unittest.TestCase):
    """
    Test measurement of T1
    """

    def test_t1(self):
        """
        Run the simulator with amplitude damping noise.
        Then verify that the calculated T1 matches the amplitude damping parameter.
        """

        # 10 numbers ranging from 100 to 1000, logarithmically spaced
        num_of_gates = (np.logspace(2, 3, 10)).astype(int)
        gate_time = 0.11
        num_of_qubits = 4
        qubit = random.randint(0, 3)

        circs = t1gen(num_of_gates, num_of_qubits, qubit)

        expected_t1 = random.randint(10, 100)
        gamma = 1 - np.exp(-gate_time/expected_t1)
        error = amplitude_damping_error(gamma)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, 'id')
        # TODO: Include SPAM errors

        backend = qiskit.Aer.get_backend('qasm_simulator')
        shots = 300
        backend_result = qiskit.execute(circs, backend,
                                        shots=shots, noise_model=noise_model).result()

        initial_t1 = expected_t1 + 20*(-1)**random.randint(0, 1)
        initial_a = 1 + 0.5*(-1)**random.randint(0, 1)
        initial_b = 0.5*(-1)**random.randint(0, 1)

        fit = T1Fitter(backend_result, shots, num_of_gates, gate_time, num_of_qubits, qubit,
                       fit_p0=[initial_a, initial_t1, initial_b],
                       fit_bounds=([0, expected_t1-30, -1], [2, expected_t1+30, 1]))

        self.assertAlmostEqual(fit.t1, expected_t1, delta=20,
                               msg='Calculated T1 is inaccurate')
        self.assertTrue(fit.t1_err < 30,
                        'Confidence in T1 calculation is too low: ' + str(fit.t1_err))

        fit.plot_coherence()


if __name__ == '__main__':
    unittest.main()
