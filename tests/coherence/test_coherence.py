# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=no-name-in-module

"""
Test coherence measurements
"""

import unittest
import numpy as np
import qiskit
from qiskit.providers.aer.noise.errors.standard_errors import \
                                    phase_damping_error
from qiskit.providers.aer.noise.errors.standard_errors import \
                                     amplitude_damping_error
from qiskit.providers.aer.noise import NoiseModel

from qiskit.ignis.characterization.coherence import \
     T2StarExpFitter, T2StarOscFitter, T1Fitter, T2Fitter

from qiskit.ignis.characterization.coherence import t1_circuits, \
                                t2_circuits, t2star_circuits


class TestT2Star(unittest.TestCase):
    """
    Test measurement of T2*
    """

    def test_t2star(self):
        """
        Run the simulator with phase damping noise.
        Then verify that the calculated T2star matches the phase damping
        parameter.
        """

        # Setting parameters

        num_of_gates = num_of_gates = np.append(
            (np.linspace(10, 150, 30)).astype(int),
            (np.linspace(160, 450, 20)).astype(int))
        gate_time = 0.1
        qubits = [0]

        expected_t2 = 10
        p = 1 - np.exp(-2*gate_time/expected_t2)
        error = phase_damping_error(p)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, 'id')

        backend = qiskit.Aer.get_backend('qasm_simulator')
        shots = 300
    
        # Estimating T2* via an exponential function
        circs, xdata, _ = t2star_circuits(num_of_gates, gate_time,
                                          qubits)
        backend_result = qiskit.execute(
            circs, backend, shots=shots,
            backend_options={'max_parallel_experiments': 0},
            noise_model=noise_model).result()

        initial_t2 = expected_t2
        initial_a = 0.5
        initial_c = 0.5

        fit = T2StarExpFitter(backend_result, shots, xdata,
                              qubits,
                              fit_p0=[initial_a, initial_t2, initial_c],
                              fit_bounds=([-0.5, 0, -0.5],
                                          [1.5, expected_t2*1.2, 1.5]))

        self.assertAlmostEqual(fit.time[0], expected_t2, delta=2,
                               msg='Calculated T2 is inaccurate')
        self.assertTrue(
            fit.time_err[0] < 2,
            'Confidence in T2 calculation is too low: ' + str(fit.time_err))

        # Estimate T2* via an oscilliator function
        circs_osc, xdata, omega = t2star_circuits(num_of_gates, gate_time,
                                                  qubits, 5)

        backend_result = qiskit.execute(
            circs_osc, backend,
            shots=shots,
            backend_options={'max_parallel_experiments': 0},
            noise_model=noise_model).result()

        initial_a = 0.5
        initial_c = 0.5
        initial_f = omega
        initial_phi = 0

        fit = T2StarOscFitter(backend_result, shots, xdata, qubits,
                              fit_p0=[initial_a, initial_t2, initial_f,
                                      initial_phi, initial_c],
                              fit_bounds=([-0.5, 0, omega-0.02, -np.pi, -0.5],
                                          [1.5, expected_t2*1.2, omega+0.02,
                                           np.pi, 1.5]))

        self.assertAlmostEqual(fit.time[0], expected_t2, delta=2,
                               msg='Calculated T2 is inaccurate')
        self.assertTrue(
            fit.time_err[0] < 2,
            'Confidence in T2 calculation is too low: ' + str(fit.time_err))

        # TODO: add SPAM


class TestT1(unittest.TestCase):
    """
    Test measurement of T1
    """

    def test_t1(self):
        """
        Run the simulator with amplitude damping noise.
        Then verify that the calculated T1 matches the amplitude damping
        parameter.
        """

        # 25 numbers ranging from 1 to 200, linearly spaced
        num_of_gates = (np.linspace(1, 200, 25)).astype(int)
        gate_time = 0.11
        qubits = [0]

        circs, xdata = t1_circuits(num_of_gates, gate_time, qubits)

        expected_t1 = 10
        gamma = 1 - np.exp(-gate_time/expected_t1)
        error = amplitude_damping_error(gamma)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, 'id')
        # TODO: Include SPAM errors

        backend = qiskit.Aer.get_backend('qasm_simulator')
        shots = 300
        backend_result = qiskit.execute(
            circs, backend,
            shots=shots,
            backend_options={'max_parallel_experiments': 0},
            noise_model=noise_model).result()

        initial_t1 = expected_t1
        initial_a = 1
        initial_c = 0

        fit = T1Fitter(backend_result, shots, xdata, qubits,
                       fit_p0=[initial_a, initial_t1, initial_c],
                       fit_bounds=([0, 0, -1], [2, expected_t1*1.2, 1]))

        self.assertAlmostEqual(fit.time[0], expected_t1, delta=2,
                               msg='Calculated T1 is inaccurate')
        self.assertTrue(
            fit.time_err[0] < 30,
            'Confidence in T1 calculation is too low: ' + str(fit.time_err))


class TestT2(unittest.TestCase):
    """
    Test measurement of T2
    """

    def test_t2(self):
        """
        Run the simulator with dephasing noise.
        Then verify that the calculated T2 matches the dephasing parameter.
        """

        # 35 numbers ranging from 1 to 300, linearly spaced
        num_of_gates = (np.linspace(1, 300, 35)).astype(int)
        gate_time = 0.11
        qubits = [0]

        circs, xdata = t2_circuits(num_of_gates, gate_time, qubits)

        expected_t2 = 20
        gamma = 1 - np.exp(-2*gate_time/expected_t2)
        error = phase_damping_error(gamma)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, 'id')
        # TODO: Include SPAM errors

        backend = qiskit.Aer.get_backend('qasm_simulator')
        shots = 300
        backend_result = qiskit.execute(
            circs, backend,
            shots=shots,
            backend_options={'max_parallel_experiments': 0},
            noise_model=noise_model).result()

        initial_t2 = expected_t2
        initial_a = 1
        initial_c = 0.5*(-1)

        fit = T2Fitter(backend_result, shots, xdata, qubits,
                       fit_p0=[initial_a, initial_t2, initial_c],
                       fit_bounds=([0, 0, -1], [2, expected_t2*1.2, 1]))

        self.assertAlmostEqual(fit.time[0], expected_t2, delta=4,
                               msg='Calculated T2 is inaccurate')
        self.assertTrue(
            fit.time_err[0] < 5,
            'Confidence in T2 calculation is too low: ' + str(fit.time_err))


if __name__ == '__main__':
    unittest.main()
