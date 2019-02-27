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
                                     thermal_relaxation_error, \
                                     coherent_unitary_error

from qiskit.providers.aer.noise import NoiseModel

from qiskit.ignis.characterization.coherence import \
     T2StarFitter, T1Fitter, T2Fitter

from qiskit.ignis.characterization.coherence import t1_circuits, \
                                t2_circuits, t2star_circuits

from qiskit.ignis.characterization.hamiltonian import zz_circuits, ZZFitter


class TestT2Star(unittest.TestCase):
    """
    Test measurement of T2*
    """

    def test_t2star(self):
        """
        Run the simulator with thermal relaxation noise.
        Then verify that the calculated T2star matches the t2
        parameter.
        """

        # Setting parameters

        num_of_gates = num_of_gates = np.append(
            (np.linspace(10, 150, 30)).astype(int),
            (np.linspace(160, 450, 20)).astype(int))
        gate_time = 0.1
        qubits = [0]

        expected_t2 = 10
        error = thermal_relaxation_error(np.inf, expected_t2, gate_time, 0.5)
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

        fit = T2Fitter(backend_result, xdata,
                       qubits,
                       fit_p0=[initial_a, initial_t2, initial_c],
                       fit_bounds=([-0.5, 0, -0.5],
                                   [1.5, expected_t2*1.2, 1.5]),
                       circbasename='t2star')

        self.assertAlmostEqual(fit.time(qid=0), expected_t2, delta=2,
                               msg='Calculated T2 is inaccurate')
        self.assertTrue(
            fit.time_err(qid=0) < 2,
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

        fit = T2StarFitter(backend_result, xdata, qubits,
                           fit_p0=[initial_a, initial_t2, initial_f,
                                   initial_phi, initial_c],
                           fit_bounds=([-0.5, 0, omega-0.02, -np.pi, -0.5],
                                       [1.5, expected_t2*1.2, omega+0.02,
                                        np.pi, 1.5]))

        self.assertAlmostEqual(fit.time(qid=0), expected_t2, delta=2,
                               msg='Calculated T2 is inaccurate')
        self.assertTrue(
            fit.time_err(qid=0) < 2,
            'Confidence in T2 calculation is too low: ' + str(fit.time_err))

        # TODO: add SPAM


class TestT1(unittest.TestCase):
    """
    Test measurement of T1
    """

    def test_t1(self):
        """
        Run the simulator with thermal relaxatoin noise.
        Then verify that the calculated T1 matches the t1
        parameter.
        """

        # 25 numbers ranging from 1 to 200, linearly spaced
        num_of_gates = (np.linspace(1, 200, 25)).astype(int)
        gate_time = 0.11
        qubits = [0]

        circs, xdata = t1_circuits(num_of_gates, gate_time, qubits)

        expected_t1 = 10
        error = thermal_relaxation_error(expected_t1, 2*expected_t1, gate_time)
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

        fit = T1Fitter(backend_result, xdata, qubits,
                       fit_p0=[initial_a, initial_t1, initial_c],
                       fit_bounds=([0, 0, -1], [2, expected_t1*1.2, 1]))

        self.assertAlmostEqual(fit.time(qid=0), expected_t1, delta=2,
                               msg='Calculated T1 is inaccurate')
        self.assertTrue(
            fit.time_err(qid=0) < 30,
            'Confidence in T1 calculation is too low: ' + str(fit.time_err))


class TestT2(unittest.TestCase):
    """
    Test measurement of T2
    """

    def test_t2(self):
        """
        Run the simulator with thermal relaxation noise.
        Then verify that the calculated T2 matches the t2 parameter.
        """

        num_of_gates = (np.linspace(1, 30, 30)).astype(int)
        gate_time = 0.11
        qubits = [0]
        n_echos = 5
        alt_phase_echo = True

        circs, xdata = t2_circuits(num_of_gates, gate_time, qubits,
                                   n_echos, alt_phase_echo)

        expected_t2 = 20
        error = thermal_relaxation_error(np.inf, expected_t2, gate_time, 0.5)
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

        fit = T2Fitter(backend_result, xdata, qubits,
                       fit_p0=[initial_a, initial_t2, initial_c],
                       fit_bounds=([0, 0, -1], [2, expected_t2*1.2, 1]))

        self.assertAlmostEqual(fit.time(qid=0), expected_t2, delta=5,
                               msg='Calculated T2 is inaccurate')
        self.assertTrue(
            fit.time_err(qid=0) < 5,
            'Confidence in T2 calculation is too low: ' + str(fit.time_err))


class TestZZ(unittest.TestCase):
    """
    Test measurement of ZZ
    """

    def test_zz(self):
        """
        Run the simulator with unitary noise.
        Then verify that the calculated ZZ matches the zz parameter.
        """

        num_of_gates = np.arange(0, 60, 3)
        gate_time = 0.1
        qubits = [0]
        spectators = [1]

        # Generate experiments
        circs, xdata, osc_freq = zz_circuits(num_of_gates,
                                             gate_time, qubits,
                                             spectators, nosc=2)

        # Set the simulator with ZZ
        zz_expected = 0.1
        zz_unitary = np.eye(4, dtype=complex)
        zz_unitary[3, 3] = np.exp(1j*2*np.pi*zz_expected*gate_time)
        error = coherent_unitary_error(zz_unitary)
        noise_model = NoiseModel()
        noise_model.add_nonlocal_quantum_error(error, 'id', [0], [0, 1])

        # Run the simulator
        backend = qiskit.Aer.get_backend('qasm_simulator')
        shots = 300
        # For demonstration purposes split the execution into two jobs
        backend_result = qiskit.execute(circs, backend,
                                        shots=shots,
                                        noise_model=noise_model).result()

        initial_a = 1
        initial_c = 0
        initial_f = osc_freq
        initial_phi = -np.pi/20

        fit = ZZFitter(backend_result, xdata, qubits, spectators,
                       fit_p0=[initial_a, initial_f,
                               initial_phi, initial_c],
                       fit_bounds=([-0.5, 0, -np.pi, -0.5],
                                   [1.5, 2*osc_freq, np.pi, 1.5]))

        self.assertAlmostEqual(abs(fit.ZZ_rate()[0]), zz_expected, delta=0.02,
                               msg='Calculated ZZ is inaccurate')


if __name__ == '__main__':
    unittest.main()
