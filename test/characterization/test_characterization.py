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
# pylint: disable=no-name-in-module

"""
Test coherence measurements
"""

import unittest

from test.characterization.generate_data import (t1_circuit_execution,
                                                 t2_circuit_execution,
                                                 t2star_circuit_execution,
                                                 zz_circuit_execution)

import numpy as np
import qiskit

from qiskit.providers.aer.noise.errors.standard_errors import \
                                      coherent_unitary_error

from qiskit.providers.aer.noise import NoiseModel

from qiskit.ignis.characterization.coherence import \
     T2StarFitter, T1Fitter, T2Fitter

from qiskit.ignis.characterization.hamiltonian import ZZFitter

from qiskit.ignis.characterization.gates import (AmpCalFitter,
                                                 ampcal_1Q_circuits,
                                                 AngleCalFitter,
                                                 anglecal_1Q_circuits,
                                                 AmpCalCXFitter,
                                                 ampcal_cx_circuits,
                                                 AngleCalCXFitter,
                                                 anglecal_cx_circuits)

from qiskit.ignis.characterization.calibrations import (rabi_schedules,
                                                        drag_schedules,
                                                        update_u_gates)

from qiskit.test.mock import FakeOpenPulse2Q


# Fix seed for simulations
SEED = 9000


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

        backend_result, xdata, qubits, t2_value, omega = t2star_circuit_execution()

        T2StarFitter(backend_result, xdata, qubits,
                     fit_p0=[0.5, t2_value, omega, 0, 0.5],
                     fit_bounds=([-0.5, 0, omega-0.02, -np.pi, -0.5],
                                 [1.5, t2_value*1.2, omega+0.02, np.pi, 1.5]))


class TestT1(unittest.TestCase):
    """
    Test measurement of T1
    """

    def test_t1(self):
        """
        Run the simulator with thermal relaxation noise.
        Then verify that the calculated T1 matches the t1
        parameter.
        """

        backend_result, xdata, qubits, t1_value = t1_circuit_execution()

        T1Fitter(backend_result, xdata, qubits,
                 fit_p0=[1, t1_value, 0],
                 fit_bounds=([0, 0, -1], [2, t1_value*1.2, 1]))


class TestT2(unittest.TestCase):
    """
    Test measurement of T2
    """

    def test_t2(self):
        """
        Run the simulator with thermal relaxation noise.
        Then verify that the calculated T2 matches the t2 parameter.
        """

        backend_result, xdata, qubits, t2_value = t2_circuit_execution()

        T2Fitter(backend_result, xdata, qubits,
                 fit_p0=[1, t2_value, -0.5],
                 fit_bounds=([0, 0, -1], [2, t2_value*1.2, 1]))


class TestZZ(unittest.TestCase):
    """
    Test measurement of ZZ
    """

    def test_zz(self):
        """
        Run the simulator with unitary noise.
        Then verify that the calculated ZZ matches the zz parameter.
        """

        backend_result, xdata, qubits, spectators, _, omega = zz_circuit_execution()

        ZZFitter(backend_result, xdata, qubits, spectators,
                 fit_p0=[0.5, omega, 0, 0.5],
                 fit_bounds=([-0.5, 0, -np.pi, -0.5],
                             [1.5, 2*omega, np.pi, 1.5]))


class TestCals(unittest.TestCase):
    """
    Test cals
    """

    def __init__(self, *args, **kwargs):
        """
        Init the test cal simulator
        """

        unittest.TestCase.__init__(self, *args, **kwargs)
        self._qubits = [0, 2]
        self._controls = [1, 3]
        self._maxrep = 10
        self._circs = []

    def run_sim(self, noise=None):
        """
        Run the simulator for these test cals
        """

        # Run the simulator
        backend = qiskit.Aer.get_backend('qasm_simulator')
        shots = 500
        # For demonstration purposes split the execution into two jobs
        backend_result = qiskit.execute(self._circs, backend,
                                        seed_simulator=SEED,
                                        shots=shots,
                                        noise_model=noise).result()

        return backend_result

    def test_ampcal1Q(self):
        """
        Run the amplitude cal circuit generation and through the
        simulator to make sure there are no errors
        """

        self._circs, xdata = ampcal_1Q_circuits(self._maxrep, self._qubits)

        # Set the simulator
        # Add a rotation error
        err_unitary = np.zeros([2, 2], dtype=complex)
        angle_err = 0.1
        for i in range(2):
            err_unitary[i, i] = np.cos(angle_err)
            err_unitary[i, (i+1) % 2] = np.sin(angle_err)
        err_unitary[0, 1] *= -1.0

        error = coherent_unitary_error(err_unitary)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, 'u2')

        initial_theta = 0.18
        initial_c = 0.5

        fit = AmpCalFitter(self.run_sim(noise_model), xdata, self._qubits,
                           fit_p0=[initial_theta, initial_c],
                           fit_bounds=([-np.pi, -1],
                                       [np.pi, 1]))
        self.assertAlmostEqual(fit.angle_err(0), 0.1, 2)

    def test_anglecal1Q(self):
        """
        Run the angle cal circuit generation and through the
        simulator to make sure there are no errors
        """

        self._circs, xdata = anglecal_1Q_circuits(self._maxrep, self._qubits,
                                                  angleerr=0.1)

        initial_theta = 0.18
        initial_c = 0.5

        fit = AngleCalFitter(self.run_sim(), xdata, self._qubits,
                             fit_p0=[initial_theta, initial_c],
                             fit_bounds=([-np.pi, -1],
                                         [np.pi, 1]))

        self.assertAlmostEqual(fit.angle_err(0), 0.1, 2)

    def test_ampcalCX(self):
        """
        Run the amplitude cal circuit generation for CX
        and through the
        simulator to make sure there are no errors
        """

        self._circs, xdata = ampcal_cx_circuits(self._maxrep,
                                                self._qubits,
                                                self._controls)

        err_unitary = np.eye(4, dtype=complex)
        angle_err = 0.15
        for i in range(2):
            err_unitary[2+i, 2+i] = np.cos(angle_err)
            err_unitary[2+i, 2+(i+1) % 2] = -1j*np.sin(angle_err)

        error = coherent_unitary_error(err_unitary)
        noise_model = NoiseModel()
        noise_model.add_nonlocal_quantum_error(error, 'cx', [1, 0], [0, 1])

        initial_theta = 0.18
        initial_c = 0.5

        fit = AmpCalCXFitter(self.run_sim(noise_model), xdata, self._qubits,
                             fit_p0=[initial_theta, initial_c],
                             fit_bounds=([-np.pi, -1],
                                         [np.pi, 1]))

        self.assertAlmostEqual(fit.angle_err(0), 0.15, 2)
        self.assertAlmostEqual(fit.angle_err(1), 0.0, 2)

    def test_anglecalCX(self):
        """
        Run the angle cal circuit generation and through the
        simulator to make sure there are no errors
        """

        self._circs, xdata = anglecal_cx_circuits(self._maxrep,
                                                  self._qubits,
                                                  self._controls,
                                                  angleerr=0.1)

        initial_theta = 0.18
        initial_c = 0.5

        fit = AngleCalCXFitter(self.run_sim(), xdata, self._qubits,
                               fit_p0=[initial_theta, initial_c],
                               fit_bounds=([-np.pi, -1],
                                           [np.pi, 1]))

        self.assertAlmostEqual(fit.angle_err(0), 0.1, 2)
        self.assertAlmostEqual(fit.angle_err(1), 0.1, 2)


class TestCalibs(unittest.TestCase):
    """
    Test calibration module which creates pulse schedules
    """

    def __init__(self, *args, **kwargs):
        """
        Init the test cal simulator
        """

        unittest.TestCase.__init__(self, *args, **kwargs)

        backend_fake = FakeOpenPulse2Q()

        self.config = backend_fake.configuration()
        self.meas_map = self.config.meas_map
        self.n_qubits = self.config.n_qubits
        back_defaults = backend_fake.defaults()
        self.inst_map = back_defaults.instruction_schedule_map

    def test_rabi(self):
        """
        make sure the rabi function will create a schedule
        """

        rabi, xdata = rabi_schedules(
            np.linspace(-1, 1, 10),
            [0, 1],
            10,
            2.5,
            drives=[self.config.drive(i) for i in range(self.n_qubits)],
            inst_map=self.inst_map,
            meas_map=self.meas_map)

        self.assertEqual(len(rabi), 10)
        self.assertEqual(len(xdata), 10)

    def test_drag(self):
        """
        Make sure the drag function will create a schedule
        """

        drag_scheds, xdata = drag_schedules(
            np.linspace(-3, 3, 11),
            [0, 1],
            [0.5, 0.6],
            10,
            pulse_sigma=2.5,
            drives=[self.config.drive(i) for i in range(self.n_qubits)],
            inst_map=self.inst_map,
            meas_map=self.meas_map)

        self.assertEqual(len(drag_scheds), 11)
        self.assertEqual(len(xdata), 11)

    def test_ugate_update(self):
        """
        Test that a cmd def gets updates
        """

        single_q_params = [{'amp': 0.1, 'duration': 15,
                            'beta': 0.1, 'sigma': 3}]

        update_u_gates(
            single_q_params,
            qubits=[0],
            inst_map=self.inst_map,
            drives=[self.config.drive(i) for i in range(self.n_qubits)])


if __name__ == '__main__':
    unittest.main()
