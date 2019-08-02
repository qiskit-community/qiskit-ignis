import unittest
import random
import numpy as np

import qiskit
from qiskit import Aer
from qiskit.ignis.measurement.discriminator.linear import LinearIQDiscriminationFitter
from qiskit.ignis.mitigation.measurement import circuits
from qiskit.result.models import ExperimentResultData


class TestLinearIQDiscriminator(unittest.TestCase):

    def setUp(self):
        self.shots = 512
        self.qubits = [0, 1]

    def test_discrimination(self):
        meas_cal, state_labels = circuits.tensored_meas_cal([[0], [1]])

        backend = Aer.get_backend('qasm_simulator')
        job = qiskit.execute(meas_cal, backend=backend, shots=self.shots, meas_level=1)
        cal_results = job.result()

        # Make up some fake data for the qubits
        def qubit_shot(i0, q0, std):
            return [i0 + random.gauss(0, std), q0 + random.gauss(0, std)]

        def create_shots(i0, q0):
            """Creates data where all qubits are centered around i0 and q0"""
            data = []
            for ii in range(self.shots):
                shot = []
                for qbit in self.qubits:
                    shot.append(qubit_shot(i0, q0, 0.1))
                data.append(shot)

            return data

        i0, q0, i1, q1 = 0., -1., 0., 1.
        cal_results.results[0].meas_level = 1
        cal_results.results[1].meas_level = 1
        cal_results.results[0].data = ExperimentResultData(memory=create_shots(i0, q0))
        cal_results.results[1].data = ExperimentResultData(memory=create_shots(i1, q1))

        discriminator_params = {'solver': 'svd'}

        discriminator = LinearIQDiscriminationFitter(cal_results, discriminator_params, self.qubits,
                                                     ['cal_00', 'cal_11'], ['00', '11'])

        self.assertEqual(discriminator.fit_fun.predict([[i1, q1, i1, q1]])[0], '11')
        self.assertEqual(discriminator.fit_fun.predict([[i0, q0, i0, q0]])[0], '00')

        discriminator = LinearIQDiscriminationFitter(cal_results, discriminator_params, [0],
                                                     ['cal_00', 'cal_11'], ['0', '1'])

        self.assertEqual(discriminator.fit_fun.predict([[i0, q0]])[0], '0')
        self.assertEqual(discriminator.fit_fun.predict([[i1, q1]])[0], '1')
