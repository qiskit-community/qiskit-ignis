import unittest

import qiskit
from qiskit import Aer
from qiskit.ignis.measurement.discriminator.linear import LinearIQDiscriminationFitter
from qiskit.ignis.mitigation.measurement import circuits
from qiskit.result.models import ExperimentResultData


class TestLinearIQDiscriminator(unittest.TestCase):

    def setUp(self):
        self.shots = 512

    def test_discrimination(self):
        # meas_cal, state_labels = circuits.complete_meas_cal([0])
        meas_cal, state_labels = circuits.tensored_meas_cal([[0], [1]])

        backend = Aer.get_backend('qasm_simulator')
        job = qiskit.execute(meas_cal, backend=backend, shots=self.shots, meas_level=1)
        cal_results = job.result()

        # Fake it till you make it
        cal_results.results[0].meas_level = 1
        cal_results.results[1].meas_level = 1
        cal_results.results[0].data = ExperimentResultData(memory=[[100, 100], [200, -200]])
        cal_results.results[1].data = ExperimentResultData(memory=[[-100, -100], [-200, 200]])

        discriminator = LinearIQDiscriminationFitter()


        pass