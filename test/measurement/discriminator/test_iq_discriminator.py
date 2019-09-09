import unittest

import qiskit
from qiskit import Aer
from qiskit.ignis.measurement.discriminator.filters import DiscriminationFilter
from qiskit.ignis.measurement.discriminator.discriminators import \
    LinearIQDiscriminator
from qiskit.ignis.mitigation.measurement import circuits
from qiskit.result.models import ExperimentResultData
import test.utils as utils


class TestLinearIQDiscriminator(unittest.TestCase):

    def setUp(self):
        self.shots = 52
        self.qubits = [0, 1]

        meas_cal, state_labels = circuits.tensored_meas_cal([[0], [1]])

        backend = Aer.get_backend('qasm_simulator')
        job = qiskit.execute(meas_cal, backend=backend, shots=self.shots,
                             meas_level=1)

        self.cal_results = job.result()

        i0, q0, i1, q1 = 0., -1., 0., 1.
        ground = utils.create_shots(i0, q0, 0.1, 0.1, self.shots, self.qubits)
        excited = utils.create_shots(i1, q1, 0.1, 0.1, self.shots, self.qubits)

        self.cal_results.results[0].meas_level = 1
        self.cal_results.results[1].meas_level = 1
        self.cal_results.results[0].data = ExperimentResultData(memory=ground)
        self.cal_results.results[1].data = ExperimentResultData(memory=excited)

    def test_get_xdata(self):
        discriminator = LinearIQDiscriminator(self.cal_results,
                                              self.qubits,
                                              ['00', '11'])

        xdata = discriminator.get_xdata(self.cal_results)

        self.assertEqual(len(xdata), self.shots*2)
        self.assertEqual(len(xdata[0]), len(self.qubits) * 2)

        xdata = discriminator.get_xdata(self.cal_results, ['cal_00'])

        self.assertEqual(len(xdata), self.shots)
        self.assertEqual(len(xdata[0]), 4)

    def test_get_ydata(self):
        discriminator = LinearIQDiscriminator(self.cal_results,
                                              self.qubits,
                                              ['00', '11'])

        xdata = discriminator.get_xdata(self.cal_results)
        ydata = discriminator.get_ydata(self.cal_results)

        self.assertEqual(len(xdata), len(ydata))

        ydata = discriminator.get_ydata(self.cal_results, ['cal_00'])

        self.assertEqual(len(ydata), self.shots)
        self.assertEqual(ydata[0], '00')

    def test_discrimination(self):
        i0, q0, i1, q1 = 0., -1., 0., 1.
        discriminator = LinearIQDiscriminator(self.cal_results,
                                              self.qubits,
                                              ['00', '11'])

        excited_predicted = discriminator.discriminate([[i1, q1, i1, q1]])
        ground_predicted = discriminator.discriminate([[i0, q0, i0, q0]])

        self.assertEqual(excited_predicted[0], '11')
        self.assertEqual(ground_predicted[0], '00')

    def filter_and_discriminate(self):
        i0, q0, i1, q1 = 0., -1., 0., 1.

        discriminator = LinearIQDiscriminator(self.cal_results,
                                              self.qubits,
                                              ['00', '11'])

        iq_filter = DiscriminationFilter(discriminator)

        new_result = iq_filter.apply(self.cal_results)

        for nr in new_result.results:
            self.assertEqual(nr.meas_level, 2)

        for state in new_result.results[0].data.counts.to_dict():
            self.assertEqual(state, '0x0')

        for state in new_result.results[1].data.counts.to_dict():
            self.assertEqual(state, '0x3')

        self.assertEqual(len(new_result.get_memory(0)), self.shots)

        self.qubits = [0]

        discriminator = LinearIQDiscriminator(self.cal_results,
                                              self.qubits,
                                              ['0', '1'])

        self.assertEqual(discriminator.discriminate([[i0, q0]])[0], '0')
        self.assertEqual(discriminator.discriminate([[i1, q1]])[0], '1')
