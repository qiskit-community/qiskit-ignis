import copy
import numpy as np
import unittest

import qiskit
from qiskit import Aer
from qiskit.ignis.measurement.discriminator.filters import DiscriminationFilter
from qiskit.ignis.measurement.discriminator.iq_discrimination import \
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

    def test_discrimination(self):
        i0, q0, i1, q1 = 0., -1., 0., 1.
        discriminator = LinearIQDiscriminator(self.cal_results,
                                              self.qubits,
                                              ['00', '11'])

        excited_predicted = discriminator.fit_fun.predict([[i1, q1, i1, q1]])
        ground_predicted = discriminator.fit_fun.predict([[i0, q0, i0, q0]])

        self.assertEqual(excited_predicted[0], '11')
        self.assertEqual(ground_predicted[0], '00')

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

        self.assertEqual(discriminator.fit_fun.predict([[i0, q0]])[0], '0')
        self.assertEqual(discriminator.fit_fun.predict([[i1, q1]])[0], '1')

    def test_extract_xdata(self):
        i0, q0, i1, q1 = 0., -1., 0., 1.
        ground = utils.create_shots(i0, q0, 0.1, 0.1, self.shots, self.qubits)

        ExperimentResultData(memory=ground)

        discriminator = LinearIQDiscriminator(self.cal_results,
                                              self.qubits,
                                              ['00', '11'])

        # Test the formatting if return data is like single-shot data
        single_shot_result = copy.deepcopy(self.cal_results.results[0])

        x_data = discriminator.extract_xdata(single_shot_result)

        self.assertEqual(len(x_data), self.shots)
        self.assertEqual(len(x_data[0]), 2*len(self.qubits))

        # Test the shape of the data for the averaged case:
        # Average over the single shots
        avg_ground = [list(qubit_iq) for qubit_iq in
                      list(np.average(ground, axis=0))]

        # Create a new result using the averaged data
        avg_shot_result = copy.deepcopy(self.cal_results.results[0])
        avg_shot_result.data = ExperimentResultData(memory=avg_ground)

        x_data = discriminator.extract_xdata(avg_shot_result)
        self.assertEqual(len(x_data), 1)
        self.assertEqual(len(x_data[0]), 2*len(self.qubits))
