import unittest

import qiskit
from qiskit import Aer
from qiskit.ignis.measurement.discriminator.filters import DiscriminationFilter
from qiskit.ignis.measurement.discriminator.iq_discrimination import \
    LinearIQDiscriminator
from qiskit.ignis.mitigation.measurement import circuits
from qiskit.result.models import ExperimentResultData
import qiskit.ignis.utils as utils


class TestLinearIQDiscriminator(unittest.TestCase):

    def setUp(self):
        self.shots = 52
        self.qubits = [0, 1]

    def test_discrimination(self):
        meas_cal, state_labels = circuits.tensored_meas_cal([[0], [1]])

        backend = Aer.get_backend('qasm_simulator')
        job = qiskit.execute(meas_cal, backend=backend, shots=self.shots,
                             meas_level=1)

        cal_results = job.result()

        i0, q0, i1, q1 = 0., -1., 0., 1.
        ground = utils.create_shots(i0, q0, 0.1, 0.1, self.shots, self.qubits)
        excited = utils.create_shots(i1, q1, 0.1, 0.1, self.shots, self.qubits)

        cal_results.results[0].meas_level = 1
        cal_results.results[1].meas_level = 1
        cal_results.results[0].data = ExperimentResultData(memory=ground)
        cal_results.results[1].data = ExperimentResultData(memory=excited)

        discriminator_params = {'solver': 'svd'}

        discriminator = LinearIQDiscriminator(cal_results,
                                              self.qubits,
                                              ['00', '11'],
                                              discriminator_parameters=discriminator_params)

        excited_predicted = discriminator.fit_fun.predict([[i1, q1, i1, q1]])
        ground_predicted = discriminator.fit_fun.predict([[i0, q0, i0, q0]])

        self.assertEqual(excited_predicted[0], '11')
        self.assertEqual(ground_predicted[0], '00')

        iq_filter = DiscriminationFilter(discriminator)

        new_result = iq_filter.apply(cal_results)

        for nr in new_result.results:
            self.assertEqual(nr.meas_level, 2)

        for state in new_result.get_memory('cal_00'):
            self.assertEqual(state, '00')

        for state in new_result.get_memory('cal_11'):
            self.assertEqual(state, '11')

        self.qubits = [0]

        discriminator = LinearIQDiscriminator(cal_results,
                                              self.qubits,
                                              ['0', '1'],
                                              discriminator_parameters=discriminator_params)

        self.assertEqual(discriminator.fit_fun.predict([[i0, q0]])[0], '0')
        self.assertEqual(discriminator.fit_fun.predict([[i1, q1]])[0], '1')
