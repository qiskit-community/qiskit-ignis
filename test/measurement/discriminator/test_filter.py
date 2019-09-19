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

"""
Test discrimination filters.
"""

import unittest
from operator import getitem

import test.utils as utils
import qiskit
from qiskit import Aer
from qiskit.ignis.measurement.discriminator.filters import DiscriminationFilter
from qiskit.result.models import ExperimentResultData
from qiskit.ignis.measurement.discriminator.iq_discriminators import \
    LinearIQDiscriminator
from qiskit.ignis.mitigation.measurement import circuits


class TestDiscriminationFilter(unittest.TestCase):
    """
    Test methods of discrimination filters.
    """

    def setUp(self) -> None:
        """
        Initialize private variables.
        """
        self.shots = 10
        self.qubits = [0, 1]

    def test_get_base(self):
        """
        Test the get_base method to see if it can properly identify the number
        of basis states per quantum element. E.g. second excited level of
        transmon.
        """
        expected_states = {'a': '02', 'b': '01', 'c': '00', 'd': '11'}
        base = DiscriminationFilter.get_base(expected_states)
        self.assertEqual(base, 3)

    def test_count(self):
        """
        Test to see if the filter properly converts the result of
        discriminator.discriminate to a dictionary of counts.
        """
        fitter = LinearIQDiscriminator([], [], [])
        d_filter = DiscriminationFilter(fitter, 2)

        raw_counts = d_filter.count(['01', '00', '01', '00', '00', '10'])
        self.assertEqual(raw_counts['0x0'], 3)
        self.assertEqual(raw_counts['0x1'], 2)
        self.assertEqual(raw_counts['0x2'], 1)
        self.assertRaises(KeyError, getitem, raw_counts, '0x3')

        d_filter = DiscriminationFilter(fitter, 3)
        raw_counts = d_filter.count(['02', '02', '20', '21', '21', '02'])
        self.assertEqual(raw_counts['0x2'], 3)
        self.assertEqual(raw_counts['0x6'], 1)
        self.assertEqual(raw_counts['0x7'], 2)

    def test_apply(self):
        """
        Set-up a discriminator based on simulated data, train it and then
        discriminate the calibration data.
        """
        meas_cal, _ = circuits.tensored_meas_cal([[0], [1]])

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

        discriminator = LinearIQDiscriminator(cal_results,
                                              self.qubits,
                                              ['00', '11'])

        d_filter = DiscriminationFilter(discriminator)

        self.assertEqual(cal_results.results[0].meas_level, 1)
        new_results = d_filter.apply(cal_results)

        self.assertEqual(new_results.results[0].meas_level, 2)

        counts_00 = new_results.results[0].data.counts.to_dict()['0x0']
        counts_11 = new_results.results[1].data.counts.to_dict()['0x3']

        self.assertEqual(counts_00, self.shots)
        self.assertEqual(counts_11, self.shots)
