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
import os
import pickle

from qiskit.ignis.measurement.discriminator.filters import DiscriminationFilter
from qiskit.result import Result
from qiskit.ignis.measurement.discriminator.iq_discriminators import \
    LinearIQDiscriminator


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
        result_pkl = os.path.join(os.path.dirname(__file__), 'test_result.pkl')
        with open(result_pkl, 'rb') as handle:
            result = Result.from_dict(pickle.load(handle))

        discriminator = LinearIQDiscriminator(result, [0, 1])

        d_filter = DiscriminationFilter(discriminator)

        self.assertEqual(result.results[0].meas_level, 1)
        new_results = d_filter.apply(result)

        self.assertEqual(new_results.results[0].meas_level, 2)

        for idx in range(3):
            counts_00 = new_results.results[idx].data.counts.to_dict()['0x0']
            counts_11 = new_results.results[idx].data.counts.to_dict()['0x3']

            self.assertEqual(counts_00 + counts_11, 512)
