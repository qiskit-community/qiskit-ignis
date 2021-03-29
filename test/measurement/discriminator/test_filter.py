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

try:
    import sklearn  # pylint: disable=unused-import
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from qiskit.ignis.measurement.discriminator.filters import DiscriminationFilter
from qiskit.result import Result
from qiskit.ignis.measurement.discriminator.iq_discriminators import \
    LinearIQDiscriminator


@unittest.skipUnless(HAS_SKLEARN, 'scikit-learn is required for these tests')
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
        cal_00_data = [[[1., 1.]], [[1.1, 0.9]], [[0.9, 1.1]], [[1., 1.1]], [[0.9, 1.2]]]
        cal_11_data = [[[-1., -1.]], [[-1.1, -0.9]], [[-0.9, -1.1]], [[-0.9, -1.2]],
                       [[-1., -1.1]]]
        x90p_data = [[[-1.1, -1.]], [[1.1, 0.9]], [[-0.8, -1.0]], [[0.9, 1.1]],
                     [[1., 1.]]]

        result = Result.from_dict({'job_id': '',
                                   'backend_version': '1.3.0',
                                   'backend_name': 'test',
                                   'qobj_id': '',
                                   'success': True,
                                   'results': [{
                                       "header": {"name": "cal_0"},
                                       "shots": 5,
                                       "status": "DONE",
                                       'meas_level': 1,
                                       "success": True,
                                       "meas_return": "single",
                                       "data": {
                                           "memory": cal_00_data
                                       }
                                   }, {
                                       "header": {"name": "cal_1"},
                                       "shots": 5,
                                       "status": "DONE",
                                       'meas_level': 1,
                                       "success": True,
                                       "meas_return": "single",
                                       "data": {
                                           "memory": cal_11_data
                                       }
                                   }, {
                                       "header": {"name": "x90p"},
                                       "shots": 5,
                                       "status": "DONE",
                                       'meas_level': 1,
                                       "success": True,
                                       "meas_return": "single",
                                       "data": {
                                           "memory": x90p_data
                                       }
                                   }]
                                   })

        discriminator = LinearIQDiscriminator(result, [0])

        d_filter = DiscriminationFilter(discriminator)

        self.assertEqual(result.results[0].meas_level, 1)
        new_results = d_filter.apply(result)

        self.assertEqual(new_results.results[0].meas_level, 2)

        for name in ['cal_0', 'cal_1', 'x90p']:
            counts = new_results.get_counts(name)
            counts_0 = counts.get('0', 0)
            counts_1 = counts.get('1', 0)

            self.assertEqual(counts_0 + counts_1, 5)

            if name == 'x90p':
                self.assertEqual(counts_0, 3)
                self.assertEqual(counts_1, 2)
