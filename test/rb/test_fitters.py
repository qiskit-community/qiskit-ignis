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

# pylint: disable=invalid-name

"""
Test the fitters
"""

import os
import unittest
from test.utils import load_results_from_json
import json

import numpy as np

from qiskit.ignis.verification.randomized_benchmarking import \
    RBFitter, InterleavedRBFitter, PurityRBFitter, CNOTDihedralRBFitter


class TestFitters(unittest.TestCase):
    """ Test the fitters """

    def compare_results_and_excpected(self, data, expected_data, tst_index):
        """ utility function to compare results """
        data_keys = data[0].keys()
        for i, expect_data in enumerate(expected_data):
            for key in data_keys:
                if expect_data[key] is None:
                    self.assertIsNone(data[i][key], 'Incorrect ' + str(key) + ' in test no. '
                                      + str(tst_index))
                else:
                    # check if the zip function is needed
                    if isinstance(data[i][key], np.ndarray):
                        self.assertTrue(all(np.isclose(a, b) for a, b in
                                            zip(data[i][key], expect_data[key])),
                                        'Incorrect ' + str(key) + ' in test no. ' + str(tst_index))
                    else:
                        self.assertTrue(np.isclose(data[i][key], expect_data[key]),
                                        'Incorrect ' + str(key) + ' in test no. ' + str(tst_index))

    def test_fitters(self):
        """ Test the fitters """

        # Use json results files
        tests_settings = [
            {
                'rb_opts': {
                    'xdata': np.array([[1, 21, 41, 61, 81, 101, 121, 141,
                                        161, 181],
                                       [2, 42, 82, 122, 162, 202, 242, 282,
                                        322, 362]]),
                    'rb_pattern': [[0, 1], [2]],
                    'shots': 1024},
                'results_file': os.path.join(os.path.dirname(__file__),
                                             'test_fitter_results_1.json'),
                'expected_results_file': os.path.join(os.path.dirname(__file__),
                                                      'test_fitter_expected_results_1.json')
            },
            {
                'rb_opts': {
                    'xdata': np.array([[1, 21, 41, 61, 81, 101, 121, 141, 161,
                                        181]]),
                    'rb_pattern': [[0]],
                    'shots': 1024},
                'results_file': os.path.join(os.path.dirname(__file__),
                                             'test_fitter_results_2.json'),
                'expected_results_file': os.path.join(os.path.dirname(__file__),
                                                      'test_fitter_expected_results_2.json')
            }

        ]

        for tst_index, tst_settings in enumerate(tests_settings):
            results_list = load_results_from_json(tst_settings['results_file'])
            with open(tst_settings['expected_results_file'], "r") as expected_results_file:
                tst_expected_results = json.load(expected_results_file)

            # RBFitter class
            rb_fit = RBFitter(results_list[0], tst_settings['rb_opts']['xdata'],
                              tst_settings['rb_opts']['rb_pattern'])

            # add the seeds in reverse order
            for seedind in range(len(results_list)-1, 0, -1):
                rb_fit.add_data([results_list[seedind]])

            ydata = rb_fit.ydata
            fit = rb_fit.fit

            self.compare_results_and_excpected(ydata, tst_expected_results['ydata'], tst_index)
            self.compare_results_and_excpected(fit, tst_expected_results['fit'], tst_index)

    def test_interleaved_fitters(self):
        """ Test the interleaved fitters """

        # Use json results files
        tests_settings = [
            {
                'rb_opts': {
                    'xdata': np.array([[1, 11, 21, 31, 41,
                                        51, 61, 71, 81, 91],
                                       [3, 33, 63, 93, 123,
                                        153, 183, 213, 243, 273]]),
                    'rb_pattern': [[0, 2], [1]],
                    'shots': 200},
                'original_results_file':
                    os.path.join(
                        os.path.dirname(__file__),
                        'test_fitter_original_results.json'),
                'interleaved_results_file':
                    os.path.join(
                        os.path.dirname(__file__),
                        'test_fitter_interleaved_results.json'),
                'expected_results_file':
                    os.path.join(os.path.dirname(__file__),
                                 'test_fitter_interleaved_expected_results.json')
            }]

        for tst_index, tst_settings in enumerate(tests_settings):
            original_result_list = load_results_from_json(
                tst_settings['original_results_file'])
            interleaved_result_list = load_results_from_json(
                tst_settings['interleaved_results_file'])
            with open(tst_settings['expected_results_file'], "r") as expected_results_file:
                tst_expected_results = json.load(expected_results_file)

            # InterleavedRBFitter class
            joint_rb_fit = InterleavedRBFitter(
                original_result_list, interleaved_result_list,
                tst_settings['rb_opts']['xdata'],
                tst_settings['rb_opts']['rb_pattern'])

            joint_fit = joint_rb_fit.fit_int
            ydata_original = joint_rb_fit.ydata[0]
            ydata_interleaved = joint_rb_fit.ydata[1]

            self.compare_results_and_excpected(ydata_original,
                                               tst_expected_results['original_ydata'],
                                               tst_index)
            self.compare_results_and_excpected(ydata_interleaved,
                                               tst_expected_results['interleaved_ydata'],
                                               tst_index)
            self.compare_results_and_excpected(joint_fit,
                                               tst_expected_results['joint_fit'],
                                               tst_index)

    def test_purity_fitters(self):
        """ Test the purity fitters """

        # Use json results files
        tests_settings = [
            {
                'npurity': 9,
                'rb_opts': {
                    'xdata': np.array([[1, 21, 41, 61, 81, 101, 121,
                                        141, 161, 181],
                                       [1, 21, 41, 61, 81, 101, 121,
                                        141, 161, 181]]),
                    'rb_pattern': [[0, 1], [2, 3]],
                    'shots': 200},
                'results_file': os.path.join(
                    os.path.dirname(__file__),
                    'test_fitter_purity_results.json'),
                'expected_results_file': os.path.join(os.path.dirname(__file__),
                                                      'test_fitter_purity_expected_results.json')
            },
            {
                'npurity': 9,
                'rb_opts': {
                    'xdata': np.array([[1, 21, 41, 61, 81, 101, 121,
                                        141, 161, 181],
                                       [1, 21, 41, 61, 81, 101, 121,
                                        141, 161, 181]]),
                    'rb_pattern': [[0, 1], [2, 3]],
                    'shots': 200},
                'results_file': os.path.join(
                    os.path.dirname(__file__),
                    'test_fitter_coherent_purity_results.json'),
                'expected_results_file':
                    os.path.join(os.path.dirname(__file__),
                                 'test_fitter_coherent_purity_expected_results.json')
            }
        ]

        for tst_index, tst_settings in enumerate(tests_settings[0:1]):
            purity_result_list = load_results_from_json(tst_settings['results_file'])
            with open(tst_settings['expected_results_file'], "r") as expected_results_file:
                tst_expected_results = json.load(expected_results_file)

            # PurityRBFitter class
            rbfit_purity = PurityRBFitter(purity_result_list,
                                          tst_settings['npurity'],
                                          tst_settings['rb_opts']['xdata'],
                                          tst_settings['rb_opts']['rb_pattern'])

            ydata = rbfit_purity.ydata
            fit = rbfit_purity.fit

            self.compare_results_and_excpected(ydata, tst_expected_results['ydata'], tst_index)
            self.compare_results_and_excpected(fit, tst_expected_results['fit'], tst_index)

    def test_cnotdihedral_fitters(self):
        """ Test the non-clifford cnot-dihedral CNOT-Dihedral
        fitters """

        # Use json results files
        tests_settings = [
            {
                'rb_opts': {
                    'xdata': np.array([[1, 21, 41, 61,
                                        81, 101, 121, 141,
                                        161, 181],
                                       [3, 63, 123, 183,
                                        243, 303, 363, 423,
                                        483, 543]]),
                    'rb_pattern': [[0, 2], [1]],
                    'shots': 200},
                'cnotdihedral_X_results_file':
                    os.path.join(
                        os.path.dirname(__file__),
                        'test_fitter_cnotdihedral_X_results.json'),
                'cnotdihedral_X_expected_results_file':
                    os.path.join(os.path.dirname(__file__),
                                 'test_fitter_cnotdihedral_X_expected_results.json'),
                'cnotdihedral_Z_results_file':
                    os.path.join(
                        os.path.dirname(__file__),
                        'test_fitter_cnotdihedral_Z_results.json'),
                'cnotdihedral_Z_expected_results_file':
                    os.path.join(os.path.dirname(__file__),
                                 'test_fitter_cnotdihedral_Z_expected_results.json')
            }
        ]

        for tst_index, tst_settings in enumerate(tests_settings):
            cnotdihedral_X_result_list = load_results_from_json(
                tst_settings['cnotdihedral_X_results_file'])
            cnotdihedral_Z_result_list = load_results_from_json(
                tst_settings['cnotdihedral_Z_results_file'])
            with open(tst_settings['cnotdihedral_X_expected_results_file'], "r")\
                    as expected_results_file:
                tst_cnotdihedral_X_expected_results = json.load(expected_results_file)
            with open(tst_settings['cnotdihedral_Z_expected_results_file'], "r")\
                    as expected_results_file:
                tst_cnotdihedral_Z_expected_results = json.load(expected_results_file)

            # CNOTDihedralRBFitter class
            joint_rb_fit = CNOTDihedralRBFitter(
                cnotdihedral_Z_result_list, cnotdihedral_X_result_list,
                tst_settings['rb_opts']['xdata'],
                tst_settings['rb_opts']['rb_pattern'])

            joint_fit = joint_rb_fit.fit_cnotdihedral
            ydata_Z = joint_rb_fit.ydata[0]
            ydata_X = joint_rb_fit.ydata[1]

            self.compare_results_and_excpected(ydata_Z,
                                               tst_cnotdihedral_Z_expected_results[
                                                   'cnotdihedral_Z_ydata'],
                                               tst_index)
            self.compare_results_and_excpected(ydata_X,
                                               tst_cnotdihedral_X_expected_results[
                                                   'cnotdihedral_X_ydata'],
                                               tst_index)
            self.compare_results_and_excpected(joint_fit,
                                               tst_cnotdihedral_X_expected_results['joint_fit'],
                                               tst_index)


if __name__ == '__main__':
    unittest.main()
