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
Test the fitters
"""

import os
import pickle
import unittest

import numpy as np

from qiskit.ignis.verification.randomized_benchmarking import RBFitter


class TestFitters(unittest.TestCase):
    """ Test the fitters """

    def test_fitters(self):
        """ Test the fitters """

        # Use pickled results files

        tests = [{
            'rb_opts': {
                'xdata': np.array([[1, 21, 41, 61, 81, 101, 121, 141,
                                    161, 181],
                                   [2, 42, 82, 122, 162, 202, 242, 282,
                                    322, 362]]),
                'rb_pattern': [[0, 1], [2]],
                'shots': 1024},
            'results_file': os.path.join(os.path.dirname(__file__),
                                         'test_fitter_results_1.pkl'),
            'expected': {
                'ydata': [{
                    'mean': np.array([0.96367187, 0.73457031,
                                      0.58066406, 0.4828125,
                                      0.41035156, 0.34902344,
                                      0.31210938, 0.2765625,
                                      0.29453125, 0.27695313]),
                    'std': np.array([0.01013745, 0.0060955, 0.00678272,
                                     0.01746491, 0.02015981, 0.02184184,
                                     0.02340167, 0.02360293, 0.00874773,
                                     0.01308156])}, {
                                         'mean': np.array([0.98925781,
                                                           0.87734375, 0.78125,
                                                           0.73066406,
                                                           0.68496094,
                                                           0.64296875,
                                                           0.59238281,
                                                           0.57421875,
                                                           0.56074219,
                                                           0.54980469]),
                                         'std': np.array(
                                             [0.00276214, 0.01602991,
                                              0.00768946, 0.01413015,
                                              0.00820777, 0.01441348,
                                              0.01272682, 0.01031649,
                                              0.02103036, 0.01224408])}],
                'fit': [{
                    'params': np.array([0.71936804, 0.98062119,
                                        0.25803749]),
                    'params_err': np.array([0.0065886, 0.00046714,
                                            0.00556488]),
                    'epc': 0.014534104912075935,
                    'epc_err': 6.923601336318206e-06},
                        {'params': np.array([0.49507094, 0.99354093,
                                             0.50027262]),
                         'params_err': np.array([0.0146191, 0.0004157,
                                                 0.01487439]),
                         'epc': 0.0032295343343508587,
                         'epc_err': 1.3512528169626024e-06}]
            }}, {
                'rb_opts': {
                    'xdata': np.array([[1, 21, 41, 61, 81, 101, 121, 141, 161,
                                        181]]),
                    'rb_pattern': [[0]],
                    'shots': 1024},
                'results_file': os.path.join(os.path.dirname(__file__),
                                             'test_fitter_results_2.pkl'),
                'expected': {
                    'ydata': [{'mean': np.array([0.99199219, 0.93867188,
                                                 0.87871094, 0.83945313,
                                                 0.79335937, 0.74785156,
                                                 0.73613281, 0.69414062,
                                                 0.67460937, 0.65664062]),
                               'std': np.array([0.00567416, 0.00791919,
                                                0.01523437, 0.01462368,
                                                0.01189002, 0.01445049,
                                                0.00292317, 0.00317345,
                                                0.00406888,
                                                0.01504794])}],
                    'fit': [{'params': np.array([0.59599995, 0.99518211,
                                                 0.39866989]),
                             'params_err': np.array([0.08843152, 0.00107311,
                                                     0.09074325]),
                             'epc': 0.0024089464034862673,
                             'epc_err': 2.5975709525210163e-06}]}}]

        for tst_index, tst in enumerate(tests):
            fo = open(tst['results_file'], 'rb')
            results_list = pickle.load(fo)
            fo.close()

            # RBFitter class
            rb_fit = RBFitter(results_list, tst['rb_opts']['xdata'],
                              tst['rb_opts']['rb_pattern'])
            ydata = rb_fit.ydata
            fit = rb_fit.fit

            for i, _ in enumerate(ydata):
                self.assertTrue(all(np.isclose(a, b) for a, b in
                                    zip(ydata[i]['mean'],
                                        tst['expected']['ydata'][i]['mean'])),
                                'Incorrect mean in test no. ' + str(tst_index))
                if tst['expected']['ydata'][i]['std'] is None:
                    self.assertIsNone(
                        ydata[i]['std'],
                        'Incorrect std in test no. ' + str(tst_index))
                else:
                    self.assertTrue(
                        all(np.isclose(a, b) for a, b in zip(
                            ydata[i]['std'],
                            tst['expected']['ydata'][i]['std'])),
                        'Incorrect std in test no. ' + str(tst_index))
                self.assertTrue(
                    all(np.isclose(a, b) for a, b in zip(
                        fit[i]['params'],
                        tst['expected']['fit'][i]['params'])),
                    'Incorrect fit parameters in test no. ' + str(tst_index))
                self.assertTrue(
                    all(np.isclose(a, b) for a, b in zip(
                        fit[i]['params_err'],
                        tst['expected']['fit'][i]['params_err'])),
                    'Incorrect fit error in test no. ' + str(tst_index))
                self.assertTrue(np.isclose(fit[i]['epc'],
                                           tst['expected']['fit'][i]['epc']),
                                'Incorrect EPC in test no. ' + str(tst_index))
                self.assertTrue(
                    np.isclose(fit[i]['epc_err'],
                               tst['expected']['fit'][i]['epc_err']),
                    'Incorrect EPC error in test no. ' + str(tst_index))


if __name__ == '__main__':
    unittest.main()
