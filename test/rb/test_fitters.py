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

from qiskit.ignis.verification.randomized_benchmarking import \
    RBFitter, InterleavedRBFitter, PurityRBFitter


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
                    'epc_err': 0.0003572769714798349},
                        {'params': np.array([0.49507094, 0.99354093,
                                             0.50027262]),
                         'params_err': np.array([0.0146191, 0.0004157,
                                                 0.01487439]),
                         'epc': 0.0032295343343508587,
                         'epc_err': 0.00020920242080699664}]
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
                             'epc_err': 0.0005391508310961153}]}}]

        for tst_index, tst in enumerate(tests):
            fo = open(tst['results_file'], 'rb')
            results_list = pickle.load(fo)
            fo.close()

            # RBFitter class
            rb_fit = RBFitter(results_list[0], tst['rb_opts']['xdata'],
                              tst['rb_opts']['rb_pattern'])

            # add the seeds in reverse order
            for seedind in range(len(results_list)-1, 0, -1):
                rb_fit.add_data([results_list[seedind]])

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

    def test_interleaved_fitters(self):
        """ Test the interleaved fitters """

        # Use pickled results files

        tests_interleaved = \
            [{
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
                        'test_fitter_original_results.pkl'),
                'interleaved_results_file':
                    os.path.join(
                        os.path.dirname(__file__),
                        'test_fitter_interleaved_results.pkl'),
                'expected': {
                    'original_ydata':
                        [{'mean': np.array([0.9775, 0.79, 0.66,
                                            0.5775, 0.5075, 0.4825,
                                            0.4075, 0.3825,
                                            0.3925, 0.325]),
                          'std': np.array([0.0125, 0.02, 0.01,
                                           0.0125, 0.0025,
                                           0.0125, 0.0225, 0.0325,
                                           0.0425, 0.])},
                         {'mean': np.array([0.985, 0.9425, 0.8875,
                                            0.8225, 0.775, 0.7875,
                                            0.7325, 0.705,
                                            0.69, 0.6175]),
                          'std': np.array([0.005, 0.0125, 0.0025,
                                           0.0025, 0.015, 0.0125,
                                           0.0075, 0.01,
                                           0.02, 0.0375])}],
                    'interleaved_ydata':
                        [{'mean': np.array([0.955, 0.7425, 0.635,
                                            0.4875, 0.44, 0.3625,
                                            0.3575, 0.2875,
                                            0.2975, 0.3075]),
                          'std': np.array([0., 0.0025, 0.015,
                                           0.0075, 0.055,
                                           0.0075, 0.0075, 0.0025,
                                           0.0025, 0.0075])},
                         {'mean': np.array([0.9775, 0.85, 0.77,
                                            0.7775, 0.6325,
                                            0.615, 0.64, 0.6125,
                                            0.535, 0.55]),
                          'std': np.array([0.0075, 0.005, 0.01,
                                           0.0025, 0.0175, 0.005,
                                           0.01, 0.0075,
                                           0.01, 0.005])}],
                    'joint_fit': [
                        {'alpha': 0.9707393978697902,
                         'alpha_err': 0.0028343593038762326,
                         'alpha_c': 0.9661036105117012,
                         'alpha_c_err': 0.003096602375173838,
                         'epc_est': 0.003581641505636224,
                         'epc_est_err': 0.0032362911276774308,
                         'systematic_err': 0.04030926168967841,
                         'systematic_err_L': -0.03672762018404219,
                         'systematic_err_R': 0.043890903195314634},
                        {'alpha': 0.9953124384370953,
                         'alpha_err': 0.0014841466685991903,
                         'alpha_c': 0.9955519189829325,
                         'alpha_c_err': 0.002194868426034655,
                         'epc_est': -0.00012030420629183247,
                         'epc_est_err': 0.001331116936065506,
                         'systematic_err': 0.004807865769196562,
                         'systematic_err_L': -0.0049281699754883945,
                         'systematic_err_R': 0.00468756156290473}]
                }}]

        for tst_index, tst in enumerate(tests_interleaved):
            fo = open(tst['original_results_file'], 'rb')
            original_result_list = pickle.load(fo)
            fo.close()

            fo = open(tst['interleaved_results_file'], 'rb')
            interleaved_result_list = pickle.load(fo)
            fo.close()

            # InterleavedRBFitter class
            joint_rb_fit = InterleavedRBFitter(
                original_result_list, interleaved_result_list,
                tst['rb_opts']['xdata'], tst['rb_opts']['rb_pattern'])

            joint_fit = joint_rb_fit.fit_int
            ydata_original = joint_rb_fit.ydata[0]
            ydata_interleaved = joint_rb_fit.ydata[1]

            for i, _ in enumerate(ydata_original):
                self.assertTrue(all(np.isclose(a, b) for a, b in
                                    zip(ydata_original[i]['mean'],
                                        tst['expected']['original_ydata']
                                        [i]['mean'])),
                                'Incorrect mean in original data test no. '
                                + str(tst_index))
                if tst['expected']['original_ydata'][i]['std'] is None:
                    self.assertIsNone(
                        ydata_original[i]['std'],
                        'Incorrect std in original data test no. ' +
                        str(tst_index))
                else:
                    self.assertTrue(
                        all(np.isclose(a, b) for a, b in zip(
                            ydata_original[i]['std'],
                            tst['expected']['original_ydata'][i]['std'])),
                        'Incorrect std in original data test no. ' +
                        str(tst_index))

            for i, _ in enumerate(ydata_interleaved):
                self.assertTrue(all(np.isclose(a, b) for a, b in
                                    zip(ydata_interleaved[i]['mean'],
                                        tst['expected']['interleaved_ydata']
                                        [i]['mean'])),
                                'Incorrect mean in interleaved data test no. '
                                + str(tst_index))
                if tst['expected']['interleaved_ydata'][i]['std'] is None:
                    self.assertIsNone(
                        ydata_interleaved[i]['std'],
                        'Incorrect std in interleaved data test no. '
                        + str(tst_index))
                else:
                    self.assertTrue(
                        all(np.isclose(a, b) for a, b in zip(
                            ydata_interleaved[i]['std'],
                            tst['expected']['interleaved_ydata']
                            [i]['std'])),
                        'Incorrect std in interleaved data test no. '
                        + str(tst_index))

            for i, _ in enumerate(joint_fit):
                self.assertTrue(
                    np.isclose(joint_fit[i]['alpha'],
                               tst['expected']['joint_fit']
                               [i]['alpha']),
                    'Incorrect fit parameter alpha in test no. '
                    + str(tst_index))
                self.assertTrue(
                    np.isclose(joint_fit[i]['alpha_err'],
                               tst['expected']['joint_fit']
                               [i]['alpha_err']),
                    'Incorrect fit parameter alpha_err in test no. '
                    + str(tst_index))
                self.assertTrue(
                    np.isclose(joint_fit[i]['alpha_c'],
                               tst['expected']['joint_fit']
                               [i]['alpha_c']),
                    'Incorrect fit parameter alpha_c in test no. '
                    + str(tst_index))
                self.assertTrue(
                    np.isclose(joint_fit[i]['alpha_c_err'],
                               tst['expected']['joint_fit']
                               [i]['alpha_c_err']),
                    'Incorrect fit parameter alpha_c_err in test no. '
                    + str(tst_index))
                self.assertTrue(
                    np.isclose(joint_fit[i]['epc_est'],
                               tst['expected']['joint_fit']
                               [i]['epc_est']),
                    'Incorrect fit parameter epc_est in test no. '
                    + str(tst_index))
                self.assertTrue(
                    np.isclose(joint_fit[i]['epc_est_err'],
                               tst['expected']['joint_fit']
                               [i]['epc_est_err']),
                    'Incorrect fit parameter epc_est_err in test no. '
                    + str(tst_index))
                self.assertTrue(
                    np.isclose(joint_fit[i]['systematic_err'],
                               tst['expected']['joint_fit']
                               [i]['systematic_err']),
                    'Incorrect fit parameter systematic_err in test no. '
                    + str(tst_index))
                self.assertTrue(
                    np.isclose(joint_fit[i]['systematic_err_R'],
                               tst['expected']['joint_fit']
                               [i]['systematic_err_R']),
                    'Incorrect fit parameter systematic_err_R in '
                    'test no. ' + str(tst_index))
                self.assertTrue(
                    np.isclose(joint_fit[i]['systematic_err_L'],
                               tst['expected']['joint_fit']
                               [i]['systematic_err_L']),
                    'Incorrect fit parameter systematic_err_L '
                    'in test no. ' + str(tst_index))

    def test_purity_fitters(self):
        """ Test the purity fitters """

        # Use pickled results files

        tests_purity = \
            [{
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
                    'test_fitter_purity_results.pkl'),
                'expected': {
                    'ydata':
                        [{'mean': np.array([0.92534849, 0.51309098,
                                            0.3622178, 0.29969053,
                                            0.26635693, 0.25874519,
                                            0.25534863, 0.25298818,
                                            0.25352012, 0.2523394]),
                          'std': np.array([0.01314403, 0.00393961,
                                           0.01189933, 0.00936296,
                                           0.00149143, 0.00248324,
                                           0.00162298, 0.00047547,
                                           0.00146307, 0.00104081])},
                         {'mean': np.array([0.92369652, 0.52535891,
                                            0.36284821, 0.28978369,
                                            0.26764608, 0.26141492,
                                            0.25365907, 0.25399547,
                                            0.25308856, 0.25243922]),
                          'std': np.array([0.01263948, 0.0139054,
                                           0.00774744, 0.00514974,
                                           0.00110454, 0.00185583,
                                           0.00103562, 0.00108479,
                                           0.00032715, 0.00067735])}],
                    'fit':
                        [{'params': np.array([0.70657607, 0.97656138,
                                              0.25222978]),
                          'params_err': np.array([0.00783723, 0.00028377,
                                                  0.00026126]),
                          'epc': 0.034745905818288264,
                          'epc_err': 0.0004410703043924575,
                          'pepc': 0.017578966279446773,
                          'pepc_err': 0.00021793530767170674},
                         {'params': np.array([0.70689622, 0.97678485,
                                              0.25258977]),
                          'params_err': np.array([0.01243358, 0.00037419,
                                                  0.00027745]),
                          'epc': 0.03441852175571036,
                          'epc_err': 0.0005814179624217788,
                          'pepc': 0.017411364623216907,
                          'pepc_err': 0.00028731473935779276}]
                }},
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
                     'test_fitter_coherent_purity_results.pkl'),
                 'expected': {
                     'ydata':
                         [{'mean': np.array([1.03547598, 1.00945614,
                                             0.9874103, 0.99794296,
                                             0.98926947, 0.98898662,
                                             0.9908188, 1.04339706,
                                             1.02311855, 1.02636139]),
                           'std': np.array([0.00349072, 0.05013115,
                                            0.01657108, 0.03048466,
                                            0.03496286, 0.02572242,
                                            0.03661921, 0.02406485,
                                            0.04192087, 0.05903551])},
                          {'mean': np.array([1.04122543, 0.98568824,
                                             0.98702183, 1.00184751,
                                             1.02116973, 0.98867042,
                                             1.06620605, 1.11332653,
                                             1.04427034, 1.0687145]),
                           'std': np.array([0.00519259, 0.02815319,
                                            0.06940576, 0.0232619,
                                            0.0442728, 0.05649533,
                                            0.05882039, 0.13732109,
                                            0.06189085, 0.0890274])}],
                     'fit':
                         [{'params': np.array([0.04050766, 0.91275946,
                                               1.00172827]),
                           'params_err': np.array([0.09520572, 1.04827404,
                                                   0.00820391]),
                           'epc': 0.12515262778294844,
                           'epc_err': 1.8031488429069056,
                           'pepc': 0.06543040590251992,
                           'pepc_err': 0.8613501881980827},
                          {'params': np.array([0.07347761, 0.68002963,
                                               1.00724559]),
                           'params_err': np.array([1.20673822e+04,
                                                   4.60490058e+04,
                                                   1.15476367e-02]),
                           'epc': 0.4031697796194298,
                           'epc_err': 123174.20450564621,
                           'pepc': 0.23997777961599784,
                           'pepc_err': 50787.13189860349}]
                 }}]

        for tst_index, tst in enumerate(tests_purity[0:1]):
            fo = open(tst['results_file'], 'rb')
            purity_result_list = pickle.load(fo)
            fo.close()

            # PurityRBFitter class
            rbfit_purity = PurityRBFitter(purity_result_list,
                                          tst['npurity'],
                                          tst['rb_opts']['xdata'],
                                          tst['rb_opts']['rb_pattern'])

            ydata = rbfit_purity.ydata
            fit = rbfit_purity.fit

            for i, _ in enumerate(ydata):
                self.assertTrue(all(np.isclose(a, b) for a, b in
                                    zip(ydata[i]['mean'],
                                        tst['expected']['ydata'][i]['mean'])),
                                'Incorrect mean in purity data test no. '
                                + str(tst_index))
                if tst['expected']['ydata'][i]['std'] is None:
                    self.assertIsNone(
                        ydata[i]['std'],
                        'Incorrect std in purity data test no. '
                        + str(tst_index))
                else:
                    self.assertTrue(
                        all(np.isclose(a, b) for a, b in zip(
                            ydata[i]['std'],
                            tst['expected']['ydata'][i]['std'])),
                        'Incorrect std in purity data test no. '
                        + str(tst_index))
                self.assertTrue(
                    all(np.isclose(a, b, atol=0.01) for a, b in zip(
                        fit[i]['params'],
                        tst['expected']['fit'][i]['params'])),
                    'Incorrect fit parameters in purity data test no. '
                    + str(tst_index) + ' ' + str(fit[i]['params']) + ' ' +
                    str(tst['expected']['fit'][i]['params']))
                self.assertTrue(
                    all(np.isclose(a, b, atol=0.01) for a, b in zip(
                        fit[i]['params_err'],
                        tst['expected']['fit'][i]['params_err'])),
                    'Incorrect fit error in purity data test no. '
                    + str(tst_index) + ' ' + str(fit[i]['params_err']) +
                    ' ' + str(tst['expected']['fit'][i]['params_err']))
                self.assertTrue(np.isclose(fit[i]['epc'],
                                           tst['expected']['fit'][i]['epc'],
                                           atol=0.01),
                                'Incorrect EPC in purity data test no. '
                                + str(tst_index))
                self.assertTrue(
                    np.isclose(fit[i]['epc_err'],
                               tst['expected']['fit'][i]['epc_err'],
                               atol=0.01),
                    'Incorrect EPC error in purity data test no. '
                    + str(tst_index))
                self.assertTrue(
                    np.isclose(fit[i]['pepc'],
                               tst['expected']['fit'][i]['pepc'],
                               atol=0.01),
                    'Incorrect PEPC in purity data test no. '
                    + str(tst_index))
                self.assertTrue(
                    np.isclose(fit[i]['pepc_err'],
                               tst['expected']['fit'][i]['pepc_err'],
                               atol=0.01),
                    'Incorrect PEPC error in purity data test no. '
                    + str(tst_index))


if __name__ == '__main__':
    unittest.main()
