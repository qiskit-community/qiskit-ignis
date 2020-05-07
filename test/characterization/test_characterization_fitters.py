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

# pylint: disable=no-name-in-module

"""
Test fitters of Ignis characterization
"""

import os
import pickle
import unittest

import numpy as np

from qiskit.ignis.characterization.coherence.fitters import T1Fitter, \
                                                            T2Fitter, \
                                                            T2StarFitter

from qiskit.ignis.characterization.hamiltonian.fitters import ZZFitter


class TestFitters(unittest.TestCase):
    """ Test the fitters """

    @unittest.skip('Pickle files are no longer valid')
    def test_fitters(self):
        """
        Test fitters of Ignis characterization
        """

        files_with_pickles = ['test_fitters_t1.pkl', 'test_fitters_t2.pkl',
                              'test_fitters_t2star.pkl',
                              'test_fitters_zz.pkl']

        for picfile in files_with_pickles:
            with open(os.path.join(os.path.dirname(__file__),
                                   picfile), 'rb') as file_obj:
                file_content = pickle.load(file_obj)

            fit_type = file_content['type']
            error_msg_prefix = ' in test of ' + fit_type

            input_to_fitter = file_content['input_to_fitter']
            if fit_type == 't1':
                fit = T1Fitter(**input_to_fitter)
            elif fit_type == 't2':
                fit = T2Fitter(**input_to_fitter)
            elif fit_type == 't2star':
                fit = T2StarFitter(**input_to_fitter)
            elif fit_type == 'zz':
                fit = ZZFitter(**input_to_fitter)
            else:
                raise NotImplementedError('Unrecognized fitter type '
                                          + fit_type)

            expected_fit = file_content['expected_fit']
            series = fit.series
            self.assertTrue(series == expected_fit.series and
                            sorted(series) ==
                            sorted(list(fit.params.keys())) and
                            sorted(series) ==
                            sorted(list(expected_fit.params.keys())),
                            'Incorrect series in ' + error_msg_prefix)

            num_of_qubits = len(input_to_fitter['qubits'])
            for serie in series:
                self.assertTrue(num_of_qubits ==
                                len(fit.params[serie]) and
                                num_of_qubits ==
                                len(expected_fit.params[serie]) and
                                num_of_qubits ==
                                len(fit.params_err[serie]) and
                                num_of_qubits ==
                                len(expected_fit.params_err[serie]),
                                'Error for serie ' + serie + error_msg_prefix)

                self.assertTrue(
                    all(
                        (all(np.isclose(a, b) for a, b in zip(
                            fit.params[serie][qubit],
                            expected_fit.params[serie][qubit]))
                         and
                         all(np.isclose(a, b) for a, b in zip(
                             fit.params_err[serie][qubit],
                             expected_fit.params_err[serie][qubit])))
                        for qubit in range(num_of_qubits)),
                    'Incorrect fit parameters for serie'
                    + serie + error_msg_prefix)


if __name__ == '__main__':
    unittest.main()
