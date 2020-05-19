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
import json
import unittest
import numpy as np

from qiskit.result import Result

from qiskit.ignis.characterization.coherence.fitters import T1Fitter, \
                                                            T2Fitter, \
                                                            T2StarFitter

from qiskit.ignis.characterization.hamiltonian.fitters import ZZFitter


class TestFitters(unittest.TestCase):
    """ Test the fitters """

    def test_t1_fitter(self):
        """
        Test T1 fitter in Ignis characterization

        This test relies on static data stored in the file t1_data.json.
        To generate t1_data.json, run 'python generate_data.py t1'.
        """

        with open(os.path.join(os.path.dirname(__file__), 't1_data.json'),
                  'r') as handle:
            data = json.load(handle)

        fit = T1Fitter(Result.from_dict(data['backend_result']),
                       data['xdata'], data['qubits'],
                       fit_p0=[1, data['t1'], 0],
                       fit_bounds=([0, 0, -1], [2, data['t1']*1.2, 1]))

        self.assertEqual(fit.series, ['0'])
        self.assertEqual(list(fit.params.keys()), ['0'])

        num_of_qubits = len(data['qubits'])
        self.assertEqual(len(fit.params['0']), num_of_qubits)
        self.assertEqual(len(fit.params_err['0']), num_of_qubits)

        for qubit in range(num_of_qubits):
            self.assertTrue(np.allclose(fit.params['0'][qubit],
                                        [1, data['t1'], 0],
                                        rtol=0.3,
                                        atol=0.1))

    def test_t2_fitter(self):
        """
        Test T2 fitter in Ignis characterization

        This test relies on static data stored in the file t2_data.json.
        To generate t2_data.json, run 'python generate_data.py t2'.
        """

        with open(os.path.join(os.path.dirname(__file__), 't2_data.json'),
                  'r') as handle:
            data = json.load(handle)

        fit = T2Fitter(Result.from_dict(data['backend_result']),
                       data['xdata'], data['qubits'],
                       fit_p0=[1, data['t2'], -0.5],
                       fit_bounds=([0, 0, -1], [2, data['t2']*1.2, 1]))

        self.assertEqual(fit.series, ['0'])
        self.assertEqual(list(fit.params.keys()), ['0'])

        num_of_qubits = len(data['qubits'])
        self.assertEqual(len(fit.params['0']), num_of_qubits)
        self.assertEqual(len(fit.params_err['0']), num_of_qubits)

        for qubit in range(num_of_qubits):
            self.assertTrue(np.allclose(fit.params['0'][qubit],
                                        [0.5, data['t2'], 0.5],
                                        rtol=0.3,
                                        atol=0.1))

    def test_t2star_fitter(self):
        """
        Test T2* fitter in Ignis characterization

        This test relies on static data stored in the file t2star_data.json.
        To generate t2star_data.json, run 'python generate_data.py t2star'.
        """

        with open(os.path.join(os.path.dirname(__file__), 't2star_data.json'),
                  'r') as handle:
            data = json.load(handle)

        fit = T2StarFitter(Result.from_dict(data['backend_result']),
                           data['xdata'], data['qubits'],
                           fit_p0=[0.5, data['t2'], data['omega'], 0, 0.5],
                           fit_bounds=([-0.5, 0, data['omega']-0.02, -np.pi, -0.5],
                                       [1.5, data['t2']*1.2, data['omega']+0.02,
                                        np.pi, 1.5]))

        self.assertEqual(fit.series, ['0'])
        self.assertEqual(list(fit.params.keys()), ['0'])

        num_of_qubits = len(data['qubits'])
        self.assertEqual(len(fit.params['0']), num_of_qubits)
        self.assertEqual(len(fit.params_err['0']), num_of_qubits)

        for qubit in range(num_of_qubits):
            self.assertTrue(np.allclose(fit.params['0'][qubit],
                                        [0.5, data['t2'], data['omega'], 0, 0.5],
                                        rtol=0.3,
                                        atol=0.1))

    def test_zz_fitter(self):
        """
        Test ZZ fitter in Ignis characterization

        This test relies on static data stored in the file zz_data.json.
        To generate zz_data.json, run 'python generate_data.py zz'.
        """

        with open(os.path.join(os.path.dirname(__file__), 'zz_data.json'),
                  'r') as handle:
            data = json.load(handle)

        fit = ZZFitter(Result.from_dict(data['backend_result']),
                       data['xdata'], data['qubits'], data['spectators'],
                       fit_p0=[0.5, data['omega'], 0, 0.5],
                       fit_bounds=([-0.5, 0, -np.pi, -0.5],
                                   [1.5, 2*data['omega'], np.pi, 1.5]))

        self.assertEqual(fit.series, ['0', '1'])
        self.assertEqual(sorted(list(fit.params.keys())), ['0', '1'])

        num_of_qubits = len(data['qubits'])
        self.assertEqual(len(fit.params['0']), num_of_qubits)
        self.assertEqual(len(fit.params_err['0']), num_of_qubits)

        self.assertTrue(np.isclose(fit.ZZ_rate(), data['zz'], rtol=0.3, atol=0.1))


if __name__ == '__main__':
    unittest.main()
