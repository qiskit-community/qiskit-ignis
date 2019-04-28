# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=undefined-loop-variable

"""
Run through Quantum volume
"""

import unittest
import os
import pickle
import qiskit.ignis.verification.quantum_volume as qv


class TestQV(unittest.TestCase):
    """ The test class """

    def test_qv_circuits(self):
        """ Test circuit generation """

        # Qubit list
        qubit_list = [0, 1, 2, 4, 7]
        depth_list = [1, 2, 3, 4, 5, 6, 7]
        ntrials = 5

        qv_circs, _ = qv.qv_circuits(qubit_list,
                                     depth_list, ntrials)

        self.assertEqual(len(qv_circs), ntrials,
                         "Error: Not enough trials")

        self.assertEqual(len(qv_circs[0]), len(depth_list),
                         "Error: Not enough circuits for the "
                         "number of specified depths")

    def test_qv_fitter(self):

        """ Test the fitter with some pickled result data"""

        os.path.join(os.path.dirname(__file__),
                     'test_fitter_results_2.pkl')

        f0 = open(os.path.join(os.path.dirname(__file__),
                               'qv_ideal_results.pkl'), 'rb')
        ideal_results = pickle.load(f0)
        f0.close()

        f0 = open(os.path.join(os.path.dirname(__file__),
                               'qv_exp_results.pkl'), 'rb')
        exp_results = pickle.load(f0)
        f0.close()

        qubit_list = [0, 1, 2, 4, 7]
        depth_list = [1, 2, 3, 4, 5, 6, 7]

        qv_fitter = qv.QVFitter(qubit_list=qubit_list, depth_list=depth_list)
        qv_fitter.add_statevectors(ideal_results)
        qv_fitter.add_data(exp_results)

        qv_success_list = qv_fitter.qv_success()
        self.assertTrue(qv_success_list[4][0])


if __name__ == '__main__':
    unittest.main()
