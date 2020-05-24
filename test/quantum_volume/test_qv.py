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

# pylint: disable=undefined-loop-variable,invalid-name

"""
Run through Quantum volume
"""

import unittest
import os
from test.utils import load_results_from_json
import qiskit.ignis.verification.quantum_volume as qv


class TestQV(unittest.TestCase):
    """ The test class """

    def test_qv_circuits(self):
        """ Test circuit generation """

        # Qubit list
        qubit_lists = [[0, 1, 2], [0, 1, 2, 4], [0, 1, 2, 4, 7]]
        ntrials = 5

        qv_circs, _ = qv.qv_circuits(qubit_lists, ntrials)

        self.assertEqual(len(qv_circs), ntrials,
                         "Error: Not enough trials")

        self.assertEqual(len(qv_circs[0]), len(qubit_lists),
                         "Error: Not enough circuits for the "
                         "number of specified qubit lists")

    def test_qv_fitter(self):

        """ Test the fitter with some result data pre-saved as json"""

        ideal_results = load_results_from_json(os.path.join(os.path.dirname(__file__),
                                                            'qv_ideal_results.json'))
        exp_results = load_results_from_json(os.path.join(os.path.dirname(__file__),
                                                          'qv_exp_results.json'))

        qubit_lists = [[0, 1, 3], [0, 1, 3, 5], [0, 1, 3, 5, 7],
                       [0, 1, 3, 5, 7, 10]]

        qv_fitter = qv.QVFitter(qubit_lists=qubit_lists)
        qv_fitter.add_statevectors(ideal_results)
        qv_fitter.add_data(exp_results)

        qv_success_list = qv_fitter.qv_success()
        self.assertFalse(qv_success_list[0][0])


if __name__ == '__main__':
    unittest.main()
