# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
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
Test entanglement functions of Ignis verification
"""

import unittest

import numpy as np

from qiskit.ignis.verification.entanglement.linear import *

class TestEntanglement(unittest.TestCase):
    """ Test the entanglement circuits """

    def test_entanglement(self):
        """
        Test entanglement circuits of Ignis verification - GHZ, MQC, parity oscillations
        """
        qn = 5  # number of qubits
        sim = BasicAer.get_backend('qasm_simulator')

        #test simple GHZ
        circ = get_ghz_simple(qn, measure=True)
        counts = execute(circ, sim).result().get_counts(circ)
        self.assertTrue((counts['00000'] + counts['11111']) == 1024)

        # test MQC
        circ, delta = get_ghz_mqc_para(qn, measure='full')
        theta_range = np.linspace(0, 2 * np.pi, 16)
        circuits = [circ.bind_parameters({delta: theta_val})
                    for theta_val in theta_range]
        for circ in circuits:
            counts = execute(circ, sim).result().get_counts(circ)
            self.assertTrue((counts['00000'] == 1024) or
                            (counts['00000'] + counts['00001']) == 1024)

        # test parity oscillations
        circ, params = get_ghz_po_para(qn, measure='full')
        theta_range = np.linspace(0, 2 * np.pi, 16)
        circuits = [circ.bind_parameters({params[0]: theta_val, params[1]: -theta_val})
                    for theta_val in theta_range]
        gap_factor = 2.0/3
        for circ in circuits:
            counts = execute(circ, sim).result().get_counts(circ)
            even_counts = 0
            odd_counts = 0
            for key in counts:
                if (key.count('1') % 2 == 0):
                    even_counts += counts[key]
                else:
                    odd_counts += counts[key]
            self.assertTrue((even_counts > gap_factor*1024) or odd_counts > gap_factor*1024)

if __name__ == '__main__':
    unittest.main()
