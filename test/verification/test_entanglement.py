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
        Test entanglement circuits of Ignis verification -
            GHZ, MQC, parity oscillations
        """
        qn = 5  # number of qubits
        sim = BasicAer.get_backend('qasm_simulator')

        # test simple GHZc
        circ = get_ghz_simple(qn, measure=True)
        counts = execute(circ, sim, shots=1024).result().get_counts(circ)
        self.assertTrue(counts.get('00000', 0) + counts.get('11111', 0) == 1024)

        # test MQC
        circ, delta = get_ghz_mqc_para(qn)
        theta_range = np.linspace(0, 2 * np.pi, 16)
        circuits = [circ.bind_parameters({delta: theta_val})
                    for theta_val in theta_range]
        for circ in circuits:
            counts = execute(circ, sim, shots=1024).result().get_counts(circ)
            self.assertTrue((counts.get('00000', 0) == 1024) or
                            (counts.get('00000', 0) + counts.get('00001', 0)) == 1024)

        # test parity oscillations
        circ, params = get_ghz_po_para(qn)
        theta_range = np.linspace(0, 2 * np.pi, 16)
        circuits = [circ.bind_parameters({params[0]: theta_val,
                                          params[1]: -theta_val})
                    for theta_val in theta_range]
        for circ in circuits:
            counts = execute(circ, sim, shots=1024).result().get_counts(circ)
            even_counts = sum(key.count('1') % 2 == 0 for key in counts.keys())
            odd_counts = sum(key.count('1') % 2 == 1 for key in counts.keys())

        self.assertTrue(even_counts in (0, 16))
        self.assertTrue(odd_counts in (0, 16))


if __name__ == '__main__':
    unittest.main()
