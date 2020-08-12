
# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
CTMP markov sampling tests
"""
import unittest

import scipy as sp
from qiskit.ignis.mitigation.measurement.ctmp_method.markov_compiled import markov_chain_int
from qiskit.ignis.mitigation.measurement import CTMPMeasMitigator


def statistical_test(num_tests: int, fraction_passes: float):
    """Wrapper for statistical test that should be run multiple times and pass
    at least a certain fraction of times.
    """
    # pylint: disable=bare-except
    def stat_test(func):
        def wrapper_func(*args, **kwargs):

            num_failures = 0
            num_passes = 0
            for _ in range(num_tests):
                try:
                    func(*args, **kwargs)
                    num_passes += 1
                except:
                    num_failures += 1
            if num_passes / num_tests < fraction_passes:
                raise ValueError('Passed {} out of {} trials, needed {}%'.format(
                    num_passes,
                    num_tests,
                    100 * fraction_passes
                ))

        return wrapper_func

    return stat_test


class TestMarkov(unittest.TestCase):
    """Test the (un)compiled markov process simulator
    """

    def setUp(self):
        self.r_dict = {
            # (final, start, qubits)
            ('0', '1', (0,)): 1e-3,
            ('1', '0', (0,)): 1e-1,
            ('0', '1', (1,)): 1e-3,
            ('1', '0', (1,)): 1e-1,

            ('00', '11', (0, 1)): 1e-3,
            ('11', '00', (0, 1)): 1e-1,
            ('01', '10', (0, 1)): 1e-3,
            ('10', '01', (0, 1)): 1e-3
        }

        mitigator = CTMPMeasMitigator(
            generators=self.r_dict.keys(),
            rates=self.r_dict.values(),
            num_qubits=2
            )
        mitigator._compute_g_mat()
        self.trans_mat = sp.linalg.expm(mitigator._g_mat)

        self.num_steps = 100

    @statistical_test(50, 0.7)
    def test_markov_chain_int_0(self):
        """Test markov process starting at specific state"""
        y = markov_chain_int(self.trans_mat, 0, self.num_steps)
        self.assertEqual(y, 3)

    @statistical_test(50, 0.7)
    def test_markov_chain_int_1(self):
        """Test markov process starting at specific state"""
        y = markov_chain_int(self.trans_mat, 1, self.num_steps)
        self.assertEqual(y, 3)

    @statistical_test(50, 0.7)
    def test_markov_chain_int_2(self):
        """Test markov process starting at specific state"""
        y = markov_chain_int(self.trans_mat, 2, self.num_steps)
        self.assertEqual(y, 3)

    @statistical_test(50, 0.7)
    def test_markov_chain_int_3(self):
        """Test markov process starting at specific state"""
        y = markov_chain_int(self.trans_mat, 3, self.num_steps)
        self.assertEqual(y, 3)


if __name__ == '__main__':
    unittest.main()
