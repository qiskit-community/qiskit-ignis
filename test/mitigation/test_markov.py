
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

import numpy as np
import scipy as sp
from qiskit.ignis.mitigation.measurement import CTMPMeasMitigator
from qiskit.ignis.mitigation.measurement.ctmp_method.ctmp_mitigator import _markov_chain_compiled


def statistical_test(num_tests: int, fraction_passes: float):
    """Wrapper for statistical test that should be run multiple times and pass
    at least a certain fraction of times.
    """
    # pylint: disable=broad-except
    def stat_test(func):
        def wrapper_func(*args, **kwargs):

            num_failures = 0
            num_passes = 0
            for _ in range(num_tests):
                try:
                    func(*args, **kwargs)
                    num_passes += 1
                except Exception:
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
        self.trans_mat = sp.linalg.expm(mitigator.generator_matrix()).tocsc()
        self.num_steps = 100

    def markov_chain_int(self, x):
        """Convert input for markov_chain_compiled function"""
        y = np.array([0])
        x = np.array([x])
        alpha = np.array([self.num_steps])
        rng = np.random.default_rng(seed=x)
        r_vals = rng.random(size=alpha.sum())
        values = self.trans_mat.data
        indices = np.asarray(self.trans_mat.indices, dtype=np.int)
        indptrs = np.asarray(self.trans_mat.indptr, dtype=np.int)
        _markov_chain_compiled(y, x, r_vals, alpha, values, indices, indptrs)
        return y[0]

    @statistical_test(50, 0.7)
    def test_markov_chain_int_0(self):
        """Test markov process starting at specific state"""
        self.assertEqual(self.markov_chain_int(0), 3)

    @statistical_test(50, 0.7)
    def test_markov_chain_int_1(self):
        """Test markov process starting at specific state"""
        self.assertEqual(self.markov_chain_int(1), 3)

    @statistical_test(50, 0.7)
    def test_markov_chain_int_2(self):
        """Test markov process starting at specific state"""
        self.assertEqual(self.markov_chain_int(2), 3)

    @statistical_test(50, 0.7)
    def test_markov_chain_int_3(self):
        """Test markov process starting at specific state"""
        self.assertEqual(self.markov_chain_int(3), 3)


if __name__ == '__main__':
    unittest.main()
