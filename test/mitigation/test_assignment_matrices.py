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
Test assignment matrices
"""

import unittest
from typing import List
from functools import lru_cache
from itertools import product

from ddt import ddt, unpack, data
import numpy as np

from qiskit import QuantumCircuit
from qiskit.ignis.mitigation.measurement import (
    MeasMitigatorGenerator,
    MeasMitigatorFitter
)

from test_mitigators import NoisySimulationTest


_QUBIT_SUBSETS = [(1, 0, 2), (3, 0), (2,), (0, 3, 1, 2)]
_PAIRINGS = [
    *product(['complete', 'tensored'], _QUBIT_SUBSETS, [True, False]),
    ('ctmp', None, True),
    ('ctmp', None, False),
]


@ddt
class TestMatrices(NoisySimulationTest):
    """Test that assignment matrices are computed correctly (close to identity)
    and produce correct expectation values.
    """

    @lru_cache(6)
    def get_mitigator(self, method: str, noise_model: bool):
        """Return the mitigator for the given parameters.
        """
        circs, meta, _ = MeasMitigatorGenerator(self.num_qubits, method=method).run()
        cal_res = self.execute_circs(circs, noise_model=self.noise_model if noise_model else None)
        mitigator = MeasMitigatorFitter(cal_res, meta).fit(method=method)
        return mitigator

    @data(*_PAIRINGS)
    @unpack
    def test_matrices(self, method: str, qubits: List[int], noise: bool):
        """Compute and compare matrices with given options.
        """
        if qubits is not None:
            num_qubits = len(qubits)
        else:
            num_qubits = self.num_qubits
        mitigator = self.get_mitigator(method, noise_model=noise)
        assignment_matrix = mitigator.assignment_matrix(qubits)
        mitigation_matrix = mitigator.mitigation_matrix(qubits)

        np.testing.assert_array_almost_equal(
            assignment_matrix,
            np.eye(2**num_qubits, 2**num_qubits),
            decimal=1 if noise else 6
        )

        np.testing.assert_array_almost_equal(
            mitigation_matrix,
            np.eye(2**num_qubits, 2**num_qubits),
            decimal=1 if noise else 6
        )

    @data(
        *product(
            ['complete', 'tensored'],
            ['1011', '1010', '1111'],
            [[0, 1, 2], [1], [1, 0, 3, 2], None]
        ),
        *product(
            ['ctmp'],
            ['1011', '1010', '1111'],
            [None]
        )
    )
    @unpack
    def test_parity_exp_vals_partial(self, method: str, bitstr: str, qubits: List[int]):
        """Compute expectation value of parity operators and
        compare with exact result.
        """
        mitigator = self.get_mitigator(method, True)

        if qubits is None:
            exp_val_exact = (-1)**(bitstr.count('1'))
        else:
            exp_val_exact = 1
            for q in qubits:
                exp_val_exact *= (-1)**int(bitstr[::-1][q])

        qc = QuantumCircuit(self.num_qubits, self.num_qubits)
        for i, b in enumerate(bitstr[::-1]):
            if b == '1':
                qc.x(i)
        qc.measure(range(4), range(4))

        counts = self.execute_circs(qc, noise_model=self.noise_model).get_counts(0)

        exp_val_mit = mitigator.expectation_value(
            counts=counts,
            clbits=qubits,
            qubits=qubits
        )

        self.assertLessEqual(abs(exp_val_exact - exp_val_mit), 0.1)


if __name__ == '__main__':
    unittest.main()
