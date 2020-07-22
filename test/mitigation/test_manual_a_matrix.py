
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
CTMP Method tests
"""
import unittest
from itertools import product
from typing import Dict

import numpy as np
import scipy as sp
from qiskit import Aer
from qiskit import execute
from qiskit.ignis.mitigation.measurement.ctmp_method import *
from qiskit.ignis.mitigation.measurement.ctmp_method.calibration import (
    BaseCalibrationCircuitSet,
    BaseGeneratorSet,
    Generator
)
from qiskit.providers.aer.noise import ReadoutError


def my_expm(mat: np.array, order: int = 200) -> np.array:
    if mat.shape[0] != mat.shape[1]:
        raise ValueError('Needs square matrix')
    res = np.eye(mat.shape[0], dtype=np.float64)
    mat_pow = res
    for k in range(1, order):
        mat_pow = mat @ mat_pow
        res += mat_pow / sp.special.factorial(k)
    return res


def compute_a_matrix(num_qubits: int, exec_opts: dict, err: ReadoutError = None) -> np.array:
    bitstrings = [''.join(bits) for bits in product(['0', '1'], repeat=num_qubits)]
    circ_set = BaseCalibrationCircuitSet(num_qubits)
    circs_dict = {bits: circ_set.bitstring_to_circ(bits) for bits in bitstrings}
    circs = list(circs_dict.values())
    if err is not None:
        for c in circs:
            c.append(err, cargs=[0, 1])

    res = execute(circs, **exec_opts).result()

    A = np.zeros((2 ** num_qubits, 2 ** num_qubits))
    for b_in, b_out in product(bitstrings, repeat=2):
        counts = res.get_counts('cal-{}'.format(b_in))
        shots = np.sum(list(counts.values()))
        try:
            c = counts[b_out]
        except KeyError:
            c = 0
        A[int(b_in, 2), int(b_out, 2)] = c / shots
    A = A.T
    return A, res, circs_dict


def noise_matrix_test(noise_mat, exec_opts, num_qubits):
    ro_error = ReadoutError(noise_mat)
    A, a_cal_res, a_cal_circs_dict = compute_a_matrix(num_qubits, exec_opts, ro_error)
    gen_set = StandardGeneratorSet.from_num_qubits(num_qubits)
    cir_set = BaseCalibrationCircuitSet.from_dict(num_qubits, a_cal_circs_dict)
    cal_res = a_cal_res
    cal = MeasurementCalibrator(cir_set, gen_set)
    cal.calibrate(cal_res)
    G = cal.total_g_matrix(cal.r_dict).toarray()
    A_empirical = sp.linalg.expm(G)
    # A_empirical = my_expm(G)
    dist = np.linalg.norm(A - A_empirical)

    return {
        'A': A,
        'A_empirical': A_empirical,
        'G': G,
        'cal': cal,
        'dist': dist,
        'r_dict': cal.r_dict
    }


class TestCompareAMatrixGeneration(unittest.TestCase):
    backend = Aer.get_backend('qasm_simulator')
    num_qubits = 2
    exec_opts = {
        'backend': backend,
        'shots': 2 ** 18
    }
    eps = 1e-2
    eps_0 = 1e-5
    eps_1 = 1e-2
    tol_decs = 3

    def dist(self, mat_1, mat_2):
        return np.max(np.abs(mat_1 - mat_2))


class Test_2(TestCompareAMatrixGeneration):

    def setUp(self):
        self.res = None
        self.noise_mat = None

    def tearDown(self):
        self.res = noise_matrix_test(self.noise_mat, self.exec_opts, self.num_qubits)
        dist = self.dist(self.res['A_empirical'], self.res['A'])
        np.testing.assert_array_almost_equal(
            self.res['A_empirical'],
            self.res['A'],
            decimal=self.tol_decs
        )

    def test_uniform_noise(self):
        noise_mat = np.ones((4, 4)) * self.eps
        for i in range(4):
            noise_mat[i, i] = 1 - self.eps * (4 - 1)
        self.noise_mat = noise_mat

    def test_symmetric_correlated_noise(self):
        noise_mat = np.eye(4)
        noise_mat[0, 0] -= self.eps
        noise_mat[3, 3] -= self.eps
        noise_mat[0, 3] += self.eps
        noise_mat[3, 0] += self.eps
        self.noise_mat = noise_mat

    def test_symmetric_uncorrelated_noise(self):
        noise_mat = np.eye(2)
        noise_mat[0, 0] = 1 - self.eps
        noise_mat[0, 1] = self.eps
        noise_mat[1, 0] = self.eps
        noise_mat[1, 1] = 1 - self.eps
        noise_mat = np.kron(noise_mat, noise_mat)
        self.noise_mat = noise_mat

    def test_asymmetric_correlated_noise(self):
        noise_mat = np.eye(4)
        noise_mat[0, 0] -= self.eps_0
        noise_mat[3, 3] -= self.eps_1
        noise_mat[0, 3] += self.eps_0
        noise_mat[3, 0] += self.eps_1
        self.noise_mat = noise_mat

    def test_asymmetric_uncorrelated_noise(self):
        noise_mat_0 = np.eye(2)
        noise_mat_0[0, 0] = 1 - self.eps_0
        noise_mat_0[0, 1] = self.eps_0
        noise_mat_0[1, 0] = self.eps_0
        noise_mat_0[1, 1] = 1 - self.eps_0

        noise_mat_1 = np.eye(2)
        noise_mat_1[0, 0] = 1 - self.eps_1
        noise_mat_1[0, 1] = self.eps_1
        noise_mat_1[1, 0] = self.eps_1
        noise_mat_1[1, 1] = 1 - self.eps_1

        noise_mat = np.kron(noise_mat_1, noise_mat_0)
        self.noise_mat = noise_mat


def compare_a_matrix_methods(
        r_dict: Dict[Generator, float],
        cir_set: BaseCalibrationCircuitSet,
        num_qubits: int,
        exec_opts: Dict
):
    # Determine A from generators
    gen_set = list(r_dict.keys())
    gen_set = BaseGeneratorSet.from_generator_list(gen_set, num_qubits)
    cal = MeasurementCalibrator(cir_set, gen_set)
    G = cal.total_g_matrix(r_dict).toarray()
    A = sp.linalg.expm(G)
    # A = my_expm(G)

    # Determine A from noise matrix
    ro_error = ReadoutError(A)
    A_empirical, _, _ = compute_a_matrix(num_qubits, exec_opts, err=ro_error)
    dist = np.linalg.norm(A - A_empirical)

    return {
        'A': A,
        'A_empirical': A_empirical,
        'dist': dist
    }


class Test_1(TestCompareAMatrixGeneration):

    def setUp(self):
        self.cir_set = StandardCalibrationCircuitSet.from_num_qubits(self.num_qubits)
        self.r_dict = None
        self.res = None

    def tearDown(self):
        self.res = compare_a_matrix_methods(self.r_dict, self.cir_set, self.num_qubits,
                                            self.exec_opts)
        dist = self.dist(self.res['A'], self.res['A_empirical'])
        np.testing.assert_array_almost_equal(
            self.res['A_empirical'],
            self.res['A'],
            decimal=self.tol_decs
        )

    def test_2_single_qubit_symmetric(self):
        self.r_dict = {
            ('1', '0', (0,)): self.eps,
            ('0', '1', (0,)): self.eps,
            ('1', '0', (1,)): self.eps,
            ('0', '1', (1,)): self.eps,
        }

    def test_2_single_qubit_asymmetric(self):
        self.r_dict = {
            ('1', '0', (0,)): self.eps_0,
            ('0', '1', (0,)): self.eps_0,
            ('1', '0', (1,)): self.eps_1,
            ('0', '1', (1,)): self.eps_1,
        }

    def test_1_two_qubit_symmetric(self):
        self.r_dict = {
            ('11', '00', (0, 1)): self.eps,
            ('00', '11', (0, 1)): self.eps,
            ('10', '01', (0, 1)): self.eps,
            ('10', '01', (1, 0)): self.eps
        }

    def test_1_two_qubit_asymmetric(self):
        self.r_dict = {
            ('11', '00', (0, 1)): self.eps_0,
            ('00', '11', (0, 1)): self.eps_0,
            ('10', '01', (0, 1)): self.eps_1,
            ('10', '01', (1, 0)): self.eps_1
        }

    def test_3_all(self):
        self.r_dict = {
            ('1', '0', (0,)): self.eps,
            ('0', '1', (0,)): self.eps,
            ('1', '0', (1,)): self.eps,
            ('0', '1', (1,)): self.eps,
            ('11', '00', (0, 1)): self.eps,
            ('00', '11', (0, 1)): self.eps,
            ('10', '01', (0, 1)): self.eps,
            ('10', '01', (1, 0)): self.eps
        }


if __name__ == '__main__':
    unittest.main()
