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

import numpy as np
from qiskit import Aer, QuantumCircuit, QuantumRegister
from qiskit import execute
from qiskit.ignis.mitigation.measurement.ctmp_method import *
from qiskit.ignis.mitigation.measurement.ctmp_method.calibration import (
    tensor_list,
    _KET_BRA_DICT,
    generator_to_sparse_matrix,
    local_a_matrix,
    compute_gamma
)
from qiskit.ignis.verification.tomography.data import expectation_counts
from qiskit.providers.aer.noise import NoiseModel, ReadoutError
from scipy import sparse


def random_bitstring(n: int) -> str:
    return ''.join(list(np.random.choice(['0', '1'], size=n, replace=True)))


class CTMPFuncs(unittest.TestCase):

    def setUp(self):
        self.sigma_plus = np.array([[0, 0], [1, 0]])
        self.eye = np.eye(2)

    def test_tensor_list(self):
        l = [self.sigma_plus, self.eye]
        np.testing.assert_array_equal(
            sparse.kron(l[0], l[1]).toarray(),
            tensor_list(l).toarray()
        )

    def test_gen_to_mat_trivial(self):
        # Zero matrix 1q
        gen = ('0', '0', (0,))
        expected_mat = np.zeros((2, 2))
        np.testing.assert_array_almost_equal(
            expected_mat,
            generator_to_sparse_matrix(gen, num_qubits=1)
        )
        # Zero matrix 2q
        gen = ('00', '00', (0, 1))
        expected_mat = np.zeros((4, 4))
        np.testing.assert_array_almost_equal(
            expected_mat,
            generator_to_sparse_matrix(gen, num_qubits=2).toarray()
        )

    def test_gen_to_mat(self):
        """Non-trivial 3q
        b = '10', a = '01', j=1, k=2, n=3
        |b><a|-|a><a| == id kron (|1><0| kron |0><1| - |0><0| kron |1><1|)
        """
        gen = ('10', '01', (1, 2))
        res = generator_to_sparse_matrix(gen, num_qubits=3).toarray()
        part_1 = tensor_list([
                                 self.eye,
                                 _KET_BRA_DICT['10'],
                                 _KET_BRA_DICT['01']
                             ][::-1]).toarray()
        part_2 = tensor_list([
                                 self.eye,
                                 _KET_BRA_DICT['00'],
                                 _KET_BRA_DICT['11']
                             ][::-1]).toarray()
        np.testing.assert_array_equal(part_1 - part_2, res)

    def test_local_a_mat(self):
        cdd = {
            '00': {'00': 9, '01': 1},
            '01': {'01': 7, '00': 3},
            '10': {'10': 7, '00': 3},
            '11': {'11': 9, '10': 1}
        }
        a_mat = local_a_matrix(0, 1, cdd)


class CTMPMultiQubitErrorRateTest(unittest.TestCase):
    num_qubits = 3
    backend = Aer.get_backend('qasm_simulator')
    # noise_model = NoiseModel()
    eps = 1e-4
    noise_mat = np.ones((4, 4)) * eps
    for i in range(4):
        noise_mat[i, i] = 1 - eps * (4 - 1)
    ro_error = ReadoutError(noise_mat)

    exec_opts = {
        # 'noise_model': noise_model,
        'backend': backend,
        'shots': 8192 * 2 ** 2
    }

    gen_set = StandardGeneratorSet.from_num_qubits(num_qubits)
    cir_set = StandardCalibrationCircuitSet.from_num_qubits(num_qubits)

    circs_to_exec = cir_set.circs

    for circ in circs_to_exec:
        circ.append(ro_error, cargs=[0, 1])

    cal_res = execute(circs_to_exec, **exec_opts).result()

    cal = MeasurementCalibrator(cir_set, gen_set)
    cal.calibrate(cal_res)


class CTMPAsymmetricErrorRates(unittest.TestCase):
    num_qubits = 2

    backend = Aer.get_backend('qasm_simulator')
    noise_model = NoiseModel(basis_gates=['u3', 'cx'])
    for j in range(2):
        eps = 1e-2
        if j == 0:
            eps = 1e-5
        if j == 1:
            eps = 1e-2
        noise_mat = np.ones((2, 2)) * eps
        for i in range(2):
            noise_mat[i, i] = 1 - eps * (2 - 1)
        ro_error = ReadoutError(noise_mat)
        noise_model.add_readout_error(ro_error, qubits=[j])

    exec_opts = {
        'basis_gates': ['u3', 'cx'],
        'noise_model': noise_model,
        'backend': backend,
        'shots': 8192 * 2 ** 2
    }

    gen_set = StandardGeneratorSet.from_num_qubits(num_qubits)
    cir_set = StandardCalibrationCircuitSet.from_num_qubits(num_qubits)

    cal_res = execute(cir_set.circs, **exec_opts).result()

    cal = MeasurementCalibrator(cir_set, gen_set)
    cal.calibrate(cal_res)

    def test_asymmetric_error_rate(self):
        r_dict = self.cal.r_dict
        self.assertLess(1, 2)
        self.assertLess(
            r_dict[('1', '0', (0,))],
            r_dict[('1', '0', (1,))],
        )
        self.assertLess(
            r_dict[('0', '1', (0,))],
            r_dict[('0', '1', (1,))],
        )


class CTMPTest(unittest.TestCase):
    num_qubits = 5

    backend = Aer.get_backend('qasm_simulator')
    noise_model = NoiseModel(basis_gates=['u3', 'cx'])
    eps = 1e-2
    noise_mat = np.ones((2, 2)) * eps
    for i in range(2):
        noise_mat[i, i] = 1 - eps * (2 - 1)
    ro_error = ReadoutError(noise_mat)
    noise_model.add_all_qubit_readout_error(ro_error)

    exec_opts = {
        'basis_gates': ['u3', 'cx'],
        'noise_model': noise_model,
        'backend': backend,
        'shots': 8192 * 2 ** 2
    }

    gen_set = StandardGeneratorSet.from_num_qubits(num_qubits)
    cir_set = StandardCalibrationCircuitSet.from_num_qubits(num_qubits)

    cal_res = execute(cir_set.circs, **exec_opts).result()

    cal = MeasurementCalibrator(cir_set, gen_set)
    cal.calibrate(cal_res)


class CTMPCalibration(CTMPTest):

    def test_matrix_log(self):
        # Branch test
        a_mat = np.eye(4)
        np.testing.assert_array_equal(
            np.zeros((4, 4)),
            self.gen_set.amat_to_gmat(a_mat)
        )

    def test_gamma_computation_diagonal(self):
        # Trivial test with no off-diagonal elements
        vec = np.random.uniform(-1, 1, size=2 ** self.num_qubits)
        gamma = np.max(-vec)
        mat = np.diag(vec)
        sparse_mat = sparse.csr_matrix(mat)
        res = compute_gamma(sparse_mat)
        self.assertEqual(res, gamma)

    def test_gamma_computation_off_diagonal(self):
        # Trivial test with off-diagonal elements
        mat = np.random.uniform(-1, 1, size=(
            2 ** self.num_qubits, 2 ** self.num_qubits))  # type: np.ndarray
        vec = mat.diagonal()
        gamma = np.max(-vec)
        sparse_mat = sparse.csr_matrix(mat)
        res = compute_gamma(sparse_mat)
        self.assertEqual(res, gamma)

    def test_calibration_end_to_end(self):
        gamma, r_dict = self.cal.gamma, self.cal.r_dict


class CTMPExpValParity(CTMPTest):

    def setUp(self):
        self.q = QuantumRegister(self.num_qubits, 'q')
        self.circ = QuantumCircuit(self.q)
        self.rand_bits = random_bitstring(self.num_qubits)
        self.exact_parity = (-1) ** self.rand_bits.count('1')
        for i, b in enumerate(self.rand_bits):
            if b == '1':
                self.circ.x(self.q[i])
        self.circ.measure_all()
        self.result = execute(self.circ, **self.exec_opts).result()
        self.counts = self.result.get_counts(self.circ)

    def test_noisy_parity_no_mitigation(self):
        shots = np.sum(list(self.counts.values()))
        expect_counts = expectation_counts(self.counts)
        z_parity = expect_counts['1' * self.num_qubits] / shots

        self.assertEqual(
            np.sign(z_parity),
            self.exact_parity
        )

    def test_mit_exp_val(self):
        exp_val = mitigated_expectation_value(self.cal, self.counts)
        self.assertEqual(
            np.sign(exp_val),
            self.exact_parity
        )


if __name__ == '__main__':
    unittest.main()
