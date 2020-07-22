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
from typing import List

import networkx as nx
import numpy as np
import scipy as sp
from qiskit import execute, Aer, QuantumCircuit, QuantumRegister
from qiskit.ignis.mitigation.measurement.ctmp_method import *
from qiskit.ignis.verification.tomography.data import expectation_counts
from qiskit.providers.aer.noise.errors import ReadoutError
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.result import Result


class NoisyTest(unittest.TestCase):
    backend = Aer.get_backend('qasm_simulator')
    exec_opts = {
        'backend': backend,
        'shots': 8192 * 4,
        'noise_model': None
    }
    lo = 1e-3
    hi = 1e-2

    num_qubits = None

    def setUp(self):
        circ_set = WeightTwoCalibrationCircuitSet.from_num_qubits(self.num_qubits)
        cal_res = self.run_circs(circ_set.circs)
        gen_set = StandardGeneratorSet.from_num_qubits(self.num_qubits)
        self.cal = MeasurementCalibrator(cal_circ_set=circ_set, gen_set=gen_set)
        self.cal.calibrate(cal_res)
        r_dict = self.cal.r_dict
        gamma = self.cal.gamma

    def run_circs(self, circs: List[QuantumCircuit]) -> Result:
        # circs_ = deepcopy(circs)
        self.apply_ro_errors(circs, lo=self.lo, hi=self.hi)
        return execute(circs, **self.exec_opts).result()

    @staticmethod
    def get_2q_ro_error(err_rate: float) -> ReadoutError:
        r_dict = {}
        _gen_set = StandardGeneratorSet.from_num_qubits(2)
        for gen in _gen_set:
            if len(gen[2]) == 2:
                r_dict[gen] = err_rate
            elif len(gen[2]) == 1:
                r_dict[gen] = 0.0
            else:
                raise ValueError('')
        G = MeasurementCalibrator(None, _gen_set).total_g_matrix(r_dict).toarray()
        A = sp.linalg.expm(G)
        ro_error = ReadoutError(A)
        return ro_error

    @staticmethod
    def get_1q_ro_error(err_rate: float) -> ReadoutError:
        _gen_set = StandardGeneratorSet.from_num_qubits(1)
        r_dict = {}
        for gen in _gen_set:
            if len(gen[2]) == 2:
                r_dict[gen] = 0.0
            elif len(gen[2]) == 1:
                r_dict[gen] = err_rate
            else:
                raise ValueError('')
        G = MeasurementCalibrator(None, _gen_set).total_g_matrix(r_dict)
        A = sp.linalg.expm(G)
        ro_error = ReadoutError(A)
        return ro_error

    def apply_ro_errors(self, circs: List[QuantumCircuit], lo: float, hi: float):
        for circ in circs:
            self.apply_ro_error_ind(circ, lo=lo, hi=hi)

    def apply_ro_error_ind(self, circ: QuantumCircuit, lo: float, hi: float):
        num_qubits = len(circ.qregs[0])
        ro_error_1q = self.get_1q_ro_error(hi)
        ro_error_2q = self.get_2q_ro_error(lo)
        for i in range(num_qubits):
            circ.append(ro_error_1q, cargs=[i])

        for i, j in product(range(num_qubits), repeat=2):
            if i != j:
                circ.append(ro_error_2q, cargs=[i, j])


class ParityTest(NoisyTest):
    num_qubits = 4

    @staticmethod
    def str_par(s: str) -> int:
        return (-1) ** (s.count('1'))

    @staticmethod
    def get_rand_bits(num_qubits: int):
        rand_bits = ''.join(np.random.choice(['0', '1'], replace=True, size=num_qubits))
        return rand_bits

    @staticmethod
    def bits_to_circ(bitstring: str) -> QuantumCircuit:
        num_qubits = len(bitstring)
        q = QuantumRegister(num_qubits, 'q')
        circ = QuantumCircuit(q)
        for i, b in enumerate(bitstring[::-1]):
            if b == '1':
                circ.x(q[i])
        circ.measure_all()
        return circ

    def test_parity(self):
        bitstring = self.get_rand_bits(self.num_qubits)
        circ = self.bits_to_circ(bitstring)
        result = self.run_circs([circ])
        counts = result.get_counts(circ)

        shots = np.sum(list(counts.values()))
        raw_exp_val = expectation_counts(counts)['1' * self.num_qubits] / shots
        mit_exp_val = mitigated_expectation_value(self.cal, counts)
        ex_par = self.str_par(bitstring)

        self.assertEqual(np.sign(raw_exp_val), ex_par)
        self.assertEqual(np.sign(mit_exp_val), ex_par)


class TestExpValSymmetric(NoisyTest):
    num_qubits = 2

    op = SparsePauliOp.from_list([('XX', 1.0), ('YY', 1.0), ('ZZ', 1.0)])
    circ = QuantumCircuit(op.num_qubits)
    circ.h(0)
    circ.cx(0, 1)
    exact_exp_val = 1.0

    """Waiting on these tests until the generator/mitigator framework
    is finalized.
    """
    # def test_exp_val_end_to_end_no_mit(self):
    #    gen = ExpValGenerator(self.op)
    #    fit = ExpValFitter(self.op)
    #    result = self.run_circs(gen.generate_circuits(self.circ))
    #    mean, _ = fit.exp_val(result)
    #    self.assertEqual(self.exact_exp_val, mean)

    # def test_exp_val_end_to_end_with_mit(self):
    #    gen = CTMPExpValGenerator(self.op)
    #    fit = CTMPExpValFitter(self.op)
    #    result = self.run_circs(gen.generate_circuits(self.circ))
    #    mean, _ = fit.exp_val(result)
    #    self.assertEqual(self.exact_exp_val, mean)


class TestExpValAsymmetric(NoisyTest):
    num_qubits = 4
    graph = nx.generators.classic.cycle_graph(4)
    num_qubits = len(graph)
    circ = QuantumCircuit(num_qubits)
    circ.h(range(num_qubits))
    for i, j in graph.edges:
        circ.cz(i, j)

    op = SparsePauliOp.from_list([('ZIZX', 1.0)])
    exact_exp_val = 1.0

    """Waiting on these tests until the generator/mitigator framework
    is finalized.
    """
    # def test_exp_val_end_to_end_no_mit(self):
    #    gen = ExpValGenerator(self.op)
    #    fit = ExpValFitter(self.op)
    #    result = self.run_circs(gen.generate_circuits(self.circ))
    #    mean, _ = fit.exp_val(result)
    #    self.assertEqual(self.exact_exp_val, mean)
    #    pass

    # def test_a_exp_val_end_to_end_with_mit(self):
    #    gen = CTMPExpValGenerator(self.op)
    #    fit = CTMPExpValFitter(self.op)
    #    result = self.run_circs(gen.generate_circuits(self.circ))
    #    mean, _ = fit.exp_val(result)
    #    self.assertEqual(self.exact_exp_val, mean)
    #    pass


if __name__ == '__main__':
    unittest.main()
