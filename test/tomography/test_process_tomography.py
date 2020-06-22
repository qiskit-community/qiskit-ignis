# -*- coding: utf-8 -*-
#
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

# pylint: disable=missing-docstring
# pylint: disable=unexpected-keyword-arg
# pylint: disable=invalid-name

import unittest

import qiskit
from qiskit import QuantumRegister, QuantumCircuit, Aer
from qiskit.quantum_info import state_fidelity
from qiskit.quantum_info import Choi

import qiskit.ignis.verification.tomography as tomo
from qiskit.ignis.verification.tomography.fitters import cvx_fit


def run_circuit_and_tomography(circuit, qubits, method):
    choi_ideal = Choi(circuit).data
    qst = tomo.process_tomography_circuits(circuit, qubits)
    job = qiskit.execute(qst, Aer.get_backend('qasm_simulator'),
                         shots=5000)
    tomo_fit = tomo.ProcessTomographyFitter(job.result(), qst)
    choi = tomo_fit.fit(method=method).data
    return (choi, choi_ideal)


class TestProcessTomography(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.method = 'lstsq'

    def test_bell_2_qubits(self):
        q2 = QuantumRegister(2)
        bell = QuantumCircuit(q2)
        bell.h(q2[0])
        bell.cx(q2[0], q2[1])

        choi, choi_ideal = run_circuit_and_tomography(bell, q2, self.method)
        F_bell = state_fidelity(choi_ideal/4, choi/4, validate=False)
        self.assertAlmostEqual(F_bell, 1, places=1)


@unittest.skipUnless(cvx_fit._HAS_CVX, 'cvxpy is required for this test')
class TestProcessTomographyCVX(TestProcessTomography):
    def setUp(self):
        super().setUp()
        self.method = 'cvx'


if __name__ == '__main__':
    unittest.main()
