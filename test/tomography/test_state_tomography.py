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

import numpy
import qiskit
from qiskit import QuantumRegister, QuantumCircuit, Aer
from qiskit.quantum_info import state_fidelity, partial_trace, Statevector
import qiskit.ignis.verification.tomography as tomo
import qiskit.ignis.verification.tomography.fitters.cvx_fit as cvx_fit


def run_circuit_and_tomography(circuit, qubits, method='lstsq'):
    psi = Statevector.from_instruction(circuit)
    qst = tomo.state_tomography_circuits(circuit, qubits)
    job = qiskit.execute(qst, Aer.get_backend('qasm_simulator'),
                         shots=5000)
    tomo_fit = tomo.StateTomographyFitter(job.result(), qst)
    rho = tomo_fit.fit(method=method)
    return (rho, psi)


@unittest.skipUnless(cvx_fit._HAS_CVX, 'cvxpy is required to run this test')
class TestFitter(unittest.TestCase):
    def test_trace_constraint(self):
        p = numpy.array([1/2, 1/2, 1/2, 1/2, 1/2, 1/2])

        # the basis matrix for 1-qubit measurement in the Pauli basis
        A = numpy.array([
            [0.5 + 0.j, 0.5 + 0.j, 0.5 + 0.j, 0.5 + 0.j],
            [0.5 + 0.j, -0.5 + 0.j, -0.5 + 0.j, 0.5 + 0.j],
            [0.5 + 0.j, 0. - 0.5j, 0. + 0.5j, 0.5 + 0.j],
            [0.5 + 0.j, 0. + 0.5j, 0. - 0.5j, 0.5 + 0.j],
            [1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
            [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j]
        ])

        for trace_value in [1, 0.3, 2, 0, 42]:
            rho = cvx_fit.cvx_fit(p, A, trace=trace_value)
            self.assertAlmostEqual(numpy.trace(rho), trace_value, places=3)


class TestStateTomography(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.method = 'lstsq'

    def test_bell_2_qubits(self):
        q2 = QuantumRegister(2)
        bell = QuantumCircuit(q2)
        bell.h(q2[0])
        bell.cx(q2[0], q2[1])

        rho, psi = run_circuit_and_tomography(bell, q2, self.method)
        F_bell = state_fidelity(psi, rho, validate=False)
        self.assertAlmostEqual(F_bell, 1, places=1)

    def test_bell_2_qubits_no_register(self):
        bell = QuantumCircuit(2)
        bell.h(0)
        bell.cx(0, 1)

        rho, psi = run_circuit_and_tomography(bell, (0, 1), self.method)
        F_bell = state_fidelity(psi, rho, validate=False)
        self.assertAlmostEqual(F_bell, 1, places=1)

    def test_different_qubit_sets(self):
        circuit = QuantumCircuit(5)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.x(2)
        circuit.s(3)
        circuit.z(4)
        circuit.cx(1, 3)

        for qubit_pair in [(0, 1), (2, 3), (1, 4), (0, 3)]:
            rho, psi = run_circuit_and_tomography(circuit, qubit_pair, self.method)
            psi = partial_trace(psi, [x for x in range(5) if x not in qubit_pair])
            F = state_fidelity(psi, rho, validate=False)
            self.assertAlmostEqual(F, 1, places=1)

    def test_bell_3_qubits(self):
        q3 = QuantumRegister(3)
        bell = QuantumCircuit(q3)
        bell.h(q3[0])
        bell.cx(q3[0], q3[1])
        bell.cx(q3[1], q3[2])

        rho, psi = run_circuit_and_tomography(bell, q3, self.method)
        F_bell = state_fidelity(psi, rho, validate=False)
        self.assertAlmostEqual(F_bell, 1, places=1)

    def test_complex_1_qubit_circuit(self):
        q = QuantumRegister(1)
        circ = QuantumCircuit(q)
        circ.u3(1, 1, 1, q[0])

        rho, psi = run_circuit_and_tomography(circ, q, self.method)
        F_bell = state_fidelity(psi, rho, validate=False)
        self.assertAlmostEqual(F_bell, 1, places=1)

    def test_complex_3_qubit_circuit(self):
        def rand_angles():
            # pylint: disable=E1101
            return tuple(2 * numpy.pi * numpy.random.random(3) - numpy.pi)

        q = QuantumRegister(3)
        circ = QuantumCircuit(q)
        for j in range(3):
            circ.u3(*rand_angles(), q[j])

        rho, psi = run_circuit_and_tomography(circ, q, self.method)
        F_bell = state_fidelity(psi, rho, validate=False)
        self.assertAlmostEqual(F_bell, 1, places=1)


@unittest.skipUnless(cvx_fit._HAS_CVX, 'cvxpy is required  to run this test')
class TestStateTomographyCVX(TestStateTomography):
    def setUp(self):
        super().setUp()
        self.method = 'cvx'

    def test_split_job(self):
        q3 = QuantumRegister(3)
        bell = QuantumCircuit(q3)
        bell.h(q3[0])
        bell.cx(q3[0], q3[1])
        bell.cx(q3[1], q3[2])

        psi = Statevector.from_instruction(bell)
        qst = tomo.state_tomography_circuits(bell, q3)
        qst1 = qst[:len(qst) // 2]
        qst2 = qst[len(qst) // 2:]

        backend = Aer.get_backend('qasm_simulator')
        job1 = qiskit.execute(qst1, backend, shots=5000)
        job2 = qiskit.execute(qst2, backend, shots=5000)

        tomo_fit = tomo.StateTomographyFitter([job1.result(), job2.result()], qst)

        rho_mle = tomo_fit.fit(method='lstsq')
        F_bell_mle = state_fidelity(psi, rho_mle, validate=False)
        self.assertAlmostEqual(F_bell_mle, 1, places=1)


if __name__ == '__main__':
    unittest.main()
