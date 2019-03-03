# -*- coding: utf-8 -*-
#
# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring

import unittest

import numpy
import qiskit
from qiskit import QuantumRegister, QuantumCircuit, Aer
from qiskit.quantum_info import state_fidelity

import qiskit.ignis.verification.tomography as tomo
import qiskit.ignis.verification.tomography.fitters.cvx_fit as cvx_fit


def run_circuit_and_tomography(circuit, qubits):
    job = qiskit.execute(circuit, Aer.get_backend('statevector_simulator'))
    psi = job.result().get_statevector(circuit)
    qst = tomo.state_tomography_circuits(circuit, qubits)
    job = qiskit.execute(qst, Aer.get_backend('qasm_simulator'),
                         shots=5000)
    tomo_fit = tomo.StateTomographyFitter(job.result(), qst)
    rho_cvx = tomo_fit.fit(method='cvx')
    rho_mle = tomo_fit.fit(method='lstsq')
    return (rho_cvx, rho_mle, psi)


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

    def test_bell_2_qubits(self):
        q2 = QuantumRegister(2)
        bell = QuantumCircuit(q2)
        bell.h(q2[0])
        bell.cx(q2[0], q2[1])

        rho_cvx, rho_mle, psi = run_circuit_and_tomography(bell, q2)
        F_bell_cvx = state_fidelity(psi, rho_cvx)
        self.assertAlmostEqual(F_bell_cvx, 1, places=1)
        F_bell_mle = state_fidelity(psi, rho_mle)
        self.assertAlmostEqual(F_bell_mle, 1, places=1)

    def test_bell_3_qubits(self):
        q3 = QuantumRegister(3)
        bell = QuantumCircuit(q3)
        bell.h(q3[0])
        bell.cx(q3[0], q3[1])
        bell.cx(q3[1], q3[2])

        rho_cvx, rho_mle, psi = run_circuit_and_tomography(bell, q3)
        F_bell_cvx = state_fidelity(psi, rho_cvx)
        self.assertAlmostEqual(F_bell_cvx, 1, places=1)
        F_bell_mle = state_fidelity(psi, rho_mle)
        self.assertAlmostEqual(F_bell_mle, 1, places=1)

    def test_complex_1_qubit_circuit(self):
        q = QuantumRegister(1)
        circ = QuantumCircuit(q)
        circ.u3(1, 1, 1, q[0])

        rho_cvx, rho_mle, psi = run_circuit_and_tomography(circ, q)
        F_bell_cvx = state_fidelity(psi, rho_cvx)
        self.assertAlmostEqual(F_bell_cvx, 1, places=1)
        F_bell_mle = state_fidelity(psi, rho_mle)
        self.assertAlmostEqual(F_bell_mle, 1, places=1)

    def test_complex_3_qubit_circuit(self):
        def rand_angles():
            return tuple(2 * numpy.pi * numpy.random.random(3) - numpy.pi)

        q = QuantumRegister(3)
        circ = QuantumCircuit(q)
        for j in range(3):
            circ.u3(*rand_angles(), q[j])

        rho_cvx, rho_mle, psi = run_circuit_and_tomography(circ, q)
        F_bell_cvx = state_fidelity(psi, rho_cvx)
        self.assertAlmostEqual(F_bell_cvx, 1, places=1)
        F_bell_mle = state_fidelity(psi, rho_mle)
        self.assertAlmostEqual(F_bell_mle, 1, places=1)


if __name__ == '__main__':
    unittest.main()
