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

import unittest

import qiskit
from qiskit import QuantumRegister, QuantumCircuit, Aer
from qiskit.quantum_info import state_fidelity
from qiskit.tools.qi.qi import outer

import qiskit.ignis.verification.tomography as tomo


def run_circuit_and_tomography(circuit, qubits):
    job = qiskit.execute(circuit, Aer.get_backend('unitary_simulator'))
    U = job.result().get_unitary(circuit)
    choi_ideal = outer(U.ravel(order='F'))
    qst = tomo.process_tomography_circuits(circuit, qubits)
    job = qiskit.execute(qst, Aer.get_backend('qasm_simulator'),
                         shots=5000)
    tomo_fit = tomo.ProcessTomographyFitter(job.result(), qst)
    choi_cvx = tomo_fit.fit(method='cvx').data
    choi_mle = tomo_fit.fit(method='lstsq').data
    return (choi_cvx, choi_mle, choi_ideal)


class TestProcessTomography(unittest.TestCase):

    def test_bell_2_qubits(self):
        q2 = QuantumRegister(2)
        bell = QuantumCircuit(q2)
        bell.h(q2[0])
        bell.cx(q2[0], q2[1])

        choi_cvx, choi_mle, choi_ideal = run_circuit_and_tomography(bell, q2)
        F_bell_cvx = state_fidelity(choi_ideal/4, choi_cvx/4)
        self.assertAlmostEqual(F_bell_cvx, 1, places=1)
        F_bell_mle = state_fidelity(choi_ideal/4, choi_mle/4)
        self.assertAlmostEqual(F_bell_mle, 1, places=1)


if __name__ == '__main__':
    unittest.main()
