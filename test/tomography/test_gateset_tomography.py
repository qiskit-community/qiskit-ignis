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

import numpy as np
from qiskit import Aer
from qiskit.compiler import transpile, assemble
from qiskit.ignis.verification.tomography.fitters.gateset_fitter import gateset_linear_inversion
from qiskit.ignis.verification.tomography.basis.circuits import gateset_tomography_circuits
from qiskit.ignis.verification.tomography.fitters.base_fitter import TomographyFitter
from qiskit.providers.aer.noise.errors.quantum_error import QuantumError
from qiskit.providers.aer.noise import NoiseModel
from qiskit.quantum_info import Kraus, PTM

class TestGatesetTomography(unittest.TestCase):
    def collect_tomography_data(self, shots=10000, noise_model=None):
        backend_qasm = Aer.get_backend('qasm_simulator')
        circuits = gateset_tomography_circuits()
        qobj = assemble(circuits, shots=shots)
        result = backend_qasm.run(qobj, noise_model=noise_model).result()
        t = TomographyFitter(result, circuits)
        return t._data

    def linear_inversion_on_gates(self, Gs, Fs, shots=10000, noise_model=None):
        # in PTM: [(1,0),(0,0)] == 0.5 I + 0.5 Z
        rho = np.array([[0.5], [0], [0], [0.5]])

        # Z-basis measurement in PTM
        E = np.array([[1, 0, 0, 1]])

        # linear inversion should result in gates of the form B^-1*G*B where the columns
        # of B are F_k * rho, F_k being the SPAM circuits

        B = np.array([(F @ rho).T[0] for F in Fs]).T
        # perform linear inversion
        data = self.collect_tomography_data(shots=shots,noise_model=noise_model)
        gates, gate_labels = zip(*gateset_linear_inversion(data))

        expected_gates = [np.linalg.inv(B) @ G @ B for G in Gs]
        for expected_gate, gate, label in zip(expected_gates, gates, gate_labels):
            hs_distance = sum([np.abs(x) ** 2 for x in np.nditer(expected_gate - gate)])
            msg = "Failure on gate {}: Expected gate = \n{}\n vs Actual gate = \n{}".format(label, expected_gate, gate)
            self.assertAlmostEqual(hs_distance, 0, delta=0.1, msg=msg)

    def test_linear_inversion(self):
        # based on linear inversion in	arXiv:1310.4492
        # PTM representation of Id, X rotation by 90 degrees, Y rotation by 90 degrees
        G0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        G1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]])
        G2 = np.array([[1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0]])

        Gs = [G0, G1, G2]
        Fs = [Gs[0], Gs[1], Gs[2], Gs[1] @ Gs[1]]
        self.linear_inversion_on_gates(Gs, Fs)

        # Pauli X noise
        A0 = np.array([[0, 1], [1, 0]])
        noise_PTM = PTM(Kraus([A0]))._data
        Gs = [G0, noise_PTM @ G1, noise_PTM @ G2]
        Fs = [Gs[0], Gs[1], Gs[2], Gs[1] @ Gs[1]]
        noise = QuantumError([([{'name': 'kraus', 'qubits': [0], 'params': [A0]}], 1)])
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(noise, ['u1', 'u2', 'u3'])
        self.linear_inversion_on_gates(Gs, Fs, noise_model=noise_model)

        # Amplitude Damping noise
        gamma = 0.05
        A0 = [[1, 0], [0, np.sqrt(1 - gamma)]]
        A1 = [[0, np.sqrt(gamma)], [0, 0]]
        noise_PTM = PTM(Kraus([A0, A1]))._data
        Gs = [G0, noise_PTM @ G1, noise_PTM @ G2]
        Fs = [Gs[0], Gs[1], Gs[2], Gs[1] @ Gs[1]]
        noise = QuantumError([([{'name': 'kraus', 'qubits': [0], 'params': [A0, A1]}], 1)])
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(noise, ['u1', 'u2', 'u3'])
        self.linear_inversion_on_gates(Gs, Fs, noise_model=noise_model)

if __name__ == '__main__':
    unittest.main()
