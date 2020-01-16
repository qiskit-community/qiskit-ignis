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
from qiskit.compiler import assemble
from qiskit.ignis.verification.tomography import GatesetTomographyFitter
from qiskit.ignis.verification.tomography import gateset_tomography_circuits
from qiskit.ignis.verification.tomography.basis import GateSetBasis
from qiskit.providers.aer.noise.errors.quantum_error import QuantumError
from qiskit.providers.aer.noise import NoiseModel


class TestGatesetTomography(unittest.TestCase):
    def collect_tomography_data(self,
                                shots=10000,
                                noise_model=None,
                                gateset_basis='Standard GST'):
        backend_qasm = Aer.get_backend('qasm_simulator')
        circuits = gateset_tomography_circuits(gateset_basis=gateset_basis)
        qobj = assemble(circuits, shots=shots)
        result = backend_qasm.run(qobj, noise_model=noise_model).result()
        fitter = GatesetTomographyFitter(result, circuits, gateset_basis)
        return fitter

    def linear_inversion_on_gates(self, Gs, Fs,
                                  shots=10000,
                                  noise_model=None,
                                  gateset_basis='Standard GST'):
        # in PTM: [(1,0),(0,0)] == 0.5 I + 0.5 Z
        rho = np.array([[0.5], [0], [0], [0.5]])

        # Z-basis |0> measurement in PTM
        # E = np.array([[1, 0, 0, 1]])

        # linear inversion should result in gates of the form B^-1*G*B where
        # the columns of B are F_k * rho, F_k being the SPAM circuits

        B = np.array([(F @ rho).T[0] for F in Fs]).T
        # perform linear inversion
        fitter = self.collect_tomography_data(shots=shots,
                                              noise_model=noise_model,
                                              gateset_basis=gateset_basis)

        print("About to start optimization")
        fitter.fit()
        print("Optimization ended")
        gates, gate_labels = zip(*fitter.linear_inversion())
        expected_gates = [np.linalg.inv(B) @ G @ B for G in Gs]
        msg = "Number of expected gates ({}) different than actual number " \
            "of gates ({})".format(len(expected_gates), len(gates))
        self.assertEqual(len(expected_gates), len(gates), msg=msg)
        compare = zip(expected_gates, gates, gate_labels)
        for expected_gate, gate, label in compare:
            hs_distance = sum([np.abs(x) ** 2 for x
                               in np.nditer(expected_gate - gate)])
            msg = "Failure on gate {}: Expected gate = \n{}\n" \
                "vs Actual gate = \n{}".format(label, expected_gate, gate)
            self.assertAlmostEqual(hs_distance, 0, delta=0.1, msg=msg)

    def test_linear_inversion(self):
        # based on linear inversion in	arXiv:1310.4492
        # PTM representation of Id
        G0 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
        # X rotation by 90 degrees
        G1 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, -1],
                       [0, 0, 1, 0]])
        # Y rotation by 90 degrees
        G2 = np.array([[1, 0, 0, 0],
                       [0, 0, 0, -1],
                       [0, 0, 1, 0],
                       [0, 1, 0, 0]])

        Gs = [G0, G1, G2]
        Fs = [Gs[0], Gs[1], Gs[2], Gs[1] @ Gs[1]]
        print("No noises")
        self.linear_inversion_on_gates(Gs, Fs)
        print("Start noises")
        # Pauli X noise
        A0 = np.array([[0, 1], [1, 0]])
        # noise_PTM = PTM(Kraus([A0]))._data
        noise_PTM = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, -1, 0],
                              [0, 0, 0, -1]])
        Gs = [G0, noise_PTM @ G1, noise_PTM @ G2]
        Fs = [Gs[0], Gs[1], Gs[2], Gs[1] @ Gs[1]]
        noise = QuantumError([([{'name': 'kraus',
                                 'qubits': [0],
                                 'params': [A0]}], 1)])
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(noise, ['u1', 'u2', 'u3'])
        self.linear_inversion_on_gates(Gs, Fs, noise_model=noise_model)

        # Amplitude Damping noise
        gamma = 0.05
        A0 = [[1, 0], [0, np.sqrt(1 - gamma)]]
        A1 = [[0, np.sqrt(gamma)], [0, 0]]
        # noise_PTM = PTM(Kraus([A0, A1]))._data
        noise_PTM = np.array([[1, 0, 0, 0],
                              [0, np.sqrt(1-gamma), 0, 0],
                              [0, 0, np.sqrt(1-gamma), 0],
                              [gamma, 0, 0, 1-gamma]])
        Gs = [G0, noise_PTM @ G1, noise_PTM @ G2]
        Fs = [Gs[0], Gs[1], Gs[2], Gs[1] @ Gs[1]]
        noise = QuantumError([([{'name': 'kraus',
                                 'qubits': [0],
                                 'params': [A0, A1]}], 1)])
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(noise, ['u1', 'u2', 'u3'])
        self.linear_inversion_on_gates(Gs, Fs, noise_model=noise_model)

        # different gateset_basis
        def alt_gates_func(circ, qubit, op):
            if op == 'Id':
                pass
            if op == 'X_Rot_90':
                circ.u2(-np.pi / 2, np.pi / 2, qubit)
            if op == 'Y_Rot_90':
                circ.u2(np.pi, np.pi, qubit)
            if op == 'X_Rot_180':
                circ.x(qubit)

        AltGatesetBasis = GateSetBasis('Standard GST',
                                       (('Id',
                                         'X_Rot_90',
                                         'Y_Rot_90',
                                         'X_Rot_180'),
                                        alt_gates_func),
                                       (('F0', 'F1', 'F2', 'F3'),
                                        {'F0': ('Id',),
                                         'F1': ('X_Rot_90',),
                                         'F2': ('Y_Rot_90',),
                                         'F3': ('X_Rot_180',)
                                         })
                                       )
        G0 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
        G1 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, -1],
                       [0, 0, 1, 0]])
        G2 = np.array([[1, 0, 0, 0],
                       [0, 0, 0, -1],
                       [0, 0, 1, 0],
                       [0, 1, 0, 0]])
        G3 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, -1]])

        Gs = [G0, G1, G2, G3]
        Fs = [Gs[0], Gs[1], Gs[2], Gs[3]]
        self.linear_inversion_on_gates(Gs, Fs, gateset_basis=AltGatesetBasis)


if __name__ == '__main__':
    unittest.main()
