# -*- coding: utf-8 -*-
#
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

# pylint: disable=missing-docstring,invalid-name
import unittest
import numpy as np
from qiskit import Aer
from qiskit.compiler import assemble
from qiskit.ignis.verification.tomography import GatesetTomographyFitter
from qiskit.ignis.verification.tomography import gateset_tomography_circuits
from qiskit.ignis.verification.tomography.basis import default_gateset_basis

from qiskit.providers.aer.noise import NoiseModel

from qiskit.extensions import HGate, SGate
from qiskit.quantum_info import PTM


class TestGatesetTomography(unittest.TestCase):
    @staticmethod
    def collect_tomography_data(shots=10000,
                                noise_model=None,
                                gateset_basis='Default'):
        backend_qasm = Aer.get_backend('qasm_simulator')
        circuits = gateset_tomography_circuits(gateset_basis=gateset_basis)
        qobj = assemble(circuits, shots=shots)
        result = backend_qasm.run(qobj, noise_model=noise_model).result()
        fitter = GatesetTomographyFitter(result, circuits, gateset_basis)
        return fitter

    @staticmethod
    def expected_linear_inversion_gates(Gs, Fs):
        rho = Gs['rho']
        E = Gs['E']
        B = np.array([(F @ rho).T[0] for F in Fs]).T
        BB = np.linalg.inv(B)
        gates = {label: BB @ G @ B for (label, G) in Gs.items()
                 if label not in ['E', 'rho']}
        gates['E'] = E @ B
        gates['rho'] = BB @ rho
        return gates

    @staticmethod
    def hs_distance(A, B):
        return sum([np.abs(x) ** 2 for x in np.nditer(A-B)])

    @staticmethod
    def convert_from_ptm(vector):
        Id = np.sqrt(0.5) * np.array([[1, 0], [0, 1]])
        X = np.sqrt(0.5) * np.array([[0, 1], [1, 0]])
        Y = np.sqrt(0.5) * np.array([[0, -1j], [1j, 0]])
        Z = np.sqrt(0.5) * np.array([[1, 0], [0, -1]])
        v = vector.reshape(4)
        return v[0] * Id + v[1] * X + v[2] * Y + v[3] * Z

    def compare_gates(self, expected_gates, result_gates, labels, delta=0.2):
        for label in labels:
            expected_gate = expected_gates[label]
            result_gate = result_gates[label].data
            msg = "Failure on gate {}: Expected gate = \n{}\n" \
                  "vs Actual gate = \n{}".format(label,
                                                 expected_gate,
                                                 result_gate)
            distance = self.hs_distance(expected_gate, result_gate)
            self.assertAlmostEqual(distance, 0, delta=delta, msg=msg)

    def run_test_on_basis_and_noise(self,
                                    gateset_basis='Default',
                                    noise_model=None,
                                    noise_ptm=None):
        if gateset_basis == 'Default':
            gateset_basis = default_gateset_basis()

        labels = gateset_basis.gate_labels
        gates = gateset_basis.gate_matrices
        gates['rho'] = np.array([[np.sqrt(0.5)], [0], [0], [np.sqrt(0.5)]])
        gates['E'] = np.array([[np.sqrt(0.5), 0, 0, np.sqrt(0.5)]])
        # apply noise if given
        for label in labels:
            if label != "Id" and noise_ptm is not None:
                gates[label] = noise_ptm @ gates[label]
        Fs = [gateset_basis.spam_matrix(label)
              for label in gateset_basis.spam_labels]

        # prepare the fitter
        fitter = self.collect_tomography_data(shots=10000,
                                              noise_model=noise_model,
                                              gateset_basis=gateset_basis)

        # linear inversion test
        result_gates = fitter.linear_inversion()
        expected_gates = self.expected_linear_inversion_gates(gates, Fs)
        self.compare_gates(expected_gates, result_gates, labels + ['E', 'rho'])

        # fitter optimization test
        result_gates = fitter.fit()
        expected_gates = gates
        expected_gates['E'] = self.convert_from_ptm(expected_gates['E'])
        expected_gates['rho'] = self.convert_from_ptm(expected_gates['rho'])
        self.compare_gates(expected_gates, result_gates, labels + ['E', 'rho'])

    def test_noiseless_standard_basis(self):
        self.run_test_on_basis_and_noise()

    def test_noiseless_h_gate_standard_basis(self):
        basis = default_gateset_basis()
        basis.add_gate(HGate())
        self.run_test_on_basis_and_noise(gateset_basis=basis)

    def test_noiseless_s_gate_standard_basis(self):
        basis = default_gateset_basis()
        basis.add_gate(SGate())
        self.run_test_on_basis_and_noise(gateset_basis=basis)

    def test_amplitude_damping_standard_basis(self):
        gamma = 0.05
        noise_ptm = PTM(np.array([[1, 0, 0, 0],
                                  [0, np.sqrt(1-gamma), 0, 0],
                                  [0, 0, np.sqrt(1-gamma), 0],
                                  [gamma, 0, 0, 1-gamma]]))
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(noise_ptm, ['u1', 'u2', 'u3'])
        self.run_test_on_basis_and_noise(noise_model=noise_model,
                                         noise_ptm=np.real(noise_ptm.data))

    def test_depolarization_standard_basis(self):
        p = 0.05
        noise_ptm = PTM(np.array([[1, 0, 0, 0],
                                  [0, 1-p, 0, 0],
                                  [0, 0, 1-p, 0],
                                  [0, 0, 0, 1-p]]))
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(noise_ptm, ['u1', 'u2', 'u3'])
        self.run_test_on_basis_and_noise(noise_model=noise_model,
                                         noise_ptm=np.real(noise_ptm.data))


if __name__ == '__main__':
    unittest.main()
