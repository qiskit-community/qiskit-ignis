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
import functools
import numpy as np
from qiskit import Aer
from qiskit.compiler import assemble
from qiskit.ignis.verification.tomography import GatesetTomographyFitter
from qiskit.ignis.verification.tomography import gateset_tomography_circuits
from qiskit.ignis.verification.tomography.basis import GateSetBasis
from qiskit.ignis.verification.tomography.basis import StandardGatesetBasis

from qiskit.providers.aer.noise.errors.quantum_error import QuantumError
from qiskit.providers.aer.noise import NoiseModel


class TestGatesetTomography(unittest.TestCase):
    @staticmethod
    def collect_tomography_data(shots=10000,
                                noise_model=None,
                                gateset_basis='Standard GST'):
        backend_qasm = Aer.get_backend('qasm_simulator')
        circuits = gateset_tomography_circuits(gateset_basis=gateset_basis)
        qobj = assemble(circuits, shots=shots)
        result = backend_qasm.run(qobj, noise_model=noise_model).result()
        fitter = GatesetTomographyFitter(result, circuits, gateset_basis)
        return fitter

    @staticmethod
    def expected_linear_inversion_gates(Gs, Fs):
        rho = np.array([[0.5], [0], [0], [0.5]])
        B = np.array([(F @ rho).T[0] for F in Fs]).T
        return {label: np.linalg.inv(B) @ G @ B for (label, G) in Gs.items()}

    @staticmethod
    def hs_distance(A,B):
        return sum([np.abs(x) ** 2 for x in np.nditer(A-B)])

    def compare_gates(self, expected_gates, result_gates, labels, delta=0.2):
        for label in labels:
            expected_gate = expected_gates[label]
            result_gate = result_gates[label]
            msg = "Failure on gate {}: Expected gate = \n{}\n" \
                  "vs Actual gate = \n{}".format(label, expected_gate, result_gate)
            self.assertAlmostEqual(self.hs_distance(expected_gate, result_gate), 0, delta=delta, msg=msg)

    def run_test_on_basis_and_noise(self,
                                    gateset_basis=StandardGatesetBasis,
                                    noise_model = None,
                                    noise_ptm = None):
        labels = gateset_basis.gate_labels
        gates = gateset_basis.gate_matrices

        # apply noise if given
        for label in labels:
            if label != "Id" and noise_ptm is not None:
                gates[label] = noise_ptm @ gates[label]

        Fs_gate_list = [[gates[label] for label in spec]
            for spec in gateset_basis.spam_spec.values()]
        Fs = [functools.reduce(lambda a,b: a @ b, gates) for gates in Fs_gate_list]

        # prepare the fitter
        fitter = self.collect_tomography_data(shots=1000,
                                              noise_model=noise_model,
                                              gateset_basis=gateset_basis)

        # linear inversion test
        expected_gates = self.expected_linear_inversion_gates(gates, Fs)
        result_gates = fitter.linear_inversion()
        self.compare_gates(expected_gates, result_gates, labels)

        # fitter optimization test
        expected_gates = gates
        result_gates = fitter.fit()
        self.compare_gates(expected_gates, result_gates, labels)

    def test_noiseless_standard_basis(self):
        self.run_test_on_basis_and_noise()

    def test_amplitude_damping_standard_basis(self):
        gamma = 0.05
        A0 = [[1, 0], [0, np.sqrt(1 - gamma)]]
        A1 = [[0, np.sqrt(gamma)], [0, 0]]
        noise_ptm = np.array([[1, 0, 0, 0],
                              [0, np.sqrt(1-gamma), 0, 0],
                              [0, 0, np.sqrt(1-gamma), 0],
                              [gamma, 0, 0, 1-gamma]])
        noise = QuantumError([([{'name': 'kraus',
                                 'qubits': [0],
                                 'params': [A0, A1]}], 1)])
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(noise, ['u1', 'u2', 'u3'])
        self.run_test_on_basis_and_noise(noise_model=noise_model, noise_ptm=noise_ptm)

if __name__ == '__main__':
    unittest.main()
