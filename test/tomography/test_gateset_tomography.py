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
from qiskit.ignis.verification.tomography.fitters.gateset_fitter import GST_Optimize
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

    def test_mle_function(self):
        probs = {('F0', 'Id', 'F0'): 1.0, ('F0', 'Id', 'F1'): 0.5036, ('F0', 'Id', 'F2'): 0.5111,
                 ('F0', 'Id', 'F3'): 0.0, ('F1', 'Id', 'F0'): 0.4921, ('F1', 'Id', 'F1'): 0.0,
                 ('F1', 'Id', 'F2'): 0.5032, ('F1', 'Id', 'F3'): 0.5049, ('F2', 'Id', 'F0'): 0.4931,
                 ('F2', 'Id', 'F1'): 0.5078, ('F2', 'Id', 'F2'): 0.0, ('F2', 'Id', 'F3'): 0.4948,
                 ('F3', 'Id', 'F0'): 0.0, ('F3', 'Id', 'F1'): 0.5024, ('F3', 'Id', 'F2'): 0.5041,
                 ('F3', 'Id', 'F3'): 1.0, ('F0', 'X_Rot_90', 'F0'): 0.4957, ('F0', 'X_Rot_90', 'F1'): 0.0,
                 ('F0', 'X_Rot_90', 'F2'): 0.5001, ('F0', 'X_Rot_90', 'F3'): 0.4996, ('F1', 'X_Rot_90', 'F0'): 0.0,
                 ('F1', 'X_Rot_90', 'F1'): 0.5103, ('F1', 'X_Rot_90', 'F2'): 0.4995, ('F1', 'X_Rot_90', 'F3'): 1.0,
                 ('F2', 'X_Rot_90', 'F0'): 0.5087, ('F2', 'X_Rot_90', 'F1'): 0.5106, ('F2', 'X_Rot_90', 'F2'): 0.0,
                 ('F2', 'X_Rot_90', 'F3'): 0.4964, ('F3', 'X_Rot_90', 'F0'): 0.5033, ('F3', 'X_Rot_90', 'F1'): 1.0,
                 ('F3', 'X_Rot_90', 'F2'): 0.487, ('F3', 'X_Rot_90', 'F3'): 0.4976, ('F0', 'Y_Rot_90', 'F0'): 0.4941,
                 ('F0', 'Y_Rot_90', 'F1'): 0.4993, ('F0', 'Y_Rot_90', 'F2'): 0.0, ('F0', 'Y_Rot_90', 'F3'): 0.4978,
                 ('F1', 'Y_Rot_90', 'F0'): 0.5047, ('F1', 'Y_Rot_90', 'F1'): 0.0, ('F1', 'Y_Rot_90', 'F2'): 0.5017,
                 ('F1', 'Y_Rot_90', 'F3'): 0.4931, ('F2', 'Y_Rot_90', 'F0'): 0.0, ('F2', 'Y_Rot_90', 'F1'): 0.4932,
                 ('F2', 'Y_Rot_90', 'F2'): 0.5024, ('F2', 'Y_Rot_90', 'F3'): 1.0, ('F3', 'Y_Rot_90', 'F0'): 0.5019,
                 ('F3', 'Y_Rot_90', 'F1'): 0.5024, ('F3', 'Y_Rot_90', 'F2'): 1.0, ('F3', 'Y_Rot_90', 'F3'): 0.4983,
                 ('F0', 'F0'): 1.0, ('F0', 'F1'): 0.5068, ('F0', 'F2'): 0.5055, ('F0', 'F3'): 0.0, ('F1', 'F0'): 0.507,
                 ('F1', 'F1'): 0.0, ('F1', 'F2'): 0.4997, ('F1', 'F3'): 0.4942, ('F2', 'F0'): 0.506,
                 ('F2', 'F1'): 0.4898, ('F2', 'F2'): 0.0, ('F2', 'F3'): 0.4995, ('F3', 'F0'): 0.0, ('F3', 'F1'): 0.491,
                 ('F3', 'F2'): 0.5053, ('F3', 'F3'): 1.0, ('F0',): 1.0, ('F1',): 0.5064, ('F2',): 0.5001, ('F3',): 0.0}
        Gs = ['Id', 'X_Rot_90', 'Y_Rot_90']
        Fs = {'F0': ('Id',), 'F1': ('X_Rot_90',), 'F2': ('Y_Rot_90',), 'F3': ('X_Rot_90', 'X_Rot_90')}
        gst = GST_Optimize(Gs, Fs, probs)
        E = [1, 0, 0, 0]
        Rho = [0.5, 0, 0, 0.5]
        G_vals = (np.array([[ 1.00340810e+00, -1.03519151e-04, -2.07629277e-02,
                 6.47958682e-03],
               [ 6.30422449e-03,  1.00372520e+00,  3.21007300e-02,
                -2.27979857e-04],
               [-1.31900393e-02, -4.94562545e-03,  1.01345762e+00,
                -1.27923692e-02],
               [ 3.38239836e-03,  5.08983941e-03, -1.12519332e-02,
                 1.00644071e+00]]), 'Id'), (np.array([[ 0.00907228, -0.00262494, -0.01875478,  0.99669682],
               [ 0.98992356,  0.01975411,  0.03653726, -0.988985  ],
               [-0.01452778, -0.01459403,  0.99913159, -0.02212587],
               [ 0.00560653,  0.99737474, -0.017777  ,  1.00485468]]), 'X_Rot_90'), (np.array([[-0.00768643,  0.00414623,  0.00337241,  1.00792034],
               [ 0.01995706,  0.99813113,  0.01564968,  0.00615055],
               [ 1.00883827, -0.01569707, -0.02251823, -1.02230445],
               [-0.01231601,  0.01169503,  1.00333928,  1.00056696]]), 'Y_Rot_90')

        x = E + Rho + list(G_vals[0][0].flatten()) + list(G_vals[1][0].flatten()) + list(G_vals[2][0].flatten())
        result = gst.obj_fn(x)[0][0]
        self.assertLess(result, 5)

if __name__ == '__main__':
    unittest.main()
