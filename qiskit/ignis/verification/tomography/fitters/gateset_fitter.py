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

import numpy as np
import itertools
from qiskit.ignis.verification.tomography.basis.gatesetbasis import StandardGatesetBasis

def compute_probs(data):
    probs = {}
    for key in data.keys():
        vals = data[key]
        probs[key] = vals.get('0', 0) / sum(vals.values())
    return probs

def gateset_linear_inversion(data, gateset_basis='Standard GST'):
    if gateset_basis == 'Standard GST':
        gateset_basis = StandardGatesetBasis

    probs = compute_probs(data)
    n = len(gateset_basis.spam_labels)
    m = len(gateset_basis.gate_labels)
    gram_matrix = np.zeros((n,n))
    gate_matrices = []
    for i in range(m):
        gate_matrices.append(np.zeros((n,n)))

    for i in range(n): # row
        for j in range(n): # column
            F_i = gateset_basis.spam_labels[i]
            F_j = gateset_basis.spam_labels[j]
            gram_matrix[i][j] = probs[(F_i, F_j)]

            for k in range(m): #gate
                G_k = gateset_basis.gate_labels[k]
                gate_matrices[k][i][j] = probs[(F_i, G_k, F_j)]
    gram_inverse = np.linalg.inv(gram_matrix)

    gates = [gram_inverse @ gate_matrix for gate_matrix in gate_matrices]
    return list(zip(gates,gateset_basis.gate_labels))


class GST_Optimize():
    def __init__(self, Gs, Fs, probs, qubits=1):
        self.probs = probs
        self.Gs = Gs
        self.Fs = Fs
        self.Fs_names = list(Fs.keys())
        self.qubits = qubits
        self.obj_fn_data = self.compute_objective_function_data()

    def compute_objective_function_data(self):
        """The objective function is sum_{ijk}(<|E*R_Fi*G_k*R_Fj*Rho|>-m_{ijk})^2
           We expand R_Fi*G_k*R_Fj to a sequence of G-gates and store the indices
           we also obtain the m_{ijk} value from the probs list
           all that remains when computing the function is thus performing
           the matrix multiplications and remaining algebra
        """
        m = len(self.Fs)
        n = len(self.Gs)
        obj_fn_data = []
        for (i, j) in itertools.product(range(m), repeat=2):
            for k in range(n):
                m_ijk = (self.probs[(self.Fs_names[i], self.Gs[k], self.Fs_names[j])])
                matrices = [self.Gs.index(gate) for gate in self.Fs[self.Fs_names[i]]] \
                           + [k] \
                           + [self.Gs.index(gate) for gate in self.Fs[self.Fs_names[j]]]
                obj_fn_data.append((matrices, m_ijk))
        return obj_fn_data

    def obj_fn(self, x):
        n = len(self.Gs)
        d = (2 ** self.qubits) * 2  # for 1 qubit this will be 4, the dimension of density operators
        expected_var_num = 2 * d + n * d ** 2  # E is 1xd; rho is dx1; each G is dxd
        if len(x) != expected_var_num:
            raise (RuntimeError("Expected length of x is {}; got {}".format(expected_var_num, len(x))))
        E = np.array([x[0:d]])
        rho = np.array([x[d:2 * d]]).T
        G_matrices = [np.array([x[2 * d + k * d ** 2 + d * i:2 * d + k * d ** 2 + d * (i + 1)]
                                for i in range(d)]) for k in range(n)]
        val = 0
        for term in self.obj_fn_data:
            term_val = E
            for G_index in term[0]:
                term_val = term_val @ G_matrices[G_index]
            term_val = term_val @ rho
            term_val = term_val - term[1]  # m_{ijk}
            term_val = term_val ** 2
            val = val + term_val

        return val