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
from qiskit.ignis.verification.tomography.basis.gatesetbasis import StandardGatesetBasis

def compute_probs(data):
    probs = {}
    for key in data.keys():
        vals = data[key]
        probs[key] = vals.get('0', 0) / sum(vals.values())
    print(probs)
    return probs

def gateset_linear_inversion(data, gateset_basis='Standard GST'):
    if gateset_basis == 'Standard GST':
        gateset_basis = StandardGatesetBasis

    probs = compute_probs(data)
    n = len(gateset_basis.spam_labels)
    m = len(gateset_basis.gate_labels)
    id_matrix = np.zeros((n,n))
    gate_matrices = []
    for i in range(m):
        gate_matrices.append(np.zeros((n,n)))

    for i in range(n): # row
        for j in range(n): # column
            F_i = gateset_basis.spam_labels[i]
            F_j = gateset_basis.spam_labels[j]
            id_matrix[i][j] = probs[(F_i, F_j)]

            for k in range(m): #gate
                G_k = gateset_basis.gate_labels[k]
                gate_matrices[k][i][j] = probs[(F_i, G_k, F_j)]

    id_inverse = np.linalg.inv(id_matrix)
    gates = [id_inverse @ gate_matrix for gate_matrix in gate_matrices]
    return list(zip(gates,gateset_basis.gate_labels))