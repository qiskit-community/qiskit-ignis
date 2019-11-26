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
import scipy.optimize as opt
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
                matrices = [self.Gs.index(gate) for gate in self.Fs[self.Fs_names[i]]] + [k] + [self.Gs.index(gate) for
                                                                                                gate in self.Fs[
                                                                                                    self.Fs_names[j]]]
                obj_fn_data.append((matrices, m_ijk))
        return obj_fn_data

    def split_input_vector(self, x):
        n = len(self.Gs)
        d = (2 ** self.qubits)
        ds = d ** 2  # d squared - the dimension of the density operator (4 for 1 qubit)
        expected_var_num = 2 * ds + n * ds ** 2  # E is 1xds; rho is dsx1; each G is dsxds
        if len(x) != expected_var_num:
            raise (RuntimeError("Expected length of x is {}; got {}".format(expected_var_num, len(x))))
        E = np.array([x[0:ds]])
        rho = np.array([x[ds:2 * ds]]).T
        G_matrices = [
            np.array([x[2 * ds + k * ds ** 2 + ds * i:2 * ds + k * ds ** 2 + ds * (i + 1)] for i in range(ds)]) for k in
            range(n)]
        return (E, rho, G_matrices)

    def obj_fn(self, x, *args):
        E, rho, G_matrices = self.split_input_vector(x)
        val = 0
        for term in self.obj_fn_data:
            term_val = E
            for G_index in term[0]:
                term_val = term_val @ G_matrices[G_index]
            term_val = term_val @ rho
            term_val = term_val[0][0]
            term_val = term_val - term[1]  # m_{ijk}
            term_val = term_val ** 2
            val = val + term_val
        return val

    def bounds_constraints(self):
        """ E and rho are not subject to bounds
            For each G, all the elements of G are in [-1,1]
            and the first row is of the form [1,0,0,...,0]
            since this is a PTM representation
        """
        n = len(self.Gs)
        d = (2 ** self.qubits)  # rho is dxd and starts at variable d^2
        ds = d ** 2
        lb = []
        ub = []

        for i in range(2 * ds):  # E and rho - no constraint
            lb.append(-np.inf)
            ub.append(np.inf)

        for k in range(n):  # iterate over all Gs
            lb.append(1)
            ub.append(1)  # G^k_{0,0} is 1
            for i in range(ds - 1):
                lb.append(0)
                ub.append(0)  # G^k_{0,i} is 0
            for i in range((ds - 1) * ds):  # rest of G^k
                lb.append(-1)
                ub.append(1)

        return opt.Bounds(lb, ub)

    def rho_trace_constraint(self):
        """ The constraint Tr(rho) = 1"""
        d = (2 ** self.qubits)  # rho is dxd and starts at variable d^2
        ds = d ** 2
        var_num = 2 * ds + n * ds ** 2
        # if rho is indexed by [[0,1,...,d-1],[d,d+1,...],...]
        # the trace is given by entries 0, d+1, 2(d+1),...,(d-1)(d+1)
        A = [0] * var_num
        for i in range(d):
            A[ds + i * (d + 1)] = 1
        lb = 1
        ub = 1
        return ((A, lb, ub))

    def linear_constraints(self):
        As = []
        lbs = []
        ubs = []

        A, lb, ub = self.rho_trace_constraint()
        As.append(A)
        lbs.append(lb)
        ubs.append(ub)

        return opt.LinearConstraint(As, lbs, ubs)

    def constraints(self):
        constraints = []
        constraints.append(self.linear_constraints())
        return constraints

    def process_result(self, x):
        E, rho, G_matrices = self.split_input_vector(x)
        result = {}
        result['E'] = E
        result['rho'] = rho
        for i in range(len(self.Gs)):
            result[self.Gs[i]] = G_matrices[i]
        return result

    def optimize(self, initial_value):
        result = opt.minimize(self.obj_fn, initial_value,
                              constraints=self.constraints(), bounds=self.bounds_constraints())
        formatted_result = self.process_result(result.x)
        return formatted_result
