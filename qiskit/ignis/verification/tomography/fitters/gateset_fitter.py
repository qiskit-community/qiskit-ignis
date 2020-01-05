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

"""
Quantum gate set tomography fitter
"""

import numpy as np
from ..basis.gatesetbasis import StandardGatesetBasis
from .base_fitter import TomographyFitter


class GatesetTomographyFitter:
    def __init__(self,
                 result,
                 circuits,
                 gateset_basis='Standard GST'):
        """Initialize gateset tomography fitter with experimental data.

        Args:
            result (Result): a Qiskit Result object obtained from executing
                            tomography circuits.
            circuits (list): a list of circuits or circuit names to extract
                            count information from the result object.
            gateset_basis (GateSetBasis, str): Representation of the gates and
            SPAM circuits of the gateset (default: 'Standard GST')
        """
        self.gateset_basis = gateset_basis
        if gateset_basis == 'Standard GST':
            self.gateset_basis = StandardGatesetBasis
        data = TomographyFitter(result, circuits).data
        self.probs = {}
        for key, vals in data.items():
            self.probs[key] = vals.get('0', 0) / sum(vals.values())

    def linear_inversion(self):
        """
        Reconstruct a gate set from measurement data using linear inversion.

        Returns:
            A list of tuples (G, label) of gate representation
            and this gate's label as given in gateset_basis

        Additional Information:
            Given a gate set (G1,...,Gm)
            and SPAM circuits (F1,...,Fn) constructed from those gates
            the data should contain the probabilities of the following types:
            p_ijk = E*F_i*G_k*F_j*rho
            p_ij = E*F_i*F_j*rho

            One constructs the Gram matrix g = (p_ij)_ij
            which can be described as a product g=AB
            where A = sum i E F_i and B=sum F_j rho j
            For each gate Gk one can also construct the matrix Mk=(pijk)_ij
            which can be described as Mk=A*Gk*B
            Inverting g we obtain g^-1 = B^-1A^-1 and so
            g^1 * Mk = B^-1 * Gk * B
            This gives us a matrix similiar to Gk's representing matrix.
            However, it will not be the same as Gk,
            since the observable results cannot distinguish
            between (G1,...,Gm) and (B^-1*G1*B,...,B^-1*Gm*B)
            a further step of *Gauge optimization* is required on the results
            of the linear inversion stage.
            One can also use the linear inversion results as a starting point
            for a MLE optimization for finding a physical gateset, since
            unless the probabilities are accurate, the resulting gateset
            need not be physical.
        """

        n = len(self.gateset_basis.spam_labels)
        m = len(self.gateset_basis.gate_labels)
        gram_matrix = np.zeros((n, n))
        gate_matrices = []
        for i in range(m):
            gate_matrices.append(np.zeros((n, n)))

        for i in range(n):  # row
            for j in range(n):  # column
                F_i = self.gateset_basis.spam_labels[i]
                F_j = self.gateset_basis.spam_labels[j]
                gram_matrix[i][j] = self.probs[(F_i, F_j)]

                for k in range(m):  # gate
                    G_k = self.gateset_basis.gate_labels[k]
                    gate_matrices[k][i][j] = self.probs[(F_i, G_k, F_j)]

        gram_inverse = np.linalg.inv(gram_matrix)

        gates = [gram_inverse @ gate_matrix for gate_matrix in gate_matrices]
        return list(zip(gates, self.gateset_basis.gate_labels))
