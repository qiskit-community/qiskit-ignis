# -*- coding: utf-8 -*-

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


"""
Maximum-Likelihood estimation quantum tomography fitter
"""

import numpy as np
from scipy import linalg as la
from scipy.linalg import lstsq


def lstsq_fit(data, basis_matrix, weights=None, PSD=True, trace=None):
    """
    Reconstruct a density matrix using MLE least-squares fitting.

    Args:
        data (vector like): vector of expectation values
        basis_matrix (matrix like): matrix of measurement operators
        weights (vector like, optional): vector of weights to apply to the
                                         objective function (default: None)
        PSD (bool, optional): Enforced the fitted matrix to be positive
                              semidefinite (default: True)
        trace (int, optional): trace constraint for the fitted matrix
                               (default: None).

    Returns:
        The fitted matrix rho that minimizes
        ||basis_matrix * vec(rho) - data||_2.

    Additional Information:

        Objective function
        ------------------
        This fitter solves the least-squares minimization:

            minimize ||a * x - b ||_2

        where:
            a is the matrix of measurement operators a[i] = vec(M_i).H
            b is the vector of expectation value data for each projector
              b[i] ~ Tr[M_i.H * x] = (a * x)[i]
            x is the vectorized density matrix (or Choi-matrix) to be fitted

        PSD Constraint
        --------------
        Since this minimization problem is unconstrained the returned fitted
        matrix may not be postive semidefinite (PSD). To enforce the PSD
        constraint the fitted matrix is rescaled using the method proposed in
        Reference [1].

        Trace constraint
        ----------------
        In general the trace of the fitted matrix will be determined by the
        input data. If a trace constraint is specified the fitted matrix
        will be rescaled to have this trace by:
            rho = trace * rho / trace(rho)

    References:
        [1] J Smolin, JM Gambetta, G Smith, Phys. Rev. Lett. 108, 070502
            (2012). Open access: arXiv:1106.5458 [quant-ph].
    """

    # We are solving the least squares fit: minimize ||a * x - b ||_2
    # where:
    #   a is the matrix of measurement operators
    #   b is the vector of expectation value data for each projector
    #   x is the vectorized density matrix (or Choi-matrix) to be fitted
    a = basis_matrix
    b = np.array(data)

    # Optionally apply a weights vector to the data and projectors
    if weights is not None:
        w = np.array(weights)
        a = w[:, None] * a
        b = w * b

    # Perform least squares fit using Scipy.linalg lstsq function
    rho_fit, _, _, _ = lstsq(a, b)

    # Reshape fit to a density matrix
    size = len(rho_fit)
    dim = int(np.sqrt(size))
    if dim * dim != size:
        raise ValueError("fitted vector is not a square matrix.")
    # Devectorize in column-major (Fortran order in Numpy)
    rho_fit = rho_fit.reshape(dim, dim, order='F')

    # Rescale fitted density matrix be positive-semidefinite
    if PSD is True:
        rho_fit = make_positive_semidefinite(rho_fit)

    # Rescale fitted density matrix to satisfy trace constraint
    if trace is not None:
        rho_fit *= trace / np.trace(rho_fit)
    return rho_fit


###########################################################################
# Wizard Method rescaling
###########################################################################

def make_positive_semidefinite(mat, epsilon=0):
    """
    Rescale a Hermitian matrix to nearest postive semidefinite matrix.

    Args:
        mat (array like): a hermitian matrix.
        epsilon (float >=0, optional): the threshold for setting
            eigenvalues to zero. If epsilon > 0 positive eigenvalues
            below epislon will also be set to zero (Default 0).
    Returns:
        The input matrix rescaled to have non-negative eigenvalues.

    References:
        [1] J Smolin, JM Gambetta, G Smith, Phys. Rev. Lett. 108, 070502
            (2012). Open access: arXiv:1106.5458 [quant-ph].
    """

    if epsilon < 0:
        raise ValueError('epsilon must be non-negative.')

    # Get the eigenvalues and eigenvectors of rho
    # eigenvalues are sorted in increasing order
    # v[i] <= v[i+1]

    dim = len(mat)
    v, w = la.eigh(mat)
    for j in range(dim):
        if v[j] < epsilon:
            tmp = v[j]
            v[j] = 0.
            # Rescale remaining eigenvalues
            x = 0.
            for k in range(j + 1, dim):
                x += tmp / (dim - (j + 1))
                v[k] = v[k] + tmp / (dim - (j + 1))

    # Build positive matrix from the rescaled eigenvalues
    # and the original eigenvectors

    mat_psd = np.zeros([dim, dim], dtype=complex)
    for j in range(dim):
        mat_psd += v[j] * np.outer(w[:, j], np.conj(w[:, j]))

    return mat_psd
