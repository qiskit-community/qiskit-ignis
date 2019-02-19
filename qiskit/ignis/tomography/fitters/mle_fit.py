# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
Maximum-Likelihood estimation quantum tomography fitter
"""

import numpy as np
from scipy.linalg import lstsq

from .utils import make_positive_semidefinite


def state_mle_fit(data, basis_matrix, weights=None):
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
        This function is a wrapper for `mle_fit`. See
        `tomography.fitters.mle_fit` documentation for additional information.
    """
    return mle_fit(data, basis_matrix, weights=weights, PSD=True, trace=1)


def process_mle_fit(data, basis_matrix, weights=None):
    """
    Reconstruct a process (Choi) matrix using MLE least-squares fitting.

    Note: due to limitations of the fitting method the returned Choi-matrix
          will be completely-positive, but not necessarily trace preserving.

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
        The fitted Choi-matrix that minimizes
        ||basis_matrix * vec(choi) - data||_2.

    Additional Information:
        Due to limitations of the fitting method the returned Choi-matrix will
        be completely-positive, but not necessarily trace preserving.

        This function is a wrapper for `mle_fit`. See
        `tomography.fitters.mle_fit` documentation for additional information.
    """
    # Calculate trace of Choi-matrix from projector length
    rows, cols = np.shape(basis_matrix)
    dim = int(np.sqrt(np.sqrt(cols)))
    if dim ** 4 != cols:
        raise ValueError("Input data does not correspond to a process matrix.")
    return mle_fit(data, basis_matrix, weights=weights, PSD=True, trace=dim)


###########################################################################
# Linear Inversion (Least-Squares) Fitter
###########################################################################

def mle_fit(data, basis_matrix, weights=None, PSD=True, trace=None):
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
    rho_fit, residues, rank, s = lstsq(a, b)

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
