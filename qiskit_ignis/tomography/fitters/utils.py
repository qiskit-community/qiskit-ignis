# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
Quantum tomography fitter data formatting.
"""

import logging
import itertools as it
import numpy as np
from scipy import linalg as la

from qiskit._qiskiterror import QISKitError

from ..data import marginal_counts, count_keys
from ..basis import TomographyBasis
from ..circuits import default_basis

# Create logger
logger = logging.getLogger(__name__)


###########################################################################
# Data formats for converting from counts to fitter data
###########################################################################

def fitter_data(tomo_data,
                meas_basis='Pauli',
                prep_basis='Pauli',
                calibration_matrix=None,
                standard_weights=True,
                beta=0.5):
    """Generate tomography fitter data from a tomography data dictionary.

    Args:
        tomo_data (dict): tomography data returned from `tomography_data`
                          function.
        meas_matrix_fn (function, optional): A function to return measurement
                       operators corresponding to measurement outcomes. See
                       Additional Information (default: 'Pauli')
        prep_matrix_fn (function, optional): A function to return preparation
                       operators. See Additional Information (default: 'Pauli')
        calibration_matrix (matrix, optional): calibration matrix of noisey
                            measurement assignment fidelities (default: None)
        standard_weights (bool, optional): Apply weights to basis matrix
                         and data based on count probability (default: True)
        beta (float): hedging parameter for 0, 1 probabilities (default: 0)

    Returns:
        tuple(data, basis_matrix) where `data` is a vector of the computed
        expectation values, and `basis_matrix` is a matrix of the preparation
        and measurement operator.

    Additional Information
    ----------------------
    standard_weights:
        Weights are calculated from from binomial distribution standard deviation

    calibration_method:
        If a calibration matrix is provided, the pseudo-inverse of this matrix
        will be used to update the counts. Note that his count correction is
        applied before the (optional) standard weights update has been applied.
    """

    # Load built-in circuit functions
    if callable(meas_basis):
        measurement = meas_basis
    else:
        measurement = default_basis(meas_basis)
        if isinstance(measurement, TomographyBasis):
            if measurement.measurement is not True:
                raise QISKitError("Invalid measurement basis")
            measurement = measurement.measurement_matrix
    if callable(prep_basis):
        preparation = prep_basis
    else:
        preparation = default_basis(prep_basis)
        if isinstance(preparation, TomographyBasis):
            if preparation.preparation is not True:
                raise QISKitError("Invalid preparation basis")
            preparation = preparation.preparation_matrix

    # If calibration matrix is specified the pseudo-inverse of the
    # calibration matrix will be applied to the data values.
    if calibration_matrix is not None:
        cal_inv = la.pinv(np.array(calibration_matrix))

    data = []
    basis_blocks = []
    # Generate counts keys for converting to np array
    ctkeys = count_keys(len(next(iter(tomo_data))[0]))

    for label, cts in tomo_data.items():

        # Convert counts dict to numpy array
        if isinstance(cts, dict):
            cts = np.array([cts.get(key, 0) for key in ctkeys])

        # Get probabilities
        shots = np.sum(cts)
        probs = np.array(cts) / shots

        # Get reconstruction basis operators
        prep_op = _preparation_op(label[1], preparation)
        meas_ops = _measurement_ops(label[0], measurement)
        block = _basis_operator_matrix([np.kron(prep_op.T, mop) for mop in meas_ops])

        # Apply calibration pseudo-inverse before weights
        if calibration_matrix is not None:
            probs = cal_inv @ probs

        # Apply weights
        if standard_weights is True:
            wts = binomial_weights(cts, beta)
            block = wts[:, None] * block
            probs = wts * probs

        # Append data
        data += list(probs)
        basis_blocks.append(block)

    return data, np.vstack(basis_blocks)


###########################################################################
# Binomial weights for count statistics
###########################################################################

def binomial_weights(counts, beta=0.5):
    """
    Compute binomial weights for list or dictionary of counts.

    Args:
        counts (dict, vector): A set of measurement counts for
                               all outcomes of a given measurement
                               configuration.
        beta (float >= 0): A hedging parameter used to bias probabilities
                           computed from input counts away from 0 or 1.

    Returns:
        A numpy array of binomial weights for the input counts and beta
        parameter.

    Additional Information:

        The weights are determined by
            w[i] = sqrt(shots / p[i] * (1 - p[i]))
            p[i] = (counts[i] + beta) / (shots + K * beta)
        where
            `shots` is the sum of all counts in the input
            `p` is the hedged probability computed for a count
            `K` is the total number of possible measurement outcomes.
    """

    # Sort counts if input is a dictionary
    if isinstance(counts, dict):
        mcts = marginal_counts(counts, pad_zeros=True)
        ordered_keys = sorted(list(mcts))
        counts = np.array([mcts[k] for k in ordered_keys])
    # Assume counts are already sorted if a list
    else:
        counts = np.array(counts)
    shots = np.sum(counts)

    # If beta is 0 check if we would be dividing by zero
    # If so change beta value and log warning.

    if beta < 0:
        raise ValueError('beta = {} must be non-negative.'.format(beta))
    if beta == 0 and (shots in counts or 0 in counts):
        beta = 0.50922
        msg = 'Counts result in probabilities of 0 or 1 in binomial weights calculation. '
        msg += ' Setting hedging parameter beta={} to prevent dividing by zero.'.format(beta)
        logger.warning(msg)

    K = len(counts)  # Number of possible outcomes.
    # Compute hedged frequencies which are shifted to never be 0 or 1.
    freqs_hedged = (counts + beta) / (shots + K * beta)

    # Return gaussian weights for 2-outcome measurements.
    return np.sqrt(shots / (freqs_hedged * (1 - freqs_hedged)))


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
        [1] J Smolin, JM Gambetta, G Smith, Phys. Rev. Lett. 108, 070502 (2012).
            Open access: arXiv:1106.5458 [quant-ph].
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


###########################################################################
# Basis projector construction functions
###########################################################################

def _basis_operator_matrix(basis):
    """
    Return a basis measurement matrix of the input basis.

    Args:
        basis (list (array like)): a list of basis matrices.

    Returns:
        A numpy array of shape (n, col * row) where n is the number
        of operators of shape (row, col) in `basis`.
    """
    # Dimensions
    num_ops = len(basis)
    nrows, ncols = basis[0].shape
    size = nrows * ncols

    ret = np.zeros((num_ops, size), dtype=complex)
    for j, b in enumerate(basis):
        ret[j] = np.array(b).reshape((1, size), order='F').conj()
    return ret


def _preparation_op(label, prep_matrix_fn):
    """
    Return the multi-qubit matrix for a state preparation label.

    Args:
        label (tuple(str)): a preparation configuration label for a
                            tomography circuit.
        prep_matrix_fn (function): a function that returns the matrix
                        corresponding to a single qubit preparation label.
                        The functions should have signature
                            prep_matrix_fn(str) -> np.array
    Returns:
        A Numpy array for the multi-qubit prepration operator specified
        by label.

    Additional Information:
        See the Pauli and SIC-POVM preparation functions
        `pauli_preparation_matrix` and `sicpovm_preparation_matrix` for
        examples.
    """

    # Trivial case
    if label is None:
        return np.eye(1, dtype=complex)

    # Construct preparation matrix
    op = np.eye(1, dtype=complex)
    for l in label:
        op = np.kron(prep_matrix_fn(l), op)
    return op


def _measurement_ops(label, meas_matrix_fn):
    """
    Return a list multi-qubit matrices for a measurement label.

    Args:
        label (tuple(str)): a measurement configuration label for a
                            tomography circuit.
        meas_matrix_fn (function): a function that returns the matrix
                        corresponding to a single qubit measurement label
                        for a given outcome. The functions should have signature
                            meas_matrix_fn(str, int) -> np.array

    Returns:
        A list of Numpy array for the multi-qubit measurement operators
        for all measurement outcomes for the measurement basis specified
        by the label. These are ordered in increasing binary order. Eg for
        2-qubits the returned matrices correspond to outcomes [00, 01, 10, 11]

    Additional Information:
        See the Pauli measurement function `pauli_measurement_matrix`
        for an example.
    """
    num_qubits = len(label)
    meas_ops = []

    # Construct measurement POVM for all measurement outcomes for a given
    # measurement label. This will be a list of 2 ** n operators.

    for l in sorted(it.product((0, 1), repeat=num_qubits)):
        op = np.eye(1, dtype=complex)
        # Reverse label to correspond to QISKit bit ordering
        for m, outcome in zip(reversed(label), l):
            op = np.kron(op, meas_matrix_fn(m, outcome))
        meas_ops.append(op)
    return meas_ops
