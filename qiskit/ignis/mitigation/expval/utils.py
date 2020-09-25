# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Measurement error mitigation utility functions.
"""

import logging
from functools import partial
from typing import Optional, List, Dict, Tuple
import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.result import Counts, Result
from qiskit.ignis.verification.tomography import marginal_counts, combine_counts
from qiskit.ignis.numba import jit_fallback

logger = logging.getLogger(__name__)


def expectation_value(counts: Counts,
                      diagonal: Optional[np.ndarray] = None,
                      qubits: Optional[List[int]] = None,
                      clbits: Optional[List[int]] = None,
                      meas_mitigator: Optional = None,
                      ) -> Tuple[float, float]:
    r"""Compute the expectation value of a diagonal operator from counts.

    This computes the estimator of
    :math:`\langle O \rangle = \mbox{Tr}[\rho. O]`, optionally with measurement
    error mitigation, of a diagonal observable
    :math:`O = \sum_{x\in\{0, 1\}^n} O(x)|x\rangle\!\langle x|`.

    Args:
        counts: counts object
        diagonal: Optional, the vector of diagonal values for summing the
                    expectation value. If ``None`` the the default value is
                    :math:`[1, -1]^\otimes n`.
        qubits: Optional, the measured physical qubits the count
                bitstrings correspond to. If None qubits are assumed to be
                :math:`[0, ..., n-1]`.
        clbits: Optional, if not None marginalize counts to the specified bits.
        meas_mitigator: Optional, a measurement mitigator to apply mitigation.

    Returns:
        (float, float): the expectation value and standard deviation.

    Additional Information:
        The diagonal observable :math:`O` is input using the ``diagonal``
        kwarg as a list or Numpy array :math:`[O(0), ..., O(2^n -1)]`. If
        no diagonal is specified the diagonal of the Pauli operator
        :math:`O = \mbox{diag}(Z^{\otimes n}) = [1, -1]^{\otimes n}` is used.

        The ``clbits`` kwarg is used to marginalize the input counts dictionary
        over the specified bit-values, and the ``qubits`` kwarg is used to specify
        which physical qubits these bit-values correspond to as
        ``circuit.measure(qubits, clbits)``.

        For calibrating a expval measurement error mitigator for the
        ``meas_mitigator`` kwarg see
        :func:`qiskit.ignis.mitigation.expval_meas_mitigator_circuits` and
        :class:`qiskit.ignis.mitigation.ExpvalMeasMitigatorFitter`.
    """
    if meas_mitigator is not None:
        # Use mitigator expectation value method
        return meas_mitigator.expectation_value(
            counts, diagonal=diagonal, clbits=clbits,
            qubits=qubits)

    # Marginalize counts
    if clbits is not None:
        counts = marginal_counts(counts, meas_qubits=clbits)

    # Get counts shots and probabilities
    probs = np.array(list(counts.values()))
    shots = probs.sum()
    probs = probs / shots

    # Get diagonal operator coefficients
    if diagonal is None:
        coeffs = np.array([(-1) ** (key.count('1') % 2)
                           for key in counts.keys()], dtype=probs.dtype)
    else:
        diagonal = np.asarray(diagonal)
        keys = [int(key, 2) for key in counts.keys()]
        coeffs = np.asarray(diagonal[keys], dtype=probs.dtype)

    return _expval_with_stddev(coeffs, probs, shots)


def counts_probability_vector(
        counts: Counts,
        qubits: Optional[List[int]] = None,
        clbits: Optional[List[int]] = None,
        num_qubits: Optional[int] = None,
        return_shots: Optional[bool] = False) -> np.ndarray:
    """Compute mitigated expectation value.

    Args:
        counts: counts object
        qubits: qubits the count bitstrings correspond to.
        clbits: Optional, marginalize counts to just these bits.
        num_qubits: the total number of qubits.
        return_shots: return the number of shots.

    Raises:
        QiskitError: if qubit and clbit kwargs are not valid.

    Returns:
        np.ndarray: a probability vector for all count outcomes.
    """
    # Marginalize counts
    if clbits is not None:
        counts = marginal_counts(counts, meas_qubits=clbits)

    # Get total number of qubits
    if num_qubits is None:
        num_qubits = len(next(iter(counts)))

    # Get vector
    vec = np.zeros(2**num_qubits, dtype=float)
    shots = 0
    for key, val in counts.items():
        shots += val
        vec[int(key, 2)] = val
    vec /= shots

    # Remap qubits
    if qubits is not None:
        if len(qubits) != num_qubits:
            raise QiskitError("Num qubits does not match vector length.")
        axes = [num_qubits - 1 - i for i in reversed(np.argsort(qubits))]
        vec = np.reshape(vec,
                         num_qubits * [2]).transpose(axes).reshape(vec.shape)
    if return_shots:
        return vec, shots
    return vec


def calibration_data(result: Result,
                     metadata: List[Dict[str, any]]) -> Tuple[
                         Dict[int, Dict[int, int]], int]:
    """Return FullMeasureErrorMitigator from result data.

    Args:
        result: Qiskit result object.
        metadata: mitigation generator metadata.

    Returns:
        Calibration data dictionary {label: Counts} and number of qubits.
    """
    # Filter mitigation calibration data
    cal_data = {}
    num_qubits = None
    method = None
    for i, meta in enumerate(metadata):
        if meta.get('experiment') == 'meas_mit':
            if num_qubits is None:
                num_qubits = len(meta['cal'])
            if method is None:
                method = meta.get('method', None)
            key = int(meta['cal'], 2)
            counts = result.get_counts(i).int_outcomes()
            if key not in cal_data:
                cal_data[key] = counts
            else:
                cal_data[key] = combine_counts(cal_data[key], counts)
    return cal_data, num_qubits, method


def assignment_matrix(cal_data: Dict[int, Dict[int, int]],
                      num_qubits: int,
                      qubits: Optional[List[int]] = None) -> np.array:
    """Computes the assignment matrix for specified qubits.

    Args:
        cal_data: calibration dataset.
        num_qubits: the number of qubits for the calibation dataset.
        qubits: Optional, the qubit subset to construct the matrix on.

    Returns:
        np.ndarray: the constructed A-matrix.

    Raises:
        QiskitError: if the calibration data is not sufficient for
                     reconstruction on the specified qubits.
    """
    # If qubits is None construct full A-matrix from calibration data
    # Otherwise we compute the local A-matrix on specified
    # qubits subset. This involves filtering the cal data
    # on no-errors occuring on qubits outside the subset

    if qubits is not None:
        qubits = np.asarray(qubits)
        dim = 1 << qubits.size
        mask = _amat_mask(qubits, num_qubits)
        accum_func = partial(_amat_accum_local, qubits, mask)

    else:
        qubits = np.array(range(num_qubits))
        dim = 1 << num_qubits
        accum_func = _amat_accum_full

    amat = np.zeros([dim, dim], dtype=float)
    for cal, counts in cal_data.items():
        counts_keys = np.array(list(counts.keys()))
        counts_values = np.array(list(counts.values()))
        accum_func(amat, cal, counts_keys, counts_values)

    renorm = amat.sum(axis=0, keepdims=True)
    if np.any(renorm == 0):
        raise QiskitError(
            'Insufficient calibration data to fit assignment matrix'
            ' on qubits {}'.format(qubits.tolist()))
    return amat / renorm


def _expval_with_stddev(coeffs: np.ndarray,
                        probs: np.ndarray,
                        shots: int) -> Tuple[float, float]:
    """Compute expectation value and standard deviation.

    Args:
        coeffs: array of diagonal operator coefficients.
        probs: array of measurement probabilities.
        shots: total number of shots to obtain probabilities.

    Returns:
        tuple: (expval, stddev) expectation value and standard deviation.
    """
    # Compute expval
    expval = coeffs.dot(probs)

    # Compute variance
    sq_expval = (coeffs ** 2).dot(probs)
    variance = (sq_expval - expval ** 2) / shots

    # Compute standard deviation
    if variance < 0 and not np.isclose(variance, 0):
        logger.warning(
            'Encountered a negative variance in expectation value calculation.'
            '(%f). Setting standard deviation of result to 0.', variance)
    stddev = np.sqrt(variance) if variance > 0 else 0.0
    return expval, stddev


@jit_fallback
def _amat_mask(qubits, num_qubits):
    """Compute bit-mask for other other non-specified qubits."""
    mask = 0
    for i in range(num_qubits):
        if not np.any(qubits == i):
            mask += 1 << i
    return mask


@jit_fallback
def _amat_index(i, qubits):
    """Compute local index for specified qubits and full index value."""
    masks = 1 << qubits
    shifts = np.arange(qubits.size - 1, -1, -1)
    val = ((i & masks) >> qubits) << shifts
    return np.sum(val)


@jit_fallback
def _amat_accum_local(qubits, mask, mat, cal, counts_keys, counts_values):
    """Accumulate calibration data on the specified matrix"""
    x_out = cal & mask
    x_ind = _amat_index(cal, qubits)
    for i in range(counts_keys.size):
        y_out = counts_keys[i] & mask
        if x_out == y_out:
            y_ind = _amat_index(counts_keys[i], qubits)
            mat[y_ind, x_ind] += counts_values[i]


@jit_fallback
def _amat_accum_full(mat, cal, counts_keys, counts_values):
    """Accumulate calibration data on the specified matrix"""
    for i in range(counts_keys.size):
        mat[counts_keys[i], cal] += counts_values[i]
