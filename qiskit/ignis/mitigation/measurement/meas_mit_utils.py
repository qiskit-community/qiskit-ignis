# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
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

from functools import partial
from typing import Optional, List, Dict, Tuple
import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.result import Counts, Result
from qiskit.ignis.verification.tomography import marginal_counts, combine_counts
from qiskit.ignis.numba import jit_fallback


def counts_probability_vector(
        counts: Counts,
        clbits: Optional[List[int]] = None,
        qubits: Optional[List[int]] = None,
        num_qubits: Optional[int] = None) -> np.ndarray:
    """Compute mitigated expectation value.

    Args:
        counts: counts object
        clbits: Optional, marginalize counts to just these bits.
        qubits: qubits the count bitstrings correspond to.
        num_qubits: the total number of qubits.

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

    return vec


def counts_expectation_value(counts: Counts,
                             clbits: Optional[List[int]] = None,
                             diagonal: Optional[np.ndarray] = None) -> float:
    r"""Convert counts to a probability vector.

    Args:
        counts: counts object.
        clbits: Optional, marginalize counts to just these bits.
        diagonal: Optional, values of the diagonal observable. If None the
                      observable is set to :math:`Z^\otimes n`.

    Returns:
        float: expectation value.
    """
    # Marginalize counts
    if clbits is not None:
        counts = marginal_counts(counts, meas_qubits=clbits)

    # Compute expval via reduction
    expval = 0
    shots = 0
    for key, val in counts.items():
        shots += val
        if diagonal is None:
            coeff = (-1) ** (key.count('1') % 2)
        else:
            coeff = diagonal[int(key, 2)]
        expval += coeff * val
    return expval / shots


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
    for i, meta in enumerate(metadata):
        if meta.get('experiment') == 'meas_mit':
            if num_qubits is None:
                num_qubits = len(meta['cal'])
            key = int(meta['cal'], 2)
            counts = result.get_counts(i).int_outcomes()
            if key not in cal_data:
                cal_data[key] = counts
            else:
                cal_data[key] = combine_counts(cal_data[key], counts)
    return cal_data, num_qubits


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
