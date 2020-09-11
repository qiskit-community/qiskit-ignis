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

import logging
from functools import partial
from typing import Optional, List, Dict, Tuple, Union
import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.result import Counts, Result
from qiskit.ignis.verification.tomography import marginal_counts, combine_counts
from qiskit.ignis.numba import jit_fallback

logger = logging.getLogger(__name__)


def expectation_value(counts: Counts,
                      diagonal: Optional[np.ndarray] = None,
                      clbits: Optional[List[int]] = None,
                      mitigator: Optional = None,
                      mitigator_qubits: Optional[List[int]] = None) -> Tuple[float, float]:
    r"""Compute the expectation value of a diagonal operator from counts.

    Args:
        counts: counts object.
        diagonal: Optional, values of the diagonal observable. If None the
                  observable is set to :math:`Z^\otimes n`.
        clbits: Optional, marginalize counts to just these bits.
        mitigator: Optional, a measurement mitigator to apply mitigation.
        mitigator_qubits: Optional, qubits the count bitstrings correspond to.
                          Only required if applying mitigation with clbits arg.

    Returns:
        (float, float): the expectation value and standard deviation.
    """
    expval, var = _expval_with_variance(counts,
                                        diagonal=diagonal,
                                        clbits=clbits,
                                        mitigator=mitigator,
                                        mitigator_qubits=mitigator_qubits)
    return expval, np.sqrt(var)


def pauli_diagonal(pauli: str) -> np.ndarray:
    """Return diagonal for given Pauli.

    Args:
        pauli: a pauli string.

    Returns:
        np.ndarray: The diagonal vector for converting the Pauli basis
                    measurement into an expectation value.
    """
    if pauli[0] in ['+', '-']:
        pauli = pauli[1:]

    diag = np.array([1])
    for i in reversed(pauli):
        if i == 'I':
            tmp = np.array([1, 1])
        else:
            tmp = np.array([1, -1])
        diag = np.kron(tmp, diag)
    return diag


def _expval_with_variance(counts: Counts,
                          diagonal: Optional[np.ndarray] = None,
                          clbits: Optional[List[int]] = None,
                          mitigator: Optional = None,
                          mitigator_qubits: Optional[List[int]] = None) -> Tuple[float, float]:
    r"""Compute the expectation value of a diagonal operator from counts.

    Args:
        counts: counts object.
        diagonal: Optional, values of the diagonal observable. If None the
                  observable is set to :math:`Z^\otimes n`.
        clbits: Optional, marginalize counts to just these bits.
        mitigator: Optional, a measurement mitigator to apply mitigation.
        mitigator_qubits: Optional, qubits the count bitstrings correspond to.
                          Only required if applying mitigation with clbits arg.

    Returns:
        (float, float): the expectation value and variance.
    """
    if mitigator is not None:
        # Use mitigator expectation value method
        return mitigator.expectation_value(
            counts, diagonal=diagonal, clbits=clbits, qubits=mitigator_qubits)

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
        keys = [int(key, 2) for key in counts.keys()]
        coeffs = np.asarray(diagonal[keys], dtype=probs.dtype)

    # Compute expval
    expval = coeffs.dot(probs)

    # Compute variance
    if diagonal is None:
        # The square of the parity diagonal is the all 1 vector
        sq_expval = np.sum(probs)
    else:
        sq_expval = (coeffs ** 2).dot(probs)
    variance = (sq_expval - expval ** 2) / shots

    # Compute standard deviation
    if variance < 0:
        if not np.isclose(variance, 0):
            logger.warning(
                'Encountered a negative variance in expectation value calculation.'
                '(%f). Setting standard deviation of result to 0.', variance)
        variance = 0.0
    return expval, variance
