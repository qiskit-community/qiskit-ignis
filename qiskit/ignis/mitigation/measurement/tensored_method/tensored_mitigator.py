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
Single-qubit tensor-product measurement error mitigation generator.
"""
from typing import Optional, List, Dict
import numpy as np

from ..meas_mit_utils import counts_probability_vector
from ..base_meas_mitigator import BaseMeasMitigator


class TensoredMeasMitigator(BaseMeasMitigator):
    """Measurement error mitigator via 1-qubit tensor product mitigation."""

    def __init__(self, amats: List[np.ndarray]):
        """Initialize a TensorMeasurementMitigator

        Args:
            amats: list of single-qubit readout error assignment matrices.
        """
        self._num_qubits = len(amats)
        self._assignment_mats = amats
        self._mitigation_mats = np.zeros([self._num_qubits, 2, 2], dtype=float)
        self._gammas = np.zeros(self._num_qubits, dtype=float)

        for i in range(self._num_qubits):
            mat = self._assignment_mats[i]
            # Compute Gamma values
            error0 = mat[1, 0]
            error1 = mat[0, 1]
            self._gammas[i] = (1 + abs(error0 - error1)) / (1 - error0 - error1)
            # Compute inverse mitigation matrix
            try:
                ainv = np.linalg.inv(mat)
            except np.linalg.LinAlgError:
                ainv = np.linalg.pinv(mat)
            self._mitigation_mats[i] = ainv

    def mitigation_matrix(self, qubits: List[int] = None) -> np.ndarray:
        r"""Return the measurement mitigation matrix for the specified qubits.

        The mitigation matrix :math:`A^{-1}` is defined as the inverse of the
        :meth:`assignment_matrix` :math:`A`.

        Args:
            qubits: Optional, qubits being measured for operator expval.

        Returns:
            np.ndarray: the measurement error mitigation matrix :math:`A^{-1}`.
        """
        if qubits is None:
            qubits = list(range(self._num_qubits))
        if isinstance(qubits, int):
            qubits = [qubits]
        mat = self._mitigation_mats[qubits[0]]
        for i in qubits[1:]:
            mat = np.kron(self._mitigation_mats[qubits[i]], mat)
        return mat

    def assignment_matrix(self, qubits: List[int] = None) -> np.ndarray:
        r"""Return the measurement assignment matrix for specified qubits.

        The assignment matrix is the stochastic matrix :math:`A` which assigns
        a noisy measurement probability distribution to an ideal input
        measurement distribution: :math:`P(i|j) = \langle i | A | j \rangle`.

        Args:
            qubits: Optional, qubits being measured for operator expval.

        Returns:
            np.ndarray: the assignment matrix A.
        """
        if qubits is None:
            qubits = list(range(self._num_qubits))
        if isinstance(qubits, int):
            qubits = [qubits]
        mat = self._assignment_mats[qubits[0]]
        for i in qubits[1:]:
            mat = np.kron(self._assignment_mats[qubits[i]], mat)
        return mat

    def expectation_value(self,
                          counts: Dict,
                          clbits: Optional[List[int]] = None,
                          qubits: Optional[List[int]] = None,
                          diagonal: Optional[np.ndarray] = None) -> float:
        r"""Compute the mitigated expectation value of a diagonal observable.

        This computes the mitigated estimator of
        :math:`\langle O \rangle = \mbox{Tr}[\rho. O]` of a diagonal observable
        :math:`O = \sum_{x\in\{0, 1\}^n} O(x)|x\rangle\!\langle x|` where
        :math:`|O(x)|\le 1`.

        Args:
            counts: counts object
            clbits: Optional, marginalize counts to just these bits.
            qubits: qubits the count bitstrings correspond to.
            diagonal: Optional, values of the diagonal observable. If None the
                      observable is set to :math:`Z^\otimes n`.
        Returns:
            float: expval.

        Additional Information:
            The observable :math:`O` is input using the ``diagonal`` kwarg as a
            Numpy array :math:`[O(0), ..., O(2^n -1)]`. If no diagonal is specified
            the Pauli-Z operator `O = Z^{\otimes n}` is used.

            The ``clbits`` kwarg is used to marginalize the input counts dictionary
            over the specified bit-values, and the ``qubits`` kwarg is used to specify
            which physical qubits these bit-values correspond to as
            ``circuit.measure(qubits, clbits)``.
        """
        # Get expectation value on specified qubits
        probs = counts_probability_vector(counts, clbits=clbits)
        num_qubits = int(np.log2(probs.shape[0]))

        if qubits is not None:
            ainvs = self._mitigation_mats[list(qubits)]
        else:
            ainvs = self._mitigation_mats

        probs = np.reshape(probs, num_qubits * [2])
        einsum_args = [probs, list(range(num_qubits))]
        for i, ainv in enumerate(reversed(ainvs)):
            einsum_args += [ainv, [num_qubits + i, i]]
        einsum_args += [list(range(num_qubits, 2 * num_qubits))]

        probs_mit = np.einsum(*einsum_args).ravel()
        if diagonal is None:
            diagonal = self._z_diagonal(2 ** num_qubits)
        return probs_mit.dot(diagonal)

    def _compute_gamma(self, qubits=None):
        """Compute gamma for N-qubit mitigation"""
        if qubits is None:
            gammas = self._gammas
        else:
            gammas = self._gammas[list(qubits)]
        return np.product(gammas)
