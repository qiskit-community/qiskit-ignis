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
Full-matrix measurement error mitigation generator.
"""
from typing import Optional, List, Dict, Tuple
import numpy as np

from qiskit.exceptions import QiskitError
from .utils import (counts_probability_vector, _expval_with_stddev)
from .base_meas_mitigator import BaseExpvalMeasMitigator


class CompleteExpvalMeasMitigator(BaseExpvalMeasMitigator):
    """N-qubit measurement error mitigator.

    This class can be used with the
    :func:`qiskit.ignis.mitigation.expectation_value` function to apply
    measurement error mitigation of general N-qubit measurement errors
    when calculating expectation values from counts. Expectation values
    can also be computed directly using the :meth:`expectation_value` method.

    For measurement mitigation to be applied the mitigator should be
    calibrated using the
    :func:`qiskit.ignis.mitigation.expval_meas_mitigator_circuits` function
    and :class:`qiskit.ignis.mitigation.ExpvalMeasMitigatorFitter` class with
    the ``'complete'`` mitigation method.
    """

    def __init__(self, amat: np.ndarray):
        """Initialize a TensorMeasurementMitigator

        Args:
            amat (np.array): readout error assignment matrix.
        """
        self._num_qubits = int(np.log2(amat.shape[0]))
        self._assignment_mat = amat
        self._mitigation_mats = {}

    def expectation_value(self,
                          counts: Dict,
                          diagonal: Optional[np.ndarray] = None,
                          qubits: Optional[List[int]] = None,
                          clbits: Optional[List[int]] = None,
                          ) -> Tuple[float, float]:
        r"""Compute the mitigated expectation value of a diagonal observable.

        This computes the mitigated estimator of
        :math:`\langle O \rangle = \mbox{Tr}[\rho. O]` of a diagonal observable
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

        Returns:
            (float, float): the expectation value and standard deviation.

        Raises:
            QiskitError: if input arguments are invalid.

        Additional Information:
            The diagonal observable :math:`O` is input using the ``diagonal`` kwarg as
            a list or Numpy array :math:`[O(0), ..., O(2^n -1)]`. If no diagonal is specified
            the diagonal of the Pauli operator
            :math`O = \mbox{diag}(Z^{\otimes n}) = [1, -1]^{\otimes n}` is used.

            The ``clbits`` kwarg is used to marginalize the input counts dictionary
            over the specified bit-values, and the ``qubits`` kwarg is used to specify
            which physical qubits these bit-values correspond to as
            ``circuit.measure(qubits, clbits)``.
        """
        # Get probability vector
        probs, shots = counts_probability_vector(
            counts, clbits=clbits, qubits=qubits, return_shots=True)
        num_qubits = int(np.log2(probs.shape[0]))

        # Get qubit mitigation matrix and mitigate probs
        if qubits is None:
            qubits = tuple(range(num_qubits))
        if len(qubits) != num_qubits:
            raise QiskitError("Num qubits does not match number of clbits.")
        mit_mat = self.mitigation_matrix(qubits)

        # Get operator coeffs
        if diagonal is None:
            diagonal = self._z_diagonal(2 ** num_qubits)
        # Apply transpose of mitigation matrix
        coeffs = mit_mat.T.dot(diagonal)

        return _expval_with_stddev(coeffs, probs, shots)

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
            qubits = tuple(range(self._num_qubits))
        else:
            qubits = tuple(sorted(qubits))

        # Check for cached mitigation matrix
        # if not present compute
        if qubits not in self._mitigation_mats:
            marginal_matrix = self.assignment_matrix(qubits)
            try:
                mit_mat = np.linalg.inv(marginal_matrix)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if matrix is singular
                mit_mat = np.linalg.pinv(marginal_matrix)
            self._mitigation_mats[qubits] = mit_mat

        return self._mitigation_mats[qubits]

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
            return self._assignment_mat

        if isinstance(qubits, int):
            qubits = [qubits]

        # Compute marginal matrix
        axis = tuple([self._num_qubits - 1 - i for i in set(
            range(self._num_qubits)).difference(qubits)])
        num_qubits = len(qubits)
        new_amat = np.zeros(2 * [2 ** num_qubits], dtype=float)
        for i, col in enumerate(self._assignment_mat.T[self._keep_indexes(qubits)]):
            new_amat[i] = np.reshape(col, self._num_qubits * [2]).sum(axis=axis).reshape(
                [2 ** num_qubits])
        new_amat = new_amat.T
        return new_amat

    @staticmethod
    def _keep_indexes(qubits):
        indexes = [0]
        for i in sorted(qubits):
            indexes += [idx + (1 << i) for idx in indexes]
        return indexes

    def _compute_gamma(self, qubits=None):
        """Compute gamma for N-qubit mitigation"""
        mitmat = self.mitigation_matrix(qubits=qubits)
        return np.max(np.sum(np.abs(mitmat), axis=0))
