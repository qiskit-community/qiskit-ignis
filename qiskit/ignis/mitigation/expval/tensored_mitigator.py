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
from typing import Optional, List, Dict, Tuple
import numpy as np

from .utils import (counts_probability_vector, _expval_with_stddev)
from .base_meas_mitigator import BaseExpvalMeasMitigator


class TensoredExpvalMeasMitigator(BaseExpvalMeasMitigator):
    """1-qubit tensor product measurement error mitigator.

    This class can be used with the
    :func:`qiskit.ignis.mitigation.expectation_value` function to apply
    measurement error mitigation of local single-qubit measurement errors.
    Expectation values can also be computed directly using the
    :meth:`expectation_value` method.

    For measurement mitigation to be applied the mitigator should be
    calibrated using the
    :func:`qiskit.ignis.mitigation.expval_meas_mitigator_circuits` function
    and :class:`qiskit.ignis.mitigation.ExpvalMeasMitigatorFitter` class with
    the ``'tensored'`` mitigation method.
    """

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
        # Get expectation value on specified qubits
        probs, shots = counts_probability_vector(
            counts, clbits=clbits, return_shots=True)
        num_qubits = int(np.log2(probs.shape[0]))

        # Get qubit mitigation matrix and mitigate probs
        if qubits is None:
            qubits = range(num_qubits)
        ainvs = self._mitigation_mats[list(qubits)]

        # Get operator coeffs
        if diagonal is None:
            diagonal = self._z_diagonal(2 ** num_qubits)
        # Apply transpose of mitigation matrix
        coeffs = np.reshape(diagonal, num_qubits * [2])
        einsum_args = [coeffs, list(range(num_qubits))]
        for i, ainv in enumerate(reversed(ainvs)):
            einsum_args += [ainv.T, [num_qubits + i, i]]
        einsum_args += [list(range(num_qubits, 2 * num_qubits))]
        coeffs = np.einsum(*einsum_args).ravel()

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

    def assignment_fidelity(self, qubits: Optional[List[int]] = None) -> float:
        r"""Return the measurement assignment fidelity on the specified qubits.

        The assignment fidelity on N-qubits is defined as
        :math:`\sum_{x\in\{0, 1\}^n} P(x|x) / 2^n`, where
        :math:`P(x|x) = \rangle x|A|x\langle`, and :math:`A` is the
        :meth:`assignment_matrix`.

        Args:
            qubits: Optional, qubits being measured for operator expval.

        Returns:
            float: the assignment fidelity.
        """
        if qubits is None:
            qubits = list(range(self._num_qubits))
        if isinstance(qubits, int):
            qubits = [qubits]
        fid = 1.0
        for i in qubits:
            fid *= np.mean(self._assignment_mats[i].diagonal())
        return fid

    def _compute_gamma(self, qubits=None):
        """Compute gamma for N-qubit mitigation"""
        if qubits is None:
            gammas = self._gammas
        else:
            gammas = self._gammas[list(qubits)]
        return np.product(gammas)
