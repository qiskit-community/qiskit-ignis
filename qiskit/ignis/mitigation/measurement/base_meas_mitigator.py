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
Full A-matrix measurement migitation generator.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict
import numpy as np


class BaseMeasMitigator(ABC):
    """Base measurement error mitigator class."""

    @abstractmethod
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

    @abstractmethod
    def _compute_gamma(self, qubits: Optional[List[int]] = None) -> float:
        """Compute gamma for N-qubit mitigation."""

    def required_shots(self, delta: float, qubits: Optional[List[int]] = None) -> int:
        r"""Return the number of shots required for expectation value estimation.

        This is the number of shots required so that
        :math:`|\langle O \rangle_{est} - \langle O \rangle_{true}| < \delta`
        with high probability (at least 2/3) and is given by
        :math:`4\delta^2 \Gamma^2` where :math:`\Gamma^2` is the
        :meth:`mitigation_overhead`.

        Args:
            delta: Error tolerance for expectation value estimator.
            qubits: Optional, qubits being measured for operator expval.

        Returns:
            int: the required shots.
        """
        gamma = self._compute_gamma(qubits=qubits)
        return int(np.ceil((4 * gamma ** 2) / (delta ** 2)))

    def mitigation_overhead(self, qubits: Optional[List[int]] = None) -> int:
        """Return the mitigation overhead for expectation value estimation.

        This is the multiplicative factor of extra shots required for
        estimating a mitigated expectation value with the same accuracy as
        an unmitigated expectation value.

        Args:
            qubits: Optional, qubits being measured for operator expval.

        Returns:
            int: the mitigation overhead factor.
        """
        gamma = self._compute_gamma(qubits=qubits)
        return int(np.ceil(gamma ** 2))

    def stddev_upper_bound(self, shots: int = 1, qubits: Optional[List[int]] = None) -> float:
        """Return an upper bound on standard deviation of expval estimator.

        Args:
            shots: Number of shots used for expectation value measurement.
            qubits: qubits being measured for operator expval.

        Returns:
            float: the standard deviation upper bound.
        """
        gamma = self._compute_gamma(qubits=qubits)
        return gamma / np.sqrt(shots)

    @staticmethod
    def _z_diagonal(dim):
        parity = np.zeros(dim, dtype=np.int)
        for i in range(dim):
            parity[i] = bin(i)[2:].count('1')
        return (-1)**np.mod(parity, 2)
