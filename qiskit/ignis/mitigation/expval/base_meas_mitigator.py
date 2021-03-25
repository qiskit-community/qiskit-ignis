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
Full A-matrix measurement migitation generator.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Tuple, Union
import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


class BaseExpvalMeasMitigator(ABC):
    """Base measurement error mitigator class."""

    @abstractmethod
    def expectation_value(self,
                          counts: Dict,
                          diagonal: Optional[
                              Union[np.ndarray, List[complex], List[float]]] = None,
                          qubits: Optional[List[int]] = None,
                          clbits: Optional[List[int]] = None,
                          ) -> Tuple[float, float]:
        r"""Compute a measurement error mitigated expectation value.

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

    @abstractmethod
    def mitigation_matrix(self, qubits: List[int] = None) -> np.ndarray:
        r"""Return the measurement mitigation matrix for the specified qubits.

        The mitigation matrix :math:`A^{-1}` is defined as the inverse of the
        :meth:`assignment_matrix` :math:`A`.

        Args:
            qubits: Optional, qubits being measured for operator expval.

        Returns:
            np.ndarray: the measurement error mitigation matrix :math:`A^{-1}`.
        """

    @abstractmethod
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

    @abstractmethod
    def _compute_gamma(self, qubits: Optional[List[int]] = None) -> float:
        """Compute gamma for N-qubit mitigation."""

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
        return self.assignment_matrix(qubits=qubits).diagonal().mean()

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

    def plot_assignment_matrix(self,
                               qubits=None,
                               ax=None):
        """Matrix plot of the readout error assignment matrix.

        Args:
            qubits (list(int)): Optional, qubits being measured for operator expval.
            ax (axes): Optional. Axes object to add plot to.

        Returns:
            plt.axes: the figure axes object.

        Raises:
            ImportError: if matplotlib is not installed.
        """
        if not _HAS_MATPLOTLIB:
            raise ImportError(
                'Plotting functions require the optional matplotlib package.'
                ' Install with `pip install matplotlib`.')

        mat = self.assignment_matrix(qubits)
        if ax is None:
            figure = plt.figure()
            ax = figure.gca()
        axim = ax.matshow(mat, cmap=plt.cm.binary, clim=[0, 1])
        ax.figure.colorbar(axim)
        ax = self._plot_axis(mat, ax=ax)
        return ax

    def plot_mitigation_matrix(self,
                               qubits=None,
                               ax=None):
        """Matrix plot of the readout error mitigation matrix.

        Args:
            qubits (list(int)): Optional, qubits being measured for operator expval.
            ax (plt.axes): Optional. Axes object to add plot to.

        Returns:
            plt.axes: the figure axes object.

        Raises:
            ImportError: if matplotlib is not installed.
        """
        if not _HAS_MATPLOTLIB:
            raise ImportError(
                'Plotting functions require the optional matplotlib package.'
                ' Install with `pip install matplotlib`.')

        # Matrix parameters
        mat = self.mitigation_matrix(qubits)
        mat_max = np.max(mat)
        mat_min = np.min(mat)
        lim = max(abs(mat_max), abs(mat_min), 1)

        if ax is None:
            figure = plt.figure()
            ax = figure.gca()
        axim = ax.matshow(mat, cmap=plt.cm.RdGy, clim=[-lim, lim])
        ax.figure.colorbar(axim, values=np.linspace(mat_min, mat_max, 100))
        ax = self._plot_axis(mat, ax=ax)
        return ax

    @staticmethod
    def _z_diagonal(dim, dtype=float):
        r"""Return the diagonal for the operator :math:`Z^\otimes n`"""
        parity = np.zeros(dim, dtype=dtype)
        for i in range(dim):
            parity[i] = bin(i)[2:].count('1')
        return (-1)**np.mod(parity, 2)

    @staticmethod
    def _int_to_bitstring(i, num_qubits=None):
        """Convert an integer to a bitstring."""
        label = bin(i)[2:]
        if num_qubits:
            pad = num_qubits - len(label)
            if pad > 0:
                label = pad * '0' + label
        return label

    @staticmethod
    def _plot_axis(mat, ax):
        """Helper function for setting up axes for plots.

        Args:
            mat (np.ndarray): the N-qubit matrix to plot.
            ax (plt.axes): Optional. Axes object to add plot to.

        Returns:
            plt.axes: the figure object and axes object.
        """
        dim = len(mat)
        num_qubits = int(np.log2(dim))
        bit_labels = [BaseExpvalMeasMitigator._int_to_bitstring(i, num_qubits) for i in range(dim)]

        ax.set_xticks(np.arange(dim))
        ax.set_yticks(np.arange(dim))
        ax.set_xticklabels(bit_labels, rotation=90)
        ax.set_yticklabels(bit_labels)
        ax.set_xlabel('Prepared State')
        ax.xaxis.set_label_position('top')
        ax.set_ylabel('Measured State')
        return ax
