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
CTMP expectation value measurement error mitigator.
"""
import logging
from typing import Dict, Optional, List, Tuple, Union
from collections import Counter

import numpy as np
import scipy.sparse as sps

from qiskit.exceptions import QiskitError
from qiskit.ignis.verification.tomography import marginal_counts

from ..base_meas_mitigator import BaseMeasMitigator
from .ctmp_generator_set import Generator
from .markov_compiled import markov_chain_int

logger = logging.getLogger(__name__)


class CTMPMeasMitigator(BaseMeasMitigator):
    """Measurement error mitigator via full N-qubit mitigation."""

    def __init__(self, generators: List[Generator],
                 rates: List[float],
                 num_qubits: Optional[int] = None):
        """Initialize a TensorMeasurementMitigator"""
        if num_qubits is None:
            self._num_qubits = 1 + max([max([max(gen[2]) for gen in generators])])
        # Filter non-zero rates for generator
        nz_rates = []
        nz_generators = []
        threshold = 1e-5
        for rate, gen in zip(rates, generators):
            if rate > threshold:
                nz_rates.append(rate)
                nz_generators.append(gen)
        self._generators = nz_generators
        self._rates = np.array(nz_rates, dtype=float)
        # Parameters to be initialized from generators and rates
        self._g_mat = None
        self._b_mat = None
        self._gamma = None
        self._compute_b_mat()  # Also computes gamma and g_mat

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

        Raises:
            NotImplementedError: if qubits or diagonal kwargs are used.

        Additional Information:
            The observable :math:`O` is input using the ``diagonal`` kwarg as a
            Numpy array :math:`[O(0), ..., O(2^n -1)]`. If no diagonal is specified
            the Pauli-Z operator `O = Z^{\otimes n}` is used.

            The ``clbits`` kwarg is used to marginalize the input counts dictionary
            over the specified bit-values, and the ``qubits`` kwarg is used to specify
            which physical qubits these bit-values correspond to as
            ``circuit.measure(qubits, clbits)``.
        """
        if qubits is not None or diagonal is not None:
            raise NotImplementedError(
                "qubits kwarg is not yet implemented for CTMP method.")
        if clbits is not None:
            counts = marginal_counts(counts, meas_qubits=clbits)
        plus_dict, minus_dict = self._ctmp_a_inverse(counts)

        norm_c1 = np.exp(2 * self._gamma)

        shots = np.sum(list(plus_dict.values())) + np.sum(
            list(minus_dict.values()))

        exp_vals = []
        if len(plus_dict) > 0:
            for key, val in plus_dict.items():
                exp_vals.append(+(-1)**(key.count('1')) * val)
        if len(minus_dict) > 0:
            for key, val in minus_dict.items():
                exp_vals.append(-(-1)**(key.count('1')) * val)

        exp_vals = np.array(exp_vals) * norm_c1 / shots * len(exp_vals)
        mean = np.mean(exp_vals)
        return mean

    def _compute_gamma(self, qubits=None):
        """Compute gamma for N-qubit mitigation"""
        if qubits is not None:
            raise NotImplementedError(
                "qubits kwarg is not yet implemented for CTMP method.")
        return np.exp(2 * self._gamma)

    def _ctmp_a_inverse(self,
                        counts: Dict[str, int],
                        return_bitstrings: bool = False,
                        min_delta: float = 0.05) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Apply CTMP algorithm to input counts dictionary, return
        sampled counts and associated shots. Equivalent to Algorithm 1 in
        Bravyi et al.

        This is a wrapper for `core_ctmp_a_inverse`.

        Args:
            counts: Counts dictionary to mitigate.
            return_bitstrings: If `True`, return
                the data as `(shots, signs)`, so similar to `core_ctmp_a_inverse`.
            min_delta: Optional, min error tolerance for sampling.

        Returns:
            Tuple[Dict[str, int], Dict[str, int]]: Output dictionaries corresponding
            to sampled values associated with +1 and -1 results. When re-combining
            to take expectation values of an operator, these dicts correspond to
            `<O>_{plus_dict} - <O>_{minus_dict}`.
        """
        # Set a minimum number of CTMP samples
        shots_delta = max(4 / (min_delta ** 2), np.sum(list(counts.values())))
        shots = int(np.ceil(shots_delta * np.exp(2 * self._gamma)))

        shots, signs = self._core_ctmp_a_inverse(counts, shots)
        signs = [-1 if s == 1 else +1 for s in signs]
        shots = [
            np.binary_repr(shot, width=self._num_qubits) for shot in shots
        ]
        if return_bitstrings:
            return shots, signs

        plus_dict = dict(
            Counter(
                map(lambda x: x[0],
                    filter(lambda x: x[1] == +1, zip(shots, signs)))))
        minus_dict = dict(
            Counter(
                map(lambda x: x[0],
                    filter(lambda x: x[1] == -1, zip(shots, signs)))))

        return plus_dict, minus_dict

    @staticmethod
    def _sample_shot(counts: Dict[Union[int, str], int], num_samples: int):
        """Re-sample a counts dictionary.

        Args:
            counts: Original counts dictionary.
            num_samples: Number of times to sample.

        Returns:
            np.array: Re-sampled counts dictionary.
        """
        shots = np.sum(list(counts.values()))
        probs = np.array(list(counts.values())) / shots
        bits = np.array(list(counts.keys()))
        return np.random.choice(bits, size=num_samples, p=probs)

    def _core_ctmp_a_inverse(self, counts: Dict[str, int],
                             n_samples: int) -> Tuple[Tuple[int], Tuple[int]]:
        """Apply CTMP algorithm to input counts dictionary, return
        sampled counts and associated shots. Equivalent to Algorithm 1 in
        Bravyi et al.

        Args:
            counts (Dict[str, int]): Counts dictionary to mitigate.
            n_samples (int): Number of times to sample in CTMP algorithm.

        Returns:
            Tuple[Tuple[int], Tuple[int]]: Resulting list of shots and associated
            signs from the CTMP algorithm.
        """
        counts_ints = {int(bits, 2): freq for bits, freq in counts.items()}
        times = np.random.poisson(lam=self._gamma, size=n_samples)
        signs = np.mod(times, 2)
        x_vals = self._sample_shot(counts_ints, num_samples=n_samples)
        y_vals = markov_chain_int(trans_mat=self._b_mat, x=x_vals, alpha=times)
        return y_vals, signs

    def _compute_g_mat(self):
        """Compute the total `G` matrix given the coefficients `r_i` and the generators `G_i`.
        """
        g_mat = sps.coo_matrix((2**self._num_qubits, 2**self._num_qubits),
                               dtype=np.float)

        logger.info('Computing sparse G matrix')
        for gen, rate in zip(self._generators, self._rates):
            g_mat += rate * self._generator_to_coo_matrix(gen)
        num_elts = g_mat.shape[0]**2
        try:
            nnz = g_mat.nnz
            if nnz == num_elts:
                sparsity = '+inf'
            else:
                sparsity = nnz / (num_elts - nnz)
            logger.info('Computed sparse G matrix with sparsity %d', sparsity)
        except AttributeError:
            pass
        self._g_mat = g_mat

    def _compute_little_gamma(self):
        """Compute the factor `gamma` for a given generator matrix."""
        if self._g_mat is None:
            self._compute_g_mat()
        g_mat = -self._g_mat.tocoo()
        gamma = np.max(g_mat.data[g_mat.row == g_mat.col])
        if gamma < 0:
            raise QiskitError(
                'gamma should be non-negative, found gamma={}'.format(gamma))
        self._gamma = gamma

    def _compute_b_mat(self):
        """Compute B matrix"""
        if self._g_mat is None:
            self._compute_g_mat()
        if self._gamma is None:
            self._compute_little_gamma()
        b_mat = sps.eye(2**self._num_qubits) + self._g_mat / self._gamma
        self._b_mat = b_mat.tocsc()

    @staticmethod
    def _tensor_list(parts: List[np.ndarray]) -> np.ndarray:
        """Compute sparse tensor product of all matrices in list"""
        res = parts[0]
        for mat in parts[1:]:
            res = sps.kron(res, mat)
        return res

    def _generator_to_coo_matrix(self, gen: Generator) -> sps.coo_matrix:
        """Compute the matrix form of a generator.

        Generators are uniquely determined by two bitstrings,
        and a list of qubits on which the bitstrings act. For instance,
        the generator `|b^i><a^i| - |a^i><a^i|` acting on the (ordered)
        set `C_i` is represented by `g = (b_i, a_i, qubits)`

        Args:
            gen: Generator to use.

        Returns:
            sps.coo_matrix: Sparse representation of generator.
        """
        ket_bra_dict = {
            '00': np.array([[1, 0], [0, 0]]),
            '01': np.array([[0, 1], [0, 0]]),
            '10': np.array([[0, 0], [1, 0]]),
            '11': np.array([[0, 0], [0, 1]])
        }
        s_b, s_a, qubits = gen
        shape = (2**self._num_qubits, ) * 2
        res = sps.coo_matrix(shape, dtype=float)
        # pylint: disable=unnecessary-lambda
        ba_strings = list(map(lambda x: ''.join(x), list(zip(*[s_b, s_a]))))
        aa_strings = list(map(lambda x: ''.join(x), list(zip(*[s_a, s_a]))))
        ba_mats = [sps.eye(2, 2).tocoo()] * self._num_qubits
        aa_mats = [sps.eye(2, 2).tocoo()] * self._num_qubits
        for qubit, s_ba, s_aa in zip(qubits, ba_strings, aa_strings):
            ba_mats[qubit] = ket_bra_dict[s_ba]
            aa_mats[qubit] = ket_bra_dict[s_aa]
        res += self._tensor_list(ba_mats[::-1])
        res -= self._tensor_list(aa_mats[::-1])
        return res
