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
from typing import Dict, Optional, List, Tuple

import numpy as np
import scipy.linalg as la
import scipy.sparse as sps

from qiskit.exceptions import QiskitError
from qiskit.ignis.numba import jit_fallback

from .base_meas_mitigator import BaseExpvalMeasMitigator
from .utils import counts_probability_vector
from .ctmp_generator_set import Generator

logger = logging.getLogger(__name__)


class CTMPExpvalMeasMitigator(BaseExpvalMeasMitigator):
    """N-qubit CTMP measurement error mitigator.

    This class can be used with the
    :func:`qiskit.ignis.mitigation.expectation_value` function to apply
    measurement error mitigation of N-qubit measurement errors caused by
    one and two-body error generators. Expectation values
    can also be computed directly using the :meth:`expectation_value` method.

    For measurement mitigation to be applied the mitigator should be
    calibrated using the
    :func:`qiskit.ignis.mitigation.expval_meas_mitigator_circuits` function
    and :class:`qiskit.ignis.mitigation.ExpvalMeasMitigatorFitter` class with
    the ``'CTMP'`` mitigation method.
    """
    def __init__(self,
                 generators: List[Generator],
                 rates: List[float],
                 num_qubits: Optional[int] = None,
                 seed: Optional = None):
        """Initialize a TensorMeasurementMitigator"""
        if num_qubits is None:
            self._num_qubits = 1 + max([max([max(gen[2]) for gen in generators])])
        else:
            self._num_qubits = num_qubits
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
        self._generator_mats = {}
        self._noise_strengths = {}
        self._sampling_mats = {}

        # RNG for CTMP sampling
        self._rng = None
        self.seed(seed)

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
        # Convert counts to probs
        probs, shots = counts_probability_vector(
            counts, clbits=clbits, qubits=qubits, return_shots=True)
        num_qubits = int(np.log2(probs.shape[0]))
        if qubits is None:
            qubits = list(range(num_qubits))

        # Ensure diagonal is a numpy vector so we can use fancy indexing
        if diagonal is None:
            diagonal = self._z_diagonal(2**len(qubits))
        diagonal = np.asarray(diagonal, dtype=probs.dtype)

        # Get arrays from CSC sparse matrix format
        # TODO: To use qubits kwarg we should compute the B-matrix and gamma
        # for the specified qubit subsystem
        gamma = self.noise_strength(qubits)
        bmat = self._sampling_matrix(qubits)
        values = bmat.data
        indices = np.asarray(bmat.indices, dtype=int)
        indptrs = np.asarray(bmat.indptr, dtype=int)

        # Set a minimum number of CTMP samples
        shots = sum(counts.values())
        min_delta = 0.05
        shots_delta = max(4 / (min_delta**2), shots)
        num_samples = int(np.ceil(shots_delta * np.exp(2 * gamma)))

        # Break total number of samples up into steps of a max number
        # of samples
        expval = 0
        batch_size = 50000
        samples_set = (num_samples // batch_size) * [batch_size] + [
            num_samples % batch_size]
        for sample_shots in samples_set:
            # Apply sampling
            samples, sample_signs = self._ctmp_inverse(
                sample_shots, probs, gamma, values, indices, indptrs, self._rng)

            # Compute expectation value
            expval += diagonal[samples[sample_signs == 0]].sum()
            expval -= diagonal[samples[sample_signs == 1]].sum()

        expval = (np.exp(2 * gamma) / num_samples) * expval

        # TODO: calculate exact standard deviation
        # For now we return the upper bound: stddev <= Gamma / sqrt(shots)
        # using Gamma ~ exp(2 * gamma)
        stddev = np.exp(2 * gamma) / np.sqrt(shots)
        return expval, stddev

    def generator_matrix(self, qubits: List[int] = None) -> sps.coo_matrix:
        r"""Return the generator matrix on the specified qubits.

        The generator matrix :math:`G` is given by :math:`\sum_i r_i G_i`
        where the sum is taken over all :math:`G_i` acting on the specified
        qubits subset.

        Args:
            qubits: Optional, qubit subset for the generators.

        Returns:
            sps.coo_matrix: the generator matrix :math:`G`.
        """
        if qubits is None:
            qubits = tuple(range(self._num_qubits))
        else:
            qubits = tuple(sorted(qubits))

        if qubits not in self._generator_mats:
            # Construct G from subset generators
            qubits_set = set(qubits)
            g_mat = sps.coo_matrix(2 * (2**len(qubits),), dtype=float)
            for gen, rate in zip(self._generators, self._rates):
                if qubits_set.issuperset(gen[2]):
                    # Keep generator
                    g_mat += rate * self._generator_to_coo_matrix(gen, qubits)

            # Add generator matrix to cache
            self._generator_mats[qubits] = sps.coo_matrix(g_mat)
        return self._generator_mats[qubits]

    def mitigation_matrix(self, qubits: List[int] = None) -> np.ndarray:
        r"""Return the measurement mitigation matrix for the specified qubits.

        The mitigation matrix :math:`A^{-1}` is defined as the inverse of the
        :meth:`assignment_matrix` :math:`A`.

        Args:
            qubits: Optional, qubits being measured for operator expval.

        Returns:
            np.ndarray: the measurement error mitigation matrix :math:`A^{-1}`.
        """
        # NOTE: the matrix definition of G is somehow flipped in both row and
        # columns compared to the canonical ordering for the A-matrix used
        # in the Complete and Tensored methods
        gmat = self.generator_matrix(qubits)
        gmat = np.flip(gmat.todense())
        return la.expm(-gmat)

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
        # NOTE: the matrix definition of G is somehow flipped in both row and
        # columns compared to the canonical ordering for the A-matrix used
        # in the Complete and Tensored methods
        gmat = self.generator_matrix(qubits)
        gmat = np.flip(gmat.todense())
        return la.expm(gmat)

    def noise_strength(self, qubits: Optional[int] = None) -> float:
        """Return the noise strength :math:`gamma` on the specified qubits"""
        if qubits is None:
            qubits = tuple(range(self._num_qubits))
        else:
            qubits = tuple(sorted(qubits))

        if qubits not in self._noise_strengths:
            # Compute gamma and cache
            g_mat = self.generator_matrix(qubits)
            # Check ideal case
            if g_mat.row.size == 0:
                gamma = 0
            else:
                gamma = np.max(-g_mat.data[g_mat.row == g_mat.col])
            if gamma < 0:
                raise QiskitError(
                    'gamma should be non-negative, found gamma={}'.format(gamma))
            self._noise_strengths[qubits] = gamma
        return self._noise_strengths[qubits]

    def seed(self, value=None):
        """Set the seed for the quantum state RNG."""
        if isinstance(value, np.random.Generator):
            self._rng = value
        else:
            self._rng = np.random.default_rng(value)

    def _compute_gamma(self, qubits=None):
        """Compute gamma for N-qubit mitigation"""
        if qubits is not None:
            raise NotImplementedError(
                "qubits kwarg is not yet implemented for CTMP method.")
        gamma = self.noise_strength(qubits)
        return np.exp(2 * gamma)

    @staticmethod
    def _ctmp_inverse(
            n_samples: int,
            probs: np.ndarray,
            gamma: float,
            csc_data: np.ndarray,
            csc_indices: np.ndarray,
            csc_indptrs: np.ndarray,
            rng: np.random.Generator) -> Tuple[Tuple[int], Tuple[int]]:
        """Apply CTMP algorithm to input counts dictionary, return
        sampled counts and associated shots. Equivalent to Algorithm 1 in
        Bravyi et al.

        Args:
            n_samples: Number of times to sample in CTMP algorithm.
            probs: probability vector constructed from counts.
            gamma: noise strength parameter
            csc_data: Sparse CSC matrix data array (`csc_matrix.data`).
            csc_indices: Sparse CSC matrix indices array (`csc_matrix.indices`).
            csc_indptrs: Sparse CSC matrix indices array (`csc_matrix.indptrs`).
            rng: RNG generator object.

        Returns:
            Tuple[Tuple[int], Tuple[int]]: Resulting list of shots and associated
            signs from the CTMP algorithm.
        """
        alphas = rng.poisson(lam=gamma, size=n_samples)
        signs = np.mod(alphas, 2)
        x_vals = rng.choice(len(probs), size=n_samples, p=probs)

        # Apply CTMP sampling
        r_vals = rng.random(size=alphas.sum())
        y_vals = np.zeros(x_vals.size, dtype=int)
        _markov_chain_compiled(y_vals, x_vals, r_vals, alphas, csc_data, csc_indices,
                               csc_indptrs)

        return y_vals, signs

    def _sampling_matrix(self, qubits: Optional[int] = None) -> sps.csc_matrix:
        """Compute the B matrix for CTMP sampling"""
        if qubits is None:
            qubits = tuple(range(self._num_qubits))
        else:
            qubits = tuple(sorted(qubits))

        if qubits not in self._sampling_mats:
            # Compute matrix and cache
            gmat = self.generator_matrix(qubits)
            gamma = self.noise_strength(qubits)
            bmat = sps.eye(2**len(qubits))
            if gamma != 0:
                bmat = bmat + gmat / gamma
            self._sampling_mats[qubits] = bmat.tocsc()

        return self._sampling_mats[qubits]

    @staticmethod
    def _tensor_list(parts: List[np.ndarray]) -> np.ndarray:
        """Compute sparse tensor product of all matrices in list"""
        res = parts[0]
        for mat in parts[1:]:
            res = sps.kron(res, mat)
        return res

    def _generator_to_coo_matrix(self, gen: Generator, qubits: Tuple[int]) -> sps.coo_matrix:
        """Compute the matrix form of a generator.

        Generators are uniquely determined by two bitstrings,
        and a list of qubits on which the bitstrings act. For instance,
        the generator `|b^i><a^i| - |a^i><a^i|` acting on the (ordered)
        set `C_i` is represented by `g = (b_i, a_i, qubits)`

        Args:
            gen: Generator to use.
            qubits: Qubit subset for generator matrix

        Returns:
            sps.coo_matrix: Sparse representation of generator.
        """
        ket_bra_dict = {
            '00': np.array([[1, 0], [0, 0]]),
            '01': np.array([[0, 1], [0, 0]]),
            '10': np.array([[0, 0], [1, 0]]),
            '11': np.array([[0, 0], [0, 1]])
        }
        s_b, s_a, gen_qubits = gen
        num_qubits = len(qubits)
        # pylint: disable=unnecessary-lambda
        ba_strings = list(map(lambda x: ''.join(x), list(zip(*[s_b, s_a]))))
        aa_strings = list(map(lambda x: ''.join(x), list(zip(*[s_a, s_a]))))
        ba_mats = [sps.eye(2, 2).tocoo()] * num_qubits
        aa_mats = [sps.eye(2, 2).tocoo()] * num_qubits
        for qubit, s_ba, s_aa in zip(gen_qubits, ba_strings, aa_strings):
            idx = qubits.index(qubit)
            ba_mats[idx] = ket_bra_dict[s_ba]
            aa_mats[idx] = ket_bra_dict[s_aa]

        res = sps.coo_matrix(2 * (2**num_qubits, ), dtype=float)
        res = res + self._tensor_list(ba_mats[::-1]) - self._tensor_list(aa_mats[::-1])
        return res


@jit_fallback
def _choice(inds: np.ndarray, probs: np.ndarray, r_val: float) -> int:
    """Choise a random array element from specified distribution.

    Given a list and associated probabilities for each element of the list,
    return a random choice. This function is required since Numpy
    random.choice cannot be compiled with Numba.

    Args:
        inds (List[int]): List of indices to choose from.
        probs (List[float]): List of probabilities for indices.
        r_val: float: pre-generated random number in [0, 1).

    Returns:
        int: Randomly sampled index from list.
    """
    probs = np.cumsum(probs)
    n_probs = len(probs)
    for i in range(n_probs):
        if r_val < probs[i]:
            return inds[i]
    return inds[-1]


@jit_fallback
def _markov_chain_compiled(y_vals: np.ndarray, x_vals: np.ndarray,
                           r_vals: np.ndarray, alpha_vals: np.ndarray,
                           csc_vals: np.ndarray, csc_indices: np.ndarray,
                           csc_indptrs: np.ndarray):
    """Simulate the Markov process for a CSC transition matrix.

    Args:
        y_vals: array to store sampled values.
        x_vals: array of initial state values.
        r_vals: pre-generated array of random numbers [0, 1) to use in sampling.
        alpha_vals: array of Markov step values for sampling.
        csc_vals: Sparse CSC matrix data array (`csc_matrix.data`).
        csc_indices: Sparse CSC matrix indices array (`csc_matrix.indices`).
        csc_indptrs: Sparse CSC matrix indices array (`csc_matrix.indptrs`).
    """
    num_samples = y_vals.size
    r_pos = 0
    for i in range(num_samples):
        y = x_vals[i]
        for _ in range(alpha_vals[i]):
            begin_slice = csc_indptrs[y]
            end_slice = csc_indptrs[y + 1]
            probs = csc_vals[begin_slice:end_slice]
            inds = csc_indices[begin_slice:end_slice]
            y = _choice(inds, probs, r_vals[r_pos])
            r_pos += 1
        y_vals[i] = y
