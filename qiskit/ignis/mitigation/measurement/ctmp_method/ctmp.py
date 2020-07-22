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
"""Apply CTMP error mitigation.
"""

from collections import Counter
from itertools import combinations
from typing import List, Tuple, Dict, Union

import numpy as np
from scipy import sparse
from qiskit.ignis.verification.tomography import marginal_counts

from .calibration import MeasurementCalibrator
from .markov_compiled import markov_chain_int

Generator = Tuple[str, str, List[int]]


def sample_shot(counts_dict: Dict[Union[int, str], int], num_samples: int):
    """Re-sample a counts dictionary.

    Args:
        counts_dict (Dict[Union[int, str], int]): Original counts dictionary.
        num_samples (int): Number of times to sample.

    Returns:
        np.array: Re-sampled counts dictionary.
    """
    shots = np.sum(list(counts_dict.values()))
    probs = np.array(list(counts_dict.values())) / shots
    bits = np.array(list(counts_dict.keys()))
    return np.random.choice(bits, size=num_samples, p=probs)


def core_ctmp_a_inverse(  # pylint: disable=invalid-name
        B: sparse.csc_matrix,
        gamma: float,
        counts_dict: Dict[str, int],
        n_samples: int,
) -> Tuple[Tuple[int], Tuple[int]]:
    """Apply CTMP algorithm to input counts dictionary, return
    sampled counts and associated shots. Equivalent to Algorithm 1 in
    Bravyi et al.

    Args:
        B (sparse.coo_matrix): Stochastic matrix `B = Id + G / gamma`.
        gamma (float): Calibrated parameter `gamma`, see Bravyi et al.
        counts_dict (Dict[str, int]): Counts dictionary to mitigate.
        n_samples (int): Number of times to sample in CTMP algorithm.

    Returns:
        Tuple[Tuple[int], Tuple[int]]: Resulting list of shots and associated
        signs from the CTMP algorithm.
    """
    counts_ints = {int(bits, 2): counts for bits, counts in counts_dict.items()}
    times = np.random.poisson(lam=gamma, size=n_samples)
    signs = np.mod(times, 2)
    x_vals = sample_shot(counts_ints, num_samples=n_samples)
    y_vals = markov_chain_int(trans_mat=B, x=x_vals, alpha=times)
    return y_vals, signs


def ctmp_a_inverse(
        calibrator: MeasurementCalibrator,
        counts_dict: Dict[str, int],
        return_bitstrings: bool = False
):
    """Apply CTMP algorithm to input counts dictionary, return
    sampled counts and associated shots. Equivalent to Algorithm 1 in
    Bravyi et al.

    This is a wrapper for `core_ctmp_a_inverse`.

    Args:
        calibrator (MeasurementCalibrator): Calibrator to use for algorithm.
        counts_dict (Dict[str, int]): Counts dictionary to mitigate.
        return_bitstrings (bool, optional): [description]. If `True`, return
            the data as `(shots, signs)`, so similar to `core_ctmp_a_inverse`.

    Returns:
        Tuple[Dict[str, int], Dict[str, int]]: Output dictionaries corresponding
        to sampled values associated with +1 and -1 results. When re-combining
        to take expectation values of an operator, these dicts correspond to
        `<O>_{plus_dict} - <O>_{minus_dict}`.
    """
    gen_set = calibrator.gen_set
    gamma = calibrator.gamma

    n = gen_set.num_qubits
    shots = np.sum(list(counts_dict.values())) * np.exp(2 * gamma)
    shots = int(np.ceil(shots))

    shots, signs = core_ctmp_a_inverse(
        B=calibrator.B,
        gamma=gamma,
        counts_dict=counts_dict,
        n_samples=shots
    )
    signs = [-1 if s == 1 else +1 for s in signs]
    shots = [np.binary_repr(shot, width=n) for shot in shots]
    if return_bitstrings:
        return shots, signs

    plus_dict = dict(Counter(map(lambda x: x[0], filter(lambda x: x[1] == +1, zip(shots, signs)))))
    minus_dict = dict(Counter(map(lambda x: x[0], filter(lambda x: x[1] == -1, zip(shots, signs)))))

    return plus_dict, minus_dict


def mitigated_expectation_value(cal: MeasurementCalibrator, counts_dict: Dict[str, int],
                                subset: List[int] = None,
                                mean_only: bool = True) -> float:
    """Given a counts dictionary corresponding to measuring an operator in the Pauli basis,
    apply the CTMP error mitigation algorithm to the result, and return the mitigated

    Args:
        cal (MeasurementCalibrator): Calibrator to use for mitigation.
        counts_dict (Dict[str, int]): Counts dictionary to mitigate.
        subset (List[int], optional): Subset of qubits to measure the expectation
        value with respec to. Defaults to None, which means use all qubits.
        mean_only (bool, optional): If True, return only the expected value,
            if false also return the variance. Currently variance is not working.
            Defaults to True.

    Returns:
        float: Mitigated expectation value.

    Raises:
        NotImplementedError: Variance is not yet implemented.
    """
    plus_dict, minus_dict = ctmp_a_inverse(
        calibrator=cal,
        counts_dict=counts_dict,
        return_bitstrings=False
    )  # type: Tuple[Dict[str, int], Dict[str, int]]

    norm_c1 = np.exp(2 * cal.gamma)
    numq = cal._num_qubits

    shots = np.sum(list(plus_dict.values())) + np.sum(list(minus_dict.values()))

    if subset is None:
        subset = list(range(numq))

    exp_vals = []
    if len(plus_dict) > 0:
        for key, val in marginal_counts(plus_dict, subset, pad_zeros=True).items():
            exp_vals.append(+(-1) ** (key.count('1')) * val)
    if len(minus_dict) > 0:
        for key, val in marginal_counts(minus_dict, subset, pad_zeros=True).items():
            exp_vals.append(-(-1) ** (key.count('1')) * val)

    exp_vals = np.array(exp_vals) * norm_c1 / shots * len(exp_vals)
    mean = np.mean(exp_vals)

    if mean_only:
        return mean
    else:
        raise NotImplementedError('Variance is not implemented')


def mitigated_expectation_values(
        cal: MeasurementCalibrator,
        counts_dict: Dict[str, int]
) -> Dict[str, float]:
    """Apply mitigation to counts dictionary with all combinations of qubits being measured.
    Mimics  `expectation_counts` in Ignis, but with mitigated expectation values.

    Args:
        cal (MeasurementCalibrator): Calibrator to use for mitigation.
        counts_dict (Dict[str, int]): Counts dictionary to mitigate.

    Returns:
        Dict[str, float]: Dictionary with keys corresponding to qubits being measured and
        values corresponding to the resulting mitigated expectation value.
    """
    numq = cal._num_qubits
    subsets = []
    for r in range(numq):
        subsets += list(combinations(range(numq), r + 1))

    res = {}

    for subset in subsets:
        exp_op = ['0'] * numq
        for qubit in subset:
            exp_op[qubit] = '1'
        exp_op_str = ''.join(exp_op)

        exp_val = mitigated_expectation_value(cal, counts_dict, subset=subset)
        res[exp_op_str] = exp_val

    return res
