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
import logging
from typing import List, Dict
import numpy as np
import scipy.linalg as la

from qiskit.exceptions import QiskitError
from .ctmp_mitigator import CTMPExpvalMeasMitigator
from .ctmp_generator_set import Generator, standard_generator_set
from .utils import assignment_matrix

logger = logging.getLogger(__name__)


def fit_ctmp_meas_mitigator(cal_data: Dict[int, Dict[int, int]],
                            num_qubits: int,
                            generators: List[Generator] = None) -> CTMPExpvalMeasMitigator:
    """Return FullMeasureErrorMitigator from result data.

    Args:
        cal_data: calibration dataset.
        num_qubits: the number of qubits for the calibation dataset.
        generators: Optional, input generator set.

    Returns:
        Measurement error mitigator object.

    Raises:
        QiskitError: if input arguments are invalid.
    """
    if not isinstance(num_qubits, int):
        raise QiskitError('Number of qubits must be an int')
    if generators is None:
        generators = standard_generator_set(num_qubits)

    gen_mat_dict = {}
    for gen in generators + _supplementary_generators(generators):
        if len(gen[2]) > 1:
            mat = _local_g_matrix(gen, cal_data, num_qubits)
            gen_mat_dict[gen] = mat

    # Compute rates for generators
    rates = [_get_ctmp_error_rate(gen, gen_mat_dict, num_qubits) for gen in generators]
    return CTMPExpvalMeasMitigator(generators, rates)


# Utility functions used for fitting (Should be moved to Fitter class)

def _ctmp_err_rate_1_q(a: str, b: str, j: int,
                       g_mat_dict: Dict[Generator, np.array],
                       num_qubits: int) -> float:
    """Compute the 1q error rate for a given generator."""
    # pylint: disable=invalid-name
    rate_list = []
    if a == '0' and b == '1':
        g1 = ('00', '10')
        g2 = ('01', '11')
    elif a == '1' and b == '0':
        g1 = ('10', '00')
        g2 = ('11', '01')
    else:
        raise ValueError('Invalid a,b encountered...')
    for k in range(num_qubits):
        if k == j:
            continue
        c = (j, k)
        for g_strs in [g1, g2]:
            gen = g_strs + (c,)
            r = _ctmp_err_rate_2_q(gen, g_mat_dict)
            rate_list.append(r)
    if len(rate_list) != 2 * (num_qubits - 1):
        raise ValueError('Rate list has wrong number of elements')
    rate = np.mean(rate_list)
    return rate


def _ctmp_err_rate_2_q(gen, g_mat_dict) -> float:
    """Compute the 2 qubit error rate for a given generator."""
    # pylint: disable=invalid-name
    g_mat = g_mat_dict[gen]
    b, a, _ = gen
    r = g_mat[int(b, 2), int(a, 2)]
    return r


def _get_ctmp_error_rate(gen: Generator,
                         g_mat_dict: Dict[Generator, np.array],
                         num_qubits: int) -> float:
    """Compute the error rate r_i for generator G_i.

    Args:
        gen (Generator): Generator to calibrate.
        g_mat_dict (Dict[Generator, np.array]): Dictionary of local G(j,k) matrices.
        num_qubits (int): number of qubits.

    Returns:
        float: The coefficient r_i for generator G_i.

    Raises:
        ValueError: The provided generator is not already in the set of generators.
    """
    # pylint: disable=invalid-name
    b, a, c = gen
    if len(b) == 1:
        rate = _ctmp_err_rate_1_q(a, b, c[0], g_mat_dict, num_qubits)
    elif len(b) == 2:
        rate = _ctmp_err_rate_2_q(gen, g_mat_dict)
    return rate


def _local_g_matrix(gen: Generator,
                    cal_data: Dict[int, Dict[int, int]],
                    num_qubits: int) -> np.array:
    """Computes the G(j,k) matrix in the basis [00, 01, 10, 11]."""
    # pylint: disable=invalid-name
    _, _, c = gen
    j, k = c
    a_mat = assignment_matrix(cal_data, num_qubits, [j, k])
    g = la.logm(a_mat)
    if np.linalg.norm(np.imag(g)) > 1e-3:
        raise QiskitError('Encountered complex entries in G_i={}'.format(g))
    g = np.real(g)
    for i in range(4):
        for j in range(4):
            if i != j:
                if g[i, j] < 0:
                    g[i, j] = 0
    return g


def _supplementary_generators(gen_list: List[Generator]) -> List[Generator]:
    """Supplementary generators needed to run 1q calibrations.

    Args:
        gen_list (List[Generator]): List of generators.

    Returns:
        List[Generator]: List of additional generators needed.
    """
    pairs = {tuple(gen[2]) for gen in gen_list}
    supp_gens = []
    for tup in pairs:
        supp_gens.append(('10', '00', tup))
        supp_gens.append(('00', '10', tup))
        supp_gens.append(('11', '01', tup))
        supp_gens.append(('01', '11', tup))
    return supp_gens
