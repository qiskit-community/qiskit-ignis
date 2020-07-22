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
"""Simulate Markov processes.
"""

import logging
import random
from typing import List, Union

import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)

try:
    import numba

    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False
    logger.info('Numba not installed, for faster mitigation install numba'
                'https://pypi.org/project/numba/')


def jit_fallback(func):
    """Decorator to try to apply numba JIT compilation.
    """
    if USE_NUMBA:
        return numba.jit(nopython=True)(func)
    else:
        return func


@jit_fallback
def choice(inds, probs):
    """Given a list and associated probabilities for
    each element of the list, return a random choice.
    This function is used in favor of other standard
    options since I could not find one that could be
    compiled with Numba.

    Args:
        inds (List[int]): List of indices to choose from.
        probs (List[float]): List of probabilities for indices.

    Returns:
        int: Randomly sampled index from list.
    """
    probs = np.cumsum(probs)
    x = random.random()
    n_probs = len(probs)
    for i in range(n_probs):
        if x < probs[i]:
            return inds[i]
    return inds[-1]


@jit_fallback
def csc_col_slice(col: int, vals, indices, indptrs):
    """Given the appropriate data for a CSC formatted sparse
    matrix, return the slice `M[:,col]`.

    Args:
        col (int): Column to slice.
        vals (List[float]): Equal to `csc_matrix.data`.
        indices (List[int]): Equal to `csc_matrix.indices`.
        indptrs (List[int]): Equal to `csc_matrix.indptrs`.

    Returns:
        List[float]: The vector corresponding to the slice `M[:,col]`.
    """
    begin_slice = indptrs[int(col)]
    end_slice = indptrs[int(col) + 1]
    vals_slice = vals[begin_slice:end_slice]
    inds_slice = indices[begin_slice:end_slice]
    return inds_slice, vals_slice


@jit_fallback
def _markov_chain_int(x: int, alpha: int, vals, indices, indptrs) -> int:
    """Apply multiple steps of the Markov chain.

    Args:
        x (int): Initial state for the chain.
        alpha (int): Number of steps to take.
        vals (List[float]): Equal to `csc_matrix.data`.
        indices (List[int]): Equal to `csc_matrix.indices`.
        indptrs (List[int]): Equal to `csc_matrix.indptrs`.

    Returns:
        int: The state after `alpha` steps of the Markov process.
    """
    y = x
    for _ in range(alpha):
        inds, probs = csc_col_slice(y, vals=vals, indices=indices, indptrs=indptrs)
        y = choice(inds, probs)
    return y


@jit_fallback
def _multi_markov_chain_int(x_list, alpha_list, vals, indices, indptrs):
    """Simulate the markov process for a list of input states. This is
    effectively just a loop over `_markov_chain_int` to avoid Python's loops.

    Args:
        x_list (List[int]): List of initial states.
        alpha_list (List[int]): List of number of steps.
        vals (List[float]): Equal to `csc_matrix.data`.
        indices (List[int]): Equal to `csc_matrix.indices`.
        indptrs (List[int]): Equal to `csc_matrix.indptrs`.

    Returns:
        List[int]: A list of states after simulated Markov processes.
    """
    res = []
    num_samples = len(x_list)
    for i in range(num_samples):
        y = _markov_chain_int(
            x=x_list[i],
            alpha=alpha_list[i],
            vals=vals,
            indices=indices,
            indptrs=indptrs
        )
        res.append(y)
    return res


def markov_chain_int(trans_mat: sparse.csc_matrix, x: Union[int, List[int]],
                     alpha: Union[int, List[int]]) -> int:
    """Apply simulate the Markov process for the transition matrix `trans_mat`.

    Args:
        trans_mat (sparse.csc_matrix): Transition matrix for Markov process.
        x (Union[int, List[int]]): List of initial states to use, or a single initial state.
        alpha (Union[int, List[int]]): List of steps to take for each initial state,
            or a single number of steps.

    Returns:
        Union[int, List[int]]: List of final states, or single final state.
    """
    if isinstance(alpha, int):
        if alpha == 0:
            return x
    if not isinstance(trans_mat, sparse.csc_matrix):
        t_mat = sparse.csc_matrix(trans_mat)
    else:
        t_mat = trans_mat
    values = t_mat.data
    indices = np.array(t_mat.indices, dtype=int)
    indptrs = np.array(t_mat.indptr, dtype=int)

    if isinstance(x, int):
        res = _markov_chain_int(
            x=x,
            alpha=alpha,
            vals=values,
            indices=indices,
            indptrs=indptrs
        )
    else:
        res = _multi_markov_chain_int(
            x_list=np.array(x),
            alpha_list=np.array(alpha),
            vals=values,
            indices=indices,
            indptrs=indptrs
        )
    return res
