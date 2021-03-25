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
"""Perform CTMP calibration for error mitigation.
"""

from itertools import combinations
from typing import List, Tuple

Generator = Tuple[str, str, Tuple[int]]


def standard_generator_set(num_qubits: int, pairs=None) -> List[Generator]:
    """Construct this generator set given a number of qubits.

    Set of generators on 1 and 2 qubits. Corresponds to the following readout errors:
    `0 -> 1`
    `1 -> 0`
    `01 -> 10`
    `11 -> 00`
    `00 -> 11`

    Args:
        num_qubits (int): Number of qubits to calibrate.
        pairs (Optional[List[Tuple[int, int]]]): The pairs of qubits to compute generators for.
            If None, then use all pairs. Defaults to None.

    Returns:
        StandardGeneratorSet: The generator set.

    Raises:
        ValueError: Either the number of qubits is negative or not an int.
        ValueError: The number of generators does not match what it should be by counting.
    """
    if not isinstance(num_qubits, int):
        raise ValueError('num_qubits needs to be an int')
    if num_qubits <= 0:
        raise ValueError('Need num_qubits at least 1')
    generators = []
    generators += single_qubit_bitstrings(num_qubits)
    if num_qubits > 1:
        generators += two_qubit_bitstrings_symmetric(num_qubits, pairs=pairs)
        generators += two_qubit_bitstrings_asymmetric(num_qubits, pairs=pairs)
        if len(generators) != 2 * num_qubits ** 2:
            raise ValueError('Incorrect length of generators. {} != 2n^2.'.format(len(generators)))
    return generators


def single_qubit_bitstrings(num_qubits: int) -> List[Generator]:
    """Returns a list of tuples `[(C_1, b_1, a_1), (C_2, b_2, a_2), ...]` that represent
    the generators .
    """
    res = [('1', '0', (i,)) for i in range(num_qubits)]
    res += [('0', '1', (i,)) for i in range(num_qubits)]
    if len(res) != 2 * num_qubits:
        raise ValueError('Should have gotten 2n qubits, got {}'.format(len(res)))
    return res


def two_qubit_bitstrings_symmetric(num_qubits: int, pairs=None) -> List[Generator]:
    """Return the 11->00 and 00->11 generators on a given number of qubits.
    """
    if pairs is None:
        pairs = list(combinations(range(num_qubits), r=2))
    res = [('11', '00', (i, j)) for i, j in pairs if i < j]
    res += [('00', '11', (i, j)) for i, j in pairs if i < j]
    if len(res) != num_qubits * (num_qubits - 1):
        raise ValueError('Should have gotten n(n-1) qubits, got {}'.format(len(res)))
    return res


def two_qubit_bitstrings_asymmetric(num_qubits: int, pairs=None) -> List[Generator]:
    """Return the 01->10 generators on a given number of qubits.
    """
    if pairs is None:
        pairs = list(combinations(range(num_qubits), r=2))
    res = [('10', '01', (i, j)) for i, j in pairs]
    res += [('10', '01', (j, i)) for i, j in pairs]
    if len(res) != num_qubits * (num_qubits - 1):
        raise ValueError('Should have gotten n(n-1) qubits, got {}'.format(len(res)))
    return res
