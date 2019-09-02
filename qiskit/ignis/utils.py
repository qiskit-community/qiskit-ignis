# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Functions of general purpose utility for Ignis.
"""
import random
from typing import List


def qubit_shot(i_mean, q_mean, i_std, q_std):
    """Creates a random IQ point using a Gaussian distribution.
    Args:
        i_mean: mean of I distribution
        q_mean: mean of Q distribution
        i_std: standard deviation of I distribution
        q_std: standard deviation of Q distribution

    Returns:
        a list of length 2 with I and Q values.
    """
    return [random.gauss(i_mean, i_std), random.gauss(q_mean, q_std)]


def create_shots(i_mean: float, q_mean: float, i_std: float, q_std: float,
                 shots: int, qubits: List[int]):
    """Creates random IQ points for qubits using a Gaussian distribution.
    Args:
        i_mean: mean of I distribution
        q_mean: mean of Q distribution
        i_std: standard deviation of I distribution
        q_std: standard deviation of Q distribution
        shots: the number of single shots
        qubits: a list of qubits.

    Returns:
        a list containing lists representing the IQ data of the qubits.
    """
    data = []
    for _ in range(shots):
        shot = []
        for _ in qubits:
            shot.append(qubit_shot(i_mean, q_mean, i_std, q_std))
        data.append(shot)

    return data
