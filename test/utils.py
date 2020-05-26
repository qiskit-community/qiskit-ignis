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

"""Functions of general purpose utility for Ignis."""
import random
from typing import List
import json
from qiskit.result.result import Result


def qubit_shot(i_mean: float, q_mean: float, i_std: float, q_std: float):
    """Creates a random IQ point using a Gaussian distribution.

    Args:
        i_mean: mean of I distribution
        q_mean: mean of Q distribution
        i_std: standard deviation of I distribution
        q_std: standard deviation of Q distribution

    Returns:
        list: a list of length 2 with I and Q values.
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
        list: a list containing lists representing the IQ data of the qubits.
    """
    data = []
    for _ in range(shots):
        shot = []
        for _ in qubits:
            shot.append(qubit_shot(i_mean, q_mean, i_std, q_std))
        data.append(shot)

    return data


def save_results_as_json(results_list: List[Result], json_path: str):
    """
    saves the list of Results in json format at the given path
    Args:
        results_list: list of run results
        json_path: the path to save the json file
    """
    results_json = [result.to_dict() for result in results_list]
    with open(json_path, "w") as results_file:
        json.dump(results_json, results_file)


def load_results_from_json(json_path: str):
    """
    loads run results from json file
    Args:
        json_path: the path of the json file to load the results from

    Returns:
        list: results object that was saved in the json file (list of qiskit Results)
    """
    with open(json_path, "r") as results_file:
        results_json = json.load(results_file)
    return [Result.from_dict(result) for result in results_json]
