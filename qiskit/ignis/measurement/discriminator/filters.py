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

# pylint: disable=cell-var-from-loop


"""
Discrimination filters.
"""
from copy import deepcopy
from typing import List

from qiskit.exceptions import QiskitError
from qiskit.ignis.measurement.discriminator.discriminators import \
    BaseDiscriminationFitter
from qiskit.result.result import Result
from qiskit.result.models import ExperimentResultData


class DiscriminationFilter:
    """
    Implements a filter based on a discriminator that takes level 1 data to
    level 2 data.

    Usage:
        my_filter = DiscriminationFilter(my_discriminator)
        new_result = filter.apply(level_1_data)
    """

    def __init__(self, discriminator: BaseDiscriminationFitter,
                 base: int = None):
        """
        Args:
            discriminator (BaseDiscriminationFitter): a discriminator that maps level 1
                data to level 2 data.
                - Level 1 data may correspond to, e. g., IQ data.
                - Level 2 data is the state counts.
            base: the base of the expected states. If it is not given the base
                is inferred from the expected_state instance of discriminator.
        """
        self.discriminator = discriminator

        if base:
            self.base = base
        else:
            self.base = DiscriminationFilter.get_base(
                discriminator.expected_states)

    def apply(self, raw_data: Result) -> Result:
        """
        Create a new result from the raw_data by converting level 1 data to
        level 2 data.

        Args:
            raw_data: list of qiskit.Result or qiskit.Result.

        Returns:
            A list of qiskit.Result or qiskit.Result.
        """
        new_results = deepcopy(raw_data)

        to_be_discriminated = []

        # Extract all the meas level 1 data from the Result.
        shots_per_experiment_result = []
        for result in new_results.results:
            if result.meas_level == 1:
                shots_per_experiment_result.append(result.shots)
                to_be_discriminated.append(result)

        new_results.results = to_be_discriminated

        x_data = self.discriminator.get_xdata(new_results, 2)
        y_data = self.discriminator.discriminate(x_data)

        start = 0
        for idx, n_shots in enumerate(shots_per_experiment_result):
            memory = y_data[start:(start+n_shots)]
            counts = self.count(memory)
            new_results.results[idx].data = ExperimentResultData(counts=counts,
                                                                 memory=memory)
            start += n_shots

        for result in new_results.results:
            result.meas_level = 2

        return new_results

    @staticmethod
    def get_base(expected_states: dict):
        """
        Returns the base inferred from expected_states.

        The intent is to allow users to discriminate states higher than 0/1.

        DiscriminationFilter infers the basis from the expected states to allow
        users to discriminate states outside of the computational sub-space.
        For example, if the discriminated states are 00, 01, 02, 10, 11, ...,
        22 the basis will be 3.

        With this implementation the basis can be at most 10.

        Args:
            expected_states:

        Returns:
            int: the base inferred from the expected states

        Raises:
            QiskitError: if there is an invalid input in the expected states
        """
        base = 0
        for key in expected_states:
            for char in expected_states[key]:
                try:
                    value = int(char)
                except ValueError:
                    raise QiskitError('Cannot parse character in ' +
                                      expected_states[key])

                base = base if base > value else value

        return base+1

    def count(self, y_data: List[str]) -> dict:
        """
        Converts discriminated results into raw counts.

        Args:
            y_data: result of a discrimination.

        Returns:
            A dict of raw counts.
        """
        raw_counts = {}

        for cnt in y_data:
            cnt_hex = hex(int(cnt, self.base))
            if cnt_hex in raw_counts:
                raw_counts[cnt_hex] += 1
            else:
                raw_counts[cnt_hex] = 1

        return raw_counts
