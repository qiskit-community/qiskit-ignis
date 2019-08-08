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

from qiskit.exceptions import QiskitError
from qiskit.ignis.characterization.fitters import BaseFitter
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

    def __init__(self, discriminator: BaseFitter):
        """
        Args:
            discriminator (BaseFitter): a discriminator that maps level 1 data
            to level 2 data.
        """
        self.discriminator = discriminator

    def apply(self, raw_data: Result) -> Result:
        """
        Create a new result from the raw_data by converting level 1 data to
        level 2 data.

        Args:
            raw_data: list of qiskit.Result or qiskit.Result.
        Returns:
            A list of qiskit.Result or qiskit.Result.
        """
        if isinstance(raw_data, Result):
            new_results = deepcopy(raw_data)

            for result in new_results.results:

                if result.meas_level == 0:
                    raise QiskitError('Cannot discriminate level 0 data.')
                elif result.meas_level == 1:
                    x_data = self.discriminator.extract_xdata(result)
                    y_data = self.discriminator.fit_fun.predict(x_data)

                    result.meas_level = 2
                    result.data = ExperimentResultData(memory=y_data)
                elif result.meas_level == 2:
                    pass

        else:
            msg = 'raw_data is not of type qiskit.result.result.Result'
            raise QiskitError(msg)

        return new_results
