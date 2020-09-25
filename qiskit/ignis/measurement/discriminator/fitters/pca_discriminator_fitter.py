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

# pylint: disable=invalid-name

"""
IQ Discriminator module to discriminate date in the IQ Plane using PCA.
"""
from typing import Union, List

from sklearn.decomposition import PCA

from qiskit.exceptions import QiskitError
from qiskit.ignis.measurement.discriminator.base_discriminator_fitter import \
    BaseDiscriminationFitter
from qiskit.result import Result
from qiskit.pulse.schedule import Schedule
try:
    from matplotlib import pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class PCADiscriminator(BaseDiscriminationFitter):
    """ Principle Component Analysis discriminator for level 1 data. """

    def __init__(self, states, cal_results: Union[Result, List[Result]],
                 qubit_mask: List[int], expected_states: List[str] = None,
                 standardize: bool = False,
                 schedules: Union[List[str], List[Schedule]] = None,
                 discriminator_parameters: dict = None,**serialized_dict):
        """
        Args:
            states (int): Number of states to classify into. Default is 2.
            cal_results (Union[Result, List[Result]]): calibration results,
                Result or list of Result used to fit the discriminator.
            qubit_mask (List[int]): determines which qubit's level 1 data to
                use in the discrimination process.
            expected_states (List[str]): a list that should have the same
                length as schedules. All results in cal_results are used if
                schedules is None. expected_states must have the corresponding
                length.
            standardize (bool): if true the discriminator will standardize the
                xdata using the internal method _scale_data.
            schedules (Union[List[str], List[Schedule]]): The schedules or a
                subset of schedules in cal_results used to train the
                discriminator. The user may also pass the name of the schedules
                instead of the schedules. If schedules is None, then all the
                schedules in cal_results are used.
            discriminator_parameters (dict): parameters for Sklearn's LDA.
        """
        if not discriminator_parameters:
            discriminator_parameters = {}

        self._threshold = discriminator_parameters.get('threshold', 0)
        solver = discriminator_parameters.get('svd_solver', 'auto')
        whiten = discriminator_parameters.get('whiten', False)
        tol = discriminator_parameters.get('tol', 0)

        self._classifier = PCA(n_components=1, svd_solver=solver,
                               whiten=whiten,
                               tol=tol)

        # Also sets the x and y data.
        super(PCADiscriminator, self).__init__(cal_results, qubit_mask,
                                        expected_states, standardize,
                                        schedules)

        self._description = 'PCA discriminator for measurement level 1.'

        self.fit()

    def fit(self):
        """Fits the discriminator using self._xdata and self._ydata."""
        if len(self._xdata) == 0:
            return

        self._classifier.fit(self._xdata, self._ydata)
        self._fitted = True

    def discriminate(self, x_data: List[List[float]]) -> List[str]:
        """Applies the discriminator to x_data.

        Args:
            x_data (List[List[float]]): list of features. Each feature is
                                        itself a list.

        Returns:
            The discriminated x_data as a list of labels.
        """
        return [int(i < self._threshold) for i in self._classifier.transform(x_data)]

    def discriminate_state(self, x_data: List[List[float]]) -> List[str]:
        """Applies the discriminator to x_data.

        Args:
            x_data (List[List[float]]): list of features. Each feature is
                                        itself a list.

        Returns:
            The discriminated x_data as a list of labels.
        """
        return [int(i > self._threshold) for i in self._classifier.transform(x_data)]
