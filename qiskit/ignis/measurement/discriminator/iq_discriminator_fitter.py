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
IQ Discriminator module to discriminate date in the IQ Plane.
"""
import numpy as np
from typing import List, Union

from fitters.lda_discriminator_fitter import LDADiscriminator
from fitters.qda_discriminator_fitter import QDADiscriminator
from fitters.pca_discriminator_fitter import PCADiscriminator

from qiskit.exceptions import QiskitError
from qiskit.pulse import PulseError
from qiskit.result import Result
from qiskit.pulse.schedule import Schedule
try:
    from matplotlib import pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class IQDiscriminationFitter:
    """
    Abstract discriminator that implements the data formatting for IQ
    level 1 data.
    """

    def __init__(self, cal_results, qubit_mask, metadata=None):
        """
        Args:
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
        """

        self._discriminator = None
        self._cal_results = cal_results
        self._qubit_mask = qubit_mask

    @property
    def discriminator(self):
        """Return the fitted mitigator object"""
        if self._discriminator is None:
            raise QiskitError("Discriminator has not been fitted. Run `fit` first.")
        return self._discriminator




    @staticmethod
    def _get_iq_grid(x_data: np.array) -> (np.meshgrid, np.meshgrid):
        """
        Create mesh grids used to plot the decision boundary.

        Args:
            x_data (np.array): IQ data.
        Returns:
            xx (np.meshgrid): xx meshgrid for plotting discriminator boundary
            yy (np.meshgrid): yy meshgrid for plotting discriminator boundary
        """
        max_i = np.max(x_data[:, 0])
        min_i = np.min(x_data[:, 0])
        max_q = np.max(x_data[:, 1])
        min_q = np.min(x_data[:, 1])

        spacing = (max_i - min_i) / 100.0
        xx, yy = np.meshgrid(
            np.arange(min_i - 10 * spacing, max_i + 10 * spacing, spacing),
            np.arange(min_q - 10 * spacing, max_q + 10 * spacing, spacing))

        return xx, yy

    def fit(self, method):
        """Fits the discriminator using self._xdata and self._ydata."""
        if method == "LDA":
            self._discriminator = LDADiscriminator(self._cal_results, self._qubit_mask)
        elif method == "QDA":
            self._discriminator = QDADiscriminator(self._cal_results, self._qubit_mask)
        elif method == "PCA":
            self._discriminator = PCADiscriminator(self._cal_results, self._qubit_mask)
        else:
            raise ValueError("Invalid method specified.")
        return self._discriminator

    def discriminate(self, x_data: List[List[float]]) -> List[str]:
        """Applies the discriminator to x_data.

        Args:
            x_data (List[List[float]]): list of features. Each feature is
                                        itself a list.

        Returns (List[str]):
            The discriminated x_data as a list of labels.
        """


class SklearnIQDiscriminator(IQDiscriminationFitter):
    """
    A generic discriminant analysis discriminator for IQ data that
    takes an sklearn classifier as an argument.
    """

    def __init__(self, classifier, cal_results: Union[Result, List[Result]],
                 qubit_mask: List[int], expected_states: List[str] = None,
                 standardize: bool = False,
                 schedules: Union[List[str], List[Schedule]] = None):
        """
        Args:
            classifier (Classifier):
                An sklearn classifier to train and do the discrimination. The
                classifier must have a fit method and a predict method
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
        """
        self._type_check_classifier(classifier)
        self._classifier = classifier

        # Also sets the x and y data.
        IQDiscriminationFitter.__init__(self, cal_results, qubit_mask,
                                        expected_states, standardize,
                                        schedules)

        self._description = (
            '{} IQ discriminator for measurement level 1.'.format(
                classifier.__class__.__name__))

        self.fit()

    @staticmethod
    def _type_check_classifier(classifier):
        for name in ['fit', 'predict']:
            if not callable(getattr(classifier, name, None)):
                raise QiskitError(
                    'Classifier of type "{}" does not have a callable "{}"'
                    ' method.'.format(type(classifier).__name__, name)
                )

    def fit(self):
        """ Fits the discriminator using self._xdata and self._ydata. """
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
            the discriminated x_data as a list of labels.
        """
        return self._classifier.predict(x_data)
