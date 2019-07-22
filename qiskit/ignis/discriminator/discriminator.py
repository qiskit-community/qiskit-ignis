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

from abc import abstractmethod

from qiskit.result import Result


class AbstractDiscriminator(object):
    """Abstract discriminator class that hold the methods signatures
    used for discrimination, e.g. to convert IQ data into 0 and 1's."""

    @abstractmethod
    def __init__(self, result: Result, **discriminator_parameters):
        self.result = result
        self._fitted = False
        self.discriminator_parameters = discriminator_parameters

    @abstractmethod
    def _extract_calibration(self):
        """
        Extracts the calibration data from result.
        :return: A named tuple of two lists: X a list of lists and y a list.
        X corresponds to the features list and y is the dependent variable.
        """
        pass

    @abstractmethod
    def _extract_data(self):
        """
        Extracts the data that will be classified using the discriminator.
        :return: A list of lists corresponding to the features.
        """
        pass

    @abstractmethod
    def fit(self):
        """
        Trains the discriminator to the calibration data.
        :return:
        """
        pass

    @abstractmethod
    def discriminate(self) -> Result:
        """
        Fit the discriminator and use it to discriminate the data in
        self.result.
        :return: A result with the result of the discriminator.
        """
        pass

    @property
    def fitted(self):
        """True if the discriminator has been fitted to the calibration data."""
        return self._fitted
