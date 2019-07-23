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

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from qiskit.result import Result
from qiskit.ignis.discriminator.discriminator import AbstractDiscriminator
from qiskit.result.models import ExperimentResult


class LinearIQDiscriminator(AbstractDiscriminator):
    """
    A linear discriminant analysis for IQ data based on scikit learn's LinearDiscriminantAnalysis.
    """

    def __init__(self, result: Result, cal_circuits: list, data_circuits: list,
                 cal_circuit_expected, **discriminator_parameters):

        super().__init__(result, cal_circuits, data_circuits,
                         cal_circuit_expected, **discriminator_parameters)

        self.lda = LinearDiscriminantAnalysis(
            solver=discriminator_parameters.get('solver', 'svd'),
            shrinkage=discriminator_parameters.get('shrinkage', None),
            store_covariance=discriminator_parameters.get('store_covariance', False),
            tol=discriminator_parameters.get('tol', 1.0e-4))

    def _extract_calibration(self):
        """
        Get the measured features corresponding to the calibration circuits.
        Returns:
            x a list of lists. Each sublist corresponds to the IQ points
            for all the qubits.
        """
        x = []
        for circuit in self._cal_circuits:
            x.append(self._get_features(circuit))

        return x

    def _extract_data(self):
        """
        Get the measured features corresponding to the measured circuits.
        Returns:
            x a list of lists. Each sublist corresponds to the IQ points
            for all the qubits.
        """
        x = []
        for circuit in self._data_circuits:
            x.append(self._get_features(circuit))

        return x

    def _get_features(self, circuit):
        """Helper routine to extract the IQ points for all qubits."""
        features = []
        for result in self.result.get_memory(circuit):
            features.extend([np.real(result), np.imag(result)])

        return features

    def fit(self):
        """Trains the discriminator to the calibration data."""
        self.lda.fit(self._extract_calibration(), self._cal_circuits_expected)
        self.fitted = True

    def discriminate(self):

        if not self.fitted:
            self.fit()

        x = self._extract_data()
        y = self.lda.predict(x)

        # TODO Should this be
        shots = None  # TODO Get shots
        self.result.append(ExperimentResult(shots, True, y, meas_level=2))
        # TODO or this
        counter = 0
        for circuit in self._data_circuits:
            shots = None  # TODO extracted somehow from self.result
            self.result.append(ExperimentResult(shots, True, y[counter], meas_level=2))
            counter += 1

        return self.result
