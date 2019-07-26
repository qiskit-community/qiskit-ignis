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

from qiskit.ignis.characterization.fitters import BaseFitter
from qiskit.exceptions import QiskitError


class LinearIQDiscriminationFitter(BaseFitter):

    def __init__(self, cal_results, discriminator_parameters,
                 qubits, circuit_names, expected_states):
        """
        Args:
            cal_results: calibration results, list of qiskit.Result or qiskit.Result
            discriminator_parameters: parameters for the discriminator.
            qubits: the qubits for which we want to discriminate.
            circuit_names: The names of the circuits in cal_results.
            expected_states: a list that should have the same length as cal_results.
                If cal_results is a Result and not a list then exoected_states should
                be a string or a float or an int.
        """

        lda = LinearDiscriminantAnalysis(
            solver=discriminator_parameters.get('solver', 'svd'),
            shrinkage=discriminator_parameters.get('shrinkage', None),
            store_covariance=discriminator_parameters.get('store_covariance', False),
            tol=discriminator_parameters.get('tol', 1.0e-4))

        # Sanity checks
        if isinstance(cal_results, list) and isinstance(expected_states, list):
            if len(cal_results) != len(expected_states):
                raise QiskitError('Inconsistent number of results and expected results.')

        description = 'Linear IQ discriminator for measurement level 1.'

        _expected_state = []
        if isinstance(expected_states, list):
            for es in expected_states:
                _expected_state.append(es)
        else:
            _expected_state.append(expected_states)

        BaseFitter.__init__(self, description, cal_results, None, qubits, lda, None,
                            None, circuit_names, expected_state=_expected_state)

    def add_data(self, result, recalc=True, refit=True, expected_state=None):
        """
        Overrides method of base class.
        Args:
            result: the Result obtained from e.g. backend.run().result()
            recalc: this parameter is irrelevant and only needed for Liskov principle
            refit: refit the discriminator if True.
            expected_state: the expected state of the discriminator.
        """
        self._expected_state.append(expected_state)
        self._backend_result_list.append(result)

        if refit:
            self.fit_data()

    def _calc_data(self):
        """
        Extract the measured IQ points (i.e. features) from the list of
        backend results and combine them into a list of lists.
        Returns:
            x a list of lists. Each sublist corresponds to the IQ points
            for all the qubits. Each sublist therefore corresponds to an
            expected result stored in self._expected_state.
        """
        self._xdata = []
        for circuit in self._circuit_names:
            features = []
            for result in self._backend_result_list:
                try:
                    iq_data = result.get_memory(circuit)
                    features.extend([np.real(iq_data), np.imag(iq_data)])
                except (QiskitError, KeyError):
                    pass

            self._xdata.append(features)

    def fit_data(self, qid=-1, p0=None, bounds=None, series=None):
        """
        Fit the Linear Discriminant.
        Args:
            qid: not needed
            p0: not needed
            bounds: not needed
            series: not needed
        """
        self._fit_fun.fit(self._xdata, self._expected_state)
