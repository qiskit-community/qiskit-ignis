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
            cal_results: calibration results, list of qiskit.Result or
            qiskit.Result
            discriminator_parameters: parameters for the discriminator.
            qubits: the qubits for which we want to discriminate.
            circuit_names: The names of the circuits in cal_results.
            expected_states: a list that should have the same length as
                cal_results. If cal_results is a Result and not a list then
                expected_states should be a string or a float or an int.
        """
        solver = discriminator_parameters.get('solver', 'svd')
        shrink = discriminator_parameters.get('shrinkage', None)
        store_cov = discriminator_parameters.get('store_covariance', False)
        tol = discriminator_parameters.get('tol', 1.0e-4)

        lda = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrink,
                                         store_covariance=store_cov, tol=tol)

        # Sanity checks
        if isinstance(cal_results, list) and isinstance(expected_states, list):
            if len(cal_results) != len(expected_states):
                msg = 'Inconsistent number of results and expected results.'
                raise QiskitError(msg)

        description = 'Linear IQ discriminator for measurement level 1.'

        _expected_state = {}
        for idx in range(len(circuit_names)):
            circ_name = circuit_names[idx]
            expected_state = expected_states[idx]
            _expected_state[circ_name] = expected_state

        BaseFitter.__init__(self, description, cal_results, None, qubits,
                            lda, None, None, circuit_names,
                            expected_state=_expected_state)

    def add_data(self, result, recalc=True, refit=True, expected_state=None):
        """
        Overrides method of base class.
        Args:
            result: the Result obtained from e.g. backend.run().result()
            recalc: this parameter is irrelevant and only needed for Liskov
            principle.
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
        self._xdata, self._ydata = [], []
        for circuit in self._circuit_names:
            for result in self._backend_result_list:
                try:
                    iq_data = result.get_memory(circuit)[:, self._qubits]
                    for shot_idx in range(iq_data.shape[0]):
                        shot_i = list(np.real(iq_data[shot_idx]))
                        shot_q = list(np.imag(iq_data[shot_idx]))
                        self._xdata.append(shot_i + shot_q)
                        self._ydata.append(self._expected_state[circuit])

                except (QiskitError, KeyError):
                    pass

    def fit_data(self, qid=-1, p0=None, bounds=None, series=None):
        """
        Fit the Linear Discriminant.
        Args:
            qid: not needed
            p0: not needed
            bounds: not needed
            series: not needed
        """
        self._fit_fun.fit(self._xdata, self._ydata)
