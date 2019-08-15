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
from typing import Union, List

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from qiskit.ignis.characterization.fitters import BaseFitter
from qiskit.exceptions import QiskitError
from qiskit.result.models import ExperimentResult
from qiskit.result import postprocess, Result
from qiskit.pulse.schedule import Schedule


class ScikitIQDiscriminationFitter(BaseFitter):
    """
    IQDiscriminatorFitter takes IQ level 1 data produced by calibration
    measurements with a known expected state to train a discriminator
    that can be used to produce level 2 data, i.e. counts of quantum states.
    """

    def __init__(self, cal_results: Union[Result, List[Result]],
                 qubits: List[int], expected_states: Union[List[str], str],
                 discriminant,
                 schedules: Union[List[str], List[Schedule]] = None):
        """
        Args:
            cal_results: calibration results, Result or list of Result
            qubits: the qubits for which to discriminate.
            expected_states: a list that should have the same length as
                cal_results. If cal_results is a Result and not a list then
                expected_states should be a string or a schedule.
            discriminant: a discriminant from e.g. scikit learn.
            schedules: The schedules in cal_results or their names. If None
            all schedules will be used.
        """
        if not schedules:
            schedules = [_.header.name for _ in cal_results.results]

        # Sanity checks
        if isinstance(cal_results, list) and isinstance(expected_states, list):
            if len(cal_results) != len(expected_states):
                raise QiskitError('Number of input results and assigned '
                                  'states must be equal.')

        _expected_state = {}
        for idx, schedule in enumerate(schedules):
            if isinstance(schedule, Schedule):
                name = schedule.name
            else:
                name = schedule
            expected_state = expected_states[idx]
            _expected_state[name] = expected_state

        BaseFitter.__init__(self, None, cal_results, None, qubits,
                            discriminant, None, None, schedules,
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
        for schedule in self._circuit_names:
            for result in self._backend_result_list:
                try:
                    iq_data = result.get_memory(schedule)[:, self._qubits]
                    for shot_idx in range(iq_data.shape[0]):
                        shot_i = list(np.real(iq_data[shot_idx]))
                        shot_q = list(np.imag(iq_data[shot_idx]))
                        self._xdata.append(shot_i + shot_q)
                        if isinstance(schedule, Schedule):
                            self._ydata.append(
                                self._expected_state[schedule.name])
                        else:
                            self._ydata.append(self._expected_state[schedule])

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

    def extract_xdata(self, result: ExperimentResult):
        """
        Takes a result (ExperimentResult) and extracts the data into a form
        that can be used by the fitter.

        Args:
            result (ExperimentResult): result from a Qiskit backend.
        Returns:
            A list of shots where each entry is a list of IQ points.
        """

        if result.meas_level == 1:
            iq_data = postprocess.format_level_1_memory(result.data.memory)
            iq_data = iq_data[:, self._qubits]
            xdata = []
            for shot_idx in range(iq_data.shape[0]):
                shot_i = list(np.real(iq_data[shot_idx]))
                shot_q = list(np.imag(iq_data[shot_idx]))
                xdata.append(shot_i + shot_q)

            return xdata
        else:
            raise QiskitError('Cannot extract IQ data for %s' %
                              result.meas_level)


class LinearScikitIQDiscriminationFitter(ScikitIQDiscriminationFitter):

    def __init__(self, cal_results: Union[Result, List[Result]],
                 qubits: List[int], expected_states: Union[List[str], str],
                 schedules: Union[List[str], List[Schedule]] = None,
                 discriminator_parameters: dict = None):
        """
        Args:
            cal_results: calibration results, list of qiskit.Result or
            qiskit.Result
            qubits: the qubits for which to discriminate.
            expected_states: a list that should have the same length as
                cal_results. If cal_results is a Result and not a list then
                expected_states should be a string or a float or an int.
            schedules: The schedules in cal_results or their names. If None
                all schedules will be used.
            discriminator_parameters: parameters for the discriminator.
        """
        if not discriminator_parameters:
            discriminator_parameters = {}

        solver = discriminator_parameters.get('solver', 'svd')
        shrink = discriminator_parameters.get('shrinkage', None)
        store_cov = discriminator_parameters.get('store_covariance', False)
        tol = discriminator_parameters.get('tol', 1.0e-4)

        lda = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrink,
                                         store_covariance=store_cov, tol=tol)

        ScikitIQDiscriminationFitter.__init__(self, cal_results, qubits,
                                              expected_states, lda,
                                              schedules=schedules)

        self._description = 'Linear IQ discriminator for measurement level 1.'


class QuadraticScikitIQDiscriminationFitter(ScikitIQDiscriminationFitter):

    def __init__(self, cal_results: Union[Result, List[Result]],
                 qubits: List[int], expected_states: Union[List[str], str],
                 schedules: Union[List[str], List[Schedule]] = None,
                 discriminator_parameters: dict = None):
        """
        Args:
            cal_results: calibration results, list of qiskit.Result or
            qiskit.Result
            qubits: the qubits for which to discriminate.
            expected_states: a list that should have the same length as
                cal_results. If cal_results is a Result and not a list then
                expected_states should be a string or a float or an int.
            schedules: The schedules in cal_results or their names. If None
                all schedules will be used.
            discriminator_parameters: parameters for the discriminator.

        """
        if not discriminator_parameters:
            discriminator_parameters = {}

        store_cov = discriminator_parameters.get('store_covariance', False)
        tol = discriminator_parameters.get('tol', 1.0e-4)

        qda = QuadraticDiscriminantAnalysis(
            store_covariance=store_cov, tol=tol)

        ScikitIQDiscriminationFitter.__init__(self, cal_results, qubits,
                                              expected_states, qda,
                                              schedules=schedules)

        self._description = 'Quadratic IQ discriminator for measurement ' \
                            'level 1.'
