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
from sklearn.preprocessing import StandardScaler

from qiskit.ignis.characterization.fitters import BaseFitter
from qiskit.exceptions import QiskitError
from qiskit.pulse import PulseError
from qiskit.result.models import ExperimentResult
from qiskit.result import postprocess, Result
from qiskit.pulse.schedule import Schedule


class ScikitIQDiscriminationFitter(BaseFitter):
    """
    IQDiscriminatorFitter takes IQ level 1 data produced by calibration
    measurements with a known expected state. It fits a discriminator
    that can be used to produce level 2 data, i.e. counts of quantum states.
    """

    def __init__(self, cal_results: Union[Result, List[Result]],
                 qubits: List[int], expected_states: Union[List[str], str],
                 discriminant,
                 schedules: Union[List[str], List[Schedule]] = None,
                 standardize: bool = False):
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

        # Used to rescale the IQ data qubit by qubit.
        self._standardize = standardize
        self._scaler = None

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

    def _scale_data(self, xdata: List[List[float]] = None):
        """
        Scales the features of xdata to have mean zero and unit variance.
        Args:
            xdata:
        Returns:
            data scaled to have zero mean and unit variance.
        """
        if not self._standardize:
            return xdata

        if not self._scaler:
            self._scaler = StandardScaler(with_std=True)
            self._scaler.fit(self._xdata)

        if not xdata:
            self._xdata = self._scaler.transform(self._xdata)
            return

        return self._scaler.transform(xdata)

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
                    iq_data = result.get_memory(schedule)

                    if len(iq_data.shape) == 2:  # meas return 'single'
                        for shot in iq_data[:, self._qubits]:
                            shot_i = list(np.real(shot))
                            shot_q = list(np.imag(shot))
                            self._xdata.append(shot_i + shot_q)
                            self._add_ydata(schedule)

                    if len(iq_data.shape) == 1:  # meas return 'avg'
                        avg_i = list(np.real(iq_data))
                        avg_q = list(np.imag(iq_data))
                        self._xdata.append(avg_i + avg_q)
                        self._add_ydata(schedule)

                except (QiskitError, KeyError):
                    pass

        self._scale_data()

    def _add_ydata(self, schedule: Union[Schedule, str]):
        """
        Adds the expected state of schedule to self._ydata.
        Args:
            schedule: schedule or schedule name.
            Used to get the expected state.
        """
        if isinstance(schedule, Schedule):
            self._ydata.append(self._expected_state[schedule.name])
        else:
            self._ydata.append(self._expected_state[schedule])

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

    def discriminate(self, x_data: List[List[float]]) -> List[str]:
        """
        Use the trained discriminator to discriminate the given data.
        :param x_data: data properly formatted for the discriminator.
        :return: the properly classified data.
        """
        return self._fit_fun.predict(x_data)

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

            xdata = []
            if len(iq_data.shape) == 2:  # meas_return 'single' case

                for shot in iq_data[:, self._qubits]:
                    shot_i = list(np.real(shot))
                    shot_q = list(np.imag(shot))
                    xdata.append(shot_i + shot_q)

            elif len(iq_data.shape) == 1:  # meas_return 'avg' case
                avg_i = list(np.real(iq_data[self._qubits]))
                avg_q = list(np.imag(iq_data[self._qubits]))
                xdata.append(avg_i + avg_q)

            else:
                raise PulseError('Unknown measurement return type.')

            return self._scale_data(xdata)
        else:
            raise QiskitError('Cannot extract IQ data for %s' %
                              result.meas_level)


class LinearIQDiscriminator(ScikitIQDiscriminationFitter):

    def __init__(self, cal_results: Union[Result, List[Result]],
                 qubits: List[int], expected_states: Union[List[str], str],
                 schedules: Union[List[str], List[Schedule]] = None,
                 discriminator_parameters: dict = None,
                 standardize: bool = False):
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
                                              schedules=schedules,
                                              standardize=standardize)

        self._description = 'Linear IQ discriminator for measurement level 1.'


class QuadraticIQDiscriminator(ScikitIQDiscriminationFitter):

    def __init__(self, cal_results: Union[Result, List[Result]],
                 qubits: List[int], expected_states: Union[List[str], str],
                 schedules: Union[List[str], List[Schedule]] = None,
                 discriminator_parameters: dict = None,
                 standardize: bool = False):
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
                                              schedules=schedules,
                                              standardize=standardize)

        self._description = 'Quadratic IQ discriminator for measurement ' \
                            'level 1.'
