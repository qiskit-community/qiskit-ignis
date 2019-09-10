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
from abc import ABC, abstractmethod
from typing import Union, List

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from qiskit.exceptions import QiskitError
from qiskit.pulse import PulseError
from qiskit.result import Result
from qiskit.pulse.schedule import Schedule


class BaseDiscriminationFitter(ABC):
    """
    IQDiscriminatorFitter takes IQ level 1 data produced by calibration
    measurements with a known expected state. It fits a discriminator
    that can be used to produce level 2 data, i.e. counts of quantum states.
    """

    def __init__(self, cal_results: Union[Result, List[Result]],
                 qubit_mask: List[int], expected_states: Union[List[str], str],
                 standardize: bool = False,
                 schedules: Union[List[str], List[Schedule]] = None):
        """
        Args:
            cal_results: calibration results, Result or list of Result
            qubit_mask: determines which qubit's level 1 data to use in the
                discrimination process.
            expected_states: a list that should have the same length as
                cal_results. If cal_results is a Result and not a list then
                expected_states should be a string or a schedule.
            discriminant: a discriminant from e.g. scikit learn.
            schedules: The schedules in cal_results or their names. If None
            all schedules will be used.
        """
        if schedules is None:
            schedules = [_.header.name for _ in cal_results.results]

        # Sanity checks
        if isinstance(cal_results, list) and isinstance(expected_states, list):
            if len(cal_results) != len(expected_states):
                raise QiskitError('Number of input results and assigned '
                                  'states must be equal.')

        self._expected_state = {}
        for idx, schedule in enumerate(schedules):
            if isinstance(schedule, Schedule):
                name = schedule.name
            else:
                name = schedule
            expected_state = expected_states[idx]
            self._expected_state[name] = expected_state

        # Used to rescale the xdata qubit by qubit.
        self._description = None
        self._standardize = standardize
        self._scaler = None
        self._qubit_mask = qubit_mask
        self._schedules = schedules
        self._backend_result_list = []
        self._fitted = False

        if cal_results is not None:
            if isinstance(cal_results, list):
                for result in cal_results:
                    self._backend_result_list.append(result)
            else:
                self._backend_result_list.append(cal_results)

        self._xdata = self.get_xdata(self._backend_result_list, schedules)
        self._ydata = self.get_ydata(self._backend_result_list, schedules)

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

    def add_data(self, result, refit=True, expected_state=None):
        """
        Args:
            result: the Result obtained from e.g. backend.run().result()
            recalc: this parameter is irrelevant and only needed for Liskov
            principle.
            refit: refit the discriminator if True.
            expected_state: the expected state of the discriminator.
        """
        pass

    @property
    def expected_states(self):
        return self._expected_state

    @property
    def fitted(self):
        """True if the discriminator has been fitted to calibration data."""
        return self._fitted

    @abstractmethod
    def _scale_data(self, xdata: List[List[float]], refit=False):
        pass

    @abstractmethod
    def get_xdata(self, results: Union[Result, List[Result]],
                  schedules: Union[List[str], List[Schedule]] = None) \
            -> List[List[float]]:
        """
        Retrieves feature data (xdata) for the discriminator.
        Args:
            results: the get_memory() method is used to retrieve the level 1
                data. If result is a list of Result then the first Result to
                return the data of schedule in schedules is used.
            schedules: Either the names of the schedules or the schedules
                themselves.
        Returns:
            x data as a list of lists.
        """
        pass

    @abstractmethod
    def get_ydata(self, results: Union[Result, List[Result]],
                  schedules: Union[List[str], List[Schedule]] = None) \
            -> List[str]:
        """
        Return the ydata as a List[str]. The number of shots in each
        ExperimentResult is taken into account so that the length of ydata,
        i.e. sum_i(schedule_i shots), matches the length of the xdata.
        Args:
            results: needed to retrieve the number of shots.
            schedules: the schedules for which to get the ydata.
        Returns:
            the y data as a list

        TODO we may be able to simplify this if we store the number of shots
        TODO in an object like expect_sates
        """
        pass

    @abstractmethod
    def fit(self):
        """ Fits the discriminator using self._xdata and self._ydata. """
        pass

    @abstractmethod
    def discriminate(self, x_data: List[List[float]]) -> List[str]:
        """ Applies the discriminator to x_data"""
        pass


class IQDiscriminationFitter(BaseDiscriminationFitter, ABC):
    """
    Abstract discriminator that implements the data formatting for IQ
    level 1 data.
    """

    def __init__(self, cal_results: Union[Result, List[Result]],
                 qubit_mask: List[int], expected_states: Union[List[str], str],
                 standardize: bool = False,
                 schedules: Union[List[str], List[Schedule]] = None):

        BaseDiscriminationFitter.__init__(self, cal_results, qubit_mask,
                                          expected_states, standardize,
                                          schedules)

    def _scale_data(self, xdata: List[List[float]], refit=False):
        """
        Scales the features of xdata to have mean zero and unit variance.
        Args:
            xdata: A list of lists containing the IQ data.
            refit: if True a new scaler is fitted to the given IQ data.
        Returns:
            data scaled to have zero mean and unit variance.
        """
        if not self._standardize:
            return xdata

        if not self._scaler or refit:
            self._scaler = StandardScaler(with_std=True)
            self._scaler.fit(xdata)

        return self._scaler.transform(xdata)

    def get_xdata(self, results: Union[Result, List[Result]],
                  schedules: Union[List[str], List[Schedule]] = None) \
            -> List[List[float]]:
        """
        Args:
            results: qiskit Result of list thereof.
            schedules:
        Returns:
            xdata, the IQ data formatted in a list of lists. Each sublist
            corresponds to a shot or the average of shots.
        """
        xdata = []
        if schedules is None:
            schedules = self._get_schedules(results)

        for schedule in schedules:
            iq_data = None
            if isinstance(results, list):
                for result in results:
                    try:
                        iq_data = result.get_memory(schedule)
                    except QiskitError:
                        pass
            else:
                iq_data = results.get_memory(schedule)

            if iq_data is None:
                raise PulseError('Could not find IQ data for %s' % schedule)

            xdata.extend(self.format_iq_data(iq_data))

        return self._scale_data(xdata)

    def get_ydata(self, results: Union[Result, List[Result]],
                  schedules: Union[List[str], List[Schedule]] = None):
        """
        Return the ydata as a List[str]. The number of shots in each
        ExperimentResult is taken into account so that the length of ydata,
        i.e. sum_i(schedule_i shots), matches the length of the xdata.
        Args:
            results: needed to retrieve the number of shots.
            schedules: the schedules for which to get the ydata.
        Returns:
            the y data as a list

        TODO we may be able to simplify this if we store the number of shots
        TODO in an object like expect_sates
        """
        ydata = []

        if schedules is None:
            schedules = self._get_schedules(results)

        for schedule in schedules:

            if isinstance(schedule, Schedule):
                shed_name = schedule.name
            else:
                shed_name = schedule

            if isinstance(results, Result):
                results = [results]

            for result in results:
                try:
                    iq_data = result.get_memory(schedule)
                    n_shots = iq_data.shape[0]
                    ydata.extend([self._expected_state[shed_name]]*n_shots)
                except QiskitError:
                    pass

        return ydata

    def _get_schedules(self,
                       results: Union[Result, List[Result]]) -> List[str]:
        """
        Args:
            results:
        Returns:
            The name of the schedules in results.
        """
        schedules = []
        if isinstance(results, Result):
            for res in results.results:
                schedules.append(res.header.name)
        else:
            for result in results:
                schedules.extend([_.header.name for _ in result.results])

        return schedules

    def format_iq_data(self, iq_data: np.ndarray) -> List[List[float]]:
        """
        Takes IQ data obtained from get_memory(), applies the qubit mask
        and formats the data as a list of lists. Each sub list is IQ data
        where the first half of the list is the I data and the second half of
        the list is the Q data.

        Args:
            iq_data: data obtained from get_memory()
        Returns:
            A list of shots where each entry is a list of IQ points.
        """
        xdata = []
        if len(iq_data.shape) == 2:  # meas_return 'single' case

            for shot in iq_data[:, self._qubit_mask]:
                shot_i = list(np.real(shot))
                shot_q = list(np.imag(shot))
                xdata.append(shot_i + shot_q)

        elif len(iq_data.shape) == 1:  # meas_return 'avg' case
            avg_i = list(np.real(iq_data[self._qubit_mask]))
            avg_q = list(np.imag(iq_data[self._qubit_mask]))
            xdata.append(avg_i + avg_q)

        else:
            raise PulseError('Unknown measurement return type.')

        return xdata


class LinearIQDiscriminator(IQDiscriminationFitter):

    def __init__(self, cal_results: Union[Result, List[Result]],
                 qubit_mask: List[int], expected_states: Union[List[str], str],
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

        self._lda = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrink,
                                               store_covariance=store_cov,
                                               tol=tol)

        # Also sets the x and y data.
        IQDiscriminationFitter.__init__(self, cal_results, qubit_mask,
                                        expected_states, standardize,
                                        schedules)

        self._description = 'Linear IQ discriminator for measurement level 1.'

        self.fit()

    def fit(self):
        """ Fits the discriminator using self._xdata and self._ydata. """
        if len(self._xdata) == 0:
            return

        self._lda.fit(self._xdata, self._ydata)
        self._fitted = True

    def discriminate(self, x_data: List[List[float]]) -> List[str]:
        """ Applies the discriminator to x_data."""
        return self._lda.predict(x_data)


class QuadraticIQDiscriminator(IQDiscriminationFitter):

    def __init__(self, cal_results: Union[Result, List[Result]],
                 qubit_mask: List[int], expected_states: Union[List[str], str],
                 schedules: Union[List[str], List[Schedule]] = None,
                 discriminator_parameters: dict = None,
                 standardize: bool = False):
        """
        Args:
            cal_results: calibration results, list of qiskit.Result or
            qiskit.Result
            qubit_mask: the qubits for which to discriminate.
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

        self._qda = QuadraticDiscriminantAnalysis(store_covariance=store_cov,
                                                  tol=tol)

        # Also sets the x and y data.
        IQDiscriminationFitter.__init__(self, cal_results, qubit_mask,
                                        expected_states, standardize,
                                        schedules)

        self._description = 'Quadratic IQ discriminator for measurement ' \
                            'level 1.'

    def fit(self):
        """ Fits the discriminator using self._xdata and self._ydata. """
        if len(self._xdata) == 0:
            return

        self._qda.fit(self._xdata, self._ydata)
        self._fitted = True

    def discriminate(self, x_data: List[List[float]]) -> List[str]:
        """ Applies the discriminator to x_data."""
        return self._qda.predict(x_data)
