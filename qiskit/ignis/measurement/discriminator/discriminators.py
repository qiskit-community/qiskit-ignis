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
from sklearn.preprocessing import StandardScaler

from qiskit.exceptions import QiskitError
from qiskit.result import Result
from qiskit.pulse.schedule import Schedule


class BaseDiscriminationFitter(ABC):
    """
    IQDiscriminatorFitter takes IQ level 1 data produced by calibration
    measurements with a known expected state. It fits a discriminator
    that can be used to produce level 2 data, i.e. counts of quantum states.
    """

    def __init__(self, cal_results: Union[Result, List[Result]],
                 qubit_mask: List[int], expected_states: List[str],
                 standardize: bool = False,
                 schedules: Union[List[str], List[Schedule]] = None):
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

        # Use all results in cal_results if schedules is None
        if schedules is None:
            schedules = self._get_schedules(cal_results)

        self._expected_state = {}
        self._add_expected_states(expected_states, schedules)

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
            schedule (Union[Schedule, str]): schedule or schedule name.
            Used to get the expected state.
        """
        if isinstance(schedule, Schedule):
            self._ydata.append(self._expected_state[schedule.name])
        else:
            self._ydata.append(self._expected_state[schedule])

    def add_data(self, result: Result, expected_states: List[str],
                 refit: bool = True,
                 schedules: Union[List[Schedule], List[str]] = None):
        """
        Args:
            result (Result): a Result containing new data to be used to
                train the discriminator.
            expected_states (List[str]): the expected states of the results in
                result.
            refit (bool): refit the discriminator if True.
            schedules (Union[List[Schedule], List[str]]):
        """
        if schedules is None:
            schedules = self._get_schedules(result)

        self._backend_result_list.append(result)
        self._add_expected_states(expected_states, schedules)
        self._schedules.extend(schedules)
        self._xdata = self.get_xdata(self._backend_result_list, schedules)
        self._ydata = self.get_ydata(self._backend_result_list, schedules)

        if refit:
            self.fit()

    def _add_expected_states(self, expected_states: List[str],
                             schedules: Union[List[Schedule], List[str]]):
        """
        Adds the given expected states to self._expected_states.
        Args:
            expected_states (List[str]): list of expected states. Must have the
                same length as the number of schedules.
            schedules (Union[List[Schedule], List[str]]): schedules or their
                names corresponding to the expected states.
        """
        if len(expected_states) != len(schedules):
            raise QiskitError('Number of input schedules and assigned '
                              'states must be equal.')

        for idx, schedule in enumerate(schedules):
            if isinstance(schedule, Schedule):
                name = schedule.name
            else:
                name = schedule
            expected_state = expected_states[idx]
            self._expected_state[name] = expected_state

    @staticmethod
    def _get_schedules(results: Union[Result, List[Result]]) -> List[str]:
        """
        Extracts the names of all Schedules in a Result or a list of Result.
        Args:
            results (Union[Result, List[Result]]): the results for which to
            extract the names,
        Returns (List[str]):
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

    @property
    def expected_states(self):
        """Returns the expected states used to train the discriminator."""
        return self._expected_state

    @property
    def schedules(self):
        """Returns the schedules with which the discriminator was fitted."""
        return self._schedules

    @property
    def fitted(self):
        """True if the discriminator has been fitted to calibration data."""
        return self._fitted

    def _scale_data(self, xdata: List[List[float]],
                    refit: bool = False) -> List[List[float]]:
        """
        Scales xdata, for instance, by transforming it to zero mean and unit
        variance data.
        Args:
            xdata (List[List[float]]): data as a list of features. Each
                feature is itself a list.
            refit (bool): if true than self._scaler is refit using the given
                xdata.
        Returns (List[List[float]]): the scaled xdata as a list of features.
        """
        if not self._standardize:
            return xdata

        if not self._scaler or refit:
            self._scaler = StandardScaler(with_std=True)
            self._scaler.fit(xdata)

        return self._scaler.transform(xdata)

    @abstractmethod
    def get_xdata(self, results: Union[Result, List[Result]],
                  schedules: Union[List[str], List[Schedule]] = None) \
            -> List[List[float]]:
        """
        Retrieves feature data (xdata) for the discriminator.
        Args:
            results (Union[Result, List[Result]]): the get_memory() method is
                used to retrieve the level 1 data. If result is a list of
                Result then the first Result to return the data of schedule in
                schedules is used.
            schedules (Union[List[str], List[Schedule]]): Either the names of
                the schedules or the schedules themselves.
        Returns (List[List[float]]): data as a list of features. Each feature
            is a list.
        """
        pass

    @abstractmethod
    def get_ydata(self, results: Union[Result, List[Result]],
                  schedules: Union[List[str], List[Schedule]] = None) \
            -> List[str]:
        """
        Args:
            results (Union[Result, List[Result]]): results for which to
                retrieve the y data (i.e. expected states).
            schedules (Union[List[str], List[Schedule]]): the schedules for
                which to get the y data.
        Returns (List[str]): the y data, i.e. expected states. get_ydata is
            designed to produce y data with the same length as the x data.
        """
        pass

    @abstractmethod
    def fit(self):
        """ Fits the discriminator using self._xdata and self._ydata. """
        pass

    @abstractmethod
    def discriminate(self, x_data: List[List[float]]) -> List[str]:
        """
        Applies the discriminator to x_data
        Args:
            x_data (List[List[float]]): list of features. Each feature is
                itself a list.
        Returns (List[str]): the discriminated x_data as a list of labels.
        """
        pass
