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

from qiskit.exceptions import QiskitError
from qiskit.ignis.measurement.discriminator.discriminators import \
    BaseDiscriminationFitter
from qiskit.pulse import PulseError
from qiskit.result import Result
from qiskit.pulse.schedule import Schedule


class IQDiscriminationFitter(BaseDiscriminationFitter):
    """
    Abstract discriminator that implements the data formatting for IQ
    level 1 data.
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

        BaseDiscriminationFitter.__init__(self, cal_results, qubit_mask,
                                          expected_states, standardize,
                                          schedules)

    def get_xdata(self, results: Union[Result, List[Result]],
                  schedules: Union[List[str], List[Schedule]] = None) \
            -> List[List[float]]:
        """
        Retrieves feature data (xdata) for the discriminator.
        Args:
            results (Union[Result, List[Result]]): the get_memory() method is
                used to retrieve the level 1 data. If result is a list of
                Result, then the first Result in the list that returns the data
                of schedule (through get_memory(schedule)) is used.
            schedules (Union[List[str], List[Schedule]]): Either the names of
                the schedules or the schedules themselves.
        Returns (List[List[float]]): data as a list of features. Each feature
            is a list.
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
        Args:
            results (Union[Result, List[Result]]): results for which to
                retrieve the y data (i.e. expected states).
            schedules (Union[List[str], List[Schedule]]): the schedules for
                which to get the y data.
        Returns (List[str]): the y data, i.e. expected states. get_ydata is
            designed to produce y data with the same length as the x data.
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

    def format_iq_data(self, iq_data: np.ndarray) -> List[List[float]]:
        """
        Takes IQ data obtained from get_memory(), applies the qubit mask
        and formats the data as a list of lists. Each sub list is IQ data
        where the first half of the list is the I data and the second half of
        the list is the Q data.

        Args:
            iq_data (np.ndarray): data obtained from get_memory().
        Returns (List[List[float]]): A list of shots where each entry is a list
            of IQ points.
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
    """Linear discriminant analysis discriminator for IQ data."""

    def __init__(self, cal_results: Union[Result, List[Result]],
                 qubit_mask: List[int], expected_states: List[str],
                 standardize: bool = False,
                 schedules: Union[List[str], List[Schedule]] = None,
                 discriminator_parameters: dict = None):
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
            discriminator_parameters (dict): parameters for Sklearn's LDA.
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
        """
        Applies the discriminator to x_data
        Args:
            x_data (List[List[float]]): list of features. Each feature is
                itself a list.
        Returns (List[str]): the discriminated x_data as a list of labels.
        """
        return self._lda.predict(x_data)


class QuadraticIQDiscriminator(IQDiscriminationFitter):

    def __init__(self, cal_results: Union[Result, List[Result]],
                 qubit_mask: List[int], expected_states: List[str],
                 standardize: bool = False,
                 schedules: Union[List[str], List[Schedule]] = None,
                 discriminator_parameters: dict = None):
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
            discriminator_parameters (dict): parameters for Sklearn's LDA.
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
        """
        Applies the discriminator to x_data
        Args:
            x_data (List[List[float]]): list of features. Each feature is
                itself a list.
        Returns (List[str]): the discriminated x_data as a list of labels.
        """
        return self._qda.predict(x_data)
