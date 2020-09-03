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

"""
Base discriminator class. All discriminators should inherite from this base class.
"""
from abc import ABC, abstractmethod
import re
from typing import Union, List
import numpy as np

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
                 qubit_mask: List[int], expected_states: List[str] = None,
                 standardize: bool = False,
                 schedules: Union[List[str], List[Schedule]] = None):
        """
        Args:
            cal_results: calibration results, Result or list of Result used to
                         fit the discriminator.
            qubit_mask: determines which qubit's level 1 data to
                        use in the discrimination process.
            expected_states: a list that should have the same length as schedules.
                             All results in cal_results are used if schedules is None.
                             expected_states must have the corresponding length.
            standardize: if true the discriminator will standardize the
                         xdata using the internal method _scale_data.
            schedules: The schedules or a subset of schedules in cal_results used
                       to train the discriminator. The user may also pass the name
                       of the schedules instead of the schedules. If schedules is None,
                       then all the schedules in cal_results are used.
        """
        # Regex pattern used to identify calibration schedules
        self._cal_pattern = r'cal_\d+$'

        # Use all results in cal_results if schedules is None
        if schedules is None:
            schedules = self._get_schedules(cal_results, 0)
        # Dict of form schedule name: expected state
        self._expected_state = {}
        if expected_states is None:
            expected_states = self._get_expected_states(schedules)

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

        self._xdata = self.get_xdata(self._backend_result_list, 0, schedules)
        self._ydata = self.get_ydata(self._backend_result_list, 0, schedules)

    def _add_ydata(self, schedule: Union[Schedule, str]):
        """
        Adds the expected state of schedule to self._ydata.

        Args:
            schedule: schedule or schedule name. Used to get the expected state.
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
            result: a Result containing new data to be used to train the discriminator.
            expected_states: the expected states of the results in result.
            refit: refit the discriminator if True.
            schedules: The schedules or a subset of schedules in cal_results used
                       to train the discriminator. The user may also pass the name
                       of the schedules instead of the schedules. If schedules is None,
                       then all the schedules in cal_results are used.
        """
        if schedules is None:
            schedules = self._get_schedules(result, 0)

        self._backend_result_list.append(result)
        self._add_expected_states(expected_states, schedules)
        self._schedules.extend(schedules)
        self._xdata = self.get_xdata(self._backend_result_list, 0, schedules)
        self._ydata = self.get_ydata(self._backend_result_list, 0, schedules)

        if refit:
            self.fit()

    def _add_expected_states(self, expected_states: List[str],
                             schedules: Union[List[Schedule], List[str]]):
        """
        Adds the given expected states to self._expected_states.

        Args:
            expected_states: list of expected states. Must have the
                             same length as the number of schedules.
            schedules: schedules or their names corresponding to the expected states.

        Raises:
            QiskitError: If the number of input schedules does not equal the
                number of expected states
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

    def is_calibration(self, result_name: str) -> bool:
        """
        Identify if a name corresponds to a calibration name identified by
        the regex pattern self._cal_pattern.

        Args:
            result_name: name of the result to be tested.

        Returns:
            bool: True if the name of the result indicates that it is a
                calibration result.
        """
        return re.match(self._cal_pattern, result_name) is not None

    def _get_schedules(self, results: Union[Result, List[Result]],
                       schedule_type_to_get: int) -> List[str]:
        """
        Extracts the names of all Schedules in a Result or a list of Result.
        Only the results with a name that matches self._cal_pattern are
        returned.

        Args:
            results: the results for which to
                extract the names,
            schedule_type_to_get: defines which schedule type to include
                                  in the returned schedules.
                                  (``0``) calibration data only
                                  (``1``) non-calibration data
                                  (``2``) both calibration and non-calibration data
        Returns:
            list: A list of strings with the name of the schedules in results.
        """
        if isinstance(results, Result):
            results_list = [results]
        else:
            results_list = results
        schedules = []
        for res in results_list:
            for result in res.results:
                self._append_to_schedules(result.header.name,
                                          schedule_type_to_get, schedules)

        return schedules

    def _append_to_schedules(self, name: str, schedule_type: int,
                             schedules: list):
        """
        Helper function to append schedule names.

        Args:
            name: name of the schedule that may be appended to schedules.
            schedule_type: defines which schedule type to include
                           in the returned schedules.
                           (``0``) calibration data only
                           (``1``) non-calibration data
                           (``2``) both calibration and non-calibration data
            schedules (list): a list of schedule names.

        Raises:
            QiskitError: If schedule type is not 0, 1, or 2
        """
        if schedule_type == 0:
            # TODO - fix this
            #if self.is_calibration(name):
            schedules.append(name)
        elif schedule_type == 1:
            if not self.is_calibration(name):
                schedules.append(name)
        elif schedule_type == 2:
            schedules.append(name)
        else:
            raise QiskitError('schedule_type must be either\n'
                              '0: get only calibration schedules\n'
                              '1: get only non-calibration schedules\n'
                              '2: get all schedules in results.')

    @staticmethod
    def _get_expected_states(schedules: Union[List[Schedule], List[str]]) \
            -> List[str]:
        """
        Get the names of the expected_states based on the schedule names by
        replacing the substring 'cal_' in the name with ''. E.g. the name
        'cal_01' becomes '01'.

        Args:
            schedules: a list of schedules or a list of schedule names.
                       These schedules are used to identify the names of
                       the expected states.

        Returns:
            expected_states extracted from the schedules.
        """
        expected_states = []
        for schedule in schedules:
            if isinstance(schedules, Schedule):
                name = schedule.name
            else:
                name = schedule

            expected_states.append(name.replace('cal_', ''))

        return expected_states

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
            xdata: data as a list of features. Each feature is itself a list.
            refit: if true than self._scaler is refit using the given xdata

        Returns:
            the scaled xdata as a list of features.

        Raises:
            ImportError: If sckit-learn is not installed
        """
        if not self._standardize:
            return xdata

        try:
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError("To scale the xdata scikit-learn must be "
                              "installed. This can be done with 'pip install "
                              "scikit-learn'")

        if not self._scaler or refit:
            self._scaler = StandardScaler(with_std=True)
            self._scaler.fit(xdata)

        return self._scaler.transform(xdata)

    def get_xdata(self, results: Union[Result, List[Result]],
                  schedule_type_to_get: int,
                  schedules: Union[List[str], List[Schedule]] = None) \
            -> List[List[float]]:
        """
        Retrieves feature data (xdata) for the discriminator.

        Args:
            results: the get_memory() method is used to retrieve the level 1 data.
                     If result is a list of Result, then the first Result in the
                     list that returns the data of schedule
                     (through get_memory(schedule)) is used.
            schedule_type_to_get: use to specify if we should return data corresponding to
                                  (``0``) calibration data only
                                  (``1``) non-calibration data
                                  (``2``) both calibration and non-calibration data
            schedules: Either the names of the schedules or the schedules themselves.

        Returns:
            data as a list of features. Each feature is a list.

        Raises:
            PulseError: if IQ data could not be found
        """
        xdata = []
        if schedules is None:
            schedules = self._get_schedules(results, schedule_type_to_get)
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
                  schedule_type_to_get: int,
                  schedules: Union[List[str], List[Schedule]] = None):
        """
        Retrieves the expected states (ydata) for the discriminator.

        Args:
            results: results for which to retrieve the y data (i.e. expected states).
            schedule_type_to_get: use to specify if we should return data corresponding to
                                  * 0 calibration data only
                                  * 1 non-calibration data
                                  * 2 both calibration and non-calibration data
            schedules: Either the names of the schedules or the schedules themselves.

        Returns:
            list: The y data, i.e. expected states. get_ydata is designed to produce
                y data with the same length as the x data.
        """
        ydata = []

        if schedules is None:
            schedules = self._get_schedules(results, schedule_type_to_get)

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
    #
    # @abstractmethod
    # def get_xdata(self, results: Union[Result, List[Result]],
    #               schedule_type_to_get: int,
    #               schedules: Union[List[str], List[Schedule]] = None) \
    #         -> List[List[float]]:
    #     """
    #     Retrieves feature data (xdata) for the discriminator.
    #
    #     Args:
    #         results: the get_memory() method is used to retrieve the level 1 data.
    #                  If result is a list of Result, then the first Result in the
    #                  list that returns the data of schedule
    #                  (through get_memory(schedule)) is used.
    #         schedule_type_to_get: use to specify if we should return data corresponding to
    #                               (``0``) calibration data only
    #                               (``1``) non-calibration data
    #                               (``2``) both calibration and non-calibration data
    #         schedules: Either the names of the schedules or the schedules themselves.
    #
    #     Returns:
    #         The xdata as a list of features. Each feature is a list.
    #     """
    #
    # @abstractmethod
    # def get_ydata(self, results: Union[Result, List[Result]],
    #               schedule_type_to_get: int,
    #               schedules: Union[List[str], List[Schedule]] = None) \
    #         -> List[str]:
    #     """
    #     Retrieves the expected states (ydata) for the discriminator.
    #
    #     Args:
    #         results: results for which to retrieve the y data (i.e. expected states).
    #         schedule_type_to_get: use to specify if we should return data corresponding to
    #                               (``0``) calibration data only
    #                               (``1``) non-calibration data
    #                               (``2``) both calibration and non-calibration data
    #         schedules: Either the names of the schedules or the schedules themselves.
    #
    #     Returns:
    #         The y data, i.e. expected states. get_ydata is designed to produce
    #         y data with the same length as the x data.
    #     """

    @abstractmethod
    def fit(self):
        """ Fits the discriminator using self._xdata and self._ydata. """

    @abstractmethod
    def discriminate(self, x_data: List[List[float]]) -> List[str]:
        """
        Applies the discriminator to x_data

        Args:
            x_data (List[List[float]]): list of features. Each feature is
                itself a list.

        Returns (List[str]):
            the discriminated x_data as a list of labels.
        """

    def plot(self, axs=None,
             show_boundary: bool = False,
             show_fitting_data: bool = True,
             flag_misclassified: bool = False,
             qubits_to_plot: list = None,
             title: bool = True):
        """
        Creates a plot of the data used to fit the discriminator.

        Args:
            axs (Union[np.ndarray, axes]): the axis to use for the plot. If it
                is none, the plot method will create its own axis instance. If
                the number of axis instances provided is less than the number
                of qubits then only the data for the first len(axs) qubits will
                be plotted.
            show_boundary (bool): plot the decision regions if true. Some
                discriminators may put additional constraints on whether the
                decision regions are plotted or not.
            show_fitting_data (bool): if True the x data and labels used to
                fit the discriminator are shown in the plot.
            flag_misclassified (bool): plot the misclassified training data
                points if true.
            qubits_to_plot (list): each qubit in this list will receive its
                own plot. The qubits in qubits to plot must be in the qubit
                mask. If qubits_to_plot is None then the qubit mask will be
                used.
            title (bool): adds a title to each subplot with the number of
                the qubit.

        Returns:
            tuple: A tuple of the form: ``(Union[List[axes], axes], figure)``
                where the axes object used for the plot as well as the figure handle.
                The figure handle returned is not None only when the figure handle
                is created by the discriminator's plot method.

        Raises:
            QiskitError: If matplotlib is not installed, or there is
                invalid input
        """
        if not HAS_MATPLOTLIB:
            raise QiskitError('please install matplotlib')

        if qubits_to_plot is None:
            qubits_to_plot = self._qubit_mask
        else:
            for q in qubits_to_plot:
                if q not in self._qubit_mask:
                    raise QiskitError('Qubit %i is not in discriminators '
                                      'qubit mask' % q)

        if axs is None:
            fig, axs = plt.subplots(len(qubits_to_plot), 1)
        else:
            fig = None

        if not isinstance(axs, np.ndarray):
            axs = np.asarray(axs)

        axs = axs.flatten()

        if len(axs) < len(qubits_to_plot):
            raise QiskitError('Not enough axis instances supplied. '
                              'Please provide one per qubit discriminated.')

        # If only one qubit is present then draw the discrimination region.
        if show_boundary and len(self._qubit_mask) != 1:
            raise QiskitError('Background can only be plotted for individual '
                              'qubit discriminators. Qubit mask has length '
                              '%i != 1' % len(self._qubit_mask))

        x_data = np.array(self._xdata)
        y_data = np.array(self._ydata)

        if show_boundary and len(self._qubit_mask) == 1:
            try:
                xx, yy = self._get_iq_grid(x_data)
                zz = self.discriminate(np.c_[xx.ravel(), yy.ravel()])
                zz = np.array(zz).astype(float).reshape(xx.shape)
                axs[0].contourf(xx, yy, zz, alpha=.2)

            except ValueError:
                raise QiskitError('Cannot convert expected state labels to '
                                  'float.')

        n_qubits = len(self._qubit_mask)
        if show_fitting_data:
            for idx, q in enumerate(qubits_to_plot):
                q_idx = self._qubit_mask.index(q)
                ax = axs[idx]

                # Different results may have the same expected state.
                # First merge all the data with the same expected state.
                data = {}
                for _, exp_state in self.expected_states.items():

                    if exp_state not in data:
                        data[exp_state] = {'I': [], 'Q': []}

                    dat = x_data[y_data == exp_state]
                    data[exp_state]['I'].extend(dat[:, q_idx])
                    data[exp_state]['Q'].extend(dat[:, n_qubits + q_idx])

                # Plot the data by expected state.
                for exp_state in data:
                    ax.scatter(data[exp_state]['I'], data[exp_state]['Q'],
                               label=exp_state, alpha=0.5)

                    if flag_misclassified:
                        y_disc = np.array(self.discriminate(self._xdata))

                        misclassified = x_data[y_disc != y_data]
                        ax.scatter(misclassified[:, q_idx],
                                   misclassified[:, n_qubits + q_idx],
                                   color='r', alpha=0.5, marker='x')

                ax.legend(frameon=True)

        if title:
            for idx, q in enumerate(qubits_to_plot):
                axs[idx].set_title('Qubit %i' % q)

        for ax in axs:
            ax.set_xlabel('I (arb. units)')
            ax.set_ylabel('Q (arb. units)')

        return axs, fig

    def plot_xdata(self, axs,
                   results: Union[Result, List[Result]], color: str = None):
        """
        Add the relevant IQ data from the Qiskit Result, or list thereof, to
        the given axes as a scatter plot.

        Args:
            axs (Union[np.ndarray, axes]): the axis to use for the plot. If
                the number of axis instances provided is less than the number
                of qubits then only the data for the first len(axs) qubits will
                be plotted.
            results (Union[Result, List[Result]]): the discriminators
                get_xdata will be used to retrieve the x data from the Result
                or list of Results.
            color (str): color of the IQ points in the scatter plot.

        Raises:
            QiskitError: If not enough axis instances are provided
        """
        if not isinstance(axs, np.ndarray):
            axs = np.asarray(axs)

        axs = axs.flatten()

        if color is None:
            color = 'C2'

        n_qubits = len(self._qubit_mask)
        if len(axs) < n_qubits:
            raise QiskitError('Not enough axis instances supplied. '
                              'Please provide one per qubit discriminated.')

        x_data = self.get_xdata(results, 1)
        data = np.array(x_data)

        for idx in range(n_qubits):
            axs[idx].scatter(data[:, idx], data[:, n_qubits + idx], alpha=0.5,
                             color=color)

    def format_iq_data(self, iq_data: np.ndarray) -> List[List[float]]:
        """
        Takes IQ data obtained from get_memory(), applies the qubit mask
        and formats the data as a list of lists. Each sub list is IQ data
        where the first half of the list is the I data and the second half of
        the list is the Q data.

        Args:
            iq_data (np.ndarray): data obtained from get_memory().

        Returns:
            A list of shots where each entry is a list of IQ points.

        Raises:
            PulseError: if the measurement return type is unknown
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

    #
    # @abstractmethod
    # def plot(self, axs=None,
    #          show_boundary: bool = False,
    #          show_fitting_data: bool = True,
    #          flag_misclassified: bool = False,
    #          qubits_to_plot: list = None,
    #          title: bool = True):
    #     """
    #     Creates a plot of the data used to fit the discriminator.
    #
    #     Args:
    #         axs (Union[ndarray, axes]): the axis to use for the plot. If it
    #             is none, the plot method will create its own axis instance. If
    #             the number of axis instances provided is less than the number
    #             of qubits then only the data for the first len(axs) qubits will
    #             be plotted.
    #         show_boundary (bool): plot the decision regions if true. Some
    #             discriminators may put additional constraints on whether the
    #             decision regions are plotted or not.
    #         show_fitting_data (bool): if True the x data and labels used to
    #             fit the discriminator are shown in the plot.
    #         flag_misclassified (bool): plot the misclassified training data
    #             points if true.
    #         qubits_to_plot (list): each qubit in this list will receive its
    #             own plot. The qubits in qubits to plot must be in the qubit
    #             mask. If qubits_to_plot is None then the qubit mask will be
    #             used.
    #         title (bool): adds a title to each subplot with the number of
    #             the qubit.
    #
    #     Returns: (Union[List[axes], axes], figure):
    #         the axes object used for the plot as well as the figure handle.
    #         The figure handle returned is not ``None`` only when the figure
    #         handle is created by the discriminator's plot method.
    #     """
    #
    # @abstractmethod
    # def plot_xdata(self, axs,
    #                results: Union[Result, List[Result]], color: str = None):
    #     """
    #     Add the relevant IQ data from the Qiskit Result, or list thereof, to
    #     the given axes as a scatter plot.
    #
    #     Args:
    #         axs (Union[ndarray, axes]): the axis to use for the plot. If
    #             the number of axis instances provided is less than the number
    #             of qubits then only the data for the first len(axs) qubits will
    #             be plotted.
    #         results (Union[Result, List[Result]]): the discriminators
    #             get_xdata will be used to retrieve the x data from the Result
    #             or list of Results.
    #         color (str): color of the IQ points in the scatter plot.
    #     """
