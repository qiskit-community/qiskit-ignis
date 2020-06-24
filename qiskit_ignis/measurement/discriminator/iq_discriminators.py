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

# pylint: disable=invalid-name

"""
IQ Discriminator module to discriminate date in the IQ Plane.
"""
from abc import abstractmethod
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
try:
    from matplotlib import pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class IQDiscriminationFitter(BaseDiscriminationFitter):
    """
    Abstract discriminator that implements the data formatting for IQ
    level 1 data.
    """

    def __init__(self, cal_results: Union[Result, List[Result]],
                 qubit_mask: List[int], expected_states: List[str] = None,
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

    @staticmethod
    def _get_iq_grid(x_data: np.array) -> (np.meshgrid, np.meshgrid):
        """
        Create mesh grids used to plot the decision boundary.

        Args:
            x_data (np.array): IQ data.
        Returns:
            xx (np.meshgrid): xx meshgrid for plotting discriminator boundary
            yy (np.meshgrid): yy meshgrid for plotting discriminator boundary
        """
        max_i = np.max(x_data[:, 0])
        min_i = np.min(x_data[:, 0])
        max_q = np.max(x_data[:, 1])
        min_q = np.min(x_data[:, 1])

        spacing = (max_i - min_i) / 100.0
        xx, yy = np.meshgrid(
            np.arange(min_i - 10 * spacing, max_i + 10 * spacing, spacing),
            np.arange(min_q - 10 * spacing, max_q + 10 * spacing, spacing))

        return xx, yy

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

    @abstractmethod
    def fit(self):
        """Fits the discriminator using self._xdata and self._ydata."""

    @abstractmethod
    def discriminate(self, x_data: List[List[float]]) -> List[str]:
        """Applies the discriminator to x_data.

        Args:
            x_data (List[List[float]]): list of features. Each feature is
                                        itself a list.

        Returns (List[str]):
            The discriminated x_data as a list of labels.
        """


class LinearIQDiscriminator(IQDiscriminationFitter):
    """Linear discriminant analysis discriminator for IQ data."""

    def __init__(self, cal_results: Union[Result, List[Result]],
                 qubit_mask: List[int], expected_states: List[str] = None,
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
        """Fits the discriminator using self._xdata and self._ydata."""
        if len(self._xdata) == 0:
            return

        self._lda.fit(self._xdata, self._ydata)
        self._fitted = True

    def discriminate(self, x_data: List[List[float]]) -> List[str]:
        """Applies the discriminator to x_data.

        Args:
            x_data (List[List[float]]): list of features. Each feature is
                                        itself a list.

        Returns:
            The discriminated x_data as a list of labels.
        """
        return self._lda.predict(x_data)


class QuadraticIQDiscriminator(IQDiscriminationFitter):
    """Quadratic discriminant analysis discriminator for IQ data."""

    def __init__(self, cal_results: Union[Result, List[Result]],
                 qubit_mask: List[int], expected_states: List[str] = None,
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

        self.fit()

    def fit(self):
        """Fits the discriminator using self._xdata and self._ydata."""
        if len(self._xdata) == 0:
            return

        self._qda.fit(self._xdata, self._ydata)
        self._fitted = True

    def discriminate(self, x_data: List[List[float]]) -> List[str]:
        """Applies the discriminator to x_data.

        Args:
            x_data (List[List[float]]): list of features. Each feature is
                                        itself a list.

        Returns:
            The discriminated x_data as a list of labels.
        """
        return self._qda.predict(x_data)


class SklearnIQDiscriminator(IQDiscriminationFitter):
    """
    A generic discriminant analysis discriminator for IQ data that
    takes an sklearn classifier as an argument.
    """

    def __init__(self, classifier, cal_results: Union[Result, List[Result]],
                 qubit_mask: List[int], expected_states: List[str] = None,
                 standardize: bool = False,
                 schedules: Union[List[str], List[Schedule]] = None):
        """
        Args:
            classifier (Classifier):
                An sklearn classifier to train and do the discrimination. The
                classifier must have a fit method and a predict method
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
        self._type_check_classifier(classifier)
        self._classifier = classifier

        # Also sets the x and y data.
        IQDiscriminationFitter.__init__(self, cal_results, qubit_mask,
                                        expected_states, standardize,
                                        schedules)

        self._description = (
            '{} IQ discriminator for measurement level 1.'.format(
                classifier.__class__.__name__))

        self.fit()

    @staticmethod
    def _type_check_classifier(classifier):
        for name in ['fit', 'predict']:
            if not callable(getattr(classifier, name, None)):
                raise QiskitError(
                    'Classifier of type "{}" does not have a callable "{}"'
                    ' method.'.format(type(classifier).__name__, name)
                )

    def fit(self):
        """ Fits the discriminator using self._xdata and self._ydata. """
        if len(self._xdata) == 0:
            return

        self._classifier.fit(self._xdata, self._ydata)
        self._fitted = True

    def discriminate(self, x_data: List[List[float]]) -> List[str]:
        """Applies the discriminator to x_data.

        Args:
            x_data (List[List[float]]): list of features. Each feature is
                                        itself a list.

        Returns:
            the discriminated x_data as a list of labels.
        """
        return self._classifier.predict(x_data)
