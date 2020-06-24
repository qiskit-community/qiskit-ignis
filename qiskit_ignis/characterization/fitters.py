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
Fitters of characteristic times
"""

from typing import Union, List, Callable, Optional, Tuple, Dict, Any
from scipy.optimize import curve_fit
import numpy as np
from qiskit import QiskitError
from qiskit.result import Result
from ..verification.tomography import marginal_counts
from ..utils import build_counts_dict_from_list

try:
    from matplotlib import pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class BaseFitter:
    """
    Base class for a data fitter

    Args:
        description: description of the fitter's purpose, e.g. 'T1'.
        backend_result: result of execution on the backend.
        xdata: a list of the independent parameter
               (which will be fit against).
        qubits: the qubits to be characterized.
        fit_fun: equivalent to parameter `f` of scipy.curve_fit.
        fit_p0: equivalent to parameter `p0` of scipy.curve_fit.
        fit_bounds: equivalent to parameter `bounds` of scipy.curve_fit.
        circuit_names: names of the circuits, should be the same length
                       as `xdata`. Full circuit name will be these plus the
                       series name.
        series: list of circuit name tags
        expected_state: is the circuit supposed to end up in '0' or '1'?
    """

    def __init__(self, description: str,
                 backend_result: Union[Result, List[Result]],
                 xdata: Union[List[float], np.array],
                 qubits: List[int],
                 fit_fun: Callable[..., float],
                 fit_p0: List[float],  # any way to enforce length 3?
                 fit_bounds: Tuple[List[float], List[float]],
                 circuit_names: List[str],
                 series: Optional[List[str]] = None,
                 expected_state: str = '0'):

        if fit_bounds is None:
            fit_bounds = ([-np.inf for e in range(len(fit_p0))],
                          [np.inf for e in range(len(fit_p0))])

        if series is None:
            self._series = ['0']
        else:
            self._series = series

        self._circuit_names = circuit_names

        self._backend_result_list = []
        autofit = False

        if backend_result is not None:
            autofit = True
            if isinstance(backend_result, list):
                for result in backend_result:
                    self._backend_result_list.append(result)
            else:
                self._backend_result_list.append(backend_result)

        self._description = description
        self._expected_state = expected_state
        self._qubits = qubits

        self._defaultp0 = fit_p0
        self._default_bounds = fit_bounds

        self._xdata = xdata
        self._fit_fun = fit_fun

        # initialize the fit parameter lists
        self._params = {i: [[] for j in
                            range(len(self._qubits))] for i in self._series}

        self._params_err = {i: [[] for j in
                                range(len(self._qubits))]
                            for i in self._series}

        if autofit:
            self._calc_data()
            self.fit_data()

    @property
    def description(self) -> str:
        """
        Return the fitter's purpose, e.g. 'T1'
        """
        return self._description

    @property
    def backend_result(self) -> Union[Result, List[Result]]:
        """
        Return the execution results
        """
        return self._backend_result_list

    @property
    def series(self) -> Optional[List[str]]:
        """
        Return the list of series for the data
        """
        return self._series

    @property
    def measured_qubits(self) -> List[int]:
        """
        Return the indices of the qubits to be characterized
        """
        return self._qubits

    @property
    def xdata(self) -> Union[List[float], np.array]:
        """
        Return the data points on the x-axis, the independenet
        parameter which is fit against
        """
        return self._xdata

    @property
    def ydata(self) -> List[Dict]:
        """Return the data points on the y-axis

        The data points are returning in the form of a list of dictionaries:

         * ydata[i]['mean'] is a list, where item
             no. j is the probability of success
             of qubit i for a circuit that lasts xdata[j].
         * ydata[i]['std'] is a list, where ydata['std'][j] is the
             standard deviation of the success of qubit i.
        """
        return self._ydata

    @property
    def fit_fun(self) -> Callable:
        """
        Return the function used in the fit,
        e.g. BaseFitter._exp_fit_fun
        """
        return self._fit_fun

    @property
    def params(self) -> List[float]:
        """
        Return the fit function parameters that were calculated by curve_fit
        """
        return self._params

    @property
    def params_err(self) -> List[float]:
        """
        Return the error of the fit function parameters
        """
        return self._params_err

    def _get_param(self,
                   param_ind: int,
                   qid: int = -1,
                   series: str = '0',
                   err: bool = False) -> Union[float, List[float]]:
        """
        Return the fitted value, or fitting error, of a given
        parameter and a given qubit

        Args:
            param_ind: the parameter index to get
            qid: the qubit index (or all qubits if -1)
            series: the series to get
            err: get param or param err

        Returns:
            The fitted value or error
        """

        if qid != -1:
            if err:
                return self._params_err[series][qid][param_ind]

            return self._params[series][qid][param_ind]

        param_list = []
        for qind, _ in enumerate(self._qubits):
            if err:
                param_list.append(self._params_err
                                  [series][qind][param_ind])
            else:
                param_list.append(self._params[series][qind][param_ind])

        return param_list

    def add_data(self,
                 results: Union[Result, List[Result]],
                 recalc: bool = True,
                 refit: bool = True):
        """
        Add new execution results to previous execution results

        Args:
            results: new execution results
            recalc: whether tp recalculate the data
            refit: whether to refit the data
        """

        if isinstance(results, list):
            for result in results:
                self._backend_result_list.append(result)
        else:
            self._backend_result_list.append(results)

        if recalc:
            self._calc_data()  # computes self._ydata

        if refit:
            self.fit_data()

    def _calc_data(self):
        """
        Retrieve probabilities of success from execution results, i.e.,
        probability to measure a state where all qubits are 0.
        Computes a list of dictionaries, see documentation of property ydata.

        Go through all results in the list, i.e., can split
        the run over several jobs or use jobs to build more statistics

        """

        circ_counts = {}
        for _, serieslbl in enumerate(self._series):
            for circ, _ in enumerate(self._xdata):
                circname = self._circuit_names[circ] + serieslbl
                count_list = []
                for result in self._backend_result_list:
                    try:
                        count_list.append(result.get_counts(circname))
                    except (QiskitError, KeyError):
                        pass

                circ_counts[circname] = \
                    build_counts_dict_from_list(count_list)

        self._ydata = {}
        for _, serieslbl in enumerate(self._series):
            self._ydata[serieslbl] = []
            for qind, _ in enumerate(self._qubits):
                self._ydata[serieslbl].append({'mean': [], 'std': []})
                for circ, _ in enumerate(self._xdata):
                    circname = self._circuit_names[circ] + serieslbl
                    shots = sum(circ_counts[circname].values())
                    counts_subspace = \
                        marginal_counts(circ_counts[circname], [qind])
                    success_prob = \
                        counts_subspace.get(self._expected_state, 0) / shots
                    self._ydata[serieslbl][-1]['mean'].append(success_prob)
                    self._ydata[serieslbl][-1]['std'].append(
                        np.sqrt(success_prob * (1-success_prob) / shots))
                    # problem for the fitter if one of the std points is
                    # exactly zero
                    if self._ydata[serieslbl][-1]['std'][-1] == 0:
                        self._ydata[serieslbl][-1]['std'][-1] = 1e-4

    def fit_data(self,
                 qid: int = -1,
                 p0: Optional[List[float]] = None,
                 bounds: Optional[Tuple[List[float], List[float]]] = None,
                 series: Optional[str] = None):
        """
        Fit the curve.

        Compute self._params and self._params_err

        Args:
            qid: qubit for fitting. If -1 fit for all the qubits
            p0: initial guess, equivalent to `p0` in `scipy.optimize`
            bounds: bounds, equivalent to `bounds` in `scipy.optimize`
            series: series to fit (if None fit all)
        """

        if series is None:
            series = self._series.copy()

        if not isinstance(series, list):
            series = [series]

        if qid == -1:
            qfit = range(len(self._qubits))
        else:
            qfit = [qid]

        if bounds is None:
            bounds = self._default_bounds

        if p0 is None:
            p0 = self._defaultp0

        for _, serieslbl in enumerate(series):
            for qind in qfit:
                tmp_params, fcov = \
                     curve_fit(self._fit_fun, self._xdata,
                               self._ydata[serieslbl][qind]['mean'],
                               sigma=self._ydata[serieslbl][qind]['std'],
                               p0=p0, bounds=bounds)

                self._params[serieslbl][qind] = tmp_params.copy()
                self._params_err[serieslbl][qind] = np.sqrt(np.diag(fcov))

    @staticmethod
    def _exp_fit_fun(x, a, tau, c):
        """
        Exponential decay
        """

        return a * np.exp(-x / tau) + c

    @staticmethod
    def _osc_fit_fun(x, a, tau, f, phi, c):
        """
        Decay cosine
        """

        return a * np.exp(-x / tau) * np.cos(2 * np.pi * f * x + phi) + c

    @staticmethod
    def _osc_nodecay_fit_fun(x, a, f, phi, c):
        """
        Oscilliator cosine
        """

        return a * np.cos(2 * np.pi * f * x + phi) + c

    @staticmethod
    def _cal_fit_fun(x, a, thetaerr, phierr, theta0, phi0, c):
        """
        Gate calibrations fitting function
        """

        return a*np.cos((theta0+thetaerr) * x + phi0 + phierr) + c

    @staticmethod
    def _quadratic(x, a, x0, c):
        """
        Drag fitting function
        """

        return a * (x-x0)**2 + c


class IQFitter(BaseFitter):
    """
    Base Fitter Class for experiments with Level 1 results
    """

    def __init__(self, description, backend_result, xdata,
                 qubits, fit_fun, fit_p0,
                 fit_bounds, circuit_names,
                 series=None):

        """
        See BaseFitter __init__
        """

        BaseFitter.__init__(self, description,
                            backend_result, xdata,
                            qubits, fit_fun,
                            fit_p0, fit_bounds, circuit_names,
                            series)

    def _build_iq_list(self):
        """
        From a list of results, calculate the mean for each
        experiment and circuit/program

        Returns:
            tuple: A tuple of the form (``iq_list_mean``, ``iq_list_var``) where
                ``iq_list_mean`` is a list of the mean of the iq data
                ``iq_list_var`` is a list of the variance of the iq data (if
                single shot IQ)

        Raises:
            QiskitError: invalid input
        """

        iq_list = {}
        iq_list_mean = {}
        iq_list_var = {}
        shots_list = {}
        meas_ret = ''

        for single_result in self._backend_result_list:
            # go through each of the schedules in this run
            for result in single_result.results:
                sname = result.header.name
                mem_slots = result.header.memory_slots

                if meas_ret == '':
                    # update
                    if result.meas_return == 'avg':
                        meas_ret = 'avg'
                    else:
                        meas_ret = 'single'

                if result.meas_level != 1:
                    raise QiskitError('This fitter works with IQ data')

                if meas_ret != result.meas_return:
                    raise QiskitError('All data must be single shot or '
                                      'averaged')

                shots_list[sname] = (result.shots +
                                     shots_list.get(sname, 0))

                if meas_ret == 'avg':

                    # data is averaged already, but if we need to
                    # further average over different runs
                    # we need to take into account the full number of shots

                    iq_list[sname] = (result.shots *
                                      single_result.get_memory(sname) +
                                      iq_list.get(sname, np.zeros(mem_slots)))

                else:

                    # data is in single shot IQ mode
                    iq_list[sname] = (result.shots *
                                      single_result.get_memory(sname) +
                                      iq_list.get(sname,
                                                  np.zeros([result.shots,
                                                            mem_slots])))

        for sname in iq_list:

            if meas_ret == 'avg':
                iq_list_mean[sname] = iq_list[sname]/shots_list[sname]
                iq_list_var[sname] = np.ones(len(iq_list_mean[sname]))*1e-4
            else:
                iq_list_mean[sname] = np.sum(iq_list[sname])/shots_list[sname]
                iq_list_var[sname] = np.std(iq_list[sname])

        return iq_list_mean, iq_list_var

    def _calc_data(self):
        """
        Take the IQ values from the list of results, get the mean
        values and then project onto a line in IQ space to give
        the maximum signal. Load into the _ydata which is the mean
        and variance as a single float.

        Overloaded from BaseFitter._calc_data which assumed shots
        """

        iq_list_mean, iq_list_var = self._build_iq_list()

        self._ydata = {}
        for _, serieslbl in enumerate(self._series):
            self._ydata[serieslbl] = []
            for qind, _ in enumerate(self._qubits):
                self._ydata[serieslbl].append({'mean': [], 'std': []})
                mean_list = self._ydata[serieslbl][-1]['mean']
                var_list = self._ydata[serieslbl][-1]['std']
                for circ, _ in enumerate(self._xdata):
                    circname = self._circuit_names[circ] + serieslbl
                    mean_list.append(iq_list_mean[circname][qind])
                    var_list.append(iq_list_var[circname][qind])
                    # problem for the fitter if one of the std points is
                    # exactly zero
                    if var_list[-1] == 0:
                        var_list[-1] = 1e-4

                # project the data onto a line
                # mean over all the experiment
                q_exp_mean = np.mean(mean_list)
                mean_list -= q_exp_mean
                real_ext = np.mean(np.abs(np.real(mean_list)))
                imag_ext = np.mean(np.abs(np.imag(mean_list)))
                crot = real_ext/(real_ext**2+imag_ext**2)**0.5
                srot = imag_ext/(real_ext**2+imag_ext**2)**0.5
                mean_list = crot*np.real(mean_list)+srot*np.imag(mean_list)
                self._ydata[serieslbl][-1]['mean'] = mean_list

    def plot(self, qind, series='0', ax=None, show_plot=True):
        """
        Plot calibration data.

        Args:
            qind (int): qubit index to plot
            series (str): The series to plot
            ax (Axes): plot axes
            show_plot (bool): call plt.show()

        Returns:
            Axes: The axes object

        Raises:
            ImportError: if matplotlib is not installed
        """

        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib must be installed and properly "
                              "configured to use this method. To install you "
                              "can run: 'pip install matplotlib'")

        if ax is None:
            plt.figure()
            ax = plt.gca()

        ax.errorbar(self._xdata, self._ydata[series][qind]['mean'],
                    self._ydata[series][qind]['std'],
                    marker='.', markersize=9,
                    c='b', linestyle='')
        ax.plot(self._xdata, self._fit_fun(self._xdata,
                                           *self._params[series][qind]),
                c='r', linestyle='--',
                label='Q%d' % (self._qubits[qind]))

        ax.tick_params(axis='x', labelsize=14, labelrotation=70)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_xlabel('X', fontsize=16)
        ax.set_ylabel('IQ Signal', fontsize=16)
        ax.set_title(self._description + ' for qubit ' +
                     str(self._qubits[qind]), fontsize=18)
        ax.legend(fontsize=12)
        ax.grid(True)
        if show_plot:
            plt.show()

        return ax


class BaseCoherenceFitter(BaseFitter):
    """
    Base class for fitters of characteristic times

    Args:
        description: description of the fitter's purpose, e.g. 'T1'.
        backend_result: result of execution on the backend.
        xdata: delay times of the circuits.
        qubits: the qubits to be characterized.
        fit_fun: equivalent to parameter `f` of scipy.curve_fit.
        fit_p0: equivalent to parameter `p0` of scipy.curve_fit.
        fit_bounds: equivalent to parameter `bounds` of scipy.curve_fit.
        circuit_names: names of the circuits, should be the same length
                       as `xdata`. Full circuit name will be these plus the
                       series name.
        series: list of circuit name tags
        expected_state: is the circuit supposed to end up in '0' or '1'?
        time_index: among parameters of `fit_fun`,
                    which one is the characteristic time.
        time_unit: unit of delay times in `xdata`.
    """

    def __init__(self, description: str,
                 backend_result: Union[Result, List[Result]],
                 xdata: Union[List[float], np.array],
                 qubits: List[int],
                 fit_fun: Callable[..., float],
                 fit_p0: List[float],  # any way to enforce length 3?
                 fit_bounds: Tuple[List[float], List[float]],
                 circuit_names: List[str],
                 series: Optional[List[str]] = None,
                 expected_state: str = '0',
                 time_index: int = 0,
                 time_unit: str = 'micro-seconds'):

        BaseFitter.__init__(self, description,
                            backend_result, xdata,
                            qubits, fit_fun,
                            fit_p0, fit_bounds, circuit_names,
                            series, expected_state)

        self._time_index = time_index
        self._time_unit = time_unit

    def time(self,
             qid: int = -1,
             series: str = '0') -> Union[float, List[float]]:
        """
        Return the characteristic time for the given qubit and series

        Args:
           qid: the qubit index (or all qubits if -1)
           series: the series to get

        Returns:
           The characteristic time of the qubit, or all qubits
        """

        return self._get_param(self._time_index, qid, series)

    def time_err(self,
                 qid: int = -1,
                 series: str = '0') -> Union[float, List[float]]:
        """
        Return the error of characteristic time for the given qubit and series

        Args:
           qid: the qubit index (or all qubits if -1)
           series: the series to get

        Returns:
            The error of the characteristic time of the qubit, or all qubits
        """

        return self._get_param(self._time_index,
                               qid, series, err=True)

    def plot(self,
             qind: int,
             series: str,
             ax: Optional[Any] = None,
             show_plot: bool = True) -> Any:
        """
        Plot coherence data.

        Args:
            qind: qubit index to plot
            series: which series to plot (if list then plot multiple)
            ax: plot axes
            show_plot: whether to call plt.show()

        Returns:
            Axes: The axes object

        Raises:
            ImportError: if matplotlib is not installed
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib must be installed and properly "
                              "configured to use this method. To install you "
                              "can run: 'pip install matplotlib'")

        if ax is None:
            plt.figure()
            ax = plt.gca()

        ax.errorbar(self._xdata, self._ydata[series][qind]['mean'],
                    self._ydata[series][qind]['std'],
                    marker='.', markersize=9, c='b', linestyle='')
        ax.plot(self._xdata, self._fit_fun(self._xdata,
                                           *self._params[series][qind]),
                c='r', linestyle='--',
                label=self._description + ': ' +
                str(np.around(self.time(qid=qind), 1)) + ' ' + self._time_unit)

        ax.tick_params(axis='x', labelsize=14, labelrotation=70)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_xlabel('Time [' + self._time_unit + ']', fontsize=16)
        ax.set_ylabel('Probability of success', fontsize=16)
        ax.set_title(self._description + ' for qubit ' +
                     str(self._qubits[qind]), fontsize=18)
        ax.legend(fontsize=12)
        ax.grid(True)
        if show_plot:
            plt.show()

        return ax


class BaseGateFitter(BaseFitter):
    """
    Base class for fitters of gate errors
    """

    def __init__(self, description, backend_result, xdata,
                 qubits, fit_fun, fit_p0,
                 fit_bounds, circuit_names,
                 series=None, expected_state='0'):

        """
        See BaseFitter __init__

        """

        BaseFitter.__init__(self, description,
                            backend_result, xdata,
                            qubits, fit_fun,
                            fit_p0, fit_bounds, circuit_names,
                            series, expected_state)

    def plot(self, qind, series='0', ax=None, show_plot=True):
        """
        Plot err data.

        Args:
            qind (int): qubit index to plot
            series (str): the series to plot
            ax (Axes): plot axes
            show_plot (bool): call plt.show()

        Returns:
            Axes: The axes object

        Raises:
            ImportError: if matplotlib is not installed
        """

        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib must be installed and properly "
                              "configured to use this method. To install you "
                              "can run: 'pip install matplotlib'")
        if ax is None:
            plt.figure()
            ax = plt.gca()

        ax.errorbar(self._xdata, self._ydata[series][qind]['mean'],
                    self._ydata[series][qind]['std'],
                    marker='.', markersize=9,
                    c='b', linestyle='')
        ax.plot(self._xdata, self._fit_fun(self._xdata,
                                           *self._params[series][qind]),
                c='r', linestyle='--',
                label='Q%d' % (self._qubits[qind]))

        ax.tick_params(axis='x', labelsize=14, labelrotation=70)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_xlabel('Number of Gate Repetitions', fontsize=16)
        ax.set_ylabel('Excited state population', fontsize=16)
        ax.set_title(self._description + ' for qubit ' +
                     str(self._qubits[qind]), fontsize=18)
        ax.legend(fontsize=12)
        ax.grid(True)
        if show_plot:
            plt.show()

        return ax
