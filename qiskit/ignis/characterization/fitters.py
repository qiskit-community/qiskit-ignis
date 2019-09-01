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
Fitters of characteristic times
"""

from scipy.optimize import curve_fit
import numpy as np
from qiskit import QiskitError
from ..verification.tomography import marginal_counts


class BaseFitter:
    """
    Base class for a data fitter
    """

    def __init__(self, description, backend_result, xdata,
                 qubits, fit_fun, fit_p0,
                 fit_bounds, circuit_names,
                 series=None, expected_state='0'):
        """
        Args:
           description: a string describing the fitter's purpose, e.g. 'T1'
           backend_result: a qiskit.result or list of results
           xdata: a list of the independent parameter
               (which will be fit against).
           qubits: the qubits for which we measured coherence
           fit_fun, fit_p0, fit_bounds: equivalent to parameters of
           scipy.curve_fit.
           circuit_names: names of the circuits, should be the same length
           as xdata. Full circuit name will be these plus the
           series name
           series: list of circuit name tags
           expected_state: is the circuit supposed to end up in '0' or '1'?
        """

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
    def description(self):
        """
        Return the fitter's purpose, e.g. 'T1'
        """
        return self._description

    @property
    def backend_result(self):
        """
        Return the execution results (qiskit.Result)
        """
        return self._backend_result

    @property
    def series(self):
        """
        Return the list of series for the data
        """
        return self._series

    @property
    def measured_qubits(self):
        """
        Return the indices of the qubits whose characteristic time is measured
        """
        return self._qubits

    @property
    def xdata(self):
        """
        Return the data points on the x-axis (a list of floats)
        """
        return self._xdata

    @property
    def ydata(self):
        """
        Return the data points on the y-axis
        In the form of a list of dictionaries:
        - ydata[i]['mean'] is a list, where item
            no. j is the probability of success
            of qubit i for a circuit that lasts xdata[j].
        - ydata[i]['std'] is a list, where ydata['std'][j] is the
            standard deviation of the success of qubit i.
        """
        return self._ydata

    @property
    def fit_fun(self):
        """
        Return the function used in the fit,
        e.g. BaseFitter._exp_fit_fun
        """
        return self._fit_fun

    @property
    def params(self):
        """
        Return the fit function parameters that were calculated by curve_fit
        """
        return self._params

    @property
    def params_err(self):
        """
        Return the error of the fit function parameters
        """
        return self._params_err

    def _get_param(self, param_ind, qid=-1, series='0', err=False):
        """
        Helper function that gets a parameter (or parameter err)
        if qid=-1 returns a list of the parameters for all qubits

        Args:
            param_ind: the parameter index to get
            qid: the qubit index (or all qubits if -1)
            series: the series to get
            err: get param or param err
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

    def add_data(self, results, recalc=True, refit=True):
        """
        Adds more data

        Args:
            results: a result (qiskit.result) or list of results
            recalc: Recalculate the data
            refit: Refit the data
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

    def fit_data(self, qid=-1, p0=None, bounds=None, series=None):
        """
        Fit the curve.
        Computes self._params and self._params_err:
        Args:
            qid: Qubit data to fit. If -1 fit all the data
            p0: initial guess
            bounds: bounds
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
        Function used to fit the exponential decay
        """

        return a * np.exp(-x / tau) + c

    @staticmethod
    def _osc_fit_fun(x, a, tau, f, phi, c):
        """
        Function used to fit the decay cosine
        """

        return a * np.exp(-x / tau) * np.cos(2 * np.pi * f * x + phi) + c

    @staticmethod
    def _osc_nodecay_fit_fun(x, a, f, phi, c):
        """
        Function used to fit the decay cosine
        """

        return a * np.cos(2 * np.pi * f * x + phi) + c

    @staticmethod
    def _cal_fit_fun(x, a, thetaerr, phierr, theta0, phi0, c):
        """
        Function used to fit gate calibrations
        """

        return a*np.cos((theta0+thetaerr) * x + phi0 + phierr) + c


class BaseCoherenceFitter(BaseFitter):
    """
    Base class for fitters of characteristic times
    """

    def __init__(self, description, backend_result, xdata,
                 qubits, fit_fun, fit_p0,
                 fit_bounds, circuit_names,
                 series=None, expected_state='0',
                 time_index=0, time_unit='micro-seconds'):

        """
        See BaseFitter __init__

        Args:
           time_index: fit parameter corresponding to the characteristic time
        """

        BaseFitter.__init__(self, description,
                            backend_result, xdata,
                            qubits, fit_fun,
                            fit_p0, fit_bounds, circuit_names,
                            series, expected_state)

        self._time_index = time_index
        self._time_unit = time_unit

    def time(self, qid=-1, series='0'):
        """
        Return the characteristic time for qid and series
        If qid==-1 return all the qubit data
        """

        return self._get_param(self._time_index, qid, series)

    def time_err(self, qid=-1, series='0'):
        """
        Return the error of the characteristic time
        """
        return self._get_param(self._time_index,
                               qid, series, err=True)

    def plot(self, qind, series, ax=None, show_plot=True):
        """
        Plot coherence data.

        Args:
            qind: qubit index to plot
            series: which series to plot (if list plots multiple)
            ax: plot axes
            show_plot: call plt.show()

        return the axes object
        """

        from matplotlib import pyplot as plt

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
            qind: qubit index to plot
            ax: plot axes
            show_plot: call plt.show()

        return the axes object
        """

        from matplotlib import pyplot as plt

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


def build_counts_dict_from_list(count_list):

    """
    Add dictionary counts together

    """

    if len(count_list) == 1:
        return count_list[0]

    new_count_dict = {}
    for countdict in count_list:
        for x in countdict:
            new_count_dict[x] = countdict[x]+new_count_dict.get(x, 0)

    return new_count_dict
