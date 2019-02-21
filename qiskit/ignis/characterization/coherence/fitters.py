# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Fitters of characteristic times
"""

from scipy.optimize import curve_fit
import numpy as np
from qiskit.tools.qcvv.tomography import marginal_counts


class BaseCoherenceFitter:
    """
    Base class for fitters of characteristic times
    """

    def __init__(self, description,
                 backend_result, shots, xdata,
                 qubits, fit_fun, fit_p0,
                 fit_bounds, expected_state = '0'):
        """
        Args:
           description: a string describing the fitter's purpose, e.g. 'T1'
           backend_result: result of backend execution (qiskit.Result).
           xdata: a list of times in micro-seconds.
           qubits: the qubits for which we measured coherence
           fit_fun, fit_p0, fir_bounds: equivalent to parameters of scipy.curve_fit.
           expected_state: is the circuit supposed to end up in '0' or '1'?
        """

        self._description = description
        self._backend_result = backend_result
        self._shots = shots
        self._expected_state = expected_state

        self._qubits = qubits

        self._xdata = xdata
        self._calc_data()  # computes self._ydata

        self._fit_fun = fit_fun
        self._calc_fit(fit_p0, fit_bounds)  # computes self._params and self._params_err


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
    def shots(self):
        """
        Return the number of shots in the execution
        """
        return self._shots

    @property
    def measured_qubit(self):
        """
        Return the index of the qubit whose characteristic time is measured
        """
        return self._qubit

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
        - ydata[i]['mean'] is a list, where item no. j is the probability of success
                           of qubit i for a circuit that lasts xdata[j].
        - ydata[i]['std'] is a list, where ydata['std'][j] is the
                          standard deviation of the success of qubit i.
        """
        return self._ydata

    @property
    def fit_fun(self):
        """
        Return the function used in the fit, e.g. BaseCoherenceFitter._exp_fit_fun
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

    @property
    def time(self):
        """
        Return the characteristic time
        """
        return self._time

    @property
    def time_err(self):
        """
        Return the error of the characteristic time
        """
        return self._time_err


    def _calc_data(self):
        """
        Rerieve probabilities of success from execution results, i.e.,
        probability to measure a state where all qubits are 0.
        Computes a list of dictionaries, see documentation of property ydata.
        """

        self._ydata = []
        for qind, _ in enumerate(self._qubits):
            self._ydata.append({'mean': [], 'std': []})
            for circ, _ in enumerate(self._xdata):
                counts = self._backend_result.get_counts(circ)
                counts_subspace = marginal_counts(counts, [qind])
                success_prob = counts_subspace.get(self._expected_state, 0) / self._shots
                self._ydata[-1]['mean'].append(success_prob)
                self._ydata[-1]['std'].append(np.sqrt(success_prob * (1-success_prob) / self._shots))
                #problem for the fitter if one of the std points is exactly zero
                if self._ydata[-1]['std'][-1] == 0:
                    self._ydata[-1]['std'][-1] = 1e-4


    def _calc_fit(self, p0, bounds):
        """
        Fit the curve.
        Computes self._params and self._params_err:
        - self._params[i] - same as the first returned value of curve_fit, for qubit i.
        - self._params_err[i] - error for each parameter, for qubit i.
        """
        self._params = []
        self._params_err = []
        for qind, _ in enumerate(self._qubits):
            tmp_params, fcov = curve_fit(self._fit_fun, self._xdata,
                                         self._ydata[qind]['mean'],
                                         sigma=self._ydata[qind]['std'],
                                         p0=p0, bounds=bounds)

            self._params.append(tmp_params.copy())
            self._params_err.append(np.sqrt(np.diag(fcov)))


    def plot_coherence(self, qind, ax=None, show_plot=True):
        """
        Plot coherence data.

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

        ax.errorbar(self._xdata, self._ydata[qind]['mean'],
                     self._ydata[qind]['std'],
                     marker='.', markersize=9, c='b', linestyle='')
        ax.plot(self._xdata, self._fit_fun(self._xdata, *self._params[qind]),
                 c='r', linestyle='--',
                 label=self._description+': '+str(np.around(self._time[qind],1))+' micro-seconds')

        ax.tick_params(axis='x', labelsize=14, labelrotation=70)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_xlabel('time [micro-seconds]', fontsize=16)
        ax.set_ylabel('Probability of success', fontsize=16)
        ax.set_title(self._description + ' for qubit ' +
                  str(self._qubits[qind]), fontsize=18)
        ax.legend(fontsize=12)
        ax.grid(True)
        if show_plot:
            plt.show()

        return ax


    @staticmethod
    def _exp_fit_fun(x, a, tau, c):
        """
        Function used to fit the exponential decay
        """

        return a * np.exp(-x / tau) + c


class T1Fitter(BaseCoherenceFitter):
    """
    T1 fitter
    """

    def __init__(self, backend_result, shots, xdata,
                 qubits,
                 fit_p0, fit_bounds):

        BaseCoherenceFitter.__init__(self, '$T_1$',
                                     backend_result, shots,
                                     xdata, qubits,
                                     BaseCoherenceFitter._exp_fit_fun,
                                     fit_p0, fit_bounds, expected_state='1')

        self._time = []
        self._time_err = []
        for qind, _ in enumerate(qubits):
            self._time.append(self._params[qind][1])
            self._time_err.append(self._params_err[qind][1])

    def plot_coherence(self, qind, ax=None):

        ax = BaseCoherenceFitter.plot_coherence(self, qind, ax, show_plot=False)
        ax.set_ylabel("Excited State Population")

        return ax


class T2Fitter(BaseCoherenceFitter):
    """
    T2 fitter
    """

    def __init__(self, backend_result, shots, xdata,
                 measured_qubit, fit_p0, fit_bounds):

        BaseCoherenceFitter.__init__(self, '$T_2$',
                                     backend_result, shots, xdata,
                                     measured_qubit,
                                     BaseCoherenceFitter._exp_fit_fun,
                                     fit_p0, fit_bounds, expected_state='0')

        self._time = self.params[1]
        self._time_err = self.params_err[1]

    def plot_coherence(self, ax=None):

        ax = BaseCoherenceFitter.plot_coherence(self, ax, show_plot=False)
        ax.set_ylabel("Ground State Population")

        return ax


class T2StarExpFitter(BaseCoherenceFitter):
    """
    T2* fitter
    """

    def __init__(self, backend_result, shots, xdata,
                 measured_qubit, fit_p0, fit_bounds):

        BaseCoherenceFitter.__init__(self, '$T_2^*$ exp',
                                     backend_result, shots, xdata,
                                     measured_qubit,
                                     BaseCoherenceFitter._exp_fit_fun,
                                     fit_p0, fit_bounds)

        self._time = self.params[1]
        self._time_err = self.params_err[1]


class T2StarOscFitter(BaseCoherenceFitter):
    """
    T2* fitter
    """

    def __init__(self, backend_result, shots, xdata,
                 measured_qubit, fit_p0, fit_bounds):

        BaseCoherenceFitter.__init__(self, '$T_2^*$',
                                     backend_result, shots, xdata,
                                     measured_qubit,
                                     T2StarOscFitter._osc_fit_fun,
                                     fit_p0, fit_bounds)

        self._time = self.params[1]
        self._time_err = self.params_err[1]


    @staticmethod
    def _osc_fit_fun(x, a, tau, f, phi, c):
        """
        Function used to fit the decay cosine
        """

        return a * np.exp(-x / tau) * np.cos(2 * np.pi * f * x + phi) + c

