# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Functions used for the analysis of randomized benchmarking results.
"""

from scipy.optimize import curve_fit
import numpy as np
from qiskit.tools.qcvv.tomography import marginal_counts

try:
    from matplotlib import pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class RBFitter:
    """
        Class for fitters for randomized benchmarking
    """

    def __init__(self, backend_result, cliff_lengths, shots=1024, \
                 rb_pattern=[[0]]):
        """
        Args:
            backend_result: list of results (qiskit.Result).
            cliff_lengths: the Clifford lengths, 2D list i x j where i is the number of patterns,
            j is the number of cliffords lengths
            rb_pattern: the pattern for the rb sequences.

        """

        self._shots = shots
        self._cliff_lengths = cliff_lengths
        self._rb_pattern = rb_pattern

        self._result_list = []
        self.add_data(backend_result)


    @property
    def raw_data(self):
        """Return raw data."""
        return self._raw_data

    @property
    def cliff_lengths(self):
        """Return clifford lengths."""
        return self.cliff_lengths

    @property
    def ydata(self):
        """Return ydata (means and std devs)."""
        return self._ydata

    @property
    def fit(self):
        """Return fit."""
        return self._fit

    @property
    def results(self):
        """Return all the results."""
        return self._result_list


    def add_data(self, new_backend_result):
        """
        Add a new result. Re calculate the raw data, means and
        fit.

        Args:
            new_backend_result: list of rb results

        Additional information:
            Assumes that 'result' was executed is
            the output of circuits generated by randomized_becnhmarking_seq,
        """

        if new_backend_result is None:
            return

        for result_ind,result in enumerate(new_backend_result):
            if not len(result.results)==len(self._cliff_lengths[0]):
                raise ValueError("The number of clifford lengths must match the number of results")


        for ind,_ in enumerate(new_backend_result):
            self._result_list.append(new_backend_result[ind])

        self._calc_data()
        self._calc_statistics()
        self._calc_rb_fit()


    @staticmethod
    def _rb_fit_fun(x, a, alpha, b):
        """Function used to fit rb."""
        # pylint: disable=invalid-name
        return a * alpha ** x + b


    def _calc_data(self):
        """
        Retrieve probabilities of success from execution results. Outputs results into an
        internal variable _raw_data which is a 3-dimensional list, where item (i,j,k) is
        the probability to measure the ground state for the set of qubits in pattern "i"
        for seed no. j and vector length self._cliff_lengths[i][k].

        Additional information:
            Assumes that 'result' was executed is
            the output of circuits generated by randomized_becnhmarking_seq,
        """

        self._raw_data = []

        for patt_ind in range(len(self._rb_pattern)):

            string_of_0s = ''
            string_of_0s = string_of_0s.zfill(len(self._rb_pattern[patt_ind]))

            self._raw_data.append([])

            for i in range(len(self._result_list)):
                self._raw_data[-1].append([])
                for k,_ in enumerate(self._cliff_lengths[patt_ind]):
                    counts_subspace = marginal_counts(self._result_list[i].get_counts(k), self._rb_pattern[patt_ind])
                    self._raw_data[-1][i].append(
                            counts_subspace.get(string_of_0s, 0) / self._shots)


    def _calc_statistics(self):
        """
        Extract averages and std dev from the raw data (self._raw_data). Assumes that
        self._calc_data has been run. Output into internal _ydata variable:

            ydata is a list of dictionaries (length number of patterns).
            Dictionary ydata[i]:
            ydata[i]['mean'] is a numpy_array of length n;
                        entry j of this array contains the mean probability of success over seeds,
                        for vector length self._cliff_lengths[i][j].
            And ydata[i]['std'] is a numpy_array of length n;
                        entry j of this array contains the std
                        of the probability of success over seeds,
                        for vector length self._cliff_lengths[i][j].
        """

        self._ydata = []
        for patt_ind in range(len(self._rb_pattern)):
            self._ydata.append({})
            self._ydata[-1]['mean'] = np.mean(self._raw_data[patt_ind], 0)

            if len(self._raw_data[patt_ind]) == 1:  # 1 seed
                self._ydata[-1]['std'] = None
            else:
                self._ydata[-1]['std'] = np.std(self._raw_data[patt_ind], 0)


    def _calc_rb_fit(self):
        """
        Fit the RB results to an exponential curve.

        Fit each of the patterns

        Puts the results into a list of fit dictionaries:
            where each dictionary corresponds to a pattern and has fields:
            'params' - three parameters of rb_fit_fun. The middle one is the exponent.
            'err' - the error limits of the parameters.
            'epc' - error per Clifford
        """

        self._fit = []
        for patt_ind, (lens, qubits) in enumerate(zip(self._cliff_lengths, self._rb_pattern)):
            # if at least one of the std values is zero, then sigma is replaced by None
            sigma = self._ydata[patt_ind]['std'].copy()
            if (len(sigma)-np.count_nonzero(sigma) > 0):
                sigma = None
            params, pcov = curve_fit(self._rb_fit_fun, lens,
                                     self._ydata[patt_ind]['mean'],
                                     sigma=sigma,
                                     p0=(1.0, 0.95, 0.0), bounds=([-2, 0, -2], [2, 1, 2]))
            alpha = params[1]  # exponent
            params_err = np.sqrt(np.diag(pcov))
            alpha_err = params_err[1]

            nrb = 2 ** len(qubits)
            epc = (nrb-1)/nrb*(1-alpha)
            epc_err = epc*alpha_err/alpha

            self._fit.append({'params': params, 'params_err': params_err,
                        'epc': epc, 'epc_err': epc_err})


    def plot_rb_data(self, pattern_index=0, ax=None, show_plt=True):
        """
        Plot randomized benchmarking data of a single pattern.

        Args:
            ax (Axes or None): plot axis (if passed in).
            show_plt (bool): display the plot.

        Raises:
            ImportError: If matplotlib is not installed.
        """

        fit_function=self._rb_fit_fun

        if not HAS_MATPLOTLIB:
            raise ImportError('The function plot_rb_data needs matplotlib. '
                              'Run "pip install matplotlib" before.')

        if ax is None:
            plt.figure()
            ax = plt.gca()

        xdata = self._cliff_lengths[pattern_index]

        # Plot the result for each sequence
        for one_seed_data in self._raw_data[pattern_index]:
            ax.plot(xdata, one_seed_data, color='gray', linestyle='none', marker='x')

        # Plot the mean with error bars
        ax.errorbar(xdata, self._ydata[pattern_index]['mean'],
                    yerr=self._ydata[pattern_index]['std'],
                    color='r', linestyle='--', linewidth=3)

        # Plot the fit
        ax.plot(xdata,
                fit_function(xdata, *self._fit[pattern_index]['params']),
                color='blue', linestyle='-', linewidth=2)
        ax.tick_params(labelsize=14)

        ax.set_xlabel('Clifford Length', fontsize=16)
        ax.set_ylabel('Z', fontsize=16)
        ax.grid(True)

        if show_plt:
            plt.show()
