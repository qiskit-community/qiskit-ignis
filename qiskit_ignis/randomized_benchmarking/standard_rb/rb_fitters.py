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

try:
    from matplotlib import pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class RBFitter:
    """
        Class for fitters for randomized benchmarking
    """

    def __init__(self, backend_result, shots, rb_circs, xdata, rb_opts_dict):
        """
        Args:
            backend_result: result of backend execution (qiskit.Result).
            shots: number of shots.
            rb_circs: list of lists of circuits for the rb sequences (separate list for each seed).
            xdata: the Clifford lengths (with multiplier if applicable).
            rb_opts_dict: a dictionary of the RB options
        """

        self.xdata = xdata
        self.raw_data = self.calc_data(backend_result, shots, rb_circs, rb_opts_dict)
        self.ydata = self.calc_statistics(self.raw_data)
        self.fit = self.calc_rb_fit(self.xdata, self.ydata, rb_opts_dict['rb_pattern'])


    @staticmethod
    def rb_fit_fun(x, a, alpha, b):
        """Function used to fit rb."""
        # pylint: disable=invalid-name
        return a * alpha ** x + b


    def calc_data(self, backend_result, shots, rb_circs, rb_opts_dict):
        """
        Retrieve probabilities of success from execution results.

        Args:
            backend_result: rb circuits results (list of list for each seed)
            rb_circs: rb circuits
            rb_opts_dict: a dictionary of RB options
            shots: number of shots

        Returns:
            A 2-dimensional list, where item (i,j) is the probability to measure the
            ground state, for seed no. i and vector length rb_opts_dict['length_vector'][j].

        Additional information:
            Assumes that 'result' was executed rb_circs,
            and that rb_circs is a set of circuits
            which is an output of randomized_becnhmarking_seq,
            where randomized_becnhmarking_seq was run
            with the given rb_opts_dict.
        """

        string_of_0s = ''
        string_of_0s = string_of_0s.zfill(rb_opts_dict['n_qubits'])

        raw_data = []
        for i in range(rb_opts_dict['nseeds']):
            raw_data.append([])
            for k,_ in enumerate(rb_opts_dict['length_vector']):
                raw_data[i].append(backend_result[i].get_counts(rb_circs[i][k]).\
                                   get(string_of_0s, 0) / shots)

        return raw_data


    def calc_statistics(self, raw_data):
        """
        Extract averages and std dev from the raw data

        Args:
            raw_data (numpy.array): m x n list,
                                    where m is the number of seeds,
                                    and n is the number of Clifford sequences

        Return:
            A dictionary ydata, where
            ydata['mean'] is a numpy_array of length n;
                        entry j of this array contains the mean probability of success over seeds,
                        for vector length rb_opts_dict['length_vector'][j].
            And ydata['std'] is a numpy_array of length n;
                        entry j of this array contains the std
                        of the probability of success over seeds,
                        for vector length rb_opts_dict['length_vector'][j].
        """

        ydata = {'mean': np.mean(raw_data, 0)}

        if len(raw_data) == 1:  # 1 seed
            ydata['std'] = None
        else:
            ydata['std'] = np.std(raw_data, 0)

        return ydata


    def calc_rb_fit(self, xdata, ydata, pattern):
        """
        Fit the RB results to an exponential curve.

        Args:
            xdata (list of lists): a list of Clifford lengths for each pattern.
            ydata (dictionary): output of calc_statistics.
            pattern (list): (see randomized benchmarking functions).
                Pattern which specifies which qubits performing RB with which qubits.
                E.g. [[1],[0,2]] is Q1 doing 1Q RB simultaneously with
                Q0/Q2 doing 2Q RB.

        Return:
            an array of dictionaries,
            where each dictionary corresponds to a pattern and has fields:
            'params' - three parameters of rb_fit_fun. The middle one is the exponent.
            'err' - the error limits of the parameters.
            'epc' - error per Clifford
        """

        fit = []
        for lens, qubits in zip(xdata, pattern):
            params, pcov = curve_fit(self.rb_fit_fun, lens, ydata['mean'], sigma=ydata['std'],
                                     p0=(1.0, 0.95, 0.0), bounds=([-2, 0, -2], [2, 1, 2]))
            alpha = params[1]  # exponent
            params_err = np.sqrt(np.diag(pcov))
            alpha_err = params_err[1]

            nrb = 2 ** len(qubits)
            epc = (nrb-1)/nrb*(1-alpha)
            epc_err = epc*alpha_err/alpha

            fit.append({'params': params, 'params_err': params_err,
                        'epc': epc, 'epc_err': epc_err})

        return fit


    def plot_rb_data(self, pattern_index,
                     raw_data, xdata, ydata, fit,
                     fit_function=rb_fit_fun, ax=None, show_plt=True):
        """
        Plot randomized benchmarking data of a single pattern.

        Args:
            pattern_index: index of the group of qubits, whose data is to be plotted.
                           In other words, an index to entries of 'xdata' and 'fit'.
            raw_data: output of calc_raw_data.
            xdata: output of randomized_benchmarking_seq.
            ydata: output of calc_statistics.
            fit: output of calc_rb_fit.
            fit_function (callable): function used by calc_rb_fit.
            ax (Axes or None): plot axis (if passed in).
            show_plt (bool): display the plot.

        Raises:
            ImportError: If matplotlib is not installed.
        """

        if not HAS_MATPLOTLIB:
            raise ImportError('The function plot_rb_data needs matplotlib. '
                              'Run "pip install matplotlib" before.')

        if ax is None:
            plt.figure()
            ax = plt.gca()

        # Plot the result for each sequence
        for one_seed_data in raw_data:
            ax.plot(xdata[pattern_index], one_seed_data, color='gray', linestyle='none', marker='x')

        # Plot the mean with error bars
        ax.errorbar(xdata[pattern_index], ydata['mean'], yerr=ydata['std'],
                    color='r', linestyle='--', linewidth=3)

        # Plot the fit
        ax.plot(xdata[pattern_index],
                fit_function(xdata[pattern_index], *fit[pattern_index]['params']),
                color='blue', linestyle='-', linewidth=2)
        ax.tick_params(labelsize=14)

        ax.set_xlabel('Clifford Length', fontsize=16)
        ax.set_ylabel('Z', fontsize=16)
        ax.grid(True)

        if show_plt:
            plt.show()
