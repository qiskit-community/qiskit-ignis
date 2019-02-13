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


def exp_fit_fun(x, a, tau, c):
    """Function used to fit the exponential decay."""
    # pylint: disable=invalid-name
    return a * np.exp(-x / tau) + c


def osc_fit_fun(x, a, tau, f, phi, c):
    """Function used to fit the decay cosine."""
    # pylint: disable=invalid-name
    return a * np.exp(-x / tau) * np.cos(2 * np.pi * f * x + phi) + c


def rb_fit_fun(x, a, alpha, b):
    """Function used to fit rb."""
    # pylint: disable=invalid-name
    return a * alpha ** x + b


def calc_raw_data(result_list, rb_circs, rb_opts, shots):
    """
    Retrieve probabilities of success from execution results.

    Args:
        result_list: rb circuits results (list of list for each seed)
        rb_circs: rb circuits
        rb_opts: a dictionary of RB options
        shots: number of shots

    Returns:
        A 2-dimensional list, where item (i,j) is the probability to measure the
        ground state, for seed no. i and vector length rb_opts['length_vector'][j].

    Additional information:
        Assumes that 'result' was executed rb_circs,
        and that rb_circs is a set of circuits
        which is an output of randomized_becnhmarking_seq,
        where randomized_becnhmarking_seq was run
        with the given rb_opts.
    """

    """
    TODOs:
    1) For correlated RB, calculate raw_data separately for each group of qubits.
       Then the raw data will be a 3-dimensional list, where item (i,j,l) is the
       probability to measure the ground state in the projection of the qubits
       in group l, for seed no. i and vector length rb_opts['length_vector'][j].
    2) Add an input parameter old_raw_data. The new raw data, generated in this function,
       will be appended to the old raw data.
    """

    string_of_0s = ''
    string_of_0s = string_of_0s.zfill(rb_opts['n_qubits'])

    raw_data = []
    for i in range(rb_opts['nseeds']):
        raw_data.append([])
        for k,_ in enumerate(rb_opts['length_vector']):
            raw_data[i].append(result_list[i].get_counts(rb_circs[i][k]).get(string_of_0s,0) / shots)


    return raw_data


def calc_statistics(raw_data):
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
                      for vector length rb_opts['length_vector'][j].
        And ydata['std'] is a numpy_array of length n;
                      entry j of this array contains the std
                      of the probability of success over seeds,
                      for vector length rb_opts['length_vector'][j].
    """

    """
    TODO:
       For correlated RB, calculate ydata separately for each group of qubits.
       Then ydata['mean'] and ydata['std'] will be 3-dimensional lists, where items (j,l)
       refer to the qubits in group l, for vector length rb_opts['length_vector'][j].
    """

    ydata = {'mean': np.mean(raw_data, 0)}

    if len(raw_data) == 1:  # 1 seed
        ydata['std'] = None
    else:
        ydata['std'] = np.std(raw_data, 0)

    return ydata


def calc_rb_fit(xdata, ydata, pattern):
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
        params, pcov = curve_fit(rb_fit_fun, lens, ydata['mean'], sigma=ydata['std'],
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


def plot_rb_data(pattern_index,
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


def plot_coherence(pattern_index, xdata, ydata, fit,
                   xunit, exp_str, qubit_label,
                   fit_function=rb_fit_fun,
                  ):
    """
    Plot coherence data.

    Args:
        pattern_index, xdata, ydata, fit, fit_function: see documentation of plot_rb_data
        xunit: TODO: complete
        exp_str: TODO: complete
        qubit_label
    Raises:
        ImportError: If matplotlib is not installed.
    """

    if not HAS_MATPLOTLIB:
        raise ImportError('The function plot_coherence needs matplotlib. '
                          'Run "pip install matplotlib" before.')

    plt.errorbar(xdata[pattern_index], ydata['mean'], ydata['std'],
                 marker='.', markersize=9, c='b', linestyle='')
    plt.plot(xdata[pattern_index],
             fit_function(xdata[pattern_index], *fit[pattern_index]['params']),
             c='r', linestyle='--',
             label=(exp_str + '= %s %s' % (str(round(fit[pattern_index]['params'][1])), xunit)))

    plt.xticks(fontsize=14, rotation=70)
    plt.yticks(fontsize=14)
    plt.xlabel('time [%s]' % (xunit), fontsize=16)
    plt.ylabel('P(1)', fontsize=16)
    plt.title(exp_str + ' measurement of Q$_{%s}$' % (str(qubit_label)), fontsize=18)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()
