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
from qiskit import QiskitError
from ..tomography import marginal_counts
from ...characterization.fitters import build_counts_dict_from_list

try:
    from matplotlib import pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class RBFitter:
    """
        Class for fitters for randomized benchmarking
    """

    def __init__(self, backend_result, cliff_lengths,
                 rb_pattern=None):
        """
        Args:
            backend_result: list of results (qiskit.Result).
            cliff_lengths: the Clifford lengths, 2D list i x j where i is the
                number of patterns, j is the number of cliffords lengths
            rb_pattern: the pattern for the rb sequences.
        """
        if rb_pattern is None:
            rb_pattern = [[0]]

        self._cliff_lengths = cliff_lengths
        self._rb_pattern = rb_pattern
        self._raw_data = []
        self._ydata = []
        self._fit = []
        self._nseeds = 0

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
    def seeds(self):
        """Return the number of loaded seeds."""
        return self._nseeds

    @property
    def results(self):
        """Return all the results."""
        return self._result_list

    def add_data(self, new_backend_result, rerun_fit=True):
        """
        Add a new result. Re calculate the raw data, means and
        fit.

        Args:
            new_backend_result: list of rb results
            rerun_fit: re caculate the means and fit the result

        Additional information:
            Assumes that 'result' was executed is
            the output of circuits generated by randomized_becnhmarking_seq,
        """

        if new_backend_result is None:
            return

        if not isinstance(new_backend_result, list):
            new_backend_result = [new_backend_result]

        for result in new_backend_result:
            self._result_list.append(result)

            # update the number of seeds *if* new ones
            # added. Note, no checking if we've done all the
            # cliffords
            for rbcirc in result.results:
                nseeds_circ = int(rbcirc.header.name.split('_')[-1])
                if (nseeds_circ+1) > self._nseeds:
                    self._nseeds = nseeds_circ+1

        for result in self._result_list:
            if not len(result.results) == len(self._cliff_lengths[0]):
                raise ValueError(
                    "The number of clifford lengths must match the number of "
                    "results")

        if rerun_fit:
            self.calc_data()
            self.calc_statistics()
            self.fit_data()

    @staticmethod
    def _rb_fit_fun(x, a, alpha, b):
        """Function used to fit rb."""
        # pylint: disable=invalid-name
        return a * alpha ** x + b

    def calc_data(self):
        """
        Retrieve probabilities of success from execution results. Outputs
        results into an internal variable _raw_data which is a 3-dimensional
        list, where item (i,j,k) is the probability to measure the ground state
        for the set of qubits in pattern "i" for seed no. j and vector length
        self._cliff_lengths[i][k].

        Additional information:
            Assumes that 'result' was executed is
            the output of circuits generated by randomized_becnhmarking_seq,
        """

        circ_counts = {}
        circ_shots = {}
        for seedidx in range(self._nseeds):
            for circ, _ in enumerate(self._cliff_lengths[0]):
                circ_name = 'rb_length_%d_seed_%d' % (circ, seedidx)
                count_list = []
                for result in self._result_list:
                    try:
                        count_list.append(result.get_counts(circ_name))
                    except (QiskitError, KeyError):
                        pass

                circ_counts[circ_name] = \
                    build_counts_dict_from_list(count_list)

                circ_shots[circ_name] = sum(circ_counts[circ_name].values())

        self._raw_data = []
        startind = 0

        for patt_ind in range(len(self._rb_pattern)):

            string_of_0s = ''
            string_of_0s = string_of_0s.zfill(len(self._rb_pattern[patt_ind]))

            self._raw_data.append([])
            endind = startind+len(self._rb_pattern[patt_ind])

            for i in range(self._nseeds):

                self._raw_data[-1].append([])
                for k, _ in enumerate(self._cliff_lengths[patt_ind]):
                    circ_name = 'rb_length_%d_seed_%d' % (k, i)
                    counts_subspace = marginal_counts(
                        circ_counts[circ_name],
                        np.arange(startind,endind))
                    self._raw_data[-1][i].append(
                        counts_subspace.get(string_of_0s, 0)
                        / circ_shots[circ_name])
            startind += (endind)

    def calc_statistics(self):
        """
        Extract averages and std dev from the raw data (self._raw_data).
        Assumes that self._calc_data has been run. Output into internal
        _ydata variable:

            ydata is a list of dictionaries (length number of patterns).
            Dictionary ydata[i]:
            ydata[i]['mean'] is a numpy_array of length n;
                        entry j of this array contains the mean probability of
                        success over seeds, for vector length
                        self._cliff_lengths[i][j].
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

    def fit_data(self):
        """
        Fit the RB results to an exponential curve.

        Fit each of the patterns

        Puts the results into a list of fit dictionaries:
            where each dictionary corresponds to a pattern and has fields:
            'params' - three parameters of rb_fit_fun. The middle one is the
                       exponent.
            'err' - the error limits of the parameters.
            'epc' - error per Clifford
        """

        self._fit = []
        for patt_ind, (lens, qubits) in enumerate(zip(self._cliff_lengths,
                                                      self._rb_pattern)):
            # if at least one of the std values is zero, then sigma is replaced
            # by None
            if not self._ydata[patt_ind]['std'] is None:
                sigma = self._ydata[patt_ind]['std'].copy()
                if len(sigma) - np.count_nonzero(sigma) > 0:
                    sigma = None
            else:
                sigma = None
            params, pcov = curve_fit(self._rb_fit_fun, lens,
                                     self._ydata[patt_ind]['mean'],
                                     sigma=sigma,
                                     p0=(1.0, 0.95, 0.0),
                                     bounds=([-2, 0, -2], [2, 1, 2]))
            alpha = params[1]  # exponent
            params_err = np.sqrt(np.diag(pcov))
            alpha_err = params_err[1]

            nrb = 2 ** len(qubits)
            epc = (nrb-1)/nrb*(1-alpha)
            epc_err = epc*alpha_err/alpha

            self._fit.append({'params': params, 'params_err': params_err,
                              'epc': epc, 'epc_err': epc_err})

    def plot_rb_data(self, pattern_index=0, ax=None,
                     add_label=True, show_plt=True):
        """
        Plot randomized benchmarking data of a single pattern.

        Args:
            pattern_index: which RB pattern to plot
            ax (Axes or None): plot axis (if passed in).
            add_label (bool): Add an EPC label
            show_plt (bool): display the plot.

        Raises:
            ImportError: If matplotlib is not installed.
        """

        fit_function = self._rb_fit_fun

        if not HAS_MATPLOTLIB:
            raise ImportError('The function plot_rb_data needs matplotlib. '
                              'Run "pip install matplotlib" before.')

        if ax is None:
            plt.figure()
            ax = plt.gca()

        xdata = self._cliff_lengths[pattern_index]

        # Plot the result for each sequence
        for one_seed_data in self._raw_data[pattern_index]:
            ax.plot(xdata, one_seed_data, color='gray', linestyle='none',
                    marker='x')

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
        ax.set_ylabel('Ground State Population', fontsize=16)
        ax.grid(True)

        if add_label:
            bbox_props = dict(boxstyle="round,pad=0.3",
                              fc="white", ec="black", lw=2)

            ax.text(0.6, 0.9,
                    "alpha: %.3f(%.1e) EPC: %.3e(%.1e)" %
                    (self._fit[pattern_index]['params'][1],
                     self._fit[pattern_index]['params_err'][1],
                     self._fit[pattern_index]['epc'],
                     self._fit[pattern_index]['epc_err']),
                    ha="center", va="center", size=14,
                    bbox=bbox_props, transform=ax.transAxes)

        if show_plt:
            plt.show()
