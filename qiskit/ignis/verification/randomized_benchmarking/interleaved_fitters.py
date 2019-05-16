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
Functions used for the analysis of interleaved
randomized benchmarking results.
"""

import numpy as np
from .fitters import RBFitter

try:
    from matplotlib import pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class InterleavedRBFitter():
    """
        Class for fitters for interleaved RB
        Derived from RBFitter class
    """

    def __init__(self, original_result, interleaved_result,
                 cliff_lengths, rb_pattern=None):
        """
        Args:
            original_result: list of results of the
            original RB sequence (qiskit.Result).
            intelreaved_result: list of results of the
            interleaved RB sequence (qiskit.Result).
            cliff_lengths: the Clifford lengths, 2D list i x j where i is the
                number of patterns, j is the number of cliffords lengths
            rb_pattern: the pattern for the rb sequences.
        """
        self._cliff_lengths = cliff_lengths
        self._rb_pattern = rb_pattern
        self._fit_interleaved = []

        self.rbfit_original = RBFitter(
            original_result, cliff_lengths, rb_pattern)
        self.rbfit_interleaved = RBFitter(
            interleaved_result, 2 * cliff_lengths, rb_pattern)

        self._raw_original_data = self.rbfit_original.raw_data
        self._raw_interleaved_data = self.rbfit_interleaved.raw_data
        self._ydata_original = self.rbfit_original.ydata
        self._ydata_interleaved = self.rbfit_interleaved.ydata


    @property
    def cliff_lengths(self):
        """Return clifford lengths."""
        return self.cliff_lengths

    @property
    def fit_interleaved(self):
        """Return fit."""
        return self._fit_interleaved

    @property
    def ydata_original(self):
        """Return ydata_original (means and std devs)."""
        return self._ydata_original

    @property
    def ydata_interleaved(self):
        """Return ydata_interleaved (means and std devs)."""
        return self._ydata_interleaved

    @property
    def raw_original_data(self):
        """Return raw original_data."""
        return self._raw_original_data

    @property
    def raw_interleaved_data(self):
        """Return raw interleaved_data."""
        return self._raw_interleaved_data


    def add_interleaved_data(self, new_original_result,
                             new_interleaved_result,
                             rerun_fit=True):
        """
        Add a new result.

        Args:
            new_original_result: list of rb results
            of the original circuits
            new_interleaved_result: list of rb results
            of the interleaved circuits
            rerun_fit: re-caculate the means and fit the result

        Additional information:
            Assumes that 'result' was executed is
            the output of circuits generated by randomized_becnhmarking_seq
        """
        self.rbfit_original.add_data(new_original_result, rerun_fit)
        self.rbfit_interleaved.add_data(new_interleaved_result, rerun_fit)

    def calc_interleaved_data(self):
        """
        Retrieve probabilities of success from execution results. Outputs
        results into an internal variables: _raw_original_data and
        _raw_interleaved_data
        """
        self.rbfit_original.calc_data()
        self.rbfit_interleaved.calc_data()

    def calc_interleaved_statistics(self):
        """
        Extract averages and std dev.
        Output into internal variables:
        _ydata_original and _ydata_interleaved
        """
        self.rbfit_original.calc_statistics()
        self.rbfit_interleaved.calc_statistics()

    def fit_interleaved_data(self):
        """
        Fit the interleaved RB results
        Fit each of the patterns

        According to the paper: "Efficient measurement of quantum gate
        error by interleaved randomized benchmarking" (arXiv:1203.4550)
        Equations (4) and (5)

        Puts the results into a list of fit dictionaries:
            where each dictionary corresponds to a pattern and has fields:
            'epc_est' - the estimated error per the interleaved Clifford
            'epc_est_error' - the estimated error derived from the params_err
            'systematic_err' - systematic error bound of epc_est
            'systematic_err_L' = epc_est - systematic_err (left error bound)
            'systematic_err_R' = epc_est + systematic_err (right error bound)
        """
        self.rbfit_original.fit_data()
        self.rbfit_interleaved.fit_data()
        self._fit_interleaved = []

        for patt_ind, (_, qubits) in enumerate(zip(self._cliff_lengths,
                                                   self._rb_pattern)):
            # calculate nrb=d=2^n:
            nrb = 2 ** len(qubits)

            # Calculate alpha (=p) and alpha_c (=p_c):
            alpha = self.rbfit_original.fit[patt_ind]['params'][1]
            alpha_c = self.rbfit_interleaved.fit[patt_ind]['params'][1]
            # Calculate their errors:
            alpha_err = self.rbfit_original.fit[patt_ind]['params_err'][1]
            alpha_c_err = self.rbfit_interleaved.fit[patt_ind]['params_err'][1]

            # Calculate epc_est (=r_c^est) - Eq. (4):
            epc_est = (nrb - 1) * (1 - alpha_c / alpha) / nrb

            # Calculate the systematic error bounds - Eq. (5):
            systematic_err_1 = (nrb - 1) * (abs(alpha - alpha_c / alpha)
                                        + (1 - alpha)) / nrb
            systematic_err_2 = 2 * (nrb * nrb - 1) * (1 - alpha) / \
                (alpha * nrb * nrb) + 4 * (np.sqrt(1 - alpha)) * \
                (np.sqrt(nrb * nrb - 1)) / alpha
            systematic_err = min(systematic_err_1, systematic_err_2)
            systematic_err_L = epc_est - systematic_err
            systematic_err_R = epc_est + systematic_err

            # Calculate epc_est_error
            alpha_err_sq = (alpha_err / alpha) * (alpha_err / alpha)
            alpha_c_err_sq = (alpha_c_err / alpha_c) * (alpha_c_err / alpha_c)
            epc_est_err = epc_est * (np.sqrt(alpha_err_sq + alpha_c_err_sq))

            self._fit_interleaved.append({'alpha': alpha,
                                          'alpha_err': alpha_err,
                                          'alpha_c': alpha_c,
                                          'alpha_c_err': alpha_c_err,
                                          'epc_est': epc_est,
                                          'epc_est_err': epc_est_err,
                                          'systematic_err': systematic_err,
                                          'systematic_err_L': systematic_err_L,
                                          'systematic_err_R': systematic_err_R})

    def plot_interleaved_rb_data(self, pattern_index=0, ax=None,
                     add_label=True, show_plt=True):
        """
        Plot interleaved randomized benchmarking data of a single pattern.

        Args:
            pattern_index: which RB pattern to plot
            ax (Axes or None): plot axis (if passed in).
            add_label (bool): Add an EPC label
            show_plt (bool): display the plot.

        Raises:
            ImportError: If matplotlib is not installed.
        """

        original_fit_function = self.rbfit_original._rb_fit_fun
        interleaved_fit_function = self.rbfit_interleaved._rb_fit_fun

        if not HAS_MATPLOTLIB:
            raise ImportError('The function plot_interleaved_rb_data \
            needs matplotlib. Run "pip install matplotlib" before.')

        if ax is None:
            plt.figure()
            ax = plt.gca()

        xdata = self._cliff_lengths[pattern_index]

        # Plot the original and interleaved result for each sequence
        for one_seed_data in self._raw_original_data[pattern_index]:
            ax.plot(xdata, one_seed_data, color='gray', linestyle='none',
                    marker='x')
        for one_seed_data in self._raw_interleaved_data[pattern_index]:
            ax.plot(xdata, one_seed_data, color='black', linestyle='none',
                    marker='+')

        # Plot the mean with error bars
        ax.errorbar(xdata, self._ydata_original[pattern_index]['mean'],
                    yerr=self._ydata_original[pattern_index]['std'],
                    color='r', linestyle='--', linewidth=3)
        ax.errorbar(xdata, self._ydata_interleaved[pattern_index]['mean'],
                    yerr=self._ydata_interleaved[pattern_index]['std'],
                    color='g', linestyle='--', linewidth=3)

        # Plot the fit
        #ax.plot(xdata,
        #        original_fit_function(xdata, self.rbfit_original._fit[pattern_index]['params']),
        #        color='blue', linestyle='-', linewidth=2)
        #ax.tick_params(labelsize=14)

        ax.set_xlabel('Clifford Length', fontsize=16)
        ax.set_ylabel('Ground State Population', fontsize=16)
        ax.grid(True)

        if add_label:
            bbox_props = dict(boxstyle="round,pad=0.3",
                              fc="white", ec="black", lw=2)

            ax.text(0.6, 0.9,
                    "alpha: %.3f(%.1e) alpha_c: %.3e(%.1e) EPC_est: %.3e(%.1e)" %
                    (self._fit_interleaved[pattern_index]['alpha'],
                     self._fit_interleaved[pattern_index]['alpha_err'],
                     self._fit_interleaved[pattern_index]['alpha_c'],
                     self._fit_interleaved[pattern_index]['alpha_c_err'],
                     self._fit_interleaved[pattern_index]['epc_est'],
                     self._fit_interleaved[pattern_index]['systematic_err']),
                    ha="center", va="center", size=14,
                    bbox=bbox_props, transform=ax.transAxes)

        if show_plt:
            plt.show()