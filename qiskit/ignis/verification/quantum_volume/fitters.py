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
Functions used for the analysis of quantum volume results.

Based on Cross et al. "Validating quantum computers using
randomized model circuits", arXiv:1811.12926
"""

import math
import warnings
import numpy as np
from qiskit import QiskitError
from qiskit.visualization import plot_histogram
from ...utils import build_counts_dict_from_list

try:
    from matplotlib import get_backend
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class QVFitter:
    """Class for fitters for quantum volume."""

    def __init__(self, backend_result=None, statevector_result=None,
                 qubit_lists=None):
        """
        Args:
            backend_result (list): list of results (qiskit.Result).
            statevector_result (list): the ideal statevectors of each circuit
            qubit_lists (list): list of qubit lists (what was passed to the
                circuit generation)
        """

        self._qubit_lists = qubit_lists
        self._depths = [len(qubit_list) for qubit_list in qubit_lists]
        self._ntrials = 0

        self._result_list = []
        self._heavy_output_counts = {}
        self._circ_shots = {}
        self._circ_counts = {}
        self._all_output_prob_ideal = {}
        self._heavy_output_prob_ideal = {}
        self._heavy_output_prob_exp = {}
        self._ydata = []
        self._heavy_outputs = {}
        self.add_statevectors(statevector_result)
        self.add_data(backend_result)

    @property
    def depths(self):
        """Return depth list."""
        return self._depths

    @property
    def qubit_lists(self):
        """Return depth list."""
        return self._qubit_lists

    @property
    def results(self):
        """Return all the results."""
        return self._result_list

    @property
    def heavy_outputs(self):
        """Return the ideal heavy outputs dictionary."""
        return self._heavy_outputs

    @property
    def heavy_output_counts(self):
        """Return the number of heavy output counts as measured."""
        return self._heavy_output_counts

    @property
    def heavy_output_prob_ideal(self):
        """Return the heavy output probability ideally."""
        return self._heavy_output_prob_ideal

    @property
    def ydata(self):
        """Return the average and std of the output probability."""
        return self._ydata

    def add_statevectors(self, new_statevector_result):
        """
        Add the ideal results and convert to the heavy outputs.

        Assume the result is from 'statevector_simulator'

        Args:
            new_statevector_result (list): ideal results

        Raises:
            QiskitError: If the result has already been added for the circuit
        """

        if new_statevector_result is None:
            return

        if not isinstance(new_statevector_result, list):
            new_statevector_result = [new_statevector_result]

        for result in new_statevector_result:
            for qvcirc in result.results:

                circ_name = qvcirc.header.name

                # get the depth/width from the circuit name
                # qv_depth_%d_trial_%d
                depth = int(circ_name.split('_')[2])

                if circ_name in self._heavy_outputs:
                    raise QiskitError("Already added the ideal result "
                                      "for circuit %s" % circ_name)

                # convert the result into probability dictionary
                qstate = result.get_statevector(circ_name)
                pvector = np.multiply(qstate, qstate.conjugate())
                format_spec = "{0:0%db}" % depth
                self._all_output_prob_ideal[circ_name] = \
                    {format_spec.format(b): float(np.real(pvector[b])) for b in range(2**depth)}

                median_prob = self._median_probabilities([self._all_output_prob_ideal[circ_name]])
                self._heavy_outputs[circ_name] = \
                    self._heavy_strings(self._all_output_prob_ideal[circ_name], median_prob[0])

                # calculate the heavy output probability
                self._heavy_output_prob_ideal[circ_name] = \
                    self._subset_probability(
                        self._heavy_outputs[circ_name],
                        self._all_output_prob_ideal[circ_name])

    def add_data(self, new_backend_result, rerun_fit=True):
        """
        Add a new result. Re calculate fit

        Args:
            new_backend_result (list): list of qv results
            rerun_fit (bool): re calculate the means and fit the result

        Raises:
            QiskitError: If the ideal distribution isn't loaded yet

        Additional information:
            Assumes that 'result' was executed is
            the output of circuits generated by qv_circuits,
        """

        if new_backend_result is None:
            return

        if not isinstance(new_backend_result, list):
            new_backend_result = [new_backend_result]

        for result in new_backend_result:
            self._result_list.append(result)

            # update the number of trials *if* new ones
            # added.
            for qvcirc in result.results:
                ntrials_circ = int(qvcirc.header.name.split('_')[-1])
                if (ntrials_circ+1) > self._ntrials:
                    self._ntrials = ntrials_circ+1

                if qvcirc.header.name not in self._heavy_output_prob_ideal:
                    raise QiskitError('Ideal distribution '
                                      'must be loaded first')

        if rerun_fit:
            self.calc_data()
            self.calc_statistics()

    def calc_data(self):
        """
        Make a count dictionary for each unique circuit from all the results.

        Calculate the heavy output probability.

        Additional information:
            Assumes that 'result' was executed is
            the output of circuits generated by qv_circuits,
        """

        for trialidx in range(self._ntrials):
            for _, depth in enumerate(self._depths):
                circ_name = 'qv_depth_%d_trial_%d' % (depth, trialidx)

                # get the counts form ALL executed circuits
                count_list = []
                for result in self._result_list:
                    try:
                        count_list.append(result.get_counts(circ_name))
                    except (QiskitError, KeyError):
                        pass

                self._circ_counts[circ_name] = \
                    build_counts_dict_from_list(count_list)

                self._circ_shots[circ_name] = \
                    sum(self._circ_counts[circ_name].values())

                # calculate the experimental heavy output counts
                self._heavy_output_counts[circ_name] = \
                    self._subset_probability(
                        self._heavy_outputs[circ_name],
                        self._circ_counts[circ_name])

                # calculate the experimental heavy output probability
                self._heavy_output_prob_exp[circ_name] = \
                    self._heavy_output_counts[circ_name]/self._circ_shots[circ_name]

                # calculate the experimental heavy output probability
                self._heavy_output_prob_exp[circ_name] = \
                    self._heavy_output_counts[circ_name] / self._circ_shots[circ_name]

    def calc_statistics(self):
        """
        Convert the heavy outputs in the different trials into mean and error
        for plotting.

        Here we assume the error is due to a binomial distribution.
        Error (standard deviation) for binomial distribution is sqrt(np(1-p)),
        where n is the number of trials (self._ntrials) and
        p is the success probability (self._ydata[0][depthidx]/self._ntrials).
        """

        self._ydata = np.zeros([4, len(self._depths)], dtype=float)

        exp_vals = np.zeros(self._ntrials, dtype=float)
        ideal_vals = np.zeros(self._ntrials, dtype=float)

        for depthidx, depth in enumerate(self._depths):

            exp_shots = 0

            for trialidx in range(self._ntrials):
                cname = 'qv_depth_%d_trial_%d' % (depth, trialidx)
                exp_vals[trialidx] = self._heavy_output_counts[cname]
                exp_shots += self._circ_shots[cname]
                ideal_vals[trialidx] = self._heavy_output_prob_ideal[cname]

            # Calculate mean and error for experimental data
            self._ydata[0][depthidx] = np.sum(exp_vals)/np.sum(exp_shots)
            self._ydata[1][depthidx] = (self._ydata[0][depthidx] *
                                        (1.0-self._ydata[0][depthidx])
                                        / self._ntrials)**0.5

            # Calculate mean and error for ideal data
            self._ydata[2][depthidx] = np.mean(ideal_vals)
            self._ydata[3][depthidx] = (self._ydata[2][depthidx] *
                                        (1.0-self._ydata[2][depthidx])
                                        / self._ntrials)**0.5

    def plot_qv_data(self, ax=None, show_plt=True, figsize=(7, 5), set_title=True, title=None):
        """Plot the qv data as a function of depth

        Args:
            ax (Axes or None): plot axis (if passed in).
            show_plt (bool): display the plot.
            figsize (tuple): Figure size in inches.
            set_title (bool): set figure title.
            title (String or None): text for setting figure title

        Raises:
            ImportError: If matplotlib is not installed.

        Returns:
            matplotlib.Figure:
                A figure of Quantum Volume data (heavy
                output probability) with two-sigma error
                bar as a function of circuit depth.
        """

        if not HAS_MATPLOTLIB:
            raise ImportError('The function plot_rb_data needs matplotlib. '
                              'Run "pip install matplotlib" before.')

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = None

        xdata = range(len(self._depths))

        # Plot the experimental data with error bars
        ax.errorbar(xdata, self._ydata[0],
                    yerr=self._ydata[1]*2,
                    color='r', marker='o',
                    markersize=6, capsize=5,
                    elinewidth=2, label='Exp (2$\\sigma$ error)')

        # Plot the ideal data with error bars
        ax.errorbar(xdata, self._ydata[2],
                    yerr=self._ydata[3]*2,
                    color='b', marker='v',
                    markersize=6, capsize=5,
                    elinewidth=2, label='Ideal (2$\\sigma$ error)')

        # Plot the threshold
        ax.axhline(2/3, color='k', linestyle='dashed', linewidth=1, label='Threshold')

        ax.set_xticks(xdata)
        ax.set_xticklabels(self._qubit_lists, rotation=45)

        ax.set_xlabel('Qubit Subset', fontsize=14)
        ax.set_ylabel('Heavy Output Probability', fontsize=14)
        ax.grid(True)

        ax.legend()

        if set_title:
            if title is None:
                title = (
                    f'Quantum Volume for up to {len(self._qubit_lists[-1])} Qubits '
                    f'and {self._ntrials} Trials')
            ax.set_title(title)

        if fig:
            if get_backend() in ['module://ipykernel.pylab.backend_inline',
                                 'nbAgg']:
                plt.close(fig)

        if show_plt:
            plt.show()

        return fig

    def plot_qv_trial(self, depth, trial_index, figsize=(7, 5), ax=None):
        """Plot individual trial.
        Args:
            depth(int): circuit depth
            trial_index(int): trial index
            figsize (tuple): Figure size in inches.
            ax (Axes or None): plot axis (if passed in).

        Returns:
            matplotlib.Figure:
                A figure for histogram of ideal and experiment probabilities.
        """
        circ_name = f"qv_depth_{depth}_trial_{trial_index}"
        ideal_data = self._all_output_prob_ideal[circ_name]
        exp_data = self._circ_counts[circ_name]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = None

        # plot experimental histogram
        plot_histogram(exp_data, legend=['Exp'], ax=ax)

        # plot idea histogram overlap with experimental values
        plot_histogram(ideal_data, legend=['Ideal'], bar_labels=False, ax=ax, color='r')

        # get ideal histograms and change to unfilled
        bars = [r for r in ax.get_children() if isinstance(r, Rectangle)]
        for i in range(int(len(bars)/2), len(bars)-1):
            bars[i].fill = False
            # set non-black edge color to increase bar labels legibility
            bars[i].set_edgecolor('saddlebrown')

        # show experimental heavy output probability to the legend
        ax.plot([], [], ' ', label=f'HOP~{self._heavy_output_prob_exp[circ_name]:.3f}')

        # plot median probability
        median_prob = self._median_probabilities([self._all_output_prob_ideal[circ_name]])
        ax.axhline(median_prob, color='r', linestyle='dashed', linewidth=1, label='Median')
        ax.legend()
        ax.set_title(f'Quantum Volume {2**depth}, Trial #{trial_index}', fontsize=14)

        # Only close mpl figures in jupyter with inline backends
        if get_backend() in ['module://ipykernel.pylab.backend_inline',
                             'nbAgg']:
            plt.close(fig)

        return fig

    def plot_hop_accumulative(self, depth, ax=None, figsize=(7, 5)):
        """Plot individual and accumulative heavy output probability (HOP)
        as a function of number of trials.

        Args:
            depth (int): depth of QV circuits
            ax (Axes or None): plot axis (if passed in).
            figsize (tuple): figure size in inches.

        Raises:
            ImportError: If matplotlib is not installed.

        Returns:
            matplotlib.Figure:
                A figure of individual and accumulative HOP as a function of number of trials,
                with 2-sigma confidence interval and 2/3 threshold.
        """

        if not HAS_MATPLOTLIB:
            raise ImportError('The function plot_hop_accumulative needs matplotlib. '
                              'Run "pip install matplotlib" before.')

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = None

        trial_list = np.arange(self._ntrials)  # x data
        hop_list = []  # y data

        for trial_index in range(self._ntrials):
            circ_name = f'qv_depth_{depth}_trial_{trial_index}'
            hop_list.append(self._heavy_output_prob_exp[circ_name])

        hop_accumulative = np.cumsum(hop_list) / np.arange(1, self._ntrials+1)
        two_sigma = 2 * (hop_accumulative * (1 - hop_accumulative) /
                         np.arange(1, self._ntrials+1))**0.5

        # plot two-sigma shaded area
        ax.errorbar(trial_list, hop_accumulative, fmt="none", yerr=two_sigma, ecolor='lightgray',
                    elinewidth=20, capsize=0, alpha=0.5, label='2$\\sigma$')
        # plot accumulative HOP
        ax.plot(trial_list, hop_accumulative, color='r', label='Cumulative HOP')
        # plot inidivual HOP as scatter
        ax.scatter(trial_list, hop_list, s=3, zorder=3, label='Individual HOP')
        # plot 2/3 success threshold
        ax.axhline(2/3, color='k', linestyle='dashed', linewidth=1, label='Threshold')

        ax.set_xlim(0, self._ntrials)
        ax.set_ylim(hop_accumulative[-1]-4*two_sigma[-1], hop_accumulative[-1]+4*two_sigma[-1])

        ax.set_xlabel('Number of Trials', fontsize=14)
        ax.set_ylabel('Heavy Output Probability', fontsize=14)

        ax.set_title(f'Quantum Volume {2**depth} Trials', fontsize=14)

        # re-arrange legend order
        handles, labels = ax.get_legend_handles_labels()
        handles = [handles[1], handles[2], handles[0], handles[3]]
        labels = [labels[1], labels[2], labels[0], labels[3]]
        ax.legend(handles, labels)

        # Only close mpl figures in jupyter with inline backends
        if fig:
            if get_backend() in ['module://ipykernel.pylab.backend_inline',
                                 'nbAgg']:
                plt.close(fig)

        return fig

    def qv_success(self):
        """Return whether each depth was successful (> 2/3 with confidence
        level > 0.977 corresponding to z_value = 2) and the confidence level.

        Returns:
            list: List of list of 2 elements for each depth:
            - success True/False
            - confidence level
        """

        success_list = []
        confidence_level_threshold = self.calc_confidence_level(z_value=2)

        for depth_ind, _ in enumerate(self._depths):
            success_list.append([False, 0.0])
            hmean = self._ydata[0][depth_ind]
            sigma = self._ydata[1][depth_ind]
            z_value = self.calc_z_value(hmean, sigma)
            confidence_level = self.calc_confidence_level(z_value)
            success_list[-1][1] = confidence_level

            if (hmean > 2/3 and confidence_level > confidence_level_threshold):
                success_list[-1][0] = True

        return success_list

    def calc_z_value(self, mean, sigma):
        """Calculate z value using mean and sigma.

        Args:
            mean (float): mean
            sigma (float): standard deviation

        Returns:
            float: z_value in standard normal distibution.
        """

        if sigma == 0:
            # assign a small value for sigma if sigma = 0
            sigma = 1e-10
            warnings.warn('Standard deviation sigma should not be zero.')

        z_value = (mean - 2/3) / sigma

        return z_value

    def calc_confidence_level(self, z_value):
        """Calculate confidence level using z value.

        Accumulative probability for standard normal distribution
        in [-z, +infinity] is 1/2 (1 + erf(z/sqrt(2))),
        where z = (X - mu)/sigma = (hmean - 2/3)/sigma

        Args:
            z_value (float): z value in in standard normal distibution.

        Returns:
            float: confidence level in decimal (not percentage).
        """

        confidence_level = 0.5 * (1 + math.erf(z_value/2**0.5))

        return confidence_level

    def quantum_volume(self):
        """Return the volume for each depth.

        Returns:
            list: List of quantum volumes
        """

        qv_list = 2**np.array(self._depths)

        return qv_list

    def _heavy_strings(self, ideal_distribution, ideal_median):
        """Return the set of heavy output strings.

        Args:
            ideal_distribution (dict): dict of ideal output distribution
                where keys are bit strings (as strings) and values are
                probabilities of observing those strings
            ideal_median (float): median probability across all outputs

        Returns:
            list: list the set of heavy output strings, i.e. those strings
                whose ideal probability of occurrence exceeds the median.
        """
        return list(filter(lambda x: ideal_distribution[x] > ideal_median,
                           list(ideal_distribution.keys())))

    def _median_probabilities(self, distributions):
        """Return a list of median probabilities.

        Args:
            distributions (list): list of dicts mapping binary strings
                (as strings) to probabilities.

        Returns:
            list: a list of median probabilities.
        """
        medians = []
        for dist in distributions:
            values = np.array(list(dist.values()))
            medians.append(float(np.real(np.median(values))))

        return medians

    def _subset_probability(self, strings, distribution):
        """Return the probability of a subset of outcomes.

        Args:
            strings (list): list of bit strings (as strings)
            distribution (dict): dict where keys are bit strings (as strings)
                and values are probabilities of observing those strings

        Returns:
            float: the probability of the subset of strings, i.e. the sum
                of the probabilities of each string as given by the
                distribution.
        """
        return sum([distribution.get(value, 0) for value in strings])
