# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Quantum volume Experiment Analysis.
"""
import warnings
import math
from typing import Optional, Dict, Union, List

import numpy as np

from qiskit.result import Result
from qiskit.ignis.experiments.base import Analysis
from qiskit.providers import BaseJob
from qiskit.exceptions import QiskitError

# for the Result object
from qiskit.visualization import plot_histogram
try:
    from matplotlib import get_backend
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

class QuantumVolumeAnalysis(Analysis):
    """Quantum volume experiment analysis.
    the experiment data must include both ideal and non-ideal results"""

    # pylint: disable=arguments-differ
    def __init__(self,
                 data: Optional[any] = None,
                 ideal_data: Optional[any] = None,
                 qubit_lists: Optional[Union[int, List[int]]] = None,
                 metadata: Optional[Dict[str, any]] = None,
                 exp_id: Optional[str] = None):
        self._qubit_lists = qubit_lists
        self._depths = [len(qubit_list) for qubit_list in qubit_lists]

        self._exp_ideal_data = []
        self._ydata = []

        super().__init__(data=data,
                         metadata=metadata,
                         exp_id=exp_id)

        # Optionally initialize with data (non ideal added in super call)
        self.add_ideal_data(ideal_data, metadata)

        self._analysis_fn = self.fit

    def add_ideal_data(self,
                       ideal_data: Union[BaseJob, Result, any],
                       metadata: Optional[Dict[str, any]] = None):
        """Add additional ideal_data to the fitter.

        Args:
                ideal_data: input ideal_data for the fitter.
                metadata: Optional, list of metadata dicts for input ideal_data.
                          if None will be taken from ideal_data Result object.

        Raises:
            QiskitError: if input ideal_data is incorrectly formatted.
        """
        # the add_data method saves the data to self._exp_data, so the data already there need to
        # temporarily move. same for self._exp_metadata
        temp = self._exp_data
        self._exp_data = self._exp_ideal_data
        temp_meta = self._exp_metadata
        # the metadata is same for ideal and non ideal data.
        # so use dummy meta for add_data function
        self._exp_metadata = [{} for _ in range(len(self._exp_ideal_data))]
        self.add_data(ideal_data, metadata)
        self._exp_ideal_data = self._exp_data
        self._exp_data = temp
        self._exp_metadata = temp_meta

    def clear_data(self):
        """Clear stored data"""
        self._exp_data = []
        self._exp_ideal_data = []
        self._exp_metadata = []
        self._result = None

    def _format_data(self, data: Result,
                     metadata: Dict[str, any],
                     index: int):
        """Format the required data from a Result.data dict
        extract count in case of non ideal data, and state-vector in case of ideal data
        """
        try:
            return data.get_statevector(index)
        except QiskitError:
            return data.get_counts(index)

    # functions to help calculate the QV:
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

    def _qv_success(self):
        """Return whether each depth was successful (> 2/3 with confidence
        level > 0.977 corresponding to z_value = 2) and the confidence level.

        Returns:
            list: List of list of 3 elements for each depth:
            - success True/False
            - hop
            - confidence level
        """
        success_list = []
        confidence_level_threshold = self._calc_confidence_level(z_value=2)

        for depth_ind, _ in enumerate(self._depths):
            success_list.append([False, 0.0, 0.0])
            hmean = self._ydata[0][depth_ind]
            sigma = self._ydata[1][depth_ind]
            z_value = self._calc_z_value(hmean, sigma)
            confidence_level = self._calc_confidence_level(z_value)
            success_list[-1][1] = hmean
            success_list[-1][2] = confidence_level

            if hmean > 2 / 3 and confidence_level > confidence_level_threshold:
                success_list[-1][0] = True

        return success_list

    def _calc_z_value(self, mean, sigma):
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

        z_value = (mean - 2 / 3) / sigma

        return z_value

    def _calc_confidence_level(self, z_value):
        """Calculate confidence level using z value.

        Accumulative probability for standard normal distribution
        in [-z, +infinity] is 1/2 (1 + erf(z/sqrt(2))),
        where z = (X - mu)/sigma = (hmean - 2/3)/sigma

        Args:
            z_value (float): z value in in standard normal distibution.

        Returns:
            float: confidence level in decimal (not percentage).
        """

        confidence_level = 0.5 * (1 + math.erf(z_value / 2 ** 0.5))

        return confidence_level

    def _quantum_volume(self):
        """Return the volume for each depth.

        Returns:
            list: List of quantum volumes
        """

        qv_list = 2 ** np.array(self._depths)

        return qv_list

    def _calc_quantum_volume(self):
        """
        calc the quantum volume of the analysed system
        Returns:
            int: quantum volume
        """
        qv_success_list = self._qv_success()
        qv_list = self._ydata
        quantum_volume = 1
        for qidx in range(len(self._qubit_lists)):
            if qv_list[0][qidx] > 2 / 3:
                if qv_success_list[qidx][0]:
                    quantum_volume = self._quantum_volume()[qidx]

        return quantum_volume

    def fit(self, data, metadata):
        """
        fit the data, calculate the heavy output probability and calculate the quantum volume.

        also convert the heavy outputs in the different trials into mean and error for plotting.
        Here we assume the error is due to a binomial distribution.
        Error (standard deviation) for binomial distribution is sqrt(np(1-p)),
        where n is the number of trials (self._ntrials) and
        p is the success probability (self._ydata[0][depthidx]/self._ntrials).

        Returns:
            QuantumVolumeResult: quantum volume result object
        """
        heavy_output_counts = {}
        circ_shots = {}
        circ_counts = {}
        all_output_prob_ideal = {}
        heavy_output_prob_ideal = {}
        heavy_output_prob_exp = {}
        heavy_outputs = {}
        median_probabilities = {}

        # analyse ideal data
        for exp_num, ideal_result in enumerate(self._exp_ideal_data):
            circ_name = metadata[exp_num]["circ_name"]
            depth = metadata[exp_num]["depth"]

            # convert the result into probability dictionary
            pvector = np.multiply(ideal_result, ideal_result.conjugate())
            format_spec = "{0:0%db}" % depth
            all_output_prob_ideal[circ_name] = \
                {format_spec.format(b): float(np.real(pvector[b])) for b in range(2 ** depth)}

            median_prob = self._median_probabilities([all_output_prob_ideal[circ_name]])
            median_probabilities[circ_name] = median_prob
            heavy_outputs[circ_name] = \
                self._heavy_strings(all_output_prob_ideal[circ_name], median_prob[0])

            # calculate the heavy output probability
            heavy_output_prob_ideal[circ_name] = \
                self._subset_probability(
                    heavy_outputs[circ_name],
                    all_output_prob_ideal[circ_name])

        # analyse non-ideal data
        for exp_num, result in enumerate(data):
            circ_name = metadata[exp_num]["circ_name"]

            circ_shots[circ_name] = sum(result.values())
            circ_counts[circ_name] = result

            # calculate the experimental heavy output counts
            heavy_output_counts[circ_name] = \
                self._subset_probability(heavy_outputs[circ_name], result)

            # calculate the experimental heavy output probability
            heavy_output_prob_exp[circ_name] = \
                heavy_output_counts[circ_name] / circ_shots[circ_name]

        # calculate the mean and error
        self._ydata = np.zeros([4, len(self._depths)], dtype=float)
        trials = int(len(data) / len(self._qubit_lists))

        exp_vals = np.zeros(trials, dtype=float)
        ideal_vals = np.zeros(trials, dtype=float)

        for depthidx, depth in enumerate(self._depths):
            exp_shots = 0

            for trialidx in range(trials):
                cname = 'qv_depth_%d_trial_%d' % (depth, trialidx + 1)
                exp_vals[trialidx] = heavy_output_counts[cname]
                exp_shots += circ_shots[cname]
                ideal_vals[trialidx] = heavy_output_prob_ideal[cname]

            # Calculate mean and error for experimental data
            self._ydata[0][depthidx] = np.sum(exp_vals) / np.sum(exp_shots)
            self._ydata[1][depthidx] = (self._ydata[0][depthidx] *
                                        (1.0 - self._ydata[0][depthidx])
                                        / trials) ** 0.5

            # Calculate mean and error for ideal data
            self._ydata[2][depthidx] = np.mean(ideal_vals)
            self._ydata[3][depthidx] = (self._ydata[2][depthidx] *
                                        (1.0 - self._ydata[2][depthidx])
                                        / trials) ** 0.5

        plot_qv_data = {'xdata': self._depths,
                        'ydata': self._ydata}
        plot_trial_data = {'all_output_prob_ideal': all_output_prob_ideal,
                           'circ_counts': circ_counts,
                           'heavy_output_prob_exp': heavy_output_prob_exp,
                           'median_probabilities': median_probabilities}
        plot_hop_accumulative_data = {'xdata': trials,
                                      'heavy_output_prob_exp': heavy_output_prob_exp}

        result = QuantumVolumeResult(self._calc_quantum_volume(),
                                     self._qv_success(),
                                     plot_qv_data,
                                     plot_trial_data,
                                     plot_hop_accumulative_data,
                                     self._qubit_lists,
                                     trials)
        return result


class QuantumVolumeResult():
    """
    Quantum volume result object
    """
    # TODO: inherit from a general ExperimentResultObject
    def __init__(self,
                 quantum_volume,
                 success_list,
                 plot_qv_data,
                 plot_trial_data,
                 plot_hop_accumulative_data,
                 qubit_lists,
                 ntrials):
        self._quantum_volume = quantum_volume
        self._success_list = success_list
        self._plot_qv_data = plot_qv_data
        self._plot_trial_data = plot_trial_data
        self._plot_hop_accumulative_data = plot_hop_accumulative_data
        self._qubit_lists = qubit_lists
        self._ntrials = ntrials

    @property
    def quantum_volume(self):
        """Return the ideal heavy outputs dictionary."""
        return self._quantum_volume

    def __repr__(self):
        """
        print the percentage of successful trials and the confidence level of each depth
        """
        res_str = ""
        for qidx, qubit_list in enumerate(self._qubit_lists):
            if self._success_list[qidx][1] > 2 / 3:
                if self._success_list[qidx][0]:
                    res_str += "Width/depth %d greater than 2/3 (%f) with confidence %f " \
                               "(successful).\r\n" % \
                               (len(qubit_list), self._success_list[qidx][1],
                                self._success_list[qidx][2])
                else:
                    res_str += "Width/depth %d greater than 2/3 (%f) with confidence %f " \
                               "(unsuccessful).\r\n" \
                               % (len(qubit_list), self._success_list[qidx][1],
                                  self._success_list[qidx][2])
            else:
                res_str += "Width/depth %d less than 2/3 (unsuccessful).\r\n" % len(qubit_list)
        return res_str

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
            raise ImportError('The function plot_qv_data needs matplotlib. '
                              'Run "pip install matplotlib" before.')

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = None

        xdata = self._plot_qv_data['xdata']
        ydata = self._plot_qv_data['ydata']

        # Plot the experimental data with error bars
        ax.errorbar(xdata, ydata[0],
                    yerr=ydata[1] * 2,
                    color='r', marker='o',
                    markersize=6, capsize=5,
                    elinewidth=2, label='Exp (2$\\sigma$ error)')

        # Plot the ideal data with error bars
        ax.errorbar(xdata, ydata[2],
                    yerr=ydata[3] * 2,
                    color='b', marker='v',
                    markersize=6, capsize=5,
                    elinewidth=2, label='Ideal (2$\\sigma$ error)')

        # Plot the threshold
        ax.axhline(2 / 3, color='k', linestyle='dashed', linewidth=1, label='Threshold')

        ax.set_xticks(xdata)
        ax.set_xticklabels(self._qubit_lists, rotation=45)

        ax.set_xlabel('Qubit Subset', fontsize=14)
        ax.set_ylabel('Heavy Output Probability', fontsize=14)
        ax.grid(True)

        ax.legend()

        if set_title:
            if title is None:
                title = ('Quantum Volume for up to {} '
                         'Qubits and {} Trials').format(len(self._qubit_lists[-1]), self._ntrials)
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

        Raises:
            QiskitError: If given depth or trial_index do not exit in the experiment.

        Returns:
            matplotlib.Figure:
                A figure for histogram of ideal and experiment probabilities.
        """
        circ_name = "qv_depth_{}_trial_{}".format(depth, trial_index)
        all_output_prob_ideal = self._plot_trial_data['all_output_prob_ideal']
        circ_counts = self._plot_trial_data['circ_counts']
        heavy_output_prob_exp = self._plot_trial_data['heavy_output_prob_exp']
        median_probabilities = self._plot_trial_data['median_probabilities']

        if circ_name not in all_output_prob_ideal.keys():
            raise QiskitError("Error - given depth and trial do not exist."
                              " possible depths are {},"
                              " and highest trial number is {}"
                              .format([len(qubits) for qubits in self._qubit_lists],
                                      self._ntrials))
        ideal_data = all_output_prob_ideal[circ_name]
        exp_data = circ_counts[circ_name]

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
        for i in range(int(len(bars) / 2), len(bars) - 1):
            bars[i].fill = False
            # set non-black edge color to increase bar labels legibility
            bars[i].set_edgecolor('saddlebrown')

        # show experimental heavy output probability to the legend
        ax.plot([], [], ' ', label='HOP~{:.3f}'.format(heavy_output_prob_exp[circ_name]))

        # plot median probability
        median_prob = median_probabilities[circ_name]
        ax.axhline(median_prob, color='r', linestyle='dashed', linewidth=1, label='Median')
        ax.legend()
        ax.set_title('Quantum Volume {}, Trial #{}'.format(2**depth, trial_index), fontsize=14)

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
            QiskitError: If given depth do not exit in the experiment.

        Returns:
            matplotlib.Figure:
                A figure of individual and accumulative HOP as a function of number of trials,
                with 2-sigma confidence interval and 2/3 threshold.
        """

        if not HAS_MATPLOTLIB:
            raise ImportError('The function plot_hop_accumulative needs matplotlib. '
                              'Run "pip install matplotlib" before.')

        if depth not in [len(qubits) for qubits in self._qubit_lists]:
            raise QiskitError("Error - given depth do not exist. possible depths are {}."
                              .format([len(qubits) for qubits in self._qubit_lists]))

        heavy_output_prob_exp = self._plot_hop_accumulative_data['heavy_output_prob_exp']
        trials = self._plot_hop_accumulative_data['xdata']

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = None

        trial_list = np.arange(1, trials + 1)  # x data
        hop_list = []  # y data

        for trial_index in trial_list:
            circ_name = 'qv_depth_{}_trial_{}'.format(depth, trial_index)
            hop_list.append(heavy_output_prob_exp[circ_name])

        hop_accumulative = np.cumsum(hop_list) / np.arange(1, trials + 1)
        two_sigma = 2 * (hop_accumulative * (1 - hop_accumulative) /
                         np.arange(1, trials + 1)) ** 0.5

        # plot two-sigma shaded area
        ax.errorbar(trial_list, hop_accumulative, fmt="none", yerr=two_sigma, ecolor='lightgray',
                    elinewidth=20, capsize=0, alpha=0.5, label='2$\\sigma$')
        # plot accumulative HOP
        ax.plot(trial_list, hop_accumulative, color='r', label='Cumulative HOP')
        # plot inidivual HOP as scatter
        ax.scatter(trial_list, hop_list, s=3, zorder=3, label='Individual HOP')
        # plot 2/3 success threshold
        ax.axhline(2 / 3, color='k', linestyle='dashed', linewidth=1, label='Threshold')

        ax.set_xlim(0, trials)
        ax.set_ylim(max([0, hop_accumulative[-1] - 4 * two_sigma[-1]]),
                    min([1, hop_accumulative[-1] + 4 * two_sigma[-1]]))

        ax.set_xlabel('Number of Trials', fontsize=14)
        ax.set_ylabel('Heavy Output Probability', fontsize=14)

        ax.set_title('Quantum Volume {} Trials'.format(2**depth), fontsize=14)

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
