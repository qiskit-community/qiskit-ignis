# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Functions used for the analysis of quantum volume results.
"""

import math
import numpy as np
from qiskit import QiskitError
from ...characterization.fitters import build_counts_dict_from_list

try:
    from matplotlib import pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class QVFitter:
    """
        Class for fitters for quantum volume
    """

    def __init__(self, backend_result=None, statevector_result=None,
                 qubit_list=None,
                 depth_list=None):
        """
        Args:
            backend_result: list of results (qiskit.Result).
            statevector_result: the ideal statevectors of each circuit
            qubit_list: list of qubits
            depth_list: the depths explored in the results
        """

        self._depths = depth_list
        self._qubit_list = qubit_list
        self._width = len(qubit_list)
        self._ntrials = 0

        self._result_list = []
        self._heavy_output_prob = {}
        self._heavy_output_prob_ideal = {}
        self._ydata = []
        self._heavy_outputs = {}
        self.add_statevectors(statevector_result)
        self.add_data(backend_result)

    @property
    def depths(self):
        """Return depth list."""
        return self._depths

    @property
    def results(self):
        """Return all the results."""
        return self._result_list

    @property
    def heavy_outputs(self):
        """Return the ideal heavy outputs dictionary."""
        return self._heavy_outputs

    @property
    def heavy_output_prob(self):
        """Return the heavy output probability as measured."""
        return self._heavy_output_prob

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
        Add the ideal results and convert to the heavy outputs
        Assume the result is from 'statevector_simulator'

        Args:
            new_statevector_result: ideal results
        """

        if new_statevector_result is None:
            return

        if not isinstance(new_statevector_result, list):
            new_statevector_result = [new_statevector_result]

        for result in new_statevector_result:
            for qvcirc in result.results:

                circname = qvcirc.header.name

                if circname in self._heavy_outputs:
                    raise QiskitError("Already added the ideal result "
                                      "for circuit %s" % circname)

                # convert the result into probability dictionary
                qstate = result.get_statevector(circname)
                pvector = np.multiply(qstate, qstate.conjugate())
                format_spec = "{0:0%db}" % self._width
                pmap = {format_spec.format(b):
                        float(np.real(pvector[b]))
                        for b in range(2**self._width)}
                median_prob = self._median_probabilities([pmap])
                self._heavy_outputs[qvcirc.header.name] = \
                    self._heavy_strings(pmap, median_prob[0])

                # calculate the heavy output probability
                self._heavy_output_prob_ideal[circname] = \
                    self._subset_probability(
                        self._heavy_outputs[circname],
                        pmap)

    def add_data(self, new_backend_result, rerun_fit=True):
        """
        Add a new result. Re calculate fit

        Args:
            new_backend_result: list of qv results
            rerun_fit: re caculate the means and fit the result

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
        Make a count dictionary for each unique circuit from all the
        results.

        Calculate the heavy output probability

        Additional information:
            Assumes that 'result' was executed is
            the output of circuits generated by qv_ciruits,
        """

        circ_counts = {}
        circ_shots = {}
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

                circ_counts[circ_name] = \
                    build_counts_dict_from_list(count_list)

                circ_shots[circ_name] = sum(circ_counts[circ_name].values())

                # normalize
                for state in circ_counts[circ_name]:
                    circ_counts[circ_name][state] /= circ_shots[circ_name]

                # calculate the heavy output probability
                self._heavy_output_prob[circ_name] = \
                    self._subset_probability(
                        self._heavy_outputs[circ_name],
                        circ_counts[circ_name])

    def calc_statistics(self):
        """
        Convert the heavy outputs in the different trials into mean and std
        for plotting
        """

        self._ydata = np.zeros([4, len(self._depths)], dtype=float)

        exp_vals = np.zeros(self._ntrials, dtype=float)
        ideal_vals = np.zeros(self._ntrials, dtype=float)

        for depthidx, depth in enumerate(self._depths):

            for trialidx in range(self._ntrials):
                cname = 'qv_depth_%d_trial_%d' % (depth, trialidx)
                exp_vals[trialidx] = self._heavy_output_prob[cname]
                ideal_vals[trialidx] = self._heavy_output_prob_ideal[cname]

            self._ydata[0][depthidx] = np.mean(exp_vals)
            self._ydata[1][depthidx] = np.std(exp_vals)
            self._ydata[2][depthidx] = np.mean(ideal_vals)
            self._ydata[3][depthidx] = np.std(ideal_vals)

    def plot_qv_data(self, ax=None, show_plt=True):
        """
        Plot the qv data as a function of depth

        Args:
            ax (Axes or None): plot axis (if passed in).
            add_label (bool): Add an EPC label
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

        xdata = self._depths

        # Plot the experimental data with error bars
        ax.errorbar(xdata, self._ydata[0],
                    yerr=self._ydata[1],
                    color='r', linestyle=None, marker='o', markersize=5,
                    label='Exp')

        # Plot the ideal data with error bars
        ax.errorbar(xdata, self._ydata[2],
                    yerr=self._ydata[3],
                    color='b', linestyle=None, marker='o', markersize=5,
                    label='Ideal')

        # Plot the threshold
        ax.plot(xdata,
                np.ones(len(xdata))*2.0/3.0,
                color='black', linestyle='--', linewidth=2, label='Threshold')
        ax.tick_params(labelsize=14)

        ax.set_xlabel('Depth', fontsize=16)
        ax.set_ylabel('Heavy Probability', fontsize=16)
        ax.grid(True)

        ax.legend()

        if show_plt:
            plt.show()

    def qv_success(self):
        """Return whether each depth was successful (>2/3) and with what
        confidence.

        Returns:
            List of lenth depth with eact element a 3 list with
            - success True/False
            - error in the mean
            - confidence
        """

        success_list = []

        for depth_ind, _ in enumerate(self._depths):
            success_list.append([])
            hmean = self._ydata[0][depth_ind]
            success_list[-1].append(hmean > (2/3))
            success_list[-1].append(self._ydata[1][depth_ind] /
                                    self._ntrials**0.5)
            if success_list[-1][0]:
                cfd = 0.5*(1 +
                           math.erf((hmean - 2/3)
                                    / (1e-10 + success_list[-1][1])/2**0.5))
                success_list[-1].append(cfd)
            else:
                success_list[-1].append(0)

        return success_list

    def quantum_volume(self):
        """Return the volume for each depth.

        Returns:
            List of quantum volumes
        """

        qv_list = np.min([self._depths, len(self._qubit_list) *
                          np.ones(len(self._depths))], axis=0)
        qv_list = 2**qv_list

        return qv_list

    def _heavy_strings(self, ideal_distribution, ideal_median):
        """Return the set of heavy output strings.

        ideal_distribution = dict of ideal output distribution
        where keys are bit strings (as strings) and values are
        probabilities of observing those strings
        ideal_mean = median probability across all outputs

        Return the set of heavy output strings, i.e. those strings
        whose ideal probability of occurrence exceeds the median.
        """
        return list(filter(lambda x: ideal_distribution[x] > ideal_median,
                           list(ideal_distribution.keys())))

    def _median_probabilities(self, distributions):
        """Return a list of median probabilities.

        distributions = list of dicts mapping binary strings
        (as strings) to probabilities.

        Return a list of median probabilities.
        """
        medians = []
        for dist in distributions:
            values = np.array(list(dist.values()))
            medians.append(float(np.real(np.median(values))))

        return medians

    def _subset_probability(self, strings, distribution):
        """Return the probability of a subset of outcomes.

        strings = list of bit strings (as strings)
        distribution = dict where keys are bit strings (as strings)
        and values are probabilities of observing those strings

        Return the probability of the subset of strings, i.e. the sum
        of the probabilities of each string as given by the
        distribution.
        """
        return sum([distribution.get(value, 0) for value in strings])
