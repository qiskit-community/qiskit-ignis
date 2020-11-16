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
Quantum volume Experiment.
"""

import copy
import warnings
import math

import numpy as np
import uuid
import qiskit

from qiskit.circuit.library import QuantumVolume
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.result import Result

from typing import Optional, Dict, Union, List
from numpy.random import Generator
from qiskit.ignis.experiments.base import Experiment, Generator, Analysis

from qiskit import transpile, assemble
from qiskit.providers import BaseJob
from qiskit.exceptions import QiskitError
from qiskit.providers import BaseBackend

# for the Result object
from qiskit.visualization import plot_histogram
try:
    from matplotlib import get_backend
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

class QuantumVolumeExperiment(Experiment):
    """Quantum volume experiment."""

    # pylint: disable=arguments-differ
    def __init__(self,
                 qubits: Union[int, List[int]],
                 trials: Optional[int] = 0,
                 job: Optional = None,
                 seed: Optional[Union[int, Generator]] = None,
                 simulation_backend: Optional[BaseBackend] = None):

        if isinstance(qubits, int):
            qubits = [range(qubits)]
        self._qubits = qubits
        self._trials = 0
        self._simulation_backend = simulation_backend
        self._exe_trials = 0
        self._circuits = []
        self._circuit_no_meas = []
        self._ideal_job = []
        self._metadata = []
        self._exp_id = str(uuid.uuid4())
        generator = QuantumVolumeGenerator(qubits, seed=seed)
        analysis = QuantumVolumeAnalysis(qubit_lists=qubits)
        super().__init__(generator=generator, analysis=analysis, job=job)
        if job is None:
            self._job = []
        else:
            self._job = [job]
        if self._simulation_backend is None:
            self._simulation_backend = qiskit.Aer.get_backend('statevector_simulator')
        if trials != 0:
            self.add_data(trials)

    @property
    def circuits(self):
        """Return experiment circuits"""
        return self._circuits

    @property
    def circuit_no_meas(self):
        """Return experiment circuits without measurement"""
        return self._circuit_no_meas

    @property
    def qubits(self):
        """Return experiment qubits"""
        return self._qubits

    @qubits.setter
    def qubits(self, value):
        """Set experiment qubits"""
        if not isinstance(value, (List[int], int)):
            raise QiskitError("Invalid qubits object. must be int or List[int]")
        self._qubits = value

    def add_data(self, new_trials: int = 1):
        """generate more trials of the Quantum volume circuits"""
        for _ in range(new_trials):
            new_circuits_no_meas = self.generator.circuits()
            for new_circ_no_meas in new_circuits_no_meas:
                new_circuit = copy.deepcopy(new_circ_no_meas)
                new_circuit.measure_active()
                self._circuits.append(new_circuit)
                self._circuit_no_meas.append(new_circ_no_meas)
            for meta in self.generator.metadata():
                meta['exp_id'] = self._exp_id
                self._metadata.append(meta)
            self._trials += 1

    def execute(self, backend: BaseBackend, **kwargs) -> BaseJob:
        """Execute the experiment on a backend.
​
        TODO: Add transpiler options

        Args:
            backend: backend to run experiment on.
            kwargs: kwargs for assemble method.
​
        Returns:
            BaseJob: the experiment job.
        """
        if self._trials == 0:
            warnings.warn("No data was generated. try to use 'add_data' function first")
        if self._exp_id is None:
            self._exp_id = str(uuid.uuid4())
        # Get circuits and metadata
        new_trials = self._trials - self._exe_trials
        init_index_new_circuits = self._exe_trials * len(self.qubits)
        # transpile and assemble only the new circuits
        circuits = transpile(self._circuits[init_index_new_circuits:],
                             backend=backend,
                             initial_layout=self.qubits * new_trials)
        # Get the circuits without the measurement
        circuits_no_meas = transpile(self._circuit_no_meas[init_index_new_circuits:],
                                     backend=self._simulation_backend,
                                     initial_layout=self.qubits * new_trials)

        # Assemble qobj of the ideal results and submit to the statevector simulator
        ideal_qobj = assemble(circuits_no_meas,
                              backend=self._simulation_backend,
                              qobj_header={'metadata': self._metadata[init_index_new_circuits:]},
                              **kwargs)
        self._ideal_job.append(self._simulation_backend.run(ideal_qobj))

        # Assemble qobj and submit to backend
        qobj = assemble(circuits,
                        backend=backend,
                        qobj_header={'metadata': self._metadata[init_index_new_circuits:]},
                        **kwargs)
        self._job.append(backend.run(qobj))
        # updated the amount of executed trials
        self._exe_trials = self._trials
        return self

    def run_analysis(self, **params):
        """Analyze the stored data.

        Returns:
            any: the output of the analysis,
        """
        # assuming only latest job has to be added
        # TODO: add support for adding multiple jobs at once to the analysis
        self.analysis.add_data(self._job[-1])
        self.analysis.add_ideal_data(self._ideal_job[-1])
        result = self.analysis.run(**params)
        return result


class QuantumVolumeGenerator(Generator):
    """Quantum volume experiment generator."""

    # pylint: disable=arguments-differ
    def __init__(self, qubits: Union[int, List[int]],
                 seed: Optional[Union[int, Generator]] = None):
        self._seed = seed
        self._trial_number = 0
        if isinstance(qubits, int):
            qubits = [range(qubits)]
        # for circuit generation, the pysical qubits numbers are not important
        qubits = [len(qubit) for qubit in qubits]
        super().__init__('qv_depth_%d_to_depth_%d' % (qubits[0], qubits[-1]),
                         qubits)

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of quantum volume circuits (circuit for each depth).
        the circuits returned by this method do not contain measurements"""
        qv_circuits = []

        # Initialize RNG
        if isinstance(self._seed, int):
            rng = np.random.default_rng(self._seed)
        else:
            _seed = self._seed

        for depth in self.qubits:
            if isinstance(self._seed, int):
                _seed = rng.integers(1000)
            qv_circ = QuantumVolume(depth, depth, seed=_seed)
            qv_circ.name = 'qv_depth_%d_trial_%d' % (depth, self._trial_number)
            qv_circuits.append(qv_circ)
        self._trial_number += 1
        return qv_circuits

    def _extra_metadata(self) -> List[dict]:
        """Generate a list of experiment circuits metadata."""
        return [{"depth": qubit,
                 "trail_number": self._trial_number,
                 "seed": self._seed,
                 "circ_name": 'qv_depth_%d_trial_%d' % (qubit, self._trial_number)}
                for qubit in self.qubits]


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

    @property
    def depths(self):
        """Return depth list."""
        return self._depths

    @property
    def qubit_lists(self):
        """Return depth list."""
        return self._qubit_lists

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

    def fit(self, data, metadata):
        """
        fit the data, calculate the heavy output probability and calculate the quantum volume.

        also convert the heavy outputs in the different trials into mean and error for plotting.
        Here we assume the error is due to a binomial distribution.
        Error (standard deviation) for binomial distribution is sqrt(np(1-p)),
        where n is the number of trials (self._ntrials) and
        p is the success probability (self._ydata[0][depthidx]/self._ntrials).

        Returns:
            int: quantum volume
        """
        heavy_output_counts = {}
        circ_shots = {}
        all_output_prob_ideal = {}
        heavy_output_prob_ideal = {}
        heavy_output_prob_exp = {}
        heavy_outputs = {}

        for exp_num, ideal_result in enumerate(self._exp_ideal_data):
            circ_name = self._exp_metadata[exp_num]["circ_name"]
            depth = self._exp_metadata[exp_num]["depth"]

            # convert the result into probability dictionary
            pvector = np.multiply(ideal_result, ideal_result.conjugate())
            format_spec = "{0:0%db}" % depth
            all_output_prob_ideal[circ_name] = \
                {format_spec.format(b): float(np.real(pvector[b])) for b in range(2 ** depth)}

            median_prob = self._median_probabilities([all_output_prob_ideal[circ_name]])
            heavy_outputs[circ_name] = \
                self._heavy_strings(all_output_prob_ideal[circ_name], median_prob[0])

            # calculate the heavy output probability
            heavy_output_prob_ideal[circ_name] = \
                self._subset_probability(
                    heavy_outputs[circ_name],
                    all_output_prob_ideal[circ_name])

        for exp_num, result in enumerate(self._exp_data):
            circ_name = self._exp_metadata[exp_num]["circ_name"]

            circ_shots[circ_name] = sum(result.values())

            # calculate the experimental heavy output counts
            heavy_output_counts[circ_name] = \
                self._subset_probability(heavy_outputs[circ_name], result)

            # calculate the experimental heavy output probability
            heavy_output_prob_exp[circ_name] = \
                heavy_output_counts[circ_name] / circ_shots[circ_name]

        # calculate the mean and error
        self._ydata = np.zeros([4, len(self._depths)], dtype=float)
        trials = int(len(self._exp_data) / len(self._qubit_lists))

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

        quantum_volume = self.calc_quantum_volume()
        success_list = self.qv_success()
        result = QuantumVolumeResult(quantum_volume,
                                     success_list,
                                     self.qubit_lists,
                                     trials,
                                     heavy_output_counts,
                                     circ_shots,
                                     result.values(),
                                     all_output_prob_ideal,
                                     heavy_output_prob_ideal,
                                     heavy_output_prob_exp,
                                     heavy_outputs,
                                     self._ydata)
        return result

    def qv_success(self):
        """Return whether each depth was successful (> 2/3 with confidence
        level > 0.977 corresponding to z_value = 2) and the confidence level.

        Returns:
            list: List of list of 2 elements for each depth:
            - success True/False
            - confidence level
        """

        success_list = []
        confidence_level_threshold = self._calc_confidence_level(z_value=2)

        for depth_ind, _ in enumerate(self._depths):
            success_list.append([False, 0.0])
            hmean = self._ydata[0][depth_ind]
            sigma = self._ydata[1][depth_ind]
            z_value = self._calc_z_value(hmean, sigma)
            confidence_level = self._calc_confidence_level(z_value)
            success_list[-1][1] = confidence_level

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

    def calc_quantum_volume(self):
        """
        calc the quantum volume of the analysed system
        Returns:
            int: quantum volume
        """
        qv_success_list = self.qv_success()
        qv_list = self._ydata
        quantum_volume = 1
        for qidx in range(len(self._qubit_lists)):
            if qv_list[0][qidx] > 2 / 3:
                if qv_success_list[qidx][0]:
                    quantum_volume = self._quantum_volume()[qidx]

        return quantum_volume

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

class QuantumVolumeResult():
    """
    Quantum volume result object
    """
    # TODO: inherit from a general ExperimentResultObject
    def __init__(self,
                 quantum_volume,
                 success_list,
                 qubit_lists,
                 ntrials,
                 heavy_output_counts,
                 circ_shots,
                 circ_counts,
                 all_output_prob_ideal,
                 heavy_output_prob_ideal,
                 heavy_output_prob_exp,
                 heavy_outputs,
                 ydata):
        self.quantum_volume = quantum_volume
        self._success_list = success_list
        self._qubit_lists = qubit_lists
        self._ntrials = ntrials
        self.circ_shots = circ_shots
        self._circ_counts = circ_counts
        self._all_output_prob_ideal = all_output_prob_ideal
        self._heavy_output_prob_exp = heavy_output_prob_exp
        self._ydata = ydata
        self._depths = [len(qubit_list) for qubit_list in qubit_lists]

        # not used in plotting
        self._heavy_output_counts = heavy_output_counts
        self._heavy_output_prob_ideal = heavy_output_prob_ideal
        self._heavy_outputs = heavy_outputs

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

    def print_results(self):
        """
        print the percentage of successful trials and the confidence level of each depth
        """
        qv_success_list = self._success_list
        qv_list = self._ydata
        for qidx, qubit_list in enumerate(self._qubit_lists):
            if qv_list[0][qidx] > 2 / 3:
                if qv_success_list[qidx][0]:
                    print("Width/depth %d greater than 2/3 (%f) with confidence %f (successful)." %
                          (len(qubit_list), qv_list[0][qidx], qv_success_list[qidx][1]))
                else:
                    print("Width/depth %d greater than 2/3 (%f) with confidence %f "
                          "(unsuccessful)." %
                          (len(qubit_list), qv_list[0][qidx], qv_success_list[qidx][1]))
            else:
                print("Width/depth %d less than 2/3 (unsuccessful)." % len(qubit_list))

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
                    yerr=self._ydata[1] * 2,
                    color='r', marker='o',
                    markersize=6, capsize=5,
                    elinewidth=2, label='Exp (2$\\sigma$ error)')

        # Plot the ideal data with error bars
        ax.errorbar(xdata, self._ydata[2],
                    yerr=self._ydata[3] * 2,
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

        Returns:
            matplotlib.Figure:
                A figure for histogram of ideal and experiment probabilities.
        """
        circ_name = "qv_depth_{}_trial_{}".format(depth, trial_index)
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
        for i in range(int(len(bars) / 2), len(bars) - 1):
            bars[i].fill = False
            # set non-black edge color to increase bar labels legibility
            bars[i].set_edgecolor('saddlebrown')

        # show experimental heavy output probability to the legend
        ax.plot([], [], ' ', label='HOP~{:.3f}'.format(self._heavy_output_prob_exp[circ_name]))

        # plot median probability
        median_prob = self._median_probabilities([self._all_output_prob_ideal[circ_name]])
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

        trial_list = np.arange(1, self._ntrials)  # x data
        hop_list = []  # y data

        for trial_index in trial_list:
            circ_name = 'qv_depth_{}_trial_{}'.format(depth, trial_index)
            hop_list.append(self._heavy_output_prob_exp[circ_name])

        hop_accumulative = np.cumsum(hop_list) / np.arange(1, self._ntrials + 1)
        two_sigma = 2 * (hop_accumulative * (1 - hop_accumulative) /
                         np.arange(1, self._ntrials + 1)) ** 0.5

        # plot two-sigma shaded area
        ax.errorbar(trial_list, hop_accumulative, fmt="none", yerr=two_sigma, ecolor='lightgray',
                    elinewidth=20, capsize=0, alpha=0.5, label='2$\\sigma$')
        # plot accumulative HOP
        ax.plot(trial_list, hop_accumulative, color='r', label='Cumulative HOP')
        # plot inidivual HOP as scatter
        ax.scatter(trial_list, hop_list, s=3, zorder=3, label='Individual HOP')
        # plot 2/3 success threshold
        ax.axhline(2 / 3, color='k', linestyle='dashed', linewidth=1, label='Threshold')

        ax.set_xlim(0, self._ntrials)
        ax.set_ylim(hop_accumulative[-1] - 4 * two_sigma[-1],
                    hop_accumulative[-1] + 4 * two_sigma[-1])

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
