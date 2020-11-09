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
import itertools
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
from ...utils import build_counts_dict_from_list


class QuantumVolumeExperiment(Experiment):
    """Quantum volume experiment."""

    # pylint: disable=arguments-differ
    def __init__(self,
                 qubits: Union[int, List[int]],
                 job: Optional = None,
                 seed: Optional[Union[int, Generator]] = None):

        if isinstance(qubits, int):
            qubits = [range(qubits)]
        self._qubits = qubits
        self._trials = 0
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
        statevector_backend = qiskit.Aer.get_backend('statevector_simulator')
        circuits_no_meas = transpile(self._circuit_no_meas[init_index_new_circuits:],
                                     backend=statevector_backend,
                                     initial_layout=self.qubits * new_trials)

        # Assemble qobj of the ideal results and submit to the statevector simulator
        ideal_qobj = assemble(circuits_no_meas,
                              backend=statevector_backend,
                              qobj_header={'metadata': self._metadata[init_index_new_circuits:]},
                              **kwargs)
        self._ideal_job.append(statevector_backend.run(ideal_qobj))

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
        self.analysis.clear_data()
        self.analysis.add_data([self._ideal_job, self._job])
        result = self.analysis.run(**params)
        return result


class QuantumVolumeGenerator(Generator):
    """Quantum volume experiment generator."""

    # pylint: disable=arguments-differ
    def __init__(self, qubits: Union[int, List[int]],
                 seed: Optional[Union[int, Generator]] = None,
                 trial_number: Optional[int] = 1):
        self._seed = seed
        self._trial_number = trial_number
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
    """Quantum volume experiment analysis."""

    # pylint: disable=arguments-differ
    def __init__(self,
                 data: Optional[any] = None,
                 ideal_data: Optional[any] = None,
                 qubit_lists: Optional[Union[int, List[int]]] = None,
                 metadata: Optional[Dict[str, any]] = None,
                 exp_id: Optional[str] = None):
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

        super().__init__(data=data,
                         metadata=metadata,
                         name='QV',
                         exp_id=exp_id)

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

    def run(self):
        """Analyze the quantum volume circuits data.

        Returns:
            any: the output of the analysis, which is the QV data
        """
        self._calc_data()
        self._calc_statistics()
        # calc the QV
        qv_success_list = self.qv_success()
        qv_list = self._ydata
        qv = 1
        for qidx in range(len(self._qubit_lists)):
            if qv_list[0][qidx] > 2 / 3:
                if qv_success_list[qidx][0]:
                    qv = self.quantum_volume()[qidx]

        self._result = qv
        return self._result

    def print_results(self):
        """
        print the percentage of successful trials and the confidence level of each depth
        """
        qv_success_list = self.qv_success()
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

    def add_data(self,
                 data: Union[BaseJob, Result, any],
                 metadata: Optional[Dict[str, any]] = None):
        """Add additional data to the Quantum Volume analyser.

        Args:
                data: input data. must include both ideal and non-ideal results, in the form of
                      a list [ideal results, non-ideal results].
                metadata: Optional, list of metadata dicts for input data.
                          if None will be taken from data Result object.
                          can have one medatada for both ideal and non-ideal data

        Raises:
            QiskitError: if input data is incorrectly formatted.
        """
        if data is None:
            return
        if not isinstance(data, list) or len(data) != 2:
            raise QiskitError("data must contain both ideal and non ideal data in the format of"
                              " [ideal data, non ideal data]")
        if isinstance(data[0], list) and isinstance(data[1], list) and \
                        len(data[0]) != len(data[1]):
            raise QiskitError("ideal data and non ideal data must be of the same length")
        self._add_ideal_data(data[0], metadata)
        self._add_nonideal_data(data[1], metadata)

    def _add_ideal_data(self, ideal_data: Union[BaseJob, Result, any],
                        metadata: Optional[Dict[str, any]] = None):
        """
        Add additional ideal data to the QV analyser.
        Args:

        Raises:
            QiskitError: if input data was already added.
        """
        if ideal_data is None:
            return

        if isinstance(ideal_data, BaseJob):
            ideal_data = ideal_data.result()

        if isinstance(ideal_data, BaseJob):
            ideal_data = ideal_data.result()

        if isinstance(ideal_data, Result):
            # Extract metadata from result object if not provided
            if metadata is None:
                if not hasattr(ideal_data.header, "metadata"):
                    raise QiskitError("Experiment is missing metadata.")
                metadata = ideal_data.header.metadata

            trials = len(ideal_data.results) / len(self._qubit_lists)
            ideal_results = np.reshape(ideal_data.results, (trials, len(self._qubit_lists)))

            exp_num = 0
            for result_trial in ideal_results:
                for result in result_trial:

                    circ_name = metadata[exp_num]["circ_name"]

                    # get the depth/width from the circuit name
                    # qv_depth_%d_trial_%d
                    depth = metadata[exp_num]["depth"]

                    if circ_name in self._heavy_outputs:
                        raise QiskitError("Already added the ideal result "
                                          "for circuit %s" % circ_name)

                    # convert the result into probability dictionary
                    qstate = result.statevector
                    pvector = np.multiply(qstate, qstate.conjugate())
                    format_spec = "{0:0%db}" % depth
                    self._all_output_prob_ideal[circ_name] = \
                        {format_spec.format(b): float(np.real(pvector[b])) for b in range(2 ** depth)}

                    median_prob = self._median_probabilities([self._all_output_prob_ideal[circ_name]])
                    self._heavy_outputs[circ_name] = \
                        self._heavy_strings(self._all_output_prob_ideal[circ_name], median_prob[0])

                    # calculate the heavy output probability
                    self._heavy_output_prob_ideal[circ_name] = \
                        self._subset_probability(
                            self._heavy_outputs[circ_name],
                            self._all_output_prob_ideal[circ_name])

                    exp_num += 1


    def _add_nonideal_data(self,
                           data: Union[BaseJob, Result, any],
                           metadata: Optional[Dict[str, any]] = None):
        """
        Add additional non ideal data to the QV analyser.
        Args:

        Raises:
            QiskitError: if input data was already added.
        """

        if data is None:
            return

        if isinstance(data, BaseJob):
            data = data.result()

        if isinstance(data, BaseJob):
            data = data.result()

        if isinstance(data, Result):
            # Extract metadata from result object if not provided
            if metadata is None:
                if not hasattr(data.header, "metadata"):
                    raise QiskitError("Experiment is missing metadata.")
                metadata = data.header.metadata

            trials = len(data.results) / len(self._qubit_lists)
            data = np.reshape(data.results, (trials, len(self._qubit_lists)))

            exp_num = 0
            for result in data:
                self._result_list.append(result)

                # update the number of trials *if* new ones
                # added.
                for _ in result.results:
                    circ_name = metadata[exp_num]["circ_name"]
                    if circ_name not in self._heavy_output_prob_ideal:
                        raise QiskitError('Ideal distribution '
                                          'must be loaded first')
                    exp_num += 1
            self._ntrials += trials

    def _calc_data(self):
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
                    self._heavy_output_counts[circ_name] / self._circ_shots[circ_name]

                # calculate the experimental heavy output probability
                self._heavy_output_prob_exp[circ_name] = \
                    self._heavy_output_counts[circ_name] / self._circ_shots[circ_name]

    def _calc_statistics(self):
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
            self._ydata[0][depthidx] = np.sum(exp_vals) / np.sum(exp_shots)
            self._ydata[1][depthidx] = (self._ydata[0][depthidx] *
                                        (1.0 - self._ydata[0][depthidx])
                                        / self._ntrials) ** 0.5

            # Calculate mean and error for ideal data
            self._ydata[2][depthidx] = np.mean(ideal_vals)
            self._ydata[3][depthidx] = (self._ydata[2][depthidx] *
                                        (1.0 - self._ydata[2][depthidx])
                                        / self._ntrials) ** 0.5

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

    def quantum_volume(self):
        """Return the volume for each depth.

        Returns:
            list: List of quantum volumes
        """

        qv_list = 2 ** np.array(self._depths)

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
