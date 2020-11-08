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
        self._exp_id = None
        generator = QuantumVolumeGenerator(qubits, seed=seed)
        analysis = QuantumVolumeAnalysis()
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

    def add_data(self, new_trials: Optional(int)=1):
        """generate more trials of the Quantum volume circuits"""
        for _ in range(new_trials):
            new_circuits_no_meas = self.generator.circuits()
            for new_circ_no_meas in new_circuits_no_meas:
                new_circuit = copy.deepcopy(new_circ_no_meas)
                new_circuit.measure_active()
                self._circuits.append(new_circuit)
                self._circuit_no_meas.append(new_circ_no_meas)
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
        # transpile and assemble only the new circuits
        circuits = transpile(self._circuits[self._exe_trials * len(self.qubits):],
                             backend=backend,
                             initial_layout=self.generator.qubits)
        # Get the circuits without the measurement
        statevector_backend = qiskit.Aer.get_backend('statevector_simulator')
        circuits_no_meas = transpile(self._circuit_no_meas[self._exe_trials * len(self.qubits):],
                                     backend=statevector_backend,
                                     initial_layout=self.generator.qubits)
        metadata = self.generator.metadata()

        for meta in metadata:
            meta['name'] = self.generator.name
            meta['exp_id'] = self._exp_id
            meta['qubits'] = self.generator.qubits

        # Assemble qobj of the ideal results and submit to the statevector simulator
        ideal_qobj = assemble(circuits_no_meas,
                              backend=statevector_backend,
                              qobj_header={'metadata': metadata},
                              **kwargs)
        self._ideal_job.append(statevector_backend.run(ideal_qobj))

        # Assemble qobj and submit to backend
        qobj = assemble(circuits,
                        backend=backend,
                        qobj_header={'metadata': metadata},
                        **kwargs)
        self._job.append(backend.run(qobj))
        return self

    def run_analysis(self, **params):
        """Analyze the stored data.

        Returns:
            any: the output of the analysis,
        """
        self.analysis.clear_data()
        self.analysis.add_ideal_data(self._ideal_job)
        self.analysis.add_data(self._job)
        result = self.analysis.run(**params)
        return result


class QuantumVolumeGenerator(Generator):
    """Quantum volume experiment generator."""

    # pylint: disable=arguments-differ
    def __init__(self, qubits: Union[int, List[int]],
                 seed: Optional[Union[int, Generator]] = None,
                 trial_number: Optional(int) = 1):
        self._seed = seed
        self._trial_number = trial_number
        if isinstance(qubits, int):
            qubits = [range(qubits)]
        super().__init__('qv_depth_%d_to_depth_%d' % (len(qubits[0]), len(qubits[-1])),
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
            qv_circ.name = 'qv_depth_%d_trial_%d' % (self.num_qubits, self._trial_number)
            qv_circuits.append(qv_circ)
        return qv_circuits

    def _extra_metadata(self) -> List[dict]:
        """Generate a list of experiment circuits metadata."""
        return [{"depth": qubit,
                 "trail_number": self._trial_number,
                 "seed": self._seed}
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

    def _analysis_fn(self, data: List[any], metadata: List[Dict[str, any]],
                     **kwargs: Dict[str, any]):
        """
        the analysis function
        Args:
            data:
            metadata:
            **kwargs:

        Returns:

        """


    def run(self):
        """Analyze the quantum volume circuits data.

        Returns:
            any: the output of the analysis, which is the QV data
        """
        self._result = self._analysis_fn(self.data, self.metadata)
        return self._result

    def add_data(self,
                 data: Union[BaseJob, Result, any],
                 ideal_data: Union[BaseJob, Result, any],
                 metadata: Optional[Dict[str, any]] = None):
        """Add additional data to the fitter.

        Args:
                data: input data for the fitter.
                metadata: Optional, list of metadata dicts for input data.
                          if None will be taken from data Result object.

        Raises:
            QiskitError: if input data is incorrectly formatted.
        """
        super().add_data(data, metadata)
        super().add_data(ideal_data, metadata)
        if data is None:
            return

        if isinstance(data, BaseJob):
            data = data.result()

        if isinstance(ideal_data, BaseJob):
            ideal_data = ideal_data.result()

        if isinstance(data, Result):
            # Extract metadata from result object if not provided
            if metadata is None:
                if not hasattr(data.header, "metadata"):
                    raise QiskitError("Experiment is missing metadata.")
                metadata = data.header.metadata

            # Get data from result
            new_data = []
            new_meta = []
            for i, meta in enumerate(metadata):
                if self._accept_data(meta):
                    new_data.append(self._format_data(data, meta, i))
                    new_meta.append(meta)
        else:
            # Add general preformatted data
            if not isinstance(data, list):
                data = [data]

            if metadata is None:
                # Empty metadata incase it is not needed for a given experiment
                metadata = len(data) * [{}]
            elif not isinstance(metadata, list):
                metadata = [metadata]

            # Filter data
            new_data = []
            new_meta = []
            for i, meta in enumerate(metadata):
                if self._accept_data(meta):
                    new_data.append(data)
                    new_meta.append(meta)

        # Add extra data
        self._exp_data += new_data
        self._exp_metadata += new_meta

        # Check metadata and data are same length
        if len(self._exp_metadata) != len(self._exp_data):
            raise QiskitError("data and metadata lists must be the same length")

    def _add_ideal_data(self, ideal_data: Union[BaseJob, Result, any],
                        metadata: Optional[Dict[str, any]] = None):
        """

        Returns:

        """
        if ideal_data is None:
            return

        if isinstance(ideal_data, BaseJob):
            ideal_data = ideal_data.result()

        if isinstance(ideal_data, BaseJob):
            ideal_data = ideal_data.result()

        if isinstance(ideal_data, Result):
            trials = len(ideal_data.results) / len(self._qubit_lists)
            ideal_results = np.reshape(ideal_data.results, (trials, len(self._qubit_lists)))

            exp_num = 0
            for result_trial in ideal_results:
                for result in result_trial:

                    circ_name = metadata[exp_num]["name"]

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

        Returns:

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
                if (ntrials_circ + 1) > self._ntrials:
                    self._ntrials = ntrials_circ + 1

                if qvcirc.header.name not in self._heavy_output_prob_ideal:
                    raise QiskitError('Ideal distribution '
                                      'must be loaded first')

    def _format_data(self, data: Result,
                     metadata: Dict[str, any],
                     index: int):
        """Format the required data from a Result.data dict

        Additional Information:
            This extracts counts from the experiment result data.
            Analysis subclasses can override this method to extract
            different data types from circuit results.
        """
        # Derived classes should override this method to filter
        # only the required data.
        # The default behavior filters on counts.
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
                {format_spec.format(b): float(np.real(pvector[b])) for b in range(2 ** depth)}

            median_prob = self._median_probabilities([self._all_output_prob_ideal[circ_name]])
            self._heavy_outputs[circ_name] = \
                self._heavy_strings(self._all_output_prob_ideal[circ_name], median_prob[0])

            # calculate the heavy output probability
            self._heavy_output_prob_ideal[circ_name] = \
                self._subset_probability(
                    self._heavy_outputs[circ_name],
                    self._all_output_prob_ideal[circ_name])


        return data.get_counts(index)

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