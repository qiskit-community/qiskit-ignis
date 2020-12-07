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
import uuid
from typing import Optional, Union, List

from numpy import random
import qiskit

from qiskit.ignis.experiments.base import Experiment
from qiskit.ignis.experiments.quantum_volume.qv_generator import QuantumVolumeGenerator
from qiskit.ignis.experiments.quantum_volume.qv_analysis import QuantumVolumeAnalysis
from qiskit import transpile, assemble
from qiskit.providers import BaseJob
from qiskit.exceptions import QiskitError
from qiskit.providers import BaseBackend


class QuantumVolumeExperiment(Experiment):
    """Quantum volume experiment."""

    # pylint: disable=arguments-differ
    def __init__(self,
                 qubits: Union[int, List[int]],
                 trials: Optional[int] = 0,
                 job: Optional = None,
                 seed: Optional[Union[int, random.Generator]] = None,
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
            QuantumVolumeResult: the output of the analysis
        """
        self.analysis.clear_data()
        for job_index in range(len(self._job)):
            self.analysis.add_data(self._job[job_index])
            self.analysis.add_ideal_data(self._ideal_job[job_index])
        result = self.analysis.run(**params)
        return result
