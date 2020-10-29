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

from typing import Optional, Dict, Union, List
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
                 seed: Optional(int)=None):

        if isinstance(qubits, int):
            qubits = [range(qubits)]
        self._qubits = qubits
        self._trials = 0
        self._seed = seed
        self._circuits = [[] for _ in range(len(self._qubits))]
        self._circuit_no_meas = [[] for _ in range(len(self._qubits))]
        self._ideal_job = None
        generator = QuantumVolumeGenerator(qubits)
        analysis = QuantumVolumeAnalysis()
        super().__init__(generator=generator, analysis=analysis, job=job)

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

    def add_data(self, trials: Optional(int)=1):
        """generate more trials of the Quantum volume circuits"""
        for _ in range(trials):
            new_circuits_no_meas = self.generator.circuits()
            for depth in range(len(self._qubits)):
                new_circuit = copy.deepcopy(new_circuits_no_meas[depth])
                new_circuit.measure_active()
                self._circuits[depth].append(new_circuit)
                self._circuit_no_meas[depth].append(new_circuits_no_meas[depth])
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
        # Get circuits and metadata
        exp_id = str(uuid.uuid4())
        circuits = transpile(self._circuits,
                             backend=backend,
                             initial_layout=self.generator.qubits)
        # Get the circuits without the measurement
        statevector_backend = qiskit.Aer.get_backend('statevector_simulator')
        circuits_no_meas = transpile(self._circuit_no_meas,
                             backend=statevector_backend,
                             initial_layout=self.generator.qubits)
        metadata = self.generator.metadata()

        for meta in metadata:
            meta['name'] = self.generator.name
            meta['exp_id'] = exp_id
            meta['qubits'] = self.generator.qubits

        # Assemble qobj of the ideal results and submit to the statevector simulator
        ideal_qobj = assemble(circuits_no_meas,
                              backend=statevector_backend,
                              qobj_header={'metadata': metadata},
                              **kwargs)
        self._ideal_job = statevector_backend.run(ideal_qobj)

        # Assemble qobj and submit to backend
        qobj = assemble(circuits,
                        backend=backend,
                        qobj_header={'metadata': metadata},
                        **kwargs)
        self._job = backend.run(qobj)
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
    def __init__(self, qubits_list: Union[int, List[int]],
                 seed: Optional(int)=None,
                 trial_number: Optional(int)=1):
        self._seed = seed
        self._trial_number = trial_number
        if isinstance(qubits_list, int):
            qubits_list = [range(qubits_list)]
        super().__init__('qv_depth_%d_to_depth_%d' % (len(qubits_list[0]), len(qubits_list[-1])),
                         qubits_list)

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of quantum volume circuits (circuit for each depth).
        the circuits returned by this method do not contain measurements"""
        qv_circuits = []

        if self._seed:
            rng = np.random.default_rng(self._seed)
        else:
            _seed = None

        for depth in self.qubits:
            if self._seed:
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
                 metadata: Optional[Dict[str, any]] = None,
                 method: Optional[Analysis] = None,
                 mitigator: Optional = None,
                 exp_id: Optional[str] = None):
        super().__init__(data=data,
                         metadata=metadata,
                         name='QV',
                         exp_id=exp_id)
