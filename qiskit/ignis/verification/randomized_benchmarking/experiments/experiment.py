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
Randomized benchmarking experiment classes
"""

import uuid
from typing import List, Optional, Union, Tuple, Dict
from numpy.random import RandomState
from qiskit.ignis.experiments.base import Experiment, Analysis, Generator
from qiskit.providers import BaseJob
from qiskit.providers import BaseBackend
from qiskit import transpile, assemble
from qiskit.exceptions import QiskitError
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.quantum_info.operators.symplectic.clifford import Clifford
from ..dihedral import CNOTDihedral
from . import (RBGenerator, PurityRBGenerator, InterleavedRBGenerator,
               RBAnalysis, InterleavedRBAnalysis, CNOTDihedralRBAnalysis, PurityRBAnalysis)


class RBExperimentBase(Experiment):
    """Base experiment class for randomized benchmarking experiments"""
    def __init__(self,
                 generator: Optional[Generator] = None,
                 analysis: Optional[Analysis] = None):
        super().__init__(generator=generator, analysis=analysis)
        self.reset()

    def get_circuits_and_metadata(self,
                                  seeds: Optional[List[int]] = None
                                  ) -> Tuple[List[QuantumCircuit], List[Dict[str, any]]]:
        """Returns the circuits and metadata related to some (or all) the seeds
            Args:
                seeds: The seeds for which to return the data (all, if `None` is given)
            Returns:
                The pair of circuit list and metadata list
        """
        circuits = []
        metadata = []
        for (circ, meta) in zip(self.generator.circuits(), self.generator.metadata()):
            if seeds is None or meta['seed'] in seeds:
                circuits.append(circ)
                metadata.append(meta)
        return (circuits, metadata)

    def default_basis_gates(self) -> List[str]:
        """The default gate basis used when transpiling the RB circuits"""
        return ['id', 'u1', 'u2', 'u3', 'cx']

    def reset(self):
        """Resets the experiment data"""
        self._jobs = []
        self._kwargs = None
        self._backend = None
        self._basis_gates = self.default_basis_gates()
        self._transpiled_circuits = None
        self._backend = None
        self._kwargs = None

    def run(self, backend: BaseBackend, reset=True, seeds=None, **kwargs) -> any:
        """Run an experiment and perform analysis"""
        # pylint: disable=arguments-differ,attribute-defined-outside-init
        if reset:
            self.reset()
        self._backend = backend
        self._basis_gates = kwargs.pop('basis_gates', self._basis_gates)
        self._kwargs = kwargs
        job = self.execute(backend, seeds, **kwargs)
        self._jobs.append(job)
        self.analysis.add_data(job)
        return self.run_analysis()

    def check_rerun_possible(self):
        """Raises an exception in the case an additional was requested before running the
        experiment for the first time
        """
        if self._backend is None:
            raise QiskitError("Cannot rerun experiment: Use Experiment.run() for the first time.")

    def run_additional_shots(self, shots: int):
        """Runs the existing circuits for additional number of shots"""
        self.check_rerun_possible()
        self._kwargs['shots'] = shots
        return self.run(backend=self._backend, reset=False, **self._kwargs)

    def run_additional_seeds(self, num_of_seeds: int):
        """Adds new seeds to the RB experiment"""
        self.check_rerun_possible()
        new_seeds = self.generator.add_seeds(num_of_seeds)
        return self.run(backend=self._backend, seeds=new_seeds, reset=False, **self._kwargs)

    def execute(self, backend: BaseBackend, seeds: Optional[List[int]] = None, **kwargs) -> BaseJob:
        """Execute the experiment on a backend.
​

        Args:
            backend: backend to run experiment on.
            seeds: which seeds to run
            kwargs: kwargs for assemble method.

​
        Returns:
            BaseJob: the experiment job.
        """
        # pylint: disable=arguments-differ
        # Get circuits and metadata
        exp_id = str(uuid.uuid4())
        circuits, metadata = self.get_circuits_and_metadata(seeds)
        circuits = transpile(circuits,
                             backend=backend,
                             basis_gates=self._basis_gates)
        for meta in metadata:
            meta['name'] = self.generator.name
            meta['exp_id'] = exp_id
            meta['qubits'] = self.generator.qubits

        # Assemble qobj and submit to backend
        qobj = assemble(circuits,
                        backend=backend,
                        qobj_header={'metadata': metadata},
                        **kwargs)
        return backend.run(qobj)

    def reset_data(self, **params):
        """Resets the analyis data and adds the data in the currently stored jobs"""
        self.analysis.clear_data()
        for job in self._jobs:
            self.analysis.add_data(job, **params)

    def run_analysis(self, **params):
        """Analyze the stored data.

        Returns:
            any: the output of the analysis,
        """
        result = self.analysis.run(**params)
        return result

    def gates_per_clifford(self) -> float:
        """Computes the average number of gates per group element in the transpiled circuits"""
        qubits = self.generator.meas_qubits()
        ngates = {qubit: {base: 0 for base in self._basis_gates} for qubit in qubits}
        transpiled_circuits_list = transpile(self.generator.circuits(),
                                             backend=self._backend,
                                             basis_gates=self._basis_gates
                                             )

        for transpiled_circuit in transpiled_circuits_list:
            for instr, qregs, _ in transpiled_circuit.data:
                for qreg in qregs:
                    if qreg.index in ngates and instr.name in ngates[qreg.index]:
                        ngates[qreg.index][instr.name] += 1

        # include inverse, ie + 1 for all clifford length
        length_per_seed = sum([length + 1 for length in self.generator.lengths()])
        total_ncliffs = self.generator.nseeds() * length_per_seed
        for qubit in qubits:
            for base in self._basis_gates:
                ngates[qubit][base] /= total_ncliffs

        return ngates


class RBExperiment(RBExperimentBase):
    """Experiment class for standard RB experiment"""
    def __init__(self,
                 nseeds: int = 1,
                 qubits: List[int] = (0,),
                 lengths: List[int] = (1, 10, 20),
                 group_gates: Optional[str] = None,
                 rand_seed: Optional[Union[int, RandomState]] = None,
                 ):
        """Initialize the RB experiment
            Args:
                nseeds: number of different seeds (random circuits) to generate
                qubits: the qubits particiapting in the experiment
                lengths: for each seed, the lengths of the circuits used for that seed.
                group_gates: which group the circuits is based on
                rand_seed: optional random number seed
        """
        generator = RBGenerator(nseeds, qubits, lengths, group_gates, rand_seed)
        if generator.rb_group_type() == 'clifford':
            analysis = RBAnalysis(qubits, lengths)
        if generator.rb_group_type() == 'cnot_dihedral':
            analysis = CNOTDihedralRBAnalysis(qubits, lengths)
        super().__init__(generator=generator, analysis=analysis)


class InterleavedRBExperiment(RBExperimentBase):
    """Experiment class for interleaved RB experiment"""
    def __init__(self,
                 interleaved_element:
                 Union[QuantumCircuit, Instruction, Clifford, CNOTDihedral],
                 nseeds: int = 1,
                 qubits: List[int] = (0,),
                 lengths: List[int] = (1, 10, 20),
                 group_gates: Optional[str] = None,
                 rand_seed: Optional[Union[int, RandomState]] = None,
                 transform_interleaved_element: Optional[bool] = False
                 ):
        """Initialize the interleaved RB experiment
           Args:
               interleaved_element: the element to interleave, given either as a group element
               or as an instruction/circuit
               nseeds: number of different seeds (random circuits) to generate
               qubits: the qubits particiapting in the experiment
               lengths: for each seed, the lengths of the circuits used for that seed.
               group_gates: which group the circuits is based on
               rand_seed: optional random number seed
               transform_interleaved_element: when the interleaved element is gate or circuit, it
               can be kept as it is (`False`) or transofrmed to the group element and then
               transformed to the canonical circuit representation of that element (`True`)
        """
        generator = InterleavedRBGenerator(interleaved_element, nseeds, qubits, lengths,
                                           group_gates, rand_seed, transform_interleaved_element)
        analysis = InterleavedRBAnalysis(qubits, lengths, group_type=generator.rb_group_type())
        super().__init__(generator=generator, analysis=analysis)


class PurityRBExperiment(RBExperimentBase):
    """Experiment class for purity RB experiment"""
    def __init__(self,
                 nseeds: int = 1,
                 qubits: List[int] = (0,),
                 lengths: List[int] = (1, 10, 20),
                 rand_seed: Optional[Union[int, RandomState]] = None,
                 ):
        """Initialize the purity RB experiment
            Args:
                nseeds: number of different seeds (random circuits) to generate
                qubits: the qubits particiapting in the experiment
                lengths: for each seed, the lengths of the circuits used for that seed.
                rand_seed: optional random number seed
        """
        generator = PurityRBGenerator(nseeds, qubits, lengths, rand_seed)
        analysis = PurityRBAnalysis(qubits, lengths)
        super().__init__(generator=generator, analysis=analysis)
