import uuid
from numpy.random import RandomState
from typing import List, Optional, Union
from qiskit.ignis.experiments.base import Experiment, Analysis, Generator
from qiskit.providers import BaseJob
from qiskit.providers import BaseBackend
from qiskit import transpile, assemble
from qiskit.exceptions import QiskitError
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
import qiskit
from ..dihedral import CNOTDihedral
from . import (RBGenerator, PurityRBGenerator, InterleavedRBGenerator, RBAnalysis, InterleavedRBAnalysis)


class RBExperimentBase(Experiment):
    def __init__(self,
                 generator: Optional[Generator] = None,
                 analysis: Optional[Analysis] = None):
        super().__init__(generator=generator, analysis=analysis)
        self.reset()

    def get_circuits_and_metadata(self, seeds=None):
        circuits = []
        metadata = []
        for (circ, meta) in zip(self.generator.circuits(), self.generator.metadata()):
            if seeds is None or meta['seed'] in seeds:
                circuits.append(circ)
                metadata.append(meta)
        return (circuits, metadata)

    def default_basis_gates(self):
        return ['u1', 'u2', 'u3', 'cx']

    def reset(self):
        self._jobs = []
        self._kwargs = None
        self._backend = None
        self._basis_gates = self.default_basis_gates()
        self._transpiled_circuits = None

    def run(self, backend: BaseBackend, reset=True, seeds=None, **kwargs) -> any:
        """Run an experiment and perform analysis"""
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
        if self._backend is None:
            raise QiskitError("Cannot rerun experiment: Use Experiment.run() for the first time.")

    def run_additional_shots(self, shots):
        self.check_rerun_possible()
        self._kwargs['shots'] = shots
        return self.run(backend=self._backend, reset=False, **self._kwargs)

    def run_additional_seeds(self, num_of_seeds):
        self.check_rerun_possible()
        new_seeds = self.generator.add_seeds(num_of_seeds)
        return self.run(backend=self._backend, seeds=new_seeds, reset=False, **self._kwargs)

    def execute(self, backend: BaseBackend, seeds=None, **kwargs) -> BaseJob:
        """Execute the experiment on a backend.
​

        Args:
            backend: backend to run experiment on.
            kwargs: kwargs for assemble method.
​
        Returns:
            BaseJob: the experiment job.
        """
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

    def gates_per_clifford(self):
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
        total_ncliffs = self.generator.nseeds() * sum([length + 1 for length in self.generator.lengths()])
        for qubit in qubits:
            for base in self._basis_gates:
                ngates[qubit][base] /= total_ncliffs

        return ngates

class RBExperiment(RBExperimentBase):
    def __init__(self,
                 nseeds: int = 1,
                 qubits: List[int] = [0],
                 lengths: List[int] = [1, 10, 20],
                 group_gates: Optional[str] = None,
                 rand_seed: Optional[Union[int, RandomState]] = None,
                 ):
        generator = RBGenerator(nseeds, qubits, lengths, group_gates, rand_seed)
        analysis = RBAnalysis(qubits, lengths)
        super().__init__(generator=generator, analysis=analysis)


class InterleavedRBExperiment(RBExperimentBase):
    def __init__(self,
                 interleaved_element:
                 Union[QuantumCircuit, Instruction,
                       qiskit.quantum_info.operators.symplectic.Clifford,
                       CNOTDihedral],
                 nseeds: int = 1,
                 qubits: List[int] = [0],
                 lengths: List[int] = [1, 10, 20],
                 group_gates: Optional[str] = None,
                 rand_seed: Optional[Union[int, RandomState]] = None,
                 transform_interleaved_element: Optional[bool] = False
                 ):
        generator = InterleavedRBGenerator(interleaved_element, nseeds, qubits, lengths,
                                           group_gates, rand_seed, transform_interleaved_element)
        analysis = InterleavedRBAnalysis(qubits, lengths)
        super().__init__(generator=generator, analysis=analysis)