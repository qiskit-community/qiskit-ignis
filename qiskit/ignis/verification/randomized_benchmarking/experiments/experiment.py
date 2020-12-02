import uuid
from numpy.random import RandomState
from typing import List, Optional, Union
from qiskit.ignis.experiments.base import Experiment, Analysis, Generator
from qiskit.providers import BaseJob
from qiskit.providers import BaseBackend
from qiskit import transpile, assemble
from . import (RBGenerator, PurityRBGenerator, InterleavedRBGenerator, RBAnalysisBase)


class RBExperiment(Experiment):
    def __init__(self,
                 nseeds: int = 1,
                 qubits: List[int] = [0],
                 lengths: List[int] = [1, 10, 20],
                 group_gates: Optional[str] = None,
                 rand_seed: Optional[Union[int, RandomState]] = None
                 ):
        generator = RBGenerator(nseeds, qubits, lengths, group_gates, rand_seed)
        analysis = RBAnalysisBase(qubits, lengths)
        super().__init__(generator=generator, analysis=analysis)
        self._jobs = []

    def get_circuits_and_metadata(self, seeds=None):
        circuits = []
        metadata = []
        for (circ, meta) in zip(self.generator.circuits(), self.generator.metadata()):
            if seeds is None or meta['seed'] in seeds:
                circuits.append(circ)
                metadata.append(meta)
        return (circuits, metadata)

    def run(self, backend: BaseBackend, seeds=None, **kwargs) -> any:
        """Run an experiment and perform analysis"""
        job = self.execute(backend, seeds, **kwargs)
        self._jobs.append(job)
        self.analysis.add_data(job)
        return self.run_analysis()

    def add_seeds_and_run(self, backend, num_of_seeds, **kwargs):
        new_seeds = self.generator.add_seeds(num_of_seeds)
        return self.run(backend, new_seeds, **kwargs)

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
                             initial_layout=self.generator.qubits)  # should change somehow?

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