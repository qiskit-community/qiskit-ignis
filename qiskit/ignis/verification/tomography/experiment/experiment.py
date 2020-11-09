from typing import List, Optional
from qiskit.ignis.experiments.base import Experiment, Analysis, Generator
from qiskit.providers import BaseJob


class TomographyExperiment(Experiment):
    def __init__(self,
                 generator: Optional[Generator] = None,
                 analysis: Optional[Analysis] = None,
                 job: Optional[BaseJob] = None):
        super().__init__(generator=generator, analysis=analysis, job=job)

    def set_target_qubits(self, qubits: List[int]):
        self.analysis.set_target_qubits(qubits)
        return self
