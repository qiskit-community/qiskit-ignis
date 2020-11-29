from numpy.random import RandomState
from typing import List, Optional, Union
from qiskit.ignis.experiments.base import Experiment, Analysis, Generator
from qiskit.providers import BaseJob
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