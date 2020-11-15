import numpy as np
from qiskit import QuantumCircuit
from qiskit.ignis.experiments.base import Generator
from typing import List, Dict, Union, Optional


class RBGenerator(Generator):
    def __init__(self,
                 length_vector: Optional[List[int]] = None,
                 rb_pattern: Optional[List[List[int]]] = None,
                 length_multiplier: Optional[List[int]] = 1,
                 ):
        self.set_pattern(rb_pattern)
        self.set_length_multiplier(length_multiplier)
        self._length_vector = length_vector if length_vector is not None else [1, 10, 20]

    def set_patten(self, rb_pattern):
        self._rb_pattern = rb_pattern if rb_pattern is not None else [[0]]
        self._pattern_flat = [item for pat in self._rb_pattern for item in pat]
        self._pattern_dim = [len(pat) for pat in self._rb_pattern]

    def set_length_multiplier(self, length_multiplier):
        if hasattr(length_multiplier, "__len__"):
            if len(length_multiplier) != len(self._rb_pattern):
                raise ValueError(
                    "Length multiplier must be the same length as the pattern")
            self._length_multiplier = np.array(length_multiplier)
            if self._length_multiplier.dtype != 'int' or (self._length_multiplier < 1).any():
                raise ValueError("Invalid length multiplier")
        else:
            self._length_multiplier = np.ones(len(self._rb_pattern), dtype='int') * length_multiplier

    def check_pattern(self):
        if len(self._pattern_flat) != len(set(self._pattern_flat)):
            raise ValueError("Invalid pattern. Duplicate qubit index.")

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of experiment circuits."""
        return self._circuits

    def _extra_metadata(self) -> List[Dict[str, any]]:
        """Generate a list of experiment metadata dicts."""
        return [{
            'circuit_name': circ.name,
         }
            for circ in self._circuits]

class PurityRBGenerator(RBGenerator):
    def check_pattern(self):
        super().check_pattern()
        if len(set(self._pattern_dim)) > 1:
            raise ValueError("Invalid pattern for purity RB. \
            All simultaneous sequences should have the same dimension.")

class InterleavedRBGenerator(RBGenerator):
    def check_pattern(self):
        super().check_pattern()
        interleaved_dim = [elem.num_qubits for elem in self._interleaved_elem]
        if self._pattern_dim != interleaved_dim:
            raise ValueError("Invalid pattern for interleaved RB.")