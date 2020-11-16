import numpy as np
from qiskit import QuantumCircuit
from qiskit.ignis.experiments.base import Generator
from qiskit.circuit import Instruction
from typing import List, Dict, Union, Optional
from ..rb_groups import RBgroup
from ..dihedral import CNOTDihedral

class RBGeneratorBase(Generator):
    def __init__(self,
                 length_vector: Optional[List[int]] = None,
                 rb_pattern: Optional[List[List[int]]] = None,
                 length_multiplier: Optional[List[int]] = 1,
                 group_gates: Optional[str] = None,
                 ):
        self.set_pattern(rb_pattern)
        self.check_pattern()
        self.set_length_multiplier(length_multiplier)
        self._length_vector = length_vector if length_vector is not None else [1, 10, 20]
        self._xdata = np.array([np.array(length_vector) * mult for mult in self._length_multiplier])
        self._rb_group = RBgroup(group_gates)

    def set_pattern(self, rb_pattern):
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

class RBGenerator(RBGeneratorBase):
    def __init__(self,
                 length_vector: Optional[List[int]] = None,
                 rb_pattern: Optional[List[List[int]]] = None,
                 length_multiplier: Optional[List[int]] = 1,
                 group_gates: Optional[str] = None,
                 ):
        super().__init__(length_vector,
                         rb_pattern,
                         length_multiplier,
                         group_gates)

class PurityRBGenerator(RBGeneratorBase):
    def __init__(self,
                 length_vector: Optional[List[int]] = None,
                 rb_pattern: Optional[List[List[int]]] = None,
                 length_multiplier: Optional[List[int]] = 1,
                 group_gates: Optional[str] = None,
                 ):
        super().__init__(length_vector,
                         rb_pattern,
                         length_multiplier,
                         group_gates)
        self.check_pattern()
        self._npurity = 3 ** max(self._pattern_dim)

    def check_pattern(self):
        if len(set(self._pattern_dim)) > 1:
            raise ValueError("Invalid pattern for purity RB. \
            All simultaneous sequences should have the same dimension.")

class InterleavedRBGenerator(RBGeneratorBase):
    def __init__(self,
                 interleaved_elem:
                 Union[List[QuantumCircuit], List[Instruction],
                       List[qiskit.quantum_info.operators.symplectic.Clifford],
                       List[CNOTDihedral]],
                 length_vector: Optional[List[int]] = None,
                 rb_pattern: Optional[List[List[int]]] = None,
                 length_multiplier: Optional[List[int]] = 1,
                 group_gates: Optional[str] = None,
                 ):
        super().__init__(length_vector,
                         rb_pattern,
                         length_multiplier,
                         group_gates)
        self.set_interleaved_elem(interleaved_elem)
        self.check_pattern()

    def set_interleaved_elem(self, interleaved_elem):
        self._interleaved_elem = interleaved_elem
        self._interleaved_elem_list = []

        group_gates_type = self._rb_group.group_gates_type()
        for elem in interleaved_elem:
            if isinstance(elem, (QuantumCircuit, Instruction)):
                num_qubits = elem.num_qubits
                qc = elem
                elem = self._rb_group.iden(num_qubits)
                elem = elem.from_circuit(qc)
            if (isinstance(elem, qiskit.quantum_info.operators.symplectic.clifford.Clifford)
                and group_gates_type == 0) or (isinstance(elem, CNOTDihedral)
                                               and group_gates_type == 1):
                self._interleaved_elem_list.append(elem)
            else:
                raise ValueError("Invalid interleaved element type.")

            if not isinstance(elem, QuantumCircuit) and \
                    not isinstance(elem,
                                   qiskit.quantum_info.operators.symplectic.clifford.Clifford) \
                    and not isinstance(elem, CNOTDihedral):
                raise ValueError("Invalid interleaved element type. "
                                 "interleaved_elem should be a list of QuantumCircuit,"
                                 "or a list of Clifford / CNOTDihedral objects")

    def check_pattern(self):
        interleaved_dim = [elem.num_qubits for elem in self._interleaved_elem]
        if self._pattern_dim != interleaved_dim:
            raise ValueError("Invalid pattern for interleaved RB.")