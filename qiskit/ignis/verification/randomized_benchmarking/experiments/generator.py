import numpy as np
import copy
from qiskit import (QuantumCircuit, QuantumRegister, ClassicalRegister)
from qiskit.ignis.experiments.base import Generator
from qiskit.circuit import Instruction
from typing import List, Dict, Union, Optional
from ..rb_groups import RBgroup
from ..dihedral import CNOTDihedral

class RBGeneratorBase(Generator):
    def __init__(self,
                 nseeds: int = 1,
                 length_vector: Optional[List[int]] = None,
                 rb_pattern: Optional[List[List[int]]] = None,
                 length_multiplier: Optional[List[int]] = 1,
                 group_gates: Optional[str] = None,
                 ):
        self._nseeds = nseeds
        self.set_pattern(rb_pattern)
        self.check_pattern()
        self.set_length_multiplier(length_multiplier)
        self._length_vector = length_vector if length_vector is not None else [1, 10, 20]
        self._xdata = np.array([np.array(length_vector) * mult for mult in self._length_multiplier])
        self._rb_group = RBgroup(group_gates)

        self._circuits = []
        self._metadata = []

    def set_pattern(self, rb_pattern):
        self._rb_pattern = rb_pattern if rb_pattern is not None else [[0]]
        self._qubits = [item for pat in self._rb_pattern for item in pat]
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
        if len(self._qubits) != len(set(self._qubits)):
            raise ValueError("Invalid pattern. Duplicate qubit index.")

    def generate_circuits(self):
        self.check_pattern()
        for seed in range(self._nseeds):
            (new_circuits, new_meta) = self.generate_circuits_for_seed(seed)
            for meta in new_meta:
                meta['seed'] = seed
            self._circuits.extend(new_circuits)
            self._metadata.extend(new_meta)

    def generate_circuits_for_seed(self, seed):
        qr = QuantumRegister(max(self._qubits) + 1, 'qr')
        cr = ClassicalRegister(len(self._qubits), 'cr')
        general_circ = QuantumCircuit(qr, cr)
        elements = [self._rb_group.iden(dim) for dim in self._pattern_dim]

        # go through and add elements to RB sequences
        length_index = 0
        for elements_index in range(self._length_vector[-1]):
            self.add_random_gates(general_circ, elements)


    def add_random_gates(self, general_circ, elements):
        for (dim_index, dim) in enumerate(self._pattern_dim):
            for _ in range(self._length_multiplier[dim_index]):
                # make the seed unique for each element
                if self._rand_seed:
                    self._rand_seed += 1
                new_element = self._rb_group.random(dim, self._rand_seed)
                self.add_element(general_circ, elements, new_element, dim_index)

    def add_element(self, general_circ, elements, new_element, dim_index):
        qr = general_circ.qregs[0]
        qubits = self._rb_pattern[dim_index]
        elements[dim_index] = self._rb_group.compose(elements[dim_index], new_element)
        general_circ += self.replace_q_indices(
            self._rb_group.to_circuit(new_element),
            qubits, qr)
        # add a barrier
        general_circ.barrier(*[qr[x] for x in qubits])

    def replace_q_indices(self, circuit, q_nums, qr):
        """
        Take a circuit that is ordered from 0,1,2 qubits and replace 0 with the
        qubit label in the first index of q_nums, 1 with the second index...

        Args:
            circuit (QuantumCircuit): circuit to operate on
            q_nums (list): list of qubit indices
            qr (QuantumRegister): A quantum register to use for the output circuit

        Returns:
            QuantumCircuit: updated circuit
        """

        new_circuit = QuantumCircuit(qr)
        for instr, qargs, cargs in circuit.data:
            new_qargs = [
                qr[q_nums[x]] for x in [arg.index for arg in qargs]]
            new_op = copy.deepcopy((instr, new_qargs, cargs))
            new_circuit.data.append(new_op)

        return new_circuit

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of experiment circuits."""
        return self._circuits

    def _extra_metadata(self) -> List[Dict[str, any]]:
        """Generate a list of experiment metadata dicts."""
        return self._metadata

class RBGenerator(RBGeneratorBase):
    def __init__(self,
                 nseeds: int = 1,
                 length_vector: Optional[List[int]] = None,
                 rb_pattern: Optional[List[List[int]]] = None,
                 length_multiplier: Optional[List[int]] = 1,
                 group_gates: Optional[str] = None,
                 ):
        super().__init__(nseeds,
                         length_vector,
                         rb_pattern,
                         length_multiplier,
                         group_gates)


class PurityRBGenerator(RBGeneratorBase):
    def __init__(self,
                 nseeds: int = 1,
                 length_vector: Optional[List[int]] = None,
                 rb_pattern: Optional[List[List[int]]] = None,
                 length_multiplier: Optional[List[int]] = 1,
                 group_gates: Optional[str] = None,
                 ):
        super().__init__(nseeds,
                         length_vector,
                         rb_pattern,
                         length_multiplier,
                         group_gates)
        self._npurity = 3 ** max(self._pattern_dim)
        self.generate_circuits()

    def check_pattern(self):
        super().check_pattern()
        if len(set(self._pattern_dim)) > 1:
            raise ValueError("Invalid pattern for purity RB. \
            All simultaneous sequences should have the same dimension.")

class InterleavedRBGenerator(RBGeneratorBase):
    def __init__(self,
                 interleaved_elem:
                 Union[List[QuantumCircuit], List[Instruction],
                       List[qiskit.quantum_info.operators.symplectic.Clifford],
                       List[CNOTDihedral]],
                 nseeds: int = 1,
                 length_vector: Optional[List[int]] = None,
                 rb_pattern: Optional[List[List[int]]] = None,
                 length_multiplier: Optional[List[int]] = 1,
                 group_gates: Optional[str] = None,
                 ):
        super().__init__(nseeds,
                         length_vector,
                         rb_pattern,
                         length_multiplier,
                         group_gates)
        self.set_interleaved_elem(interleaved_elem)
        self.generate_circuits()

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
        super().check_pattern()
        interleaved_dim = [elem.num_qubits for elem in self._interleaved_elem]
        if self._pattern_dim != interleaved_dim:
            raise ValueError("Invalid pattern for interleaved RB.")

    def add_element(self, general_circ, elements, new_element, dim_index):
        qr = general_circ.qregs[0]
        interleaved_element = self._interleaved_elem_list[dim_index]
        qubits = self._rb_pattern[dim_index]

        # adding the new random element
        elements[dim_index] = self._rb_group.compose(elements[dim_index], new_element)
        general_circ += self.replace_q_indices(
            self._rb_group.to_circuit(new_element), qubits, qr)
        general_circ.barrier(*[qr[x] for x in qubits])

        # adding the interleaved element
        elements[dim_index] = self._rb_group.compose(elements[dim_index], interleaved_element)
        general_circ += self.replace_q_indices(
            self._rb_group.to_circuit(interleaved_element), qubits, qr)
        general_circ.barrier(*[qr[x] for x in qubits])