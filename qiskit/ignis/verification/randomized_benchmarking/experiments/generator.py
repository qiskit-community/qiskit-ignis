import numpy as np
import copy
import qiskit
from qiskit import (QuantumCircuit, QuantumRegister, ClassicalRegister)
from qiskit.ignis.experiments.base import Generator
from qiskit.circuit import Instruction
from typing import List, Dict, Union, Optional
from ..rb_groups import RBgroup
from ..dihedral import CNOTDihedral

class RBGeneratorBase(Generator):
    def __init__(self,
                 nseeds: int = 1,
                 qubits: List[int] = [0],
                 lengths: List[int] = [1, 10, 20],
                 group_gates: Optional[str] = None,
                 ):
        self._nseeds = nseeds
        self._qubits = list(set(qubits))
        self._lengths = lengths
        self._rb_group = RBgroup(group_gates)
        self._rand_seed = None
        self._circuits = []
        self._metadata = []

    def num_qubits(self):
        return len(self._qubits)

    def generate_circuits(self):
        for seed in range(self._nseeds):
            circuit_and_meta = self.generate_circuits_for_seed()
            for data in circuit_and_meta:
                circuit = data['circuit']
                meta = data['meta']
                meta['seed'] = seed
                self._circuits.append(circuit)
                self._metadata.append(meta)

    def generate_circuits_for_seed(self):
        element_list = self.generate_random_element_list(self._lengths[-1])
        element_lists = self.split_element_list(element_list, self._lengths)
        circuits_and_meta = self.generate_circuits_from_elements(element_lists)
        return circuits_and_meta

    def generate_circuits_from_elements(self, element_lists):
        # adds the elements to a new circuit
        # at the end of every element list, outputs a circuit from what was currently built
        # with an inverse gate appended at the end
        result = []
        qr = QuantumRegister(max(self._qubits) + 1, 'qr')
        cr = ClassicalRegister(self.num_qubits(), 'cr')
        circ = QuantumCircuit(qr, cr)
        current_element = self._rb_group.iden(self.num_qubits())
        for element_list in element_lists:
            for element in element_list:
                current_element = self._rb_group.compose(element, element)
                circ += self.replace_q_indices(
                    self._rb_group.to_circuit(element),
                    self._qubits, qr)
                # add a barrier
                circ.barrier(*[qr[x] for x in self._qubits])
            # finished with the current list - output a circuit based on what we have
            output_circ = QuantumCircuit(qr, cr)
            output_meta = {}
            output_circ += circ
            inv_circuit = self._rb_group.inverse(current_element)
            output_circ += self.replace_q_indices(inv_circuit, self._qubits, qr)
            result.append({'circuit': output_circ, 'meta': output_meta})
        return result

    def generate_random_element_list(self, length):
        element_list = []
        for _ in range(length):
            if self._rand_seed is not None:
                self._rand_seed += 1
            element_list.append(self._rb_group.random(self.num_qubits(), self._rand_seed))
        return element_list

    def split_element_list(self, element_list, lengths):
        element_lists = []
        current_element_list = []
        stop_indexes = [x - 1 for x in lengths]
        for index, element in enumerate(element_list):
            current_element_list.append(element)
            if index == stop_indexes[0]:
                element_lists.append(current_element_list)
                current_element_list = []
                stop_indexes.pop(0)
        return element_lists

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
                 qubits: List[int] = [0],
                 lengths: List[int] = [1, 10, 20],
                 group_gates: Optional[str] = None,
                 ):
        super().__init__(nseeds,
                         qubits,
                         lengths,
                         group_gates)


class PurityRBGenerator(RBGeneratorBase):
    def __init__(self,
                 nseeds: int = 1,
                 qubits: List[int] = [0],
                 lengths: List[int] = [1, 10, 20],
                 group_gates: Optional[str] = None,
                 ):
        super().__init__(nseeds,
                         qubits,
                         lengths,
                         group_gates)
        self._npurity = 3 ** len(self._qubits)
        self.generate_circuits()

class InterleavedRBGenerator(RBGeneratorBase):
    def __init__(self,
                 interleaved_element:
                 Union[QuantumCircuit, Instruction,
                       qiskit.quantum_info.operators.symplectic.Clifford,
                       CNOTDihedral],
                 nseeds: int = 1,
                 qubits: List[int] = [0],
                 lengths: List[int] = [1, 10, 20],
                 group_gates: Optional[str] = None,
                 ):
        super().__init__(nseeds,
                         qubits,
                         lengths,
                         group_gates)
        self.set_interleaved_element(interleaved_element)
        self.generate_circuits()

    def set_interleaved_element(self, interleaved_element):
        group_gates_type = self._rb_group.group_gates_type()
        if isinstance(interleaved_element, (QuantumCircuit, Instruction)):
            num_qubits = interleaved_element.num_qubits
            qc = interleaved_element
            interleaved_element = self._rb_group.iden(num_qubits)
            interleaved_element = interleaved_element.from_circuit(qc)
        if (not isinstance(interleaved_element, qiskit.quantum_info.operators.symplectic.clifford.Clifford)
            and group_gates_type == 0) and not (isinstance(interleaved_element, CNOTDihedral)
                                           and group_gates_type == 1):
            raise ValueError("Invalid interleaved element type.")

        if not isinstance(interleaved_element, QuantumCircuit) and \
                not isinstance(interleaved_element,
                               qiskit.quantum_info.operators.symplectic.clifford.Clifford) \
                and not isinstance(interleaved_element, CNOTDihedral):
            raise ValueError("Invalid interleaved element type. "
                             "interleaved_elem should be a list of QuantumCircuit,"
                             "or a list of Clifford / CNOTDihedral objects")
        self._interleaved_element = interleaved_element
