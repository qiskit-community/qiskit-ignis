import numpy as np
from numpy.random import RandomState
import copy
import qiskit
from itertools import product
from qiskit import (QuantumCircuit, QuantumRegister, ClassicalRegister)
from qiskit.ignis.experiments.base import Generator
from qiskit.quantum_info.operators.symplectic.clifford import Clifford
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
                 rand_seed: Optional[Union[int, RandomState]] = None,
                 name = "randomized benchmarking base"
                 ):
        self._nseeds = nseeds
        self._meas_qubits = list(set(qubits))
        self._lengths = lengths
        self._rb_group = RBgroup(group_gates)
        if self._rb_group.group_gates_type() == 0:
            self._rb_group_type = 'clifford'
        if self._rb_group.group_gates_type() == 1:
            self._rb_group_type = 'cnot_dihedral'
        self._rand_seed = rand_seed
        self._circuits = []
        self._metadata = []
        super().__init__(name, self.num_all_qubits())

    def num_qubits(self):
        return len(self._meas_qubits)

    def num_all_qubits(self):
        return max(self._meas_qubits) + 1

    def generate_circuits(self):
        for seed in range(self._nseeds):
            circuit_and_meta = self.generate_circuits_for_seed()
            for data in circuit_and_meta:
                circuit = data['circuit']
                meta = data['meta']
                meta['seed'] = seed
                meta['group_type'] = self._rb_group_type
                self.add_measurements(circuit)
                self.set_circuit_name(circuit, meta)
                meta['circuit_name'] = circuit.name
                self._circuits.append(circuit)
                self._metadata.append(meta)

    def set_circuit_name(self, circuit, meta):
        name = "rb_"
        if meta['group_type'] == 'cnot_dihedral':
            name += "cnotdihedral_{}_".format(meta['cnot_basis'])
        if self.circuit_type_string(meta) is not None:
            name += (self.circuit_type_string(meta) + "_")
        name += "length_{}_seed_{}".format(meta['length_index'], meta['seed'])
        circuit.name = name

    def circuit_type_string(self, meta):
        return None

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
        qr = QuantumRegister(self.num_all_qubits(), 'qr')
        cr = ClassicalRegister(self.num_qubits(), 'cr')
        circ = QuantumCircuit(qr, cr)
        current_element = self._rb_group.iden(self.num_qubits())
        for length_index, element_list in enumerate(element_lists):
            for element in element_list:
                current_element = self._rb_group.compose(current_element, element)
                circ += self.replace_q_indices(
                    self._rb_group.to_circuit(element),
                    self._meas_qubits, qr)
                # add a barrier
                circ.barrier(*[qr[x] for x in self._meas_qubits])
            # finished with the current list - output a circuit based on what we have
            output_circ = QuantumCircuit(qr, cr)
            output_meta = {'length_index': length_index}
            output_circ += circ
            inv_circuit = self._rb_group.inverse(current_element)
            output_circ += self.replace_q_indices(inv_circuit, self._meas_qubits, qr)
            if self._rb_group_type == 'cnot_dihedral':
                output_meta['cnot_basis'] = 'Z'
                cnot_circuit, cnot_meta = self.generate_cnot_circuit(output_circ, output_meta)
                result.append({'circuit': cnot_circuit, 'meta': cnot_meta})
            result.append({'circuit': output_circ, 'meta': output_meta})
        return result

    def generate_cnot_circuit(self, circuit, meta):
        cnot_circuit = QuantumCircuit(circuit.qregs[0], circuit.cregs[0])
        cnot_meta = copy.copy(meta)
        for qubit in self._meas_qubits:
            cnot_circuit.h(qubit)
            cnot_circuit.barrier(qubit)
        cnot_circuit += circuit
        for qubit in self._meas_qubits:
            cnot_circuit.barrier(qubit)
            cnot_circuit.h(qubit)

        cnot_meta['cnot_basis'] = 'X'
        return (cnot_circuit, cnot_meta)

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

    def add_measurements(self, circuit):
        for clbit, qubit in enumerate(self._meas_qubits):
            circuit.measure(qubit, clbit)

    def set_meta(self, circuit_and_meta_list, extra_meta):
        for data in circuit_and_meta_list:
            for key, value in extra_meta.items():
                data['meta'][key] = value

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
                 rand_seed: Optional[Union[int, RandomState]] = None,
                 ):
        super().__init__(nseeds,
                         qubits,
                         lengths,
                         group_gates,
                         rand_seed,
                         name="randomized benchmarking")
        self.generate_circuits()

    def generate_circuits_for_seed(self):
        circuits_and_meta = super().generate_circuits_for_seed()
        self.set_meta(circuits_and_meta, {
            'experiment_type': 'standard',
        })
        return circuits_and_meta


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
                         group_gates,
                         name="purity randomized benchmarking")
        self.generate_circuits()

    def generate_circuits_for_seed(self):
        circuits_and_meta = super().generate_circuits_for_seed()
        # each standard circuit gives rise to 3**qubits circuits
        # with corresponding pre-measure operators
        new_circuits_and_meta = []
        for data in circuits_and_meta:
            new_circuits_and_meta += self.add_purity_measurements(data['circuit'], data['meta'])
        self.set_meta(new_circuits_and_meta, {
            'experiment_type': 'purity',
        })
        return new_circuits_and_meta

    def add_purity_measurements(self, circuit, meta):
        meas_op_names = ['Z', 'X', 'Y']
        result = []
        for meas_ops in product(meas_op_names, repeat=self.num_qubits()):
            new_meta = copy.copy(meta)
            new_meta['purity_meas_ops'] = "".join(meas_ops)
            new_circuit = QuantumCircuit(circuit.qregs[0], circuit.cregs[0])
            new_circuit += circuit
            for qubit_index, meas_op in enumerate(meas_ops):
                qubit = self._meas_qubits[qubit_index]
                if meas_op == 'Z':
                    pass  # do nothing
                if meas_op == 'X':
                    new_circuit.rx(np.pi / 2, qubit)
                if meas_op == 'Y':
                    new_circuit.ry(np.pi / 2, qubit)
            result.append({'circuit': new_circuit, 'meta': new_meta})
        return result

    def circuit_type_string(self, meta):
        return "purity_{}".format(meta['purity_meas_ops'])


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
                         group_gates,
                         name="interleaved randomized benchmarking")
        self.set_interleaved_element(interleaved_element)
        self.generate_circuits()

    def set_interleaved_element(self, interleaved_element):
        group_gates_type = self._rb_group.group_gates_type()
        if isinstance(interleaved_element, (QuantumCircuit, Instruction)):
            num_qubits = interleaved_element.num_qubits
            qc = interleaved_element
            interleaved_element = self._rb_group.iden(num_qubits)
            interleaved_element = interleaved_element.from_circuit(qc)
        if (not isinstance(interleaved_element, Clifford) and group_gates_type == 0) \
                and not (isinstance(interleaved_element, CNOTDihedral) and group_gates_type == 1):
            raise ValueError("Invalid interleaved element type.")

        if not isinstance(interleaved_element, QuantumCircuit) and \
                not isinstance(interleaved_element, Clifford) \
                and not isinstance(interleaved_element, CNOTDihedral):
            raise ValueError("Invalid interleaved element type. "
                             "interleaved_elem should be a list of QuantumCircuit,"
                             "or a list of Clifford / CNOTDihedral objects")
        self._interleaved_element = interleaved_element

    def generate_circuits_for_seed(self):
        element_list = self.generate_random_element_list(self._lengths[-1])
        element_lists = self.split_element_list(element_list, self._lengths)
        circuits_and_meta = self.generate_circuits_from_elements(element_lists)
        self.set_meta(circuits_and_meta, {
            'experiment_type': 'interleaved',
            'circuit_type': 'standard'
        })

        element_list = self.interleave(element_list)
        element_lists = self.split_element_list(element_list, [2*x for x in self._lengths])
        interleaved_circuits_and_meta = self.generate_circuits_from_elements(element_lists)
        self.set_meta(interleaved_circuits_and_meta, {
            'experiment_type': 'interleaved',
            'circuit_type': 'interleaved'
        })
        return circuits_and_meta + interleaved_circuits_and_meta

    def interleave(self, element_list):
        new_element_list = []
        for element in element_list:
            new_element_list.append(element)
            new_element_list.append(self._interleaved_element)
        return new_element_list

    def circuit_type_string(self, meta):
        if meta['circuit_type'] == 'interleaved':
            return "interleaved"
        return None
