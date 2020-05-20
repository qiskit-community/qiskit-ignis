# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Gate set tomography preparation and measurement basis
"""

# Needed for functions
import functools
from typing import Tuple, Callable, Union, Optional, Dict
import numpy as np

# Import QISKit classes
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Gate
from qiskit.quantum_info import PTM
from .tomographybasis import TomographyBasis


class GateSetBasis:
    """
    This class contains the gateset data needed to perform gateset tomgography.
    The gateset tomography data consists of two sets, G and F
    G = (G1,...,Gn) is the set of the gates we wish to characterize
    F = (F1,..,Fm) is a set of SPAM (state preparation and measurement)
    circuits. The SPAM circuits are constructed from the elements of G
    and all the SPAM combinations are appended before and after elements of G
    when performing the tomography measurements
    (i.e. we measure all circuits of the form Fi * Gk * Fj)

    The gateset data is comprised of four elements:
    1) The labels (strings) of the gates
    2) A function f(circ, qubit, op)
        which adds to circ at qubit the gate labeled by op
    3) The labels of the SPAM circuits for the gate set tomography
    4) For SPAM label, tuple of gate labels for the gates in this SPAM circuit
    """
    def __init__(self,
                 name: str,
                 gates: Dict[str, Union[Callable, Gate]],
                 spam: Dict[str, Tuple[str]]
                 ):
        """
        Initialize the gate set basis data

        Args:
            name: Name of the basis.
            gates: The gate data (name -> gate/gate function)
            spam: The spam data (name -> sequence of gate names)
        """
        self.name = name
        self.gate_labels = list(gates.keys())
        self.gates = gates
        self.gate_matrices = {name: np.real(self._gate_matrix(gate))
                              for (name, gate) in gates.items()}
        self.spam_labels = tuple(sorted(spam.keys()))
        self.spam_spec = spam

    def _gate_matrix(self, gate):
        """Gets a PTM representation of the gate"""
        if isinstance(gate, Gate):
            return PTM(gate).data
        if callable(gate):
            c = QuantumCircuit(1)
            gate(c, c.qubits[0])
            return PTM(c).data
        return None

    def add_gate(self, gate: Union[Callable, Gate], name: Optional[str] = None):
        """Adds a new gate to the gateset
            Args:
                gate: Either a qiskit gate object or a function taking
                (QuantumCircuit, QuantumRegister)
                and adding the gate to the circuit
                name: the name of the new gate
            Raises:
                RuntimeError: If the gate is given as a function but without
                a name.
        """
        if name is None:
            if isinstance(gate, Gate):
                name = gate.name
            else:
                raise RuntimeError("Gate name is missing")
        self.gate_labels.append(name)
        self.gates[name] = gate
        self.gate_matrices[name] = self._gate_matrix(gate)

    def add_gate_to_circuit(self,
                            circ: QuantumCircuit,
                            qubit: QuantumRegister,
                            op: str
                            ):
        """
        Adds the gate op to circ at qubit

        Args:
            circ: the circuit to apply op on
            qubit: qubit to be operated on
            op: gate name

        Raises:
            RuntimeError: if `op` does not describe a gate
        """
        if op not in self.gates:
            raise RuntimeError("{} is not a SPAM circuit".format(op))
        gate = self.gates[op]
        if callable(gate):
            gate(circ, qubit)
        if isinstance(gate, Gate):
            circ.append(gate, [qubit], [])

    def add_spam_to_circuit(self,
                            circ: QuantumCircuit,
                            qubit: QuantumRegister,
                            op: str
                            ):
        """
        Adds the SPAM circuit op to circ at qubit

        Args:
            circ: the circuit to apply op on
            qubit: qubit to be operated on
            op: SPAM circuit name

        Raises:
            RuntimeError: if `op` does not describe a SPAM circuit
        """
        if op not in self.spam_spec:
            raise RuntimeError("{} is not a SPAM circuit".format(op))
        op_gates = self.spam_spec[op]
        for gate_name in op_gates:
            self.add_gate_to_circuit(circ, qubit, gate_name)

    def measurement_circuit(self,
                            op: str,
                            qubit: QuantumRegister,
                            clbit: ClassicalRegister
                            ) -> QuantumCircuit:
        """
        Creates a measurement circuit for the SPAM op

        Params:
            op: SPAM circuit name
            qubit: qubit to be operated on and measured
            clbit: clbit for measurement outcome.

        Returns:
            The measurement circuit
        """
        circ = QuantumCircuit(qubit.register, clbit.register)
        self.add_spam_to_circuit(circ, qubit, op)
        circ.measure(qubit, clbit)
        return circ

    def measurement_matrix(self, label: str) -> np.array:
        """
         Returns the matrix corresponding to a gate label

        Args:
            label: Gate label

        Returns:
            The corresponding matrix
        """
        return self.gate_matrices[label]

    def preparation_circuit(self,
                            op: str,
                            qubit: QuantumRegister
                            ) -> QuantumCircuit:
        """
        Creates a preperation circuit for the SPAM op

        Params:
            op: SPAM circuit name
            qubit: qubit to be operated on

        Returns:
            The preperation circuit
        """
        circ = QuantumCircuit(qubit.register)
        self.add_spam_to_circuit(circ, qubit, op)
        return circ

    def preparation_matrix(self, label: str) -> np.array:
        """
        Returns the matrix corresponding to a gate label

        Params:
            label: Gate label

        Returns:
            The corresponding matrix
        """
        return self.gate_matrices[label]

    def spam_matrix(self, label: str) -> np.array:
        """
        Returns the matrix corresponding to a spam label
        Every spam is a sequence of gates, and so the result matrix
        is the product of the matrices corresponding to those gates

        Params:
            label: Spam label

        Returns:
            The corresponding matrix
        """
        spec = self.spam_spec[label]
        f_matrices = [self.gate_matrices[gate_label] for gate_label in spec]
        result = functools.reduce(lambda a, b: a @ b, f_matrices)
        return result

    def get_tomography_basis(self) -> TomographyBasis:
        """
        Returns a TomographyBasis object
        corresponding to the gate set tomography data

        A TomographyBasis object should have
        for both measurements and preperations
        the sets of labels (the SPAM labels in our case),
        circuit creation functions and corresponding matrices

        Returns:
            The gateset tomography data formatted as a TomographyBasis object
        """
        return TomographyBasis(self.name,
                               measurement=(self.spam_labels,
                                            self.measurement_circuit,
                                            self.measurement_matrix),
                               preparation=(self.spam_labels,
                                            self.preparation_circuit,
                                            self.preparation_matrix))


def default_gateset_basis():
    """Returns a default tomographically-complete gateset basis
        Return value: The gateset given as example 3.4.1 in arXiv:1509.02921

    """
    default_gates = {
        'Id': lambda circ, qubit: None,
        'X_Rot_90': lambda circ, qubit: circ.u2(-np.pi / 2, np.pi / 2, qubit),
        'Y_Rot_90': lambda circ, qubit: circ.u2(np.pi, np.pi, qubit)
    }
    default_spam = {
        'F0': ('Id',),
        'F1': ('X_Rot_90',),
        'F2': ('Y_Rot_90',),
        'F3': ('X_Rot_90', 'X_Rot_90')
    }
    return GateSetBasis('Default GST', default_gates, default_spam)
