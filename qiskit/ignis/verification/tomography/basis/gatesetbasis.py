# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
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
from typing import Tuple
import numpy as np

# Import QISKit classes
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
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
    def __init__(self, name: str,
                 gates: Tuple,
                 spam: Tuple):
        """
        Initialize the gate set basis data

        The gates data is a tuple (names, circuit_fn, matrix_list) containing
        * **names** - the names (strings) of the elements of the gateset.
        * **circuit_fns** a dictionary str -> function, that for every
            gate name returns a function taking a pair
            (QuantumCircuit, QuantumRegister) and appends to the circuit
            the gate on the given qubits denotes by the given string.
        * **matrix_list** - a tuple containing the matrices describing
            the gateset elements.

        The SPAM data is a tuple (names, specs) containing
        * **names** - the names (strings) of the elements of the SPAM circuits.
        * **specs** a dictionary that for every SPAM circuit holds a
            tuple of the gate names for the gates comprising the circuit
            in the order they appear.
        Args:
            name: Name of the basis.
            gates: a tuple containing the gate data.
            spam: a tuple containing the SPAM data
        """
        self.name = name
        self.gate_labels = gates[0]
        self.gate_funcs = gates[1]
        self.gate_matrices = dict(zip(self.gate_labels, gates[2]))
        self.spam_labels = spam[0]
        self.spam_spec = spam[1]

    def add_to_circuit(self,
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
        for gate in op_gates:
            self.gate_funcs[gate](circ, qubit)

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
        self.add_to_circuit(circ, qubit, op)
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
        self.add_to_circuit(circ, qubit, op)
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
        F_matrices = [self.gate_matrices[gate_label] for gate_label in spec]
        result = functools.reduce(lambda a, b: a @ b, F_matrices)
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


standard_gates_funcs = {
    'Id': lambda circ, qubit: None,
    'X_Rot_90': lambda circ, qubit: circ.u2(-np.pi / 2, np.pi / 2, qubit),
    'Y_Rot_90': lambda circ, qubit: circ.u2(np.pi, np.pi, qubit)
}

# PTM representation of Id
G0 = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])
# X rotation by 90 degrees
G1 = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 0, -1],
               [0, 0, 1, 0]])
# Y rotation by 90 degrees
G2 = np.array([[1, 0, 0, 0],
               [0, 0, 0, -1],
               [0, 0, 1, 0],
               [0, 1, 0, 0]])
standard_gates_matrices = (G0, G1, G2)
StandardGatesetBasis = GateSetBasis('Standard GST',
                                    (('Id', 'X_Rot_90', 'Y_Rot_90'),
                                     standard_gates_funcs,
                                     standard_gates_matrices),
                                    (('F0', 'F1', 'F2', 'F3'),
                                     {'F0': ('Id',),
                                      'F1': ('X_Rot_90',),
                                      'F2': ('Y_Rot_90',),
                                      'F3': ('X_Rot_90', 'X_Rot_90')
                                      }))
