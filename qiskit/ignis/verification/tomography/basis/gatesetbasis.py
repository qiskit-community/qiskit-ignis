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
import numpy as np

# Import QISKit classes
from qiskit import QuantumCircuit
from .tomographybasis import TomographyBasis


class GateSetBasis:
    """
    This class contains the gateset data needed to perform gateset tomgography.
    This data is comprised of four elements:
    1) The labels (strings) of the gates
    2) A function f(circ, qubit, op)
        which adds to circ at qubit the gate labeled by op
    3) The labels of the SPAM circuits for the gate set tomography
    4) For SPAM label, tuple of gate labels for the gates in this SPAM circuit
    """
    def __init__(self, name, gates, spam):
        """
        Initialize the gate set basis data

        Params:
            name (str): Name of the basis.
            gates (tuple, function): the gate labels and gate function
            spam (tuple, dict): tuple of SPAM circuits names
            and dict of SPAM definitions
        """
        self.name = name
        self.gate_labels = gates[0]
        self.gate_func = gates[1]
        self.gate_matrices = dict(zip(self.gate_labels, gates[2]))
        self.spam_labels = spam[0]
        self.spam_spec = spam[1]

    def add_to_circuit(self, circ, qubit, op):
        """
        Adds the SPAM circuit op to circ at qubit

        Params:
            circ (QuantumCircuit): the circuit to apply op on
            qubit (QuantumRegister tuple): qubit to be operated on
            op (str): SPAM circuit name
        """
        if op not in self.spam_spec:
            raise RuntimeError("{} is not a SPAM circuit".format(op))
        op_gates = self.spam_spec[op]
        for gate in op_gates:
            self.gate_func(circ, qubit, gate)

    def measurement_circuit(self, op, qubit, clbit):
        """
        Creates a measurement circuit for the SPAM op

        Params:
            op (str): SPAM circuit name
            qubit (QuantumRegister tuple): qubit to be operated on and measured
            clbit (ClassicalRegister tuple): clbit for measurement outcome.

        Returns:
            A QuantumCircuit object.
        """
        circ = QuantumCircuit(qubit.register, clbit.register)
        self.add_to_circuit(circ, qubit, op)
        circ.measure(qubit, clbit)
        return circ

    def measurement_matrix(self, label):
        """
         Returns the matrix corresponding to a gate label

        Params:
            label (str): Gate label

        Returns:
            The corresponding matrix (usually numpy.array)
        """
        return self.gate_matrices[label]

    def preparation_circuit(self, op, qubit):
        """
        Creates a preperation circuit for the SPAM op

        Params:
            op (str): SPAM circuit name
            qubit (QuantumRegister tuple): qubit to be operated on

        Returns:
            A QuantumCircuit object.
        """
        circ = QuantumCircuit(qubit.register)
        self.add_to_circuit(circ, qubit, op)
        return circ

    def preparation_matrix(self, label):
        """
        Returns the matrix corresponding to a gate label

        Params:
            label (str): Gate label

        Returns:
            The corresponding matrix (usually numpy.array)
        """
        return self.gate_matrices[label]

    def get_tomography_basis(self):
        """
        Returns a TomographyBasis object
        corresponding to the gate set tomography data

        A TomographyBasis object should have
        for both measurements and preperations
        the sets of labels (the SPAM labels in our case),
        circuit creation functions
        and corresponding matrices (here given as stubs)

        Returns:
            A TomographyBasis object.
        """
        return TomographyBasis(self.name,
                               measurement=(self.spam_labels,
                                            self.measurement_circuit,
                                            self.measurement_matrix),
                               preparation=(self.spam_labels,
                                            self.preparation_circuit,
                                            self.preparation_matrix))


def standard_gates_func(circ, qubit, op):
    """
        The gate creation function for the default set of gates
        we use for gate set tomography:
        Id and rotations by 90 degrees around the X and Y axis

        Params:
            circ (QuantumCircuit): the circuit to add the gate to
            op (str): the gate name
            qubit (QuantumRegister tuple): qubit to be operated on

        Returns:
            A QuantumCircuit object.
    """
    if op == 'Id':
        pass
    if op == 'X_Rot_90':
        circ.u2(-np.pi / 2, np.pi / 2, qubit)
    if op == 'Y_Rot_90':
        circ.u2(np.pi, np.pi, qubit)


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
                                     standard_gates_func,
                                     standard_gates_matrices),
                                    (('F0', 'F1', 'F2', 'F3'),
                                     {'F0': ('Id',),
                                      'F1': ('X_Rot_90',),
                                      'F2': ('Y_Rot_90',),
                                      'F3': ('X_Rot_90', 'X_Rot_90')
                                      }))
