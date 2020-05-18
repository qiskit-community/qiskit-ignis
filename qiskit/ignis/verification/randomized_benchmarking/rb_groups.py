# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Methods for handling groups (Clifford, CNOT Dihedral etc.)
in randomized benchmarking"""

import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.symplectic import Clifford
from qiskit.quantum_info.random import random_clifford
from .dihedral import CNOTDihedral, random_cnotdihedral


class RBgroup():
    """Class that handles the group operations needed for RB."""

    def __init__(self, num_qubits, group_gates):
        """Initialization from num_qubits and group_gates"""
        self.num_qubit = num_qubits
        self.group_gates = group_gates

        if group_gates is None or group_gates in ('0',
                                                  'Clifford',
                                                  'clifford'):
            self.rb_group = Clifford
            self.rb_circ_type = 'rb'
            self.group_gates_type = 0
        elif group_gates in ('1', 'Non-Clifford',
                             'NonClifford'
                             'CNOTDihedral',
                             'CNOT-Dihedral'):
            self.rb_group = CNOTDihedral
            self.rb_circ_type = 'rb_cnotdihedral'
            self.group_gates_type = 1
            assert num_qubits <= 2, "num_qubits for CNOT-Dihedral RB should be 1 or 2"
        else:
            raise QiskitError("Unknown group or set of gates.")

    def num_qubits(self):
        """Return the number of qubits."""
        return self.num_qubits

    def group_gate_type(self):
        """Return an integer of the group type"""
        return self.group_gate_type

    def rb_circ_type(self):
        """Return a string of type for the circuit name"""
        return self.rb_circ_type

    def iden(self):
        """Initialize an identity group element"""
        if self.group_gates_type:
            return CNOTDihedral(self.num_qubits)
        else:
            return Clifford(np.eye(2 * self.num_qubits))

    def random(self):
        """Generate a random group element"""
        if self.group_gates_type:
            return random_cnotdihedral(self.num_qubits)
        else:
            return random_clifford(self.num_qubits)

    def compose(self, orig, other):
        """Compose two group elements: orig and other"""
        orig_elem = self.rb_group(orig)
        other_elem = self.rb_group(other)
        return orig_elem.compose(other_elem)

    def inverse(self, orig):
        """Computes the inverse QuantumCircuit"""
        elem = self.rb_group(orig)
        # decompose the group element into a QuantumCircuit
        circ = elem.to_circuit()
        # invert the QuantumCircuit
        return circ.inverse()

    def to_circuit(self, orig):
        """Returns the corresponding QuantumCircuit"""
        elem = self.rb_group(orig)
        return elem.to_circuit()
