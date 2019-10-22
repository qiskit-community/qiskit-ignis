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

# NOTE(mtreinish): Needed to avoid error on logical_xor where pylint thinks it
# doesn't have a return.
# pylint: disable=assignment-from-no-return

"""
    Clifford Operator class
"""

import numpy as np
from qiskit.quantum_info import Pauli


class Clifford:

    """Clifford class"""

    def __init__(self, num_qubits=None, table=None, phases=None):
        """Initialize an n-qubit Clifford table."""
        # Use index for initializing 1 and 2 qubit Cliffords
        # If index is none initilize to identity.

        # It will be easier for internal functions to use a single np array
        # for the Clifford table rather than the Pauli class, and we can
        # convert rows to the Pauli class as needed.

        # Initialize internal variables
        self._num_qubits = None
        self._table = None
        self._phases = None

        # Initialize from table and phases
        if table is not None:
            # Store symplectic table
            self._table = np.array(table, dtype=bool)
            shape = self._table.shape
            if len(shape) != 2 or shape[0] != shape[1] or (shape[0] % 2):
                raise ValueError("Invalid symplectic table input")
            self._num_qubits = shape[0] // 2
            # Store phases
            if phases:
                self._phases = np.array(phases, dtype=np.bool)
                shape = self._phases.shape
                if len(shape) != 1 or shape[0] != 2 * self._num_qubits:
                    raise ValueError("Invalid phases")
            else:
                # Initialize all phases as zero
                self._phases = np.zeros(2 * self._num_qubits, dtype=np.bool)

        # Initialize from num_qubits only:
        else:
            if num_qubits is not None:
                self._num_qubits = num_qubits

                # Initialize symplectic table
                zeros = np.zeros((num_qubits, num_qubits), dtype=np.bool)
                iden = np.eye(num_qubits, dtype=np.bool)
                self._table = np.block([[zeros, iden], [iden, zeros]])
                # Initialize phases
                self._phases = np.zeros(2 * num_qubits, dtype=np.bool)

    def __repr__(self):
        # TODO: truncate output for large tables
        output = 'Clifford(phases={},\n'.format(repr(self._phases.tolist()))
        table_str = '    table=['
        pad = "".join(len(table_str) * [' '])
        for j, row in enumerate(self._table):
            if j > 0:
                table_str += pad
            table_str += repr(row.tolist())
            if j < 2 * self.num_qubits - 1:
                table_str += ",\n"
            else:
                table_str += "])"
        return output + table_str

    # ---------------------------------------------------------------------
    # Data accessors
    # ---------------------------------------------------------------------

    @property
    def num_qubits(self):
        """Return the number of qubits for the Clifford."""
        return self._num_qubits

    @property
    def table(self):
        """Return the the Clifford table."""
        return self._table

    @property
    def phases(self):
        """Return the Clifford phases."""
        return self._phases

    def __getitem__(self, index):
        """Get element from internal symplectic table"""
        return self._table[index]

    def __setitem__(self, index, value):
        """Set element of internal symplectic table"""
        if isinstance(value, Pauli):
            # Update from Pauli object
            self._table[index] = np.block([value.z, value.x])
        else:
            # Update table as Numpy array
            self._table[index] = value

    # ---------------------------------------------------------------------
    # Interface with Pauli object
    # ---------------------------------------------------------------------

    def stabilizer(self, qubit):
        """Return the qubit stabilizer as a Pauli object"""
        nq = self._num_qubits
        z = self._table[nq + qubit, 0:nq]
        x = self._table[nq + qubit, nq:2 * nq]
        return Pauli(z=z, x=x)

    def update_stabilizer(self, qubit, pauli):
        """Update the qubit stabilizer row from a Pauli object"""
        self[self._num_qubits + qubit] = pauli

    def destabilizer(self, row):
        """Return the destabilizer as a Pauli object"""
        nq = self._num_qubits
        z = self._table[row, 0:nq]
        x = self._table[row, nq:2 * nq]
        return Pauli(z=z, x=x)

    def update_destabilizer(self, qubit, pauli):
        """Update the qubit destabilizer row from a Pauli object"""
        self[qubit] = pauli

    # ---------------------------------------------------------------------
    # JSON / Dict conversion
    # ---------------------------------------------------------------------

    def as_dict(self):
        """Return dictionary (JSON) represenation of Clifford object"""
        # Modify later if we want to include i and -i.
        phase_coeffs = ['', '-']
        stabilizers = []
        for qubit in range(self.num_qubits):
            label = self.stabilizer(qubit).to_label()
            phase = self.phases[self.num_qubits + qubit]
            stabilizers.append(phase_coeffs[phase] + label)
        destabilizers = []
        for qubit in range(self.num_qubits):
            label = self.destabilizer(qubit).to_label()
            phase = self.phases[qubit]
            destabilizers.append(phase_coeffs[phase] + label)
        return {"stabilizers": stabilizers, "destabilizers": destabilizers}

    @classmethod
    def from_dict(cls, clifford_dict):
        """Load a Clifford from a dictionary"""

        # Validation
        if not isinstance(clifford_dict, dict) or \
           'stabilizers' not in clifford_dict or \
           'destabilizers' not in clifford_dict:
            raise ValueError("Invalid input Clifford dictionary.")

        stabilizers = clifford_dict['stabilizers']
        destabilizers = clifford_dict['destabilizers']
        if len(stabilizers) != len(destabilizers):
            raise ValueError(
                "Invalid Clifford dict: length of stabilizers and "
                "destabilizers do not match.")
        num_qubits = len(stabilizers)

        # Helper function
        def get_row(label):
            """Return the Pauli object and phase for stabilizer"""
            if label[0] in ['I', 'X', 'Y', 'Z']:
                pauli = Pauli.from_label(label)
                phase = 0
            elif label[0] == '+':
                pauli = Pauli.from_label(label[1:])
                phase = 0
            elif label[0] == '-':
                pauli = Pauli.from_label(label[1:])
                phase = 1
            return pauli, phase

        # Generate identity Clifford on number of qubits
        clifford = cls(num_qubits)
        # Update stabilizers
        for qubit, label in enumerate(stabilizers):
            pauli, phase = get_row(label)
            clifford[num_qubits + qubit] = pauli
            clifford.phases[num_qubits + qubit] = phase
        # Update destabilizers
        for qubit, label in enumerate(destabilizers):
            pauli, phase = get_row(label)
            clifford[qubit] = pauli
            clifford.phases[qubit] = phase
        return clifford

    # ---------------------------------------------------------------------
    # Unique Clifford index
    # ---------------------------------------------------------------------
    def index(self):
        """
        Returns a unique index for the Clifford.

        Returns:
            A unique index (integer).
        """
        mat = self.table
        mat = mat.reshape(mat.size)
        ret = int(0)
        for bit in mat:
            ret = (ret << 1) | int(bit)
        mat = self.phases
        mat = mat.reshape(mat.size)
        for bit in mat:
            ret = (ret << 1) | int(bit)
        return ret

    # ---------------------------------------------------------------------
    # Canonical gate operations
    # ---------------------------------------------------------------------

    # NOTE: These might change based on changes to QuantumCircuit API.
    # They should mimic the circuit API as much as possible.
    def x(self, qubit):
        """Apply a Pauli "x" gate to a qubit"""
        iz = qubit
        self._phases = np.logical_xor(self._phases, self._table[:, iz])

    def y(self, qubit):
        """Apply an Pauli "y" gate to a qubit"""
        iz, ix = qubit, self._num_qubits + qubit
        zx_xor = np.logical_xor(self._table[:, iz], self._table[:, ix])
        self._phases = np.logical_xor(self._phases, zx_xor)

    def z(self, qubit):
        """Apply an Pauli "z" gate to qubit"""
        ix = self._num_qubits + qubit
        self._phases = np.logical_xor(self._phases, self._table[:, ix])

    def h(self, qubit):
        """Apply an Hadamard "h" gate to qubit"""
        iz, ix = qubit, self._num_qubits + qubit
        zx_and = np.logical_and(self._table[:, ix], self._table[:, iz])
        self._phases = np.logical_xor(self._phases, zx_and)
        # Cache X column for qubit
        x_cache = self._table[:, ix].copy()
        # Swap X and Z columns for qubit
        self._table[:, ix] = self._table[:, iz]  # Does this need to be a copy?
        self._table[:, iz] = x_cache

    def s(self, qubit):
        """Apply an phase "s" gate to qubit"""
        iz, ix = qubit, self._num_qubits + qubit
        zx_and = np.logical_and(self._table[:, ix], self._table[:, iz])
        self._phases = np.logical_xor(self._phases, zx_and)
        self._table[:, iz] = np.logical_xor(self._table[:, ix],
                                            self._table[:, iz])

    def sdg(self, qubit):
        """Apply an adjoint phase "sdg" gate to qubit"""
        # TODO: change direct table update if more efficient
        self.z(qubit)
        self.s(qubit)

    def v(self, qubit):
        """Apply v gate sd.h"""
        # TODO: change direct table update if more efficient
        self.sdg(qubit)
        self.h(qubit)

    def w(self, qubit):
        """Apply w gate v.v"""
        # TODO: change direct table update if more efficient
        self.h(qubit)
        self.s(qubit)

    def cx(self, qubit_ctrl, qubit_trgt):
        """Apply a Controlled-NOT "cx" gate"""
        # Helper indices for stabilizer columns
        iz_c, ix_c = qubit_ctrl, self.num_qubits + qubit_ctrl
        iz_t, ix_t = qubit_trgt, self.num_qubits + qubit_trgt
        # Compute phase
        tmp = np.logical_xor(self._table[:, ix_t], self._table[:, iz_c])
        tmp = np.logical_xor(1, tmp)  # Shelly: fixed misprint in logical
        tmp = np.logical_and(self._table[:, iz_t], tmp)
        tmp = np.logical_and(self._table[:, ix_c], tmp)
        self._phases ^= tmp
        # Update stabilizers
        self._table[:, ix_t] = np.logical_xor(self._table[:, ix_t],
                                              self._table[:, ix_c])
        self._table[:, iz_c] = np.logical_xor(self._table[:, iz_t],
                                              self._table[:, iz_c])

    def cz(self, qubit_ctrl, qubit_trgt):
        """Apply a Controlled-z "cx" gate"""
        # TODO: change direct table update if more efficient
        self.h(qubit_trgt)
        self.cx(qubit_ctrl, qubit_trgt)
        self.h(qubit_trgt)

    def swap(self, qubit0, qubit1):
        """Apply SWAP gate between two qubits"""
        # TODO: change direct swap of required rows and cols in table
        self.cx(qubit0, qubit1)
        self.cx(qubit1, qubit0)
        self.cx(qubit0, qubit1)
