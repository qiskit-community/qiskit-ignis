# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
    Clifford Operator class
"""

import numpy as np
#from qiskit import QiskitError
from qiskit.quantum_info import Pauli
#from qiskit_ignis.randomized_benchmarking.standard_rb.BinaryVector import BinaryVector


class Clifford(object):
    """Clifford Class"""

    def __init__(self, num_qubits=None, table=None, phases=None):
        """Initialize an n-qubit Clifford table."""
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
                self._phases = np.array(phases, dtype=np.int8)
                shape = self._phases.shape
                if len(shape) != 1 or shape[0] != 2 * self._num_qubits:
                    raise ValueError("Invalid phases")
            else:
                # Initialize all phases as zero
                self._phases = np.zeros(2 * self._num_qubits, dtype=np.int8)

        # Initialize from num_qubits only:
        else:
            if num_qubits is not None:
                num_qubits = 0
            self._num_qubits = num_qubits
            # Initialize symplectic table: [[Z(0), X(1)], [Z(1), X(0)]]
            zeros = np.zeros((num_qubits, num_qubits), dtype=np.bool)
            ones = np.ones((num_qubits, num_qubits), dtype=np.bool)
            self._table = np.block([[zeros, ones], [ones, zeros]])
            # Initialize phases
            self._phases = np.zeros(2 * num_qubits, dtype=np.int8)

    # ---------------------------------------------------------------------
    # Data accessors
    # ---------------------------------------------------------------------

    @property
    def num_qubits(self):
        """Return the number of qubits for the Clifford."""
        return self._num_qubits

    @property
    def table(self):
        """Return the Clifford tableau."""
        return self._table

    @property
    def phases(self):
        """Return the phases for the Clifford."""
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
        phase_coeffs = ['', '-']  # Modify later if we want to include i and -i.
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
            raise ValueError("Invalid Clifford dict: length of stabilizers and destabilizers do not match.")
        cls._num_qubits = len(stabilizers) #Shelly: added cls

    # ---------------------------------------------------------------------
    # Canonical gate operations
    # ---------------------------------------------------------------------

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
        """Apply v gate h.s.h.s"""
        # NOTE: This is convention is probably wrong, check definition of v gate
        #       from randomizedbenchmarking.py (possibly r gate there)
        # TODO: change direct table update if more efficient
        # Shelly: changed to hshs (instead of shsh)
        self.h(qubit)
        self.s(qubit)
        self.h(qubit)
        self.s(qubit)

    def w(self, qubit):
        """Apply w gate v.v"""
        # TODO: change direct table update if more efficient
        self.v(qubit)
        self.v(qubit)

    def cx(self, qubit_ctrl, qubit_trgt):
        """Apply a Controlled-NOT "cx" gate"""
        # Helper indicies for stabilizer columns
        iz_c, ix_c = qubit_ctrl, self.num_qubits + qubit_ctrl
        iz_t, ix_t = qubit_trgt, self.num_qubits + qubit_trgt
        # Compute phase
        tmp = np.logical_xor(self._table[:, ix_t], self._table[:, iz_c])
        tmp = np.logical_xor(1, tmp) #Shelly: fixed misprint in logical_xor
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


    #    def __init__(self, nqubits):
#        self.n = nqubits
#        self.table = []
#        # initial state = all-zeros
        # add destabilizers
#        for i in range(self.n):
#            pauli = {'X': BinaryVector(self.n), 'Z': BinaryVector(self.n), 'phase': 0}
#            pauli['X'].setvalue(1, i)
#            self.table.append(pauli)
#        # add stabilizers
#        for i in range(self.n):
#            pauli = {'X': BinaryVector(self.n), 'Z': BinaryVector(self.n), 'phase': 0}
#            pauli['Z'].setvalue(1, i)
#            self.table.append(pauli)
        # add auxiliary row
#        pauli = {'X': BinaryVector(self.n), 'Z': BinaryVector(self.n), 'phase': 0}
#        self.table.append(pauli)
#        self.circuit = []
#        self.cannonical = None
#        self.matrix = None



# ----------------------------------------------------------------------------------------
# Clifford properties
# ----------------------------------------------------------------------------------------
#    def get_table(self):
#        """return the table representation, updating from a matrix if needed"""
#        if self.table is None:
#            return self.to_table()
#        return self.table

#    def set_cannonical(self, cannonical):
#        """set internal memory of cannonical order
#        --must be set externally, does not decide for itself"""
#        self.cannonical = cannonical

#    def get_cannonical(self):
#        """get cannoical order, if set"""
#        return self.cannonical

#    def get_circuit(self):
#        """give circuit structure -- does not compute the circuit"""
#        return self.circuit

#    def circuit_append(self, gatelist):
#        """add to circuit list"""
#        self.circuit = self.circuit + gatelist

#    def circuit_prepend(self, gatelist):
#        """add to circuit list -- not used"""
#        self.circuit = gatelist + self.circuit

#    def size(self):
#        """report size in number of qubits"""
#        return self.n

#    def print_table(self):
#        """print out the table form of the Clifford -- used for debug"""
#        table = self.get_table()
#        for i in range(2*self.n):
#            print(table[i]['X'].m_data, table[i]['Z'].m_data, table[i]['phase'])
#        print("--------------------------------------------")

#    def print_matrix(self):
#        """Print out the matrix form of the Clifford tableau -- used for debug"""
#        matrix = self.get_matrix()
#        print(matrix)
#        print("--------------------------------------------")

#    def get_matrix(self):
#        """Return and set 2n+1 X 2n dimensional matrix for of a Clifford tableau"""
#        self.matrix = np.zeros([2*self.n, 2*self.n+1], dtype=np.uint8, order='F')
#        for i in range(2*self.n):
#            for j in range(self.n):
#                self.matrix[i, j] = self.table[i]['X'][j]
#                self.matrix[i, j+self.n] = self.table[i]['Z'][j]
#        for i in range(2*self.n):
#            self.matrix[i, 2*self.n] = self.table[i]['phase']
#        return self.matrix

#    def to_table(self):
#        """Return and set tableau in table form (used in get_table)"""
#        self.table = []
#        for i in range(2*self.n):
#            pauli = {'X': BinaryVector(self.n), 'Z': BinaryVector(self.n), 'phase': 0}
#            for j in range(self.n):
#                pauli['X'][j] = self.matrix[i, j]
#                pauli['Z'][j] = self.matrix[i, j+self.n]
#            pauli['phase'] = self.matrix[i, 2*self.n]
#            self.table.append(pauli)
        # append auxiallary row
#        pauli = {'X': BinaryVector(self.n), 'Z': BinaryVector(self.n), 'phase': 0}
#        self.table.append(pauli)
#        self.matrix = None
#        return self.table

# ----------------------------------------------------------------------------------------
# Quantum gates operations
# ----------------------------------------------------------------------------------------
#    def cx(self, con, tar):
#        """Controlled-x gate"""
#        self.get_table()
#        for i in range(2*self.n):
#            self.table[i]['phase'] ^= self.table[i]['X'][con] and\
#                self.table[i]['Z'][tar] and (self.table[i]['X'][tar] ^ self.table[i]['Z'][con] ^ 1)
#        for i in range(2*self.n):
#            self.table[i]['X'].setvalue(self.table[i]['X'][tar] ^ self.table[i]['X'][con], tar)
#            self.table[i]['Z'].setvalue(self.table[i]['Z'][tar] ^ self.table[i]['Z'][con], con)

#    def s(self, q):
#        """Phase-gate"""
#        self.get_table()
#        for i in range(2*self.n):
#            self.table[i]['phase'] ^= (self.table[i]['X'][q] and self.table[i]['Z'][q])
#            self.table[i]['Z'].setvalue(self.table[i]['Z'][q] ^ self.table[i]['X'][q], q)

#    def z(self, q):
#        """Z-Gate"""
#        self.get_table()
#        for i in range(2*self.n):
#            self.table[i]['phase'] ^= self.table[i]['X'][q]

#    def x(self, q):
#        """X-Gate"""
#        self.get_table()
#        for i in range(2*self.n):
#            self.table[i]['phase'] ^= self.table[i]['Z'][q]

#    def y(self, q):
#        """Y-Gate"""
#        self.get_table()
#        for i in range(2*self.n):
#            self.table[i]['phase'] ^= (self.table[i]['Z'][q] ^ self.table[i]['X'][q])

#    def h(self, q):
#        """Hadamard Gate"""
#        self.get_table()
#        for i in range(2*self.n):
#            self.table[i]['phase'] ^= (self.table[i]['X'][q] and self.table[i]['Z'][q])
            # exchange X and Z
#            b = self.table[i]['X'][q]
#            self.table[i]['X'].setvalue(self.table[i]['Z'][q], q)
#           self.table[i]['Z'].setvalue(b, q)

#    def sdg(self, q):
#        """Inverse phase Gate, sdg=s.z"""
#        self.s(q)
#        self.z(q)

#    def v(self, q):
#        """V Gate, V=HSHS"""
#        self.h(q)
#        self.s(q)
#        self.h(q)
#        self.s(q)

#    def w(self, q):
#        """W Gate, the inverse of V-gate"""
#        self.sdg(q)
#        self.h(q)
#        self.sdg(q)
#        self.h(q)

