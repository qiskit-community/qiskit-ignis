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
Generates accreditation circuits
"""

import numpy as np
from numpy import random

from qiskit import (QuantumCircuit, QuantumRegister, ClassicalRegister,
                    QiskitError)
from qiskit.quantum_info.synthesis import euler_angles_1q


# pylint: disable=no-member
def routine_one(gate, cz_gate, type_circ):
    """
    Routine 1.
    It appends QOTP to gate in the circuit.
        Args:
            gate (list): list of all 1-qubit gate in circuit
            cz_gate (list): list of all cz gate in circuit
            type_circ (integer): 0 if circuit is target, 1 if circuit is trap
        Returns:
            qotp_gate (list): list of all 1-qubit gate and QOTP
            postp (list): string used in classical post-processing
    """

    # Size of un-encoded circuit
    n_qb = len(gate)
    m_bands = len(gate[0])

    # Increase the lenght of cz_gate for further use in while loop
    cz_gate = np.vstack([cz_gate, [0, 0, 0]])

    # TARGET encoding
    if type_circ == 0:
        # Define a new set of 1-qubit gate initialized to identity
        qotp_gate = [[[0, 0, 0] for j in range(3*m_bands)]
                     for i in range(n_qb)]

        # Place 1-qubit gate from the target
        for i in range(n_qb):
            for j in range(m_bands):
                qotp_gate[i][3*j+1] = gate[i][j]

        # QOTP before first band of gate (Pauli-Z)
        for i in range(n_qb):
            if random.randint(0, 2) == 1:
                qotp_gate[i][0] = [0, np.pi/2, np.pi/2]

        # QOTP in middle circuit
        qotp = [[0, 0, 0], [np.pi, np.pi/2, -np.pi/2], [0, np.pi/2, np.pi/2],
                [np.pi, 0, 0]]
        for j in range(m_bands-1):
            alpha = [random.randint(0, 4) for i in range(n_qb)]
            for i in range(n_qb):
                # This undoes QOTP for qubits not connected by cZ
                qotp_gate[i][3*j+2] = qotp[alpha[i]]
                qotp_gate[i][3*j+3] = qotp[alpha[i]]

            # This undoes QOTP for qubits connected by a cZ
            # It erases rows in matrix cz_gate
            if len(cz_gate) != 1:
                while cz_gate[0][0] == j:
                    if alpha[cz_gate[0][1]] == 1 or alpha[cz_gate[0][1]] == 3:
                        qotp_gate[cz_gate[0][2]][3*j+3] = \
                            qotp[(alpha[cz_gate[0][2]]+2) % 4]
                    if alpha[cz_gate[0][2]] == 1 or alpha[cz_gate[0][2]] == 3:
                        qotp_gate[cz_gate[0][1]][3*j+3] = \
                            qotp[(alpha[cz_gate[0][1]] + 2) % 4]
                    cz_gate = np.delete(cz_gate, (0), axis=0)

        # QOTP before the measurements
        alpha = [random.randint(0, 4) for i in range(n_qb)]
        postp = [(alpha[i]) % 2 for i in range(n_qb)]
        for i in range(n_qb):
            qotp_gate[i][3*m_bands-1] = qotp[alpha[i]]

    # TRAP encoding
    else:
        n_qb = len(gate)
        m_bands = int(len(gate[0])/3)

        # Define a new set of 1-qubit gate initialized to identity
        qotp_gate = [[[0, 0, 0] for j in range(4*m_bands+4)]
                     for i in range(n_qb)]

        # Place 1-qubit gate from the target
        for j in range(m_bands+1):
            for i in range(n_qb):
                qotp_gate[i][4*j+1] = gate[i][2*j+1]
                qotp_gate[i][4*j+2] = gate[i][2*j+2]

        # QOTP before first band of gate (Pauli-Z)
        for i in range(n_qb):
            if random.randint(0, 2) == 1:
                qotp_gate[i][0] = [0, np.pi/2, np.pi/2]

        # QOTP
        qotp = [[0, 0, 0], [np.pi, np.pi/2, -np.pi/2], [0, np.pi/2, np.pi/2],
                [np.pi, 0, 0]]
        for j in range(m_bands):
            alpha = [random.randint(0, 4) for i in range(n_qb)]
            for i in range(n_qb):
                qotp_gate[i][4*j+3] = qotp[alpha[i]]
                qotp_gate[i][4*j+4] = qotp[alpha[i]]

            # This undoes QOTP for qubits not connected by cZ.
            # It follows the instructions given by matrix cz_gate, erasing
            # lines in cz_gate when band is completed
            if len(cz_gate) != 1:
                while cz_gate[0][0] == j:
                    if alpha[cz_gate[0][1]] == 1 or alpha[cz_gate[0][1]] == 3:
                        qotp_gate[cz_gate[0][2]][4*j+4] = \
                            qotp[(alpha[cz_gate[0][2]] + 2) % 4]
                    if alpha[cz_gate[0][2]] == 1 or alpha[cz_gate[0][2]] == 3:
                        qotp_gate[cz_gate[0][1]][4*j+4] = \
                            qotp[(alpha[cz_gate[0][1]] + 2) % 4]
                    cz_gate = np.delete(cz_gate, (0), axis=0)

        # QOTP before the measurements
        alpha = [random.randint(0, 4) for i in range(n_qb)]
        postp = [(alpha[i]) % 2 for i in range(n_qb)]
        for i in range(n_qb):
            qotp_gate[i][4*m_bands+3] = qotp[alpha[i]]

    # Flip
    post1 = [0 for i in range(n_qb)]
    for i in range(n_qb):
        post1[i] = postp[n_qb-1-i]
    postp = post1

    return qotp_gate, postp


def routine_two(n_qb, m_bands, cz_gate):
    """
    Routine 2.
    It returns random 1-qubit gate for trap circuits
        Args:
            n_qb (int): number of qubits
            m_bands (int): number of bands
            cz_gate (list): list of all cz gate in circuit

        Returns:
            gate_trap (list): list of all 1-qubit gates in trap circuit
    """

    # Define a new set of 1-qubit gates initialized to identity
    gate_trap = [[[0, 0, 0] for j in range(2*m_bands)] for i in range(n_qb)]

    # Place gate in the trap circ
    for j in range(m_bands-1):
        for i in range(n_qb):
            if random.randint(0, 2) == 0:
                gate_trap[i][2*j+1] = [np.pi/2, 0, np.pi]
                gate_trap[i][2*j+2] = [np.pi/2, 0, np.pi]
            else:
                gate_trap[i][2*j+1] = [0, 0, np.pi/2]
                gate_trap[i][2*j+2] = [0, 0, -np.pi/2]

    # This fixes the gate for qubits connected by cZ gate
    for i_cz, _ in enumerate(cz_gate):
        if random.randint(0, 2) == 0:
            gate_trap[cz_gate[i_cz][1]][2*(cz_gate[i_cz][0])+1] =\
                [np.pi/2, 0, np.pi]
            gate_trap[cz_gate[i_cz][1]][2*(cz_gate[i_cz][0])+2] =\
                [np.pi/2, 0, np.pi]
            gate_trap[cz_gate[i_cz][2]][2*(cz_gate[i_cz][0])+1] =\
                [0, 0, np.pi/2]
            gate_trap[cz_gate[i_cz][2]][2*(cz_gate[i_cz][0])+2] =\
                [0, 0, -np.pi/2]
        else:
            gate_trap[cz_gate[i_cz][1]][2*(cz_gate[i_cz][0])+1] =\
                [0, 0, np.pi/2]
            gate_trap[cz_gate[i_cz][1]][2*(cz_gate[i_cz][0])+2] =\
                [0, 0, -np.pi/2]
            gate_trap[cz_gate[i_cz][2]][2*(cz_gate[i_cz][0])+1] =\
                [np.pi/2, 0, np.pi]
            gate_trap[cz_gate[i_cz][2]][2*(cz_gate[i_cz][0])+2] =\
                [np.pi/2, 0, np.pi]

    # Place random H gate at beginning and end of circuit
    if random.randint(0, 2) == 1:
        for i in range(n_qb):
            gate_trap[i][0] = [np.pi/2, 0, np.pi]
            gate_trap[i][2*m_bands-1] = [np.pi/2, 0, np.pi]

    gate_trap = np.hstack([[[[0, 0, 0]] for i in range(n_qb)], gate_trap])

    return gate_trap


def accreditation_circuits(target_circuit, num_trap):
    """
    Simulation of quantum circuit on backend
        Args:
            target_circuit (QuantumCircuit): Quantum circuit consisting of
                cZ gates and arbitrary single qubit gates, followed by Z
                measurements on all qubits
            num_trap (int): number of trap circuits

        Returns:
            circuit_list (list): accreditation circuits
            postp_list (list): strings used for classical post-processing
            v_zero (int): position of target circuit
    """
    gate_target, cz_gate = accreditation_parser(target_circuit)

    n_qb = len(gate_target)
    m_bands = len(gate_target[0])

    # Check if cz_gate is valid
    if cz_gate[len(cz_gate)-1][0] >= m_bands-1:
        QiskitError('ERROR: The last band must contain no cZ gates.')

    # Position of the target
    v_zero = random.randint(0, num_trap)

    circuit_list = [[] for e in range(num_trap+1)]
    postp_list = [[] for e in range(num_trap+1)]

    for k in range(num_trap+1):

        if k == v_zero:  # Generating the target circuit

            # Create a Quantum Register with n_qb qubits.
            q_reg = QuantumRegister(n_qb, 'q')
            # Create a Classical Register with n_qb bits.
            c_reg = ClassicalRegister(n_qb, 's')
            # Create a Quantum Circuit acting on the q register
            circ = QuantumCircuit(q_reg, c_reg)

            # Append one-time-pad to gates in target circuit
            gate_target_qotp, postp = routine_one(gate_target, cz_gate, 0)

            # Increase the size of conn for further use in while loop
            cz_gate_aux = np.vstack([cz_gate, [0, 0, 0]])

            band = 0

            # Generate target circuit
            for j in range(len(gate_target_qotp[0])):
                for i in range(n_qb):
                    circ.u3(gate_target_qotp[i][j][0],
                            gate_target_qotp[i][j][1],
                            gate_target_qotp[i][j][2],
                            i)

                # This places the cZ gate in the circuit.
                # It erases rows of conn
                if (j+1) % 3 == 0 and j != len(gate_target_qotp[0])-1:
                    circ.barrier()
                    if len(cz_gate_aux) != 1:
                        while cz_gate_aux[0][0] == band:
                            circ.cz(q_reg[int(cz_gate_aux[0][1])],
                                    q_reg[int(cz_gate_aux[0][2])])
                            cz_gate_aux = np.delete(cz_gate_aux, (0), axis=0)
                    circ.barrier()
                    band = band+1

            # Map the quantum measurement to the classical bits
            circ.measure(q_reg, c_reg)

            circuit_list[k].append(circ)
            postp_list[k].append(postp)

        else:  # Generating a trap circuit

            # Create a Quantum Register with n_qb qubits.
            q_reg = QuantumRegister(n_qb, 'q')
            # Create a Classical Register with n_qb bits.
            c_reg = ClassicalRegister(n_qb, 's')
            # Create a Quantum Circuit acting on the q register
            circ = QuantumCircuit(q_reg, c_reg)

            # Generate 1-qubit gates for trap
            gate_trap = routine_two(n_qb, m_bands, cz_gate)
            # Append one-time-pad to gates in trap circuit
            gate_trap_qotp, postp = routine_one(gate_trap, cz_gate, 1)

            # Increase the size of conn for further use in while loop
            cz_gate_aux = np.vstack([cz_gate, [0, 0, 0]])

            band = 0

            for j in range(len(gate_trap_qotp[0])):
                for i in range(n_qb):
                    circ.u3(gate_trap_qotp[i][j][0],
                            gate_trap_qotp[i][j][1],
                            gate_trap_qotp[i][j][2],
                            i)
                # This places the cZ gates
                if j % 4 == 3 and j != len(gate_trap_qotp[0])-1:
                    circ.barrier()
                    if len(cz_gate_aux) != 1:
                        while cz_gate_aux[0][0] == band:
                            circ.cz(q_reg[int(cz_gate_aux[0][1])],
                                    q_reg[int(cz_gate_aux[0][2])])
                            cz_gate_aux = np.delete(cz_gate_aux, (0), axis=0)
                    circ.barrier()
                    band = band+1

            # Map the quantum measurement to the classical bits
            circ.measure(q_reg, c_reg)

            circuit_list[k].append(circ)
            postp_list[k].append(postp)

    return circuit_list, postp_list, v_zero


def accreditation_parser(target_circuit):
    """
        Converts an input quantum circuit to lists representing the input
            Args:
                target_circuit (QuantumCircuit): Quantum circuit consisting of
                cZ gates and single qubit gates, followed by Pauli-Z measure-
                ments on all qubits
            Returns:
                gates_target (list): A 2D list of all 1-qubit gates in the
                    target circuit
                cz_gate (list): list of all cz gate in target circuit
        """
    # Initialize empty lists gates_target and cz_gate
    gates_target = []
    cz_gate = []

    # Qubits in the circuit
    circuit_qubits = target_circuit.qubits

    # Initialize empty list single_qubit_gates
    # This list will be used to store 1-qubit gates in the circuit
    single_qubit_gates = [[] for _ in circuit_qubits]

    # Initialize empty list
    # This is used to check if in a band, a qubit can still be entangled with
    # other qubits (qubits can be entanged one time per band)
    unavailiable_qubits = []

    # Keep track of current band
    current_band_no = 0

    # Loops over all gates in the circuit. An extra element is added so the
    # last band is closed at the end
    for gate in target_circuit.data + ['END STRING']:

        # Checks for special cases that need to be handled differently
        gate_qubits = gate[1]
        last_element = (gate == 'END STRING')
        is_measure = ((len(gate_qubits) == len(gate[2]))
                      and not gate == 'END STRING')

        # Records the position of the last cz gate
        if cz_gate:
            last_cz = cz_gate[-1][0]
        else:
            last_cz = -1

        # Makes sure measurements are ignored
        if not is_measure:
            circuit_end_band = last_element\
                and ((last_cz == current_band_no)
                     or (single_qubit_gates != [[] for _ in circuit_qubits]))

            # If a new band is required, converts the current band's single
            # qubit gates to Euler angles and prepares for the next band
            if((not set(gate_qubits).isdisjoint(set(unavailiable_qubits)))
               or circuit_end_band):
                band_gates_angles = []
                u3_gates_temp = []
                for qubit in single_qubit_gates:
                    matrix = np.array([[1, 0], [0, 1]])
                    for gate_index in qubit[::-1]:
                        matrix = np.matmul(matrix, gate_index[0].to_matrix())
                    band_gates_angles.append(euler_angles_1q(matrix))
                    u3_gates_temp.append(band_gates_angles[0])
                    band_gates_angles = []
                gates_target.append(u3_gates_temp)
                current_band_no += 1
                unavailiable_qubits = []
                single_qubit_gates = [[] for _ in circuit_qubits]
            if last_element:
                break

            # Adds single gates' gate object to the array corresponding to
            # their qubit in single_qubit_gates
            if len(gate_qubits) == 1:
                single_qubit_gates[gate_qubits[0].index].append(gate)
            # Records the location of 2 qubit gates
            else:
                cz_gate.append([current_band_no, gate_qubits[0].index,
                                gate_qubits[1].index])
                unavailiable_qubits += gate_qubits

    # Adds a band of identity gates if circuit ends with cz gates
    if last_cz == current_band_no - 1:
        last_band = []
        for qubit in circuit_qubits:
            last_band.append((0.0, 0.0, 0.0))
        gates_target.append(last_band)

    # Transposes the u3_gates matrix
    gates_target = [list(i) for i in zip(*gates_target)]

    return gates_target, cz_gate
