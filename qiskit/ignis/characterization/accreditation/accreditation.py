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
ACCREDITING OUTPUTS OF NISQ COMPUTING DEVICES (arXiv.1811.09709)
"""

import random
import numpy as np

from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    execute,
)


def str_to_num(output):
    """
    It transorms a string of characters into a list of integers.
        Args:
            output (list): A list of strings

        Returns:
            A list of integers, each one being 0 or 1
    """
    return [1 if s == "1" else 0 for s in output]


def check_cz(cz_gate):
    """
    It checks whether the cz gate in the circuit are compatible with the proto-
    col. It returns errors if the same qubit undergoes two cz gate in the same
    band of the circuit.
        Args:
            cz_gate (list): list of all cz gate in circuit

        Returns:
            0 if cz_gate is fine, 1 if cz_gate is incorrect
    """
    for i_cz, _ in enumerate(cz_gate):
        for j_cz in range(i_cz+1, len(cz_gate)):
            if cz_gate[i_cz][0] == cz_gate[j_cz][0]:
                if cz_gate[i_cz][1] == cz_gate[j_cz][1]\
                   or cz_gate[i_cz][2] == cz_gate[j_cz][2]:
                    print("ERROR: Check input ``cz_gate''.")
                    print("No qubit must undergo two cz within same band!\n")
                    return 1
    return 0


def routine_one(gate, cz_gate, b_num):
    """
    Routine 1.
    It appends QOTP to gate in the circuit.
        Args:
            gate (list): list of all 1-qubit gate in circuit
            cz_gate (list): list of all cz gate in circuit
            b_num (integer): 0 if circuit is target, 1 if circuit is trap
        Returns:
            qotp_gate (list): list of all 1-qubit gate and QOTP
            postp (list): string used in classical post-processing
    """

    # Size of un-encoded circuit
    n_qb = len(gate)
    depth = len(gate[0])

    # Increase the lenght of cz_gate for further use in while loop
    cz_gate = np.vstack([cz_gate, [0, 0, 0]])

    # TARGET encoding
    if b_num == 0:
        # Define a new set of 1-qubit gate initialized to identity
        qotp_gate = [['iden' for j in range(3*depth)] for i in range(n_qb)]

        # Place 1-qubit gate from the target
        for i in range(n_qb):
            for j in range(depth):
                qotp_gate[i][3*j+1] = gate[i][j]

        # QOTP before first band of gate (Pauli-Z)
        for i in range(n_qb):
            if random.randint(0, 1) == 1:
                qotp_gate[i][0] = 'z'

        # QOTP in middle circuit
        qotp = ['iden', 'x', 'z', 'y']
        for j in range(depth-1):
            alpha = [random.randint(0, 3) for i in range(n_qb)]
            for i in range(n_qb):
                # This undoes QOTP for qubits not connected by cZ
                qotp_gate[i][3*j+2] = qotp[alpha[i]]
                qotp_gate[i][3*j+3] = qotp[alpha[i]]

            # This undoes QOTP for qubits connected by a cZ
            # It erases rows in matrix cz_gate
            if len(cz_gate) != 1:
                while cz_gate[0][0] == j:
                    if alpha[cz_gate[0][1]] == 1 or alpha[cz_gate[0][1]] == 3:
                        qotp_gate[cz_gate[0][2]][3*j+3] =\
                            qotp[(alpha[cz_gate[0][2]]+2) % 4]
                    if alpha[cz_gate[0][2]] == 1 or alpha[cz_gate[0][2]] == 3:
                        qotp_gate[cz_gate[0][1]][3*j+3] =\
                            qotp[(alpha[cz_gate[0][1]] + 2) % 4]
                    cz_gate = np.delete(cz_gate, (0), axis=0)

        # QOTP before the measurements
        # postp is used in the post-processing
        alpha = [random.randint(0, 3) for i in range(n_qb)]
        postp = [(alpha[i]) % 2 for i in range(n_qb)]
        for i in range(n_qb):
            qotp_gate[i][3*depth-1] = qotp[alpha[i]]

    # TRAP encoding
    else:
        n_qb = len(gate)
        depth = int(len(gate[0])/3)

        # Define a new set of 1-qubit gate initialized to identity
        qotp_gate = [['iden' for j in range(4*depth+4)] for i in range(n_qb)]

        # Place 1-qubit gate from the target
        for j in range(depth+1):
            for i in range(n_qb):
                qotp_gate[i][4*j+1] = gate[i][2*j+1]
                qotp_gate[i][4*j+2] = gate[i][2*j+2]

        # QOTP before first band of gate (Pauli-Z)
        for i in range(n_qb):
            if random.randint(0, 1) == 1:
                qotp_gate[i][0] = 'z'

        # QOTP
        qotp = ['iden', 'x', 'z', 'y']
        for j in range(depth):
            alpha = [random.randint(0, 3) for i in range(n_qb)]
            for i in range(n_qb):
                qotp_gate[i][4*j+3] = qotp[alpha[i]]
                qotp_gate[i][4*j+4] = qotp[alpha[i]]

            # This undoes QOTP for qubits not connected by cZ.
            # It follows the instructions given by matrix cz_gate, erasing
            # lines in cz_gate when band is completed
            if len(cz_gate) != 1:
                while cz_gate[0][0] == j:
                    if alpha[cz_gate[0][1]] == 1 or alpha[cz_gate[0][1]] == 3:
                        qotp_gate[cz_gate[0][2]][4*j+4] =\
                            qotp[(alpha[cz_gate[0][2]] + 2) % 4]
                    if alpha[cz_gate[0][2]] == 1 or alpha[cz_gate[0][2]] == 3:
                        qotp_gate[cz_gate[0][1]][4*j+4] =\
                            qotp[(alpha[cz_gate[0][1]] + 2) % 4]
                    cz_gate = np.delete(cz_gate, (0), axis=0)

        # QOTP before the measurements
        # postp is used in the post-processing
        alpha = [random.randint(0, 3) for i in range(n_qb)]
        postp = [(alpha[i]) % 2 for i in range(n_qb)]
        for i in range(n_qb):
            qotp_gate[i][4*depth+3] = qotp[alpha[i]]

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
            gate_trap (list): list of all 1-qubit gate in trap circuit
    """
    # Define a new set of 1-qubit gate initialized to identity
    gate_trap = [['iden' for j in range(2*m_bands)] for i in range(n_qb)]

    # Place gate in the trap circ
    for j in range(m_bands-1):
        for i in range(n_qb):
            if random.randint(0, 1) == 0:
                gate_trap[i][2*j+1] = 'h'
                gate_trap[i][2*j+2] = 'h'
            else:
                gate_trap[i][2*j+1] = 's'
                gate_trap[i][2*j+2] = 'sdg'

    # This fixes the gate for qubits connected by cZ gate
    for i_cz, _ in enumerate(cz_gate):
        if random.randint(0, 1) == 0:
            gate_trap[cz_gate[i_cz][1]][2*(cz_gate[i_cz][0])+1] = 'h'
            gate_trap[cz_gate[i_cz][1]][2*(cz_gate[i_cz][0])+2] = 'h'
            gate_trap[cz_gate[i_cz][2]][2*(cz_gate[i_cz][0])+1] = 's'
            gate_trap[cz_gate[i_cz][2]][2*(cz_gate[i_cz][0])+2] = 'sdg'
        else:
            gate_trap[cz_gate[i_cz][1]][2*(cz_gate[i_cz][0])+1] = 's'
            gate_trap[cz_gate[i_cz][1]][2*(cz_gate[i_cz][0])+2] = 'sdg'
            gate_trap[cz_gate[i_cz][2]][2*(cz_gate[i_cz][0])+1] = 'h'
            gate_trap[cz_gate[i_cz][2]][2*(cz_gate[i_cz][0])+2] = 'h'

    # Place random H gate at beginning and end of circuit
    if random.randint(0, 1) == 1:
        for i in range(n_qb):
            gate_trap[i][0] = 'h'
            gate_trap[i][2*m_bands-1] = 'h'

    gate_trap = np.hstack([[['iden'] for i in range(n_qb)], gate_trap])

    return gate_trap


def simulate_circ(gate, postp, b_num, cz_gate, simul):
    # the bit b_num is 0 if circuit is target, 1 if it is trap
    """
    Simulation of quantum circuit on backend
        Args:
            gate (list): list of all 1-qubit gate in circuit
            postp (list): binary vector used to postprocess outputs
            b_num (int): equals 0 if circuit is target, 1 if circuit is trap
            cz_gate (list): list of all cz gate in circuit
            simul: backend

        Returns:
            if b_num=0, returns output (list): outputs of target circuit
            if b_num=1, returns flag (int): 0 if trap circuit gives correct
            output, 1 otherwise
    """
    n_qb = len(gate)
    depth = len(gate[0])

    # Create a Quantum Register with n_qb qubits.
    q_reg = QuantumRegister(n_qb, 'q')
    # Create a Classical Register with n_qb bits.
    c_reg = ClassicalRegister(n_qb, 's')
    # Create a Quantum Circuit acting on the q register
    circ = QuantumCircuit(q_reg, c_reg)

    # Increase the size of conn for further use in while loop
    cz_gate = np.vstack([cz_gate, [0, 0, 0]])

    band = 0

    if b_num == 0:
        # Build target circuit
        for j in range(depth):
            for i in range(n_qb):
                getattr(circ, gate[i][j])(q_reg[i])

            # This places the cZ gate in the circuit.
            # It erases rows of conn
            if (j+1) % 3 == 0 and j != depth-1:
                circ.barrier()
                if len(cz_gate) != 1:
                    while cz_gate[0][0] == band:
                        circ.cz(q_reg[int(cz_gate[0][1])],
                                q_reg[int(cz_gate[0][2])])
                        cz_gate = np.delete(cz_gate, (0), axis=0)
                circ.barrier()
                band = band+1

        # Map the quantum measurement to the classical bits
        circ.measure(q_reg, c_reg)

        # Execute the circuit on the simul
        job = execute(circ, simul, shots=1, memory=True)
        output = job.result().get_memory()  # returns a list of str
        output = str_to_num(output[0])  # turns string into number

        # Classical postprocessing
        for i in range(n_qb):
            output[i] = (output[i] + postp[i]) % 2

        return output

    # Build trap circuit
    for j in range(depth):
        for i in range(n_qb):
            getattr(circ, gate[i][j])(q_reg[i])
        # This places the cZ gate looking at conn
        if j % 4 == 3 and j != depth-1:
            circ.barrier()
            if len(cz_gate) != 1:
                while cz_gate[0][0] == band:
                    circ.cz(q_reg[int(cz_gate[0][1])],
                            q_reg[int(cz_gate[0][2])])
                    cz_gate = np.delete(cz_gate, (0), axis=0)
            circ.barrier()
            band = band+1

    # Map the quantum measurement to the classical bits
    circ.measure(q_reg, c_reg)

    # Execute the circuit on the simul
    job = execute(circ, simul, shots=1, memory=True)
    output = job.result().get_memory()  # returns a list of str
    output = str_to_num(output[0])  # turns string into number

    # Classical postprocessing
    for i in range(n_qb):
        output[i] = (output[i] + postp[i]) % 2

    # Check if trap returns correct output
    if output != [0] * n_qb:
        return 1

    return 0


def accreditation(gate_target, cz_gate, v_trap, d_run, theta, g_num, simul):
    """
    Accreditation protocol
        Args:
            gate_target (list): list of all 1-qubit gate in target circuit
            cz_gate (list): list of all cz gate in target circuit
            v_trap (int): number of trap circuits
            d_run (int): number of run
            theta (float): number between 0 and 1 to calculate confidence
            g_num (float): number between 0 and 1 containing error rates
                           in 1-qubit gate
            simul: backend
        Returns:
            bound (float): upper-bound on variation distance
            confidence (float): confidence in the bound

            It also prints all accepted outputs in a txt file

    input 1-q. gates in the target circuit as [band 1 gate, band 2 gate,..],
    e.g.:
    gate_target = [ ['h', 's', 'h'],
                   ['h', 'tdg', 'h'],
                   ['h', 'sdg', 'h'],
                   ['h', 't', 'h'],
                   ['h', 'z', 'h'],
                   ]

    input cZ gates in target circuit as [band, qubit 1, qubit 2], e.g.:
    cz_gate = [[0, 0, 1],
                [0, 2, 3],
                [0, 4, 5],
                [1, 1, 2],
                [1, 3, 4],
                ]

    """

    # check if connections are OK
    if check_cz(cz_gate) == 1:
        import sys
        sys.exit(0)

    # Accepted outputs are stored in the following string
    acc_outputs = [0] * len(gate_target)
    n_acc = 0

    # PROTOCOL IMPLEMENTATION
    for _ in range(d_run):
        # position of the target
        v_zero = random.randint(0, v_trap)
        for k in range(v_trap+1):
            if k == v_zero:
                gate_target_qotp, postp = routine_one(gate_target, cz_gate, 0)
                output_target = simulate_circ(gate_target_qotp, postp, 0,
                                              cz_gate, simul)
            else:
                gate_trap = routine_two(len(gate_target), len(gate_target[1]),
                                        cz_gate)
                gate_trap_qotp, postp = routine_one(gate_trap, cz_gate, 1)
                flag = simulate_circ(gate_trap_qotp, postp, 1, cz_gate,
                                     simul)
                if flag == 1:
                    k = v_trap+1

        if flag == 0:
            acc_outputs = np.vstack((acc_outputs, output_target))
            n_acc = n_acc + 1

    acc_outputs = np.delete(acc_outputs, (0), axis=0)

    if n_acc == 0:
        print('\nThe protocol has never accepted.')
        bound = 1
        confidence = 1
        return bound, confidence
    if n_acc/d_run <= theta:
        print('\nn_acc/d_run is smaller than theta.')
        bound = 1
        confidence = 1
        return bound, confidence
    myfile = 'accredit_' + str(v_trap) + 'trap-' + str(d_run) + 'run.txt'
    file = open(myfile, 'w+')
    file.write(str(acc_outputs))

    print('\nVariation Distance upper-bounded by: '
          + str(round(g_num*1.7/(v_trap+1)/(n_acc/d_run-theta)+1-g_num, 4))
          + ', confidence: '
          + str(round(1-2*np.exp(-2*theta*d_run*d_run), 4)))
    file.close()
    bound = g_num*1.7/(v_trap+1)/(n_acc/d_run-theta)+1-g_num
    confidence = 1-2*np.exp(-2*theta*d_run*d_run)
    return bound, confidence
