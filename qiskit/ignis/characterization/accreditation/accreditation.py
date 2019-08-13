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
    BasicAer,
    IBMQ,
)

def str_to_num(string):
    """
        Args:
            string (): string of characters, each character being 0 or 1
            
        Returns:
            A list of integers, each one being 0 or 1
    """
    return [1 if s == "1" else 0 for s in string]

def check_cz(cz_gates):
    error = 0
    for i_cz, _ in enumerate(cz_gates):
        for j_cz in range(i_cz+1, len(cz_gates)):
            if cz_gates[i_cz][0] == cz_gates[j_cz][0]:
                if cz_gates[i_cz][1] == cz_gates[j_cz][1] or cz_gates[i_cz][2] == cz_gates[j_cz][2]:
                    print("ERROR: Check matrix ``cz_gates''.")
                    print("Remember that in every band cZ must be executed on different qubits!")
                    error = 1
    return error

def routine_one(gates, cz_gates, b_num):
    """
    Routine 1 of accreditation protocol

    Args:
        gates: matrix with all 1-qubit gates in circuit
        cz_gates: matrix with all cz gates in circuit
        b_num (integer): 0 if circuit is target, 1 if circuit is trap

    Returns:
        padded_gates: matrix with all 1-qubit gates and QOTP
        postp: string used in classical post-processing
    """

    # Size of un-encoded circuit
    n_qubits = len(gates)
    depth = len(gates[0])

    # Increase the lenght of cz_gates for further use in while loop
    cz_gates = np.vstack([cz_gates, [0, 0, 0]])

    #----------------
    # TARGET encoding
    #----------------
    if b_num == 0:
        # Define a new set of 1-qubit gates initialized to identity
        padded_gates = [['iden' for j in range(3*depth)] for i in range(n_qubits)]

        # Place 1-qubit gates from the target
        for i in range(n_qubits):
            for j in range(depth):
                padded_gates[i][3*j+1] = gates[i][j]

        # QOTP before first band of gates (Pauli-Z)
        for i in range(n_qubits):
            if random.randint(0, 1) == 1:
                padded_gates[i][0] = 'z'

        # QOTP in middle circuit
        qotp = ['iden', 'x', 'z', 'y']
        for j in range(depth-1):
            alpha = [random.randint(0, 3) for i in range(n_qubits)]
            for i in range(n_qubits):
                padded_gates[i][3*j+2] = qotp[alpha[i]]
                padded_gates[i][3*j+3] = qotp[alpha[i]]
                #This undoes QOTP for qubits not connected by cZ

            # This undoes QOTP for qubits connected by a cZ
            # It erases rows in matrix cz_gates
            if len(cz_gates) != 1:
                while cz_gates[0][0] == j:
                    if alpha[cz_gates[0][1]] == 1 or alpha[cz_gates[0][1]] == 3:
                        padded_gates[cz_gates[0][2]][3*j+3] = qotp[(alpha[cz_gates[0][2]] + 2) % 4]
                    if alpha[cz_gates[0][2]] == 1 or alpha[cz_gates[0][2]] == 3:
                        padded_gates[cz_gates[0][1]][3*j+3] = qotp[(alpha[cz_gates[0][1]] + 2) % 4]
                    cz_gates = np.delete(cz_gates, (0), axis=0)

        # QOTP before the measurements
        # postp is used in the post-processing
        alpha = [random.randint(0, 3) for i in range(n_qubits)]
        postp = [(alpha[i]) % 2 for i in range(n_qubits)]
        for i in range(n_qubits):
            padded_gates[i][3*depth-1] = qotp[alpha[i]]

    #--------------
    # TRAP encoding
    #--------------
    else:
        n_qubits = len(gates)
        depth = int(len(gates[0])/3)

        # Define a new set of 1-qubit gates initialized to identity
        padded_gates = [['iden' for j in range(4*depth+4)] for i in range(n_qubits)]

        # Place 1-qubit gates from the target
        for j in range(depth+1):
            for i in range(n_qubits):
                padded_gates[i][4*j+1] = gates[i][2*j+1]
                padded_gates[i][4*j+2] = gates[i][2*j+2]

        # QOTP before first band of gates (Pauli-Z)
        for i in range(n_qubits):
            if random.randint(0, 1) == 1:
                padded_gates[i][0] = 'z'

        # QOTP
        qotp = ['iden', 'x', 'z', 'y']
        for j in range(depth):
            alpha = [random.randint(0, 3) for i in range(n_qubits)]
            for i in range(n_qubits):
                padded_gates[i][4*j+3] = qotp[alpha[i]]
                padded_gates[i][4*j+4] = qotp[alpha[i]]

            # This undoes QOTP for qubits not connected by cZ.
            # It follows the instructions given by matrix cz_gates, erasing lines in cz_gates
            # when band is compleated
            if len(cz_gates) != 1:
                while cz_gates[0][0] == j:
                    if alpha[cz_gates[0][1]] == 1 or alpha[cz_gates[0][1]] == 3:
                        padded_gates[cz_gates[0][2]][4*j+4] = qotp[(alpha[cz_gates[0][2]] + 2) % 4]
                    if alpha[cz_gates[0][2]] == 1 or alpha[cz_gates[0][2]] == 3:
                        padded_gates[cz_gates[0][1]][4*j+4] = qotp[(alpha[cz_gates[0][1]] + 2) % 4]
                    cz_gates = np.delete(cz_gates, (0), axis=0)

        # QOTP before the measurements
        # postp is used in the post-processing
        alpha = [random.randint(0, 3) for i in range(n_qubits)]
        postp = [(alpha[i]) % 2 for i in range(n_qubits)]
        for i in range(n_qubits):
            padded_gates[i][4*depth+3] = qotp[alpha[i]]

    # Flip
    post1 = [0 for i in range(n_qubits)]
    for i in range(n_qubits):
        post1[i] = postp[n_qubits-1-i]
    postp = post1

    return padded_gates, postp

# Routine 2 of accreditation protocol
#------------------------------------
# Takes as input the size of the target circuit and a matrix cz_gates descibing the position of
# cZ gates, outputs a set of gates for trap circuit

def routine_two(n_qubits, m_bands, cz_gates):

    # Define a new set of 1-qubit gates initialized to identity
    gates = [['iden' for j in range(2*m_bands)] for i in range(n_qubits)]

    # Place gates in the trap circ
    for j in range(m_bands-1):
        for i in range(n_qubits):
            if random.randint(0, 1) == 0:
                gates[i][2*j+1] = 'h'
                gates[i][2*j+2] = 'h'
            else:
                gates[i][2*j+1] = 's'
                gates[i][2*j+2] = 'sdg'

    # This fixes the gates for qubits connected by cZ gate
    for i_cz, _ in enumerate(cz_gates):
        if random.randint(0, 1) == 0:
            gates[cz_gates[i_cz][1]][2*(cz_gates[i_cz][0])+1] = 'h'
            gates[cz_gates[i_cz][1]][2*(cz_gates[i_cz][0])+2] = 'h'
            gates[cz_gates[i_cz][2]][2*(cz_gates[i_cz][0])+1] = 's'
            gates[cz_gates[i_cz][2]][2*(cz_gates[i_cz][0])+2] = 'sdg'
        else:
            gates[cz_gates[i_cz][1]][2*(cz_gates[i_cz][0])+1] = 's'
            gates[cz_gates[i_cz][1]][2*(cz_gates[i_cz][0])+2] = 'sdg'
            gates[cz_gates[i_cz][2]][2*(cz_gates[i_cz][0])+1] = 'h'
            gates[cz_gates[i_cz][2]][2*(cz_gates[i_cz][0])+2] = 'h'

    # Place random H gates at beginning and end of circuit
    if random.randint(0, 1) == 1:
        for i in range(n_qubits):
            gates[i][0] = 'h'
            gates[i][2*m_bands-1] = 'h'

    gates = np.hstack([[['iden'] for i in range(n_qubits)], gates])

    return gates

# Simulation of circuit on backend
#------------------------------------

def simulate_circ(gates, postp, b_num, flag, cz_gates, simulator):
    # the bit b_num is 0 if circuit is target, 1 if it is trap

    n_qubits = len(gates)
    depth = len(gates[0])

    # Create a Quantum Register with n_qubits qubits.
    q_reg = QuantumRegister(n_qubits, 'q')
    # Create a Classical Register with n_qubits bits.
    c_reg = ClassicalRegister(n_qubits, 's')
    # Create a Quantum Circuit acting on the q register
    circ = QuantumCircuit(q_reg, c_reg)

    # Increase the size of conn for further use in while loop
    cz_gates = np.vstack([cz_gates, [0, 0, 0]])

    if b_num == 0:
        band = 0
        # Build target circuit
        for j in range(depth):
            for i in range(n_qubits):
                getattr(circ, gates[i][j])(q_reg[i])

            # This places the cZ gates in the circuit.
            # It erases rows of conn
            if (j+1) % 3 == 0 and j != depth-1:
                circ.barrier()
                if len(cz_gates) != 1:
                    while cz_gates[0][0] == band:
                        circ.cz(q_reg[int(cz_gates[0][1])], q_reg[int(cz_gates[0][2])])
                        cz_gates = np.delete(cz_gates, (0), axis=0)
                circ.barrier()
                band = band+1

        # Map the quantum measurement to the classical bits
        circ.measure(q_reg, c_reg)

        # Execute the circuit on the simulator
        job = execute(circ, simulator, shots=1, memory=True)
        output = job.result().get_memory() # returns a list of str
        output = str_to_num(output[0]) # turns string into number

        # Classical postprocessing
        for i in range(n_qubits):
            output[i] = (output[i] + postp[i]) % 2

        return output

    # Build trap circuit
    band = 0
    for j in range(depth):
        for i in range(n_qubits):
            getattr(circ, gates[i][j])(q_reg[i])
        # This places the cZ gates looking at conn
        if j % 4 == 3 and j != depth-1:
            circ.barrier()
            if len(cz_gates) != 1:
                while cz_gates[0][0] == band:
                    circ.cz(q_reg[int(cz_gates[0][1])], q_reg[int(cz_gates[0][2])])
                    cz_gates = np.delete(cz_gates, (0), axis=0)
            circ.barrier()
            band = band+1

    # Map the quantum measurement to the classical bits
    circ.measure(q_reg, c_reg)

    # Execute the circuit on the simulator
    job = execute(circ, simulator, shots=1, memory=True)
    output = job.result().get_memory() # returns a list of str
    output = str_to_num(output[0]) # turns string into number

    # Classical postprocessing
    for i in range(n_qubits):
        output[i] = (output[i] + postp[i]) % 2

    # Check if trap returns correct output
    if output != [0] * n_qubits:
        flag = 1

    return flag

#-----------------------
# Accreditation Protocol
#-----------------------

#-----------------------------------------------------------
# INPUT:
# A description of the target circuit, i.e. matrices "gates_target" and "cz_gates"
# The number v of traps
# The number d of runs
# A number theta \in (0, 1)
# The number g (cfr. Theorem 2)
# Simulator

# input 1-q. gates in the target circuit as [band 1 gates, band 2 gates, ..], e.g.:
# gates_target = [ ['h', 's', 'h'],
#                ['h', 'tdg', 'h'],
#                ['h', 'sdg', 'h'],
#                ['h', 't', 'h'],
#                ['h', 'z', 'h'],
#                ['h', 't', 'h'],
#              ]

# cZ gate between qubits in circuit as [band, qubit 1, qubit 2], e.g.:
# cz_gates = [[0, 0, 1],
#           [0, 2, 3],
#           [0, 4, 5],
#           [1, 1, 2],
#           [1, 3, 4],
#          ]
#-----------------------------------------------------------

def accreditation(gates_target, cz_gates, v, d, theta, g, simulator):

    # check if connections are OK
    error = check_cz(cz_gates)
    if error == 1:
        import sys
        sys.exit(0)

    # n_qubits = number of qubits, m_bands = number of bands
    n_qubits = len(gates_target)
    m_bands = len(gates_target[1])

    # Accepted outputs are stored in the following string
    acc_outputs = [0] * n_qubits
    n_acc = 0

    # PROTOCOL IMPLEMENTATION
    for _ in range(d):
        # position of the target
        v0 = random.randint(0, v)
        flag = 0
        for k in range(v+1):
            if k == v0:
                gates_target_padded, postp = routine_one(gates_target, cz_gates, 0)
                output_target = simulate_circ(gates_target_padded, postp, 0, flag,
                                              cz_gates, simulator)
            else:
                gatesTrap = routine_two(n_qubits, m_bands, cz_gates)
                gatesTrapPadded, postp = routine_one(gatesTrap, cz_gates, 1)
                flag = simulate_circ(gatesTrapPadded, postp, 1, flag, cz_gates,
                                     simulator)
                if flag == 1:
                    k = v+1

        if flag == 0:
            acc_outputs = np.vstack((acc_outputs, output_target))
            n_acc = n_acc + 1
        else:
            True

    acc_outputs = np.delete(acc_outputs, (0), axis=0)

    if len(acc_outputs) == 0:
        print('\nThe protocol has never accepted.')
    elif len(acc_outputs) > 0 and n_acc/d < theta:
        print('\nn_acc/d is smaller than theta.')
    elif len(acc_outputs) > 0 and n_acc/d > theta:
        myfile = 'accredit-' + str(n_qubits) + 'qubits-' + str(v) + 'traps-' + str(d) + 'runs.txt'
        file = open(myfile, 'w+')
        file.write(str(acc_outputs))

        bound = g*1.7/(v+1)/(n_acc/d-theta)+1-g
        confidence = 1-2*np.exp(-2*theta*d*d)
        print('\nVariation Distance upper-bounded by: '
              + str(round(bound, 4))
              + ', confidence: '
              + str(round(confidence, 4)))
        file.close()
