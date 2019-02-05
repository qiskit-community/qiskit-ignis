# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Circuits generation for T1
"""

import qiskit
from .coherence_utils import pad_id_gates

def t1_generate_circuits(num_of_gates, num_of_qubits, qubit):
    """
    num_of_gates is a vector of integers in an increasing order.
    len(num_of_gates) circuits will be generated.
    In the first circuit the initial X gate will be followed by num_of_gates[0] identity gates.
    In the second circuit it will be followed by num_of_gates[1] identity gates.
    And so on.
    'qubit' is the qubit whose T1 is to be measured.
    """

    qr = qiskit.QuantumRegister(num_of_qubits)
    cr = qiskit.ClassicalRegister(num_of_qubits)

    circuits = []

    for circ_index, circ_length in enumerate(num_of_gates):
        circ = qiskit.QuantumCircuit(qr, cr)
        circ.name = 'circuit_' + str(circ_index)
        circ.x(qr[qubit])
        circ = pad_id_gates(circ, qr, circ_length)
        circ.barrier(qr[qubit])
        circ.measure(qr[qubit], cr[qubit])
        circuits.append(circ)

    return circuits
