# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Circuits generation for T2*
"""

import numpy as np
import qiskit
from .coherence_utils import pad_id_gates

def t2starexp_generate_circuits_bygates(num_of_gates, gate_time, num_of_qubits, qubit):
    """
    Generates circuit for T2* measurement.
    Each circuit consists of a Hadamard gate, followed by a sequence of identity gates,
    and an additional Hadamard gate.

    Args:
       num_of_gates (list of integers):  the number of identity gates in each circuit.
                                         Must be in an increasing order.
       gate_time (float): time in micro-seconds of running a single gate.
       num_of_qubits (integer): the number of qubits in the circuit.
       qubit (integer): index of the qubit whose T1 is to be measured.
    Returns:
       A list of QuantumCircuit
    """

    xdata = gate_time * num_of_gates

    qr = qiskit.QuantumRegister(num_of_qubits)
    cr = qiskit.ClassicalRegister(num_of_qubits)

    circuits = []

    for circ_index, circ_length in enumerate(num_of_gates):
        circ = qiskit.QuantumCircuit(qr, cr)
        circ.name = 'circuit_' + str(circ_index)
        circ.h(qr[qubit])
        circ = pad_id_gates(circ, qr, circ_length)
        circ.barrier(qr[qubit])
        circ.h(qr[qubit])
        circ.measure(qr[qubit], cr[qubit])
        circuits.append(circ)

    return circuits, xdata


def t2starosc_generate_circuits_bygates(num_of_gates, gate_time, num_of_qubits, qubit):
     """
     Generates circuit for T2* measurement.
     Each circuit consists of a Hadamard gate, followed by a u1 gate,
     a sequence of identity gates, and an additional Hadamard gate.

     Args:
        num_of_gates (list of integers):  the number of identity gates in each circuit.
                                          Must be in an increasing order.
        gate_time (float): time in micro-seconds of running a single gate.
        num_of_qubits (integer): the number of qubits in the circuit.
        qubit (integer): index of the qubit whose T1 is to be measured.
     Returns:
        A list of QuantumCircuit
     """

     xdata = gate_time * num_of_gates

     # TODO: Consider not to hard-code
     omega = 0.003/gate_time

     qr = qiskit.QuantumRegister(num_of_qubits)
     cr = qiskit.ClassicalRegister(num_of_qubits)

     circuits = []

     for circ_index, circ_length in enumerate(num_of_gates):
         circ = qiskit.QuantumCircuit(qr, cr)
         circ.name = 'circuit_' + str(circ_index)
         circ.h(qr[qubit])
         circ.u1(2*np.pi*omega*circ_length, qr[qubit])
         circ = pad_id_gates(circ, qr, circ_length)
         circ.h(qr[qubit])
         circ.barrier(qr[qubit])
         circ.measure(qr[qubit], cr[qubit])
         circuits.append(circ)

     return circuits, xdata, omega
