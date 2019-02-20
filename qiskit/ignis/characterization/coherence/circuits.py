# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Circuit generation for coherence experiments
"""

import numpy as np
import qiskit
from .coherence_utils import pad_id_gates

def t1(num_of_gates, gate_time, num_of_qubits, qubit):
    """
    Generates circuit for T1 measurement.
    Each circuit consists of an X gate, followed by a sequence of identity gates.

    Args:
       num_of_gates (list of integers):  the number of identity gates in each circuit.
                                         Must be in an increasing order.
       gate_time (float): time in micro-seconds of running a single gate.
       num_of_qubits (integer): the number of qubits in the circuit.
       qubit (integer): index of the qubit whose T1 is to be measured.
    Returns:
       A list of QuantumCircuit
       xdata: a list of delay times in seconds
    """

    xdata = gate_time * num_of_gates

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

    return circuits, xdata

def t2star(num_of_gates, gate_time, num_of_qubits, qubit, nosc=0):
    """
    Generates circuit for T2* measurement.
    Each circuit consists of a Hadamard gate, followed by a sequence of identity gates,
    a phase gate (with a linear phase), and an additional Hadamard gate.

    Args:
       num_of_gates (list of integers):  the number of identity gates in each circuit.
                                         Must be in an increasing order.
       gate_time (float): time in micro-seconds of running a single gate.
       num_of_qubits (integer): the number of qubits in the circuit.
       qubit (integer): index of the qubit whose T1 is to be measured.
       nosc: number of oscillations to induce using the phase gate
    Returns:
       A list of QuantumCircuit
       xdata: the delay times
       osc_freq: the induced oscillation frequency
    """

    xdata = gate_time * num_of_gates

    qr = qiskit.QuantumRegister(num_of_qubits)
    cr = qiskit.ClassicalRegister(num_of_qubits)

    osc_freq = nosc/xdata[-1]

    circuits = []

    for circ_index, circ_length in enumerate(num_of_gates):
        circ = qiskit.QuantumCircuit(qr, cr)
        circ.name = 'circuit_' + str(circ_index)
        circ.h(qr[qubit])
        circ = pad_id_gates(circ, qr, circ_length)
        circ.barrier(qr[qubit])
        circ.u1(2*np.pi*osc_freq*xdata[circ_index], qr[qubit])
        circ.h(qr[qubit])
        circ.measure(qr[qubit], cr[qubit])
        circuits.append(circ)

    return circuits, xdata, osc_freq

def t2(num_of_gates, gate_time, num_of_qubits, qubit):
    """
    Generates circuit for T2 (echo) measurement.
    Each circuit consists of a Y90 gate, followed by a sequence of identity gates,
    an Y gate, a sequence of identity gates and
    an additional Y90 gate.

    Args:
       num_of_gates (list of integers):  the number of identity gates in each circuit.
                                         Must be in an increasing order.
                                         This is the number of gates between the H and echo
                                         (i.e. total length is twice)
       gate_time (float): time in micro-seconds of running a single gate.
       num_of_qubits (integer): the number of qubits in the circuit.
       qubit (integer): index of the qubit whose T1 is to be measured.
    Returns:
       A list of QuantumCircuit
       xdata: the delay times (TOTAL delay time)
    """

    xdata = gate_time * num_of_gates * 2.0

    qr = qiskit.QuantumRegister(num_of_qubits)
    cr = qiskit.ClassicalRegister(num_of_qubits)


    circuits = []

    for circ_index, circ_length in enumerate(num_of_gates):
        circ = qiskit.QuantumCircuit(qr, cr)
        circ.name = 'circuit_' + str(circ_index)
        circ.u2(0.0,0.0,qr[qubit])
        circ = pad_id_gates(circ, qr, circ_length)
        circ.barrier(qr[qubit])
        circ.y(qr[qubit])
        circ = pad_id_gates(circ, qr, circ_length)
        circ.barrier(qr[qubit])
        circ.u2(0.0,0.0,qr[qubit])
        circ.measure(qr[qubit], cr[qubit])
        circuits.append(circ)

    return circuits, xdata