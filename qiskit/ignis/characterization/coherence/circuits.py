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


def t1_circuits(num_of_gates, gate_time, qubits):
    """
    Generates circuit for T1 measurement.
    Each circuit consists of an X gate, followed by a sequence of identity
    gates.

    Args:
       num_of_gates (list of integers): the number of identity gates in each
                                        circuit. Must be in an increasing
                                        order.
       gate_time (float): time in micro-seconds of running a single gate.
       qubits (list of integers): indices of the qubits whose T1 are
       to be measured.

    Returns:
       A list of QuantumCircuit
       xdata: a list of delay times in seconds
    """

    xdata = gate_time * num_of_gates

    qr = qiskit.QuantumRegister(max(qubits)+1)
    cr = qiskit.ClassicalRegister(len(qubits))

    circuits = []

    for circ_index, circ_length in enumerate(num_of_gates):
        circ = qiskit.QuantumCircuit(qr, cr)
        circ.name = 'circuit_' + str(circ_index)
        for _, qubit in enumerate(qubits):
            circ.x(qr[qubit])
            circ = pad_id_gates(circ, qr, qubit, circ_length)
        circ.barrier(qr)
        for qind, qubit in enumerate(qubits):
            circ.measure(qr[qubit], cr[qind])
        circuits.append(circ)

    return circuits, xdata


def t2star_circuits(num_of_gates, gate_time, qubits, nosc=0):
    """
    Generates circuit for T2* measurement.
    Each circuit consists of a Hadamard gate, followed by a sequence of
    identity gates, a phase gate (with a linear phase), and an additional
    Hadamard gate.

    Args:
       num_of_gates (list of integers): the number of identity gates in each
                                        circuit. Must be in an increasing
                                        order.
       gate_time (float): time in micro-seconds of running a single gate.
       qubits (list of integers): indices of the qubits
       whose T2* are to be measured.
       nosc: number of oscillations to induce using the phase gate
    Returns:
       A list of QuantumCircuit
       xdata: a list of delay times in seconds
       osc_freq: the induced oscillation frequency
    """

    xdata = gate_time * num_of_gates

    qr = qiskit.QuantumRegister(max(qubits)+1)
    cr = qiskit.ClassicalRegister(len(qubits))

    osc_freq = nosc/xdata[-1]

    circuits = []

    for circ_index, circ_length in enumerate(num_of_gates):
        circ = qiskit.QuantumCircuit(qr, cr)
        circ.name = 'circuit_' + str(circ_index)
        for qind, qubit in enumerate(qubits):
            circ.h(qr[qubit])
            circ = pad_id_gates(circ, qr, qubit, circ_length)
            circ.u1(2*np.pi*osc_freq*xdata[circ_index], qr[qubit])
            circ.h(qr[qubit])
        circ.barrier(qr)
        for qind, qubit in enumerate(qubits):
            circ.measure(qr[qubit], cr[qind])
        circuits.append(circ)

    return circuits, xdata, osc_freq


def t2_circuits(num_of_gates, gate_time, qubits, n_echos=1, phase_alt_echo=False):
    """
    Generates circuit for T2 (echo) measurement, by a CPMG sequence.
    Each circuit consists of:
    - A Y90 gate.
    - n_echos repetitions of:
                  a sequence of identity gates followed by a Y gate,
                  and optionally (if phase_alt_echo is True):
                  another sequence of identity gates followed by an X gate.
    - One more sequence of identity gates.

    Args:
       num_of_gates (list of integers):
          Each element of the list corresponds to a circuit.
          num_of_gates[i] is the number of identity gates in each
          sequence of identity gates in circuit no. i.
          Must be in an increasing order.
       gate_time (float): time in micro-seconds of running a single gate.
       qubits (list of integers): indices of the qubits whose
       T2 are to be measured.
       n_echos (integer): number of repetitions. Must be an odd number.
       phase_alt_echo (bool): if True then include X gates.
    Returns:
       A list of QuantumCircuit
       xdata: the delay times
    """

    if n_echos % 2 == 0:
        raise ValueError('Number of echos must be odd')

    xdata = gate_time * num_of_gates * (1 + n_echos * (1 + int(phase_alt_echo)))

    qr = qiskit.QuantumRegister(max(qubits)+1)
    cr = qiskit.ClassicalRegister(len(qubits))

    circuits = []

    for circ_index, circ_length in enumerate(num_of_gates):
        circ = qiskit.QuantumCircuit(qr, cr)
        circ.name = 'circuit_' + str(circ_index)
        for qind, qubit in enumerate(qubits):
            circ.u2(0.0, 0.0, qr[qubit])  # Y90
            for _ in range(n_echos):  # repeat
                circ = pad_id_gates(circ, qr, qubit, circ_length)  # ids
                circ.y(qr[qubit])  # Y
                if phase_alt_echo:  # optionally
                    circ = pad_id_gates(circ, qr, qubit, circ_length)  # ids
                    circ.x(qr[qubit])  # X
            circ = pad_id_gates(circ, qr, qubit, circ_length)  # ids
            circ.u2(0.0, 0.0, qr[qubit])  # Y90
        circ.barrier(qr)
        for qind, qubit in enumerate(qubits):
            circ.measure(qr[qubit], cr[qind])  # measure
        circuits.append(circ)

    return circuits, xdata
