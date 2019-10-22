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
Circuit generation for coherence experiments
"""

import numpy as np
import qiskit
from ..characterization_utils import pad_id_gates


def t1_circuits(num_of_gates, gate_time, qubits):
    """
    Generates circuit for T1 measurement.
    Each circuit consists of an X gate, followed by a sequence of identity
    gates.

    Args:
       num_of_gates (list of integers): the number of identity gates in each
                                        circuit. Must be in an increasing
                                        order.
       gate_time (float): time of running a single gate.
       qubits (list of integers): indices of the qubits whose T1 are
       to be measured.

    Returns:
       A list of QuantumCircuit
       xdata: a list of delay times
    """

    xdata = gate_time * num_of_gates

    qr = qiskit.QuantumRegister(max(qubits)+1)
    cr = qiskit.ClassicalRegister(len(qubits))

    circuits = []

    for circ_index, circ_length in enumerate(num_of_gates):
        circ = qiskit.QuantumCircuit(qr, cr)
        circ.name = 't1circuit_' + str(circ_index) + '_0'
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
       gate_time (float): time of running a single gate.
       qubits (list of integers): indices of the qubits
       whose T2* are to be measured.
       nosc: number of oscillations to induce using the phase gate
    Returns:
       A list of QuantumCircuit
       xdata: a list of delay times
       osc_freq: the induced oscillation frequency
    """

    xdata = gate_time * num_of_gates

    qr = qiskit.QuantumRegister(max(qubits)+1)
    cr = qiskit.ClassicalRegister(len(qubits))

    osc_freq = nosc/xdata[-1]

    circuits = []

    for circ_index, circ_length in enumerate(num_of_gates):
        circ = qiskit.QuantumCircuit(qr, cr)
        circ.name = 't2starcircuit_' + str(circ_index) + '_0'
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


def t2_circuits(num_of_gates, gate_time, qubits, n_echos=1,
                phase_alt_echo=False):
    """
    Generates circuit for T2 (echo) measurement, by a CPMG sequence.
    Each circuit consists of:
    - Y90-t-Y-[t-t-X/Y]^m-t-Y90
    - n_echos = n+1
    - if phase_alt_echo the X/Y alternate, if phase_alt_echo=False the
    pulses are always Y

    Standard T2 echo is n_echos=1

    Args:
       num_of_gates (list of integers):
          Each element of the list corresponds to a circuit.
          num_of_gates[i] is the number of identity gates in each section
          "t" of the pulse sequeence in circuit no. i.
          Must be in an increasing order.
       gate_time (float): time of running a single gate.
       qubits (list of integers): indices of the qubits whose
       T2 are to be measured.
       n_echos (integer): number of echo gates (X or Y).
       phase_alt_echo (bool): if True then alternate the echo between
       X and Y.
    Returns:
       A list of QuantumCircuit
       xdata: the delay times
    """

    if n_echos < 1:
        raise ValueError('Must be at least one echo')

    xdata = 2 * gate_time * num_of_gates * n_echos

    qr = qiskit.QuantumRegister(max(qubits)+1)
    cr = qiskit.ClassicalRegister(len(qubits))

    circuits = []

    for circ_index, circ_length in enumerate(num_of_gates):
        circ = qiskit.QuantumCircuit(qr, cr)
        circ.name = 't2circuit_' + str(circ_index) + '_0'
        for qind, qubit in enumerate(qubits):

            # First Y90 and Y echo
            circ.u2(0.0, 0.0, qr[qubit])  # Y90
            circ = pad_id_gates(circ, qr, qubit, circ_length)  # ids
            circ.y(qr[qubit])

            for echoid in range(n_echos-1):  # repeat
                circ = pad_id_gates(circ, qr, qubit, 2*circ_length)  # ids
                if phase_alt_echo and (not echoid % 2):  # optionally
                    circ.x(qr[qubit])  # X
                else:
                    circ.y(qr[qubit])

            circ = pad_id_gates(circ, qr, qubit, circ_length)  # ids
            circ.u2(0.0, 0.0, qr[qubit])  # Y90
        circ.barrier(qr)
        for qind, qubit in enumerate(qubits):
            circ.measure(qr[qubit], cr[qind])  # measure
        circuits.append(circ)

    return circuits, xdata
