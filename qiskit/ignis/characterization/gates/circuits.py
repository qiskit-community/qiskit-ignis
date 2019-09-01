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
Circuit generation for measuring gate errors
"""

import numpy as np
import qiskit


def ampcal_1Q_circuits(max_reps, qubits):
    """
    Generates circuit for measuring the amplitude error of
    the single qubit gates

    The U2 gate is repeatedly applied (in groups of 2)
    and we look at the population of the
    qubit in the xy axis (amplitude erorr amplification sequence)

    Y90-(Y90-Y90)^n

    Args:
       max_reps: the maximum number of repetitions. Circuits will increment
       by 1 rep up to max_rep
       qubits (list of integers): indices of the qubits to perform the
       calibration on
    Returns:
       A list of QuantumCircuit
       xdata: a list of gate repetitions (number of u2 gates)
    """

    xdata = np.arange(max_reps)*2

    qr = qiskit.QuantumRegister(max(qubits)+1)
    cr = qiskit.ClassicalRegister(len(qubits))
    circuits = []

    for circ_index, circ_length in enumerate(xdata):
        circ = qiskit.QuantumCircuit(qr, cr)
        circ.name = 'ampcal1Qcircuit_' + str(circ_index) + '_0'
        for qind, qubit in enumerate(qubits):
            circ.u2(0.0, 0.0, qr[qubit])
            for _ in range(circ_length):
                circ.barrier(qr[qubit])
                circ.u2(0.0, 0.0, qr[qubit])

        for qind, qubit in enumerate(qubits):
            circ.measure(qr[qubit], cr[qind])
        circuits.append(circ)

    return circuits, xdata


def anglecal_1Q_circuits(max_reps, qubits, angleerr=0.0):
    """
    Generates circuit for measuring the angle error of
    the single qubit gate

    Y90-(X90-X90-Y90-Y90)^n - X90

    Args:
       max_reps: the maximum number of repetitions. Circuits will increment
       by 1 rep up to max_rep
       qubits (list of integers): indices of the qubits to perform the
       calibration on
       angleerr: put in an artificial angle error (for testing)
    Returns:
       A list of QuantumCircuit
       xdata: a list of gate repetitions
    """

    xdata = np.arange(max_reps)*2

    qr = qiskit.QuantumRegister(max(qubits)+1)
    cr = qiskit.ClassicalRegister(len(qubits))
    circuits = []

    for circ_index, circ_length in enumerate(np.arange(max_reps)):
        circ = qiskit.QuantumCircuit(qr, cr)
        circ.name = 'anglecal1Qcircuit_' + str(circ_index) + '_0'
        for qind, qubit in enumerate(qubits):
            circ.u2(0.0, 0.0, qr[qubit])  # Y90p
            for _ in range(circ_length):
                if angleerr != 0:
                    circ.u1(-2*angleerr, qr[qubit])
                for _ in range(2):
                    circ.barrier(qr[qubit])
                    circ.u2(-np.pi/2, np.pi/2, qr[qubit])  # Xp
                if angleerr != 0:
                    circ.u1(2*angleerr, qr[qubit])
                for _ in range(2):
                    circ.barrier(qr[qubit])
                    circ.u2(0.0, 0.0, qr[qubit])  # Yp

            if angleerr != 0:
                circ.u1(-angleerr, qr[qubit])
            circ.u2(-np.pi/2, np.pi/2, qr[qubit])  # X90p
        for qind, qubit in enumerate(qubits):
            circ.measure(qr[qubit], cr[qind])
        circuits.append(circ)

    return circuits, xdata


def ampcal_cx_circuits(max_reps, qubits, control_qubits):
    """
    Generates circuit for measuring the amplitude error of
    the cx gate

    The cx gate is repeatedly applied
    and we look at the population of the target
    qubit in the xy axis (amplitude erorr amplification sequence)

    X(control)-X90(target)-(CX)^n

    Note: the circuit may not behave as intended if the
    target-control pairs are not in the coupling map

    Args:
       max_reps: the maximum number of repetitions. Circuits will increment
       by 1 rep up to max_rep
       qubits (list of integers): indices of the target qubits
       to perform the calibration on
       contorl_qubits (list of integers): indices of the control qubits
       to perform the calibration on
    Returns:
       A list of QuantumCircuit
       xdata: a list of gate repetitions
    """
    xdata = np.arange(max_reps)

    qr = qiskit.QuantumRegister(max([max(qubits), max(control_qubits)])+1)
    cr = qiskit.ClassicalRegister(len(qubits))

    circuits = []

    for circ_index, circ_length in enumerate(xdata):
        circ = qiskit.QuantumCircuit(qr, cr)
        circ.name = 'ampcalcxcircuit_' + str(circ_index) + '_0'
        for qind, qubit in enumerate(qubits):
            circ.x(qr[control_qubits[qind]])
            circ.u2(-np.pi/2, np.pi/2, qr[qubit])  # X90p
            for _ in range(circ_length):
                circ.barrier([qr[control_qubits[qind]], qr[qubit]])
                circ.cx(qr[control_qubits[qind]], qr[qubit])

        for qind, qubit in enumerate(qubits):
            circ.measure(qr[qubit], cr[qind])
        circuits.append(circ)

    return circuits, xdata


def anglecal_cx_circuits(max_reps, qubits, control_qubits, angleerr=0.0):
    """
    Generates circuit for measuring the angle error of
    the cx gate

    The cx gate is repeatedly applied
    and we look at the population of the target
    qubit in the xy axis (amplitude erorr amplification sequence)

    X(control)-Y90(target)-(CX - Yp(target))^n - X90(target)

    Note: the circuit may not behave as intended if the
    target-control pairs are not in the coupling map

    Args:
       max_reps: the maximum number of repetitions. Circuits will increment
       by 1 rep up to max_rep
       qubits (list of integers): indices of the target qubits
       to perform the calibration on
       to perform the calibration on
       angleerr: Injected angle error for testing
    Returns:
       A list of QuantumCircuit
       xdata: a list of gate repetitions
    """

    xdata = np.arange(max_reps)

    qr = qiskit.QuantumRegister(max([max(qubits), max(control_qubits)])+1)
    cr = qiskit.ClassicalRegister(len(qubits))

    circuits = []

    for circ_index, circ_length in enumerate(xdata):
        circ = qiskit.QuantumCircuit(qr, cr)
        circ.name = 'anglecalcxcircuit_' + str(circ_index) + '_0'
        for qind, qubit in enumerate(qubits):
            circ.x(qr[control_qubits[qind]])
            circ.u2(0.0, 0.0, qr[qubit])  # Y90p (target)
            for _ in range(circ_length):
                if angleerr != 0:
                    circ.u1(-angleerr, qr[qubit])
                circ.barrier([qr[control_qubits[qind]], qr[qubit]])
                circ.cx(qr[control_qubits[qind]], qr[qubit])
                if angleerr != 0:
                    circ.u1(angleerr, qr[qubit])
                circ.y(qr[qubit])  # Yp (target)

            circ.u2(-np.pi/2., np.pi/2., qr[qubit])  # X90p (target)
        for qind, qubit in enumerate(qubits):
            circ.measure(qr[qubit], cr[qind])
        circuits.append(circ)

    return circuits, xdata
