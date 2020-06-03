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
Circuit generation for measuring hamiltonian parametes
"""

import numpy as np
import qiskit
from ..characterization_utils import pad_id_gates


def zz_circuits(num_of_gates, gate_time, qubits, spectators, nosc=5):
    """
    Generates circuit for measuring ZZ.
    Two Ramsey experiments, the first with the spectator in the
    ground state, the second with the spectator in the excited state

    Args:
        num_of_gates (list of integers): the number of identity gates in each
            circuit. Must be in an increasing order.
        gate_time (float): time of running a single gate.
        qubits (list of integers): indices of the qubits to measure and
            perform the Ramsey
        spectators (list of integers): indices of the qubits to flip the
            state (ie measure the ZZ between qubits and spectators). Must
            be the same length as qubits
        nosc (int): number of oscillations to induce using the phase gate

    Returns:
        A list of QuantumCircuit
        xdata: a list of delay times
        osc_freq: the induced oscillation frequency

    Raises:
        ValueError: If the number of qubits differs from the number of
            spectators
    """

    if len(qubits) != len(spectators):
        raise ValueError("Number of qubits and spectators must be the same")

    for _, qubit in enumerate(qubits):
        if qubit in spectators:
            raise ValueError("Qubits and spectators must be different")

    xdata = gate_time * num_of_gates

    qr = qiskit.QuantumRegister(max([max(qubits), max(spectators)])+1)
    cr = qiskit.ClassicalRegister(len(qubits))

    osc_freq = nosc/xdata[-1]

    circuits = []

    for qflip in [False, True]:
        for circ_index, circ_length in enumerate(num_of_gates):
            circ = qiskit.QuantumCircuit(qr, cr)
            circ.name = 'zzcircuit_' + str(circ_index) + '_%d' % qflip
            for _, qspec in enumerate(spectators):
                if qflip:
                    circ.x(qr[qspec])
            circ.barrier(qr)
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
