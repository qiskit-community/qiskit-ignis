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

from typing import List, Union, Tuple
import numpy as np
import qiskit
from ..characterization_utils import pad_id_gates


def t1_circuits(num_of_gates: Union[List[int], np.array],
                gate_time: float,
                qubits: List[int]) -> Tuple[List[qiskit.QuantumCircuit], np.array]:
    r"""
    Generate circuits for T\ :sub:`1` measurement.

    Each circuit consists of an X gate, followed by a sequence of identity
    gates.

    Args:
        num_of_gates: the number of identity gates in each
            circuit. Must be in an increasing order.
        gate_time: time of running a single identity gate.
        qubits: indices of the qubits whose T\ :sub:`1`\ 's are
            to be measured.

    Returns:
        *   Generated circuits
        *   Delay times, i.e., `gate_time` multiplied by the numbers in `num_of_gates`

    """

    xdata = gate_time * np.array(num_of_gates)

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


def t2star_circuits(num_of_gates: Union[List[int], np.array],
                    gate_time: float,
                    qubits: List[int],
                    nosc: int = 0) -> Tuple[List[qiskit.QuantumCircuit], np.array, float]:
    r"""
    Generate circuits for T\ :sub:`2`:sup:`*` measurement.

    Each circuit consists of a Hadamard gate, followed by a sequence of
    identity gates, a phase gate (with a linear phase), and an additional
    Hadamard gate.

    Args:
        num_of_gates: the number of identity gates in each
            circuit. Must be in an increasing order.
        gate_time: time of running a single identity gate.
        qubits: indices of the qubits
            whose T\ :sub:`2`:sup:`*`\ 's are to be measured.
        nosc: number of oscillations to induce using the phase gate
    Returns:
        *   The generated circuits
        *   Delay times, i.e., `gate_time` multiplied by the numbers in `num_of_gates`
        *   The induced oscillation frequency
    """

    xdata = gate_time * np.array(num_of_gates)

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


def t2_circuits(num_of_gates: Union[List[int], np.array],
                gate_time: float,
                qubits: List[int],
                n_echos: int = 1,
                phase_alt_echo: bool = False) -> Tuple[List[qiskit.QuantumCircuit], np.array]:
    r"""
    Generate circuits for T\ :sub:`2` (echo) measurement, by a CPMG sequence.

    Each circuit consists of:
       *   :math:`Y90-t-Y-[t-t-X/Y]^m-t-Y90`
       *   :math:`n_{echos} = n+1`
       *   if `phase_alt_echo` then the `X/Y` alternate, if `phase_alt_echo=False` \
           tthen he pulses are always `Y`

    Standard T\ :sub:`2`:sup:`*` echo is :math:`n_echos=1`

    Args:
        num_of_gates:
            Each element of the list corresponds to a circuit.
            `num_of_gates[i]` is the number of identity gates in each section
            "t" of the pulse sequence in circuit no. i.
            Must be in an increasing order.
        gate_time: time of running a single identity gate.
        qubits: indices of the qubits whose
            T\ :sub:`2`:sup:`*`\ 's are to be measured.
        n_echos: number of echo gates (`X` or `Y`).
        phase_alt_echo: if True then alternate the echo between
            `X` and `Y`.
    Returns:
        *   Generated circuits
        *   Delay times, i.e., `gate_time` multiplied by the numbers in `num_of_gates`
    Raises:
        ValueError: If n_echos is less than 1
    """

    if n_echos < 1:
        raise ValueError('Must be at least one echo')

    xdata = 2 * gate_time * np.array(num_of_gates) * n_echos

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
