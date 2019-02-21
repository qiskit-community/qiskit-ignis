# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
Measurement correction circuits. To apply the measurement correction
use the fitters.
"""

import qiskit.tools.qcvv.tomography as tomo
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.ignis.tomography.basis.circuits import _format_registers


def measurement_calibration(qubits):

    """
    Return a list of measurement calibration circuits.

    Each circuits creates a basis state

    2 ** n circuits

    Args:
        qubits (QuantumRegister): the qubits to be measured.
            This can also be a list of whole QuantumRegisters or
            individual QuantumRegister qubit tuples.

            The calibration states will be labelled according to this ordering

    Returns:
        A list of QuantumCircuit objects containing the calibration circuits.

        A list of calibration state labels

    Additional Information:
        The returned circuits are named cal_XXX where XXX is the basis state,
        e.g., cal_1001

        Pass the results of these circuits to "generate_calibration_matrix()"
    """

    # Expand out the registers if not already done
    if isinstance(qubits, list):
        qubits = _format_registers(*qubits)  # Unroll list of registers
    else:
        qubits = _format_registers(qubits)

    qubit_registers = set(q[0] for q in qubits)

    cal_circuits = []
    nqubits = len(qubits)

    # create classical bit registers
    clbits = ClassicalRegister(nqubits)

    # labels for 2**n qubit states
    state_labels = tomo.count_keys(nqubits)

    for basis_state in state_labels:
        qc_circuit = QuantumCircuit(*qubit_registers, clbits,
                                    name='cal_%s' % basis_state)
        for qind, _ in enumerate(basis_state):
            if int(basis_state[nqubits-qind-1]):
                # the index labeling of the label is backwards with
                # the list
                qc_circuit.x(qubits[qind])

            # add measurements
            qc_circuit.measure(qubits[qind], clbits[qind])

        cal_circuits.append(qc_circuit)

    return cal_circuits, state_labels
