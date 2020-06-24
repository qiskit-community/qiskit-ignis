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

"""Utilities for circuits generation."""

import numpy as np


def pad_id_gates(circuit, qr, qubit, num_of_id_gates):
    """
    A function for padding a circuit with single-qubit identity gates.

    Args:
        circuit (QuantumCircuit): the quantum circuit that the gates should be
            appended to
        qr (QuantumRegister): the qubit register where the gates should be added
        qubit (int): index of qubit where the gates should be added
        num_of_id_gates (int): the number of identity gates to add

    Returns:
        circuit: The original circuit object, but with identity
            gates added to the qubit register qr at qubit 'qubit'
    """

    for _ in range(num_of_id_gates):
        circuit.barrier(qr[qubit])
        # Maintain compatibility with 0.12 stable terra
        # the case of .iden should be removed once terra 0.13 is stable
        if hasattr(circuit, 'i'):
            circuit.i(qr[qubit])
        else:
            circuit.iden(qr[qubit])

    circuit.barrier(qr[qubit])
    return circuit


def time_to_ngates(times, gate_time):
    """
    A function to convert a list of times into an integer list of gates.

    Args:
       times (list): A list of times (in the same unit of time as used to
       specify gate_time)
       gate_time (float): the gate_time

    Returns:
       num_gates: integer list of the number of gates
    """

    return (np.round(np.array(times)/gate_time)).astype(int)
