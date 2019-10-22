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
       circuit: the quantum circuit that the gates should be appended to
       qr: the qubit register where the gates should be added
       qubit: index of qubit where the gates should be added
       num_of_id_gates: the number of identity gates to add

    Returns:
       circuit: The original circuit object, but with identity
                gates added to the qubit register qr at qubit 'qubit'
    """

    for _ in range(num_of_id_gates):
        circuit.barrier(qr[qubit])
        circuit.iden(qr[qubit])

    circuit.barrier(qr[qubit])
    return circuit


def time_to_ngates(times, gate_time):
    """
    A function to convert a list of times into an integer list of gates.

    Args:
       times: A list of times (in the same unit of time as used to
       specify gate_time)
       gate_time: the gate_time

    Returns:
       num_gates: integer list of the number of gates
    """

    return (np.round(np.array(times)/gate_time)).astype(int)
