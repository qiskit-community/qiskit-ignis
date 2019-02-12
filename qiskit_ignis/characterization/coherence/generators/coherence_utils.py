# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Utilities for circuits generation
"""

def pad_id_gates(circuit, qr, num_of_id_gates):
    """
    A function for padding a circuit with identity gates.

    Args:
       circuit: the quantum circuit that the gates should be appended to
       qr: the qubit register that the gates should be added to
       num_of_id_gates: the number of identity gates to add

    Returns:
       circuit: The original circuit object, but with identity
                gates added to the qubit register qr
    """

    for _ in range(num_of_id_gates):
        circuit.barrier(qr)
        circuit.iden(qr)

    return circuit
