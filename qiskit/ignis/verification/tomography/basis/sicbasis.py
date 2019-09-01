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
Symmetric informationally complete (SIC)-POVM tomography preparation basis.
"""

# Needed for functions
import numpy as np

# Import QISKit classes
from qiskit import QuantumCircuit
from .tomographybasis import TomographyBasis


###########################################################################
# Built-in circuit functions
###########################################################################

def sicpovm_preparation_circuit(op, qubit):
    """
    Return a SIC-POVM projector preparation circuit.

    This circuit assumes the qubit is initialized in the Zp eigenstate [1, 0].

    Params:
        op (str): SIC-POVM element label 'S0', 'S1', 'S2' or 'S3'.
        qubit (QuantumRegister tuple): qubit to be prepared.

    Returns:
        A QuantumCircuit object.
    """
    circ = QuantumCircuit(qubit.register)
    theta = -2 * np.arctan(np.sqrt(2))
    if op == 'S1':
        circ.u3(theta, np.pi, 0.0, qubit)
    if op == 'S2':
        circ.u3(theta, np.pi / 3, 0.0, qubit)
    if op == 'S3':
        circ.u3(theta, -np.pi / 3, 0.0, qubit)
    return circ


###########################################################################
# Matrix functions for built-in bases
###########################################################################

def sicpovm_preparation_matrix(label):
    """
    Return the matrix corresonding to a SIC-POVM preparation.

    Args:
        label (str): single-qubit SIC-POVM element label.

    Returns:
        A Numpy array for the SIC-POVM element.
        Allowed inputs and corresponding returned matrices are:

            'S0' : [[1, 0], [0, 0]]
            'S1' : [[1, np.sqrt(2)], [np.sqrt(2), 2]]/ 3
            'S2' : [[1, exp(pi * 2j / 3) * sqrt(2)],
                     [exp(-pi * 2j / 3) * sqrt(2), 2]] / 3
            'S3' : [[1, exp(-pi * 2j / 3) * sqrt(2)],
                     [exp(pi * 2j / 3) * sqrt(2), 2]] / 3
    """
    res = np.array([])
    # Return matrix for allowed label
    if label == 'S0':
        res = np.array([[1, 0], [0, 0]], dtype=complex)
    if label == 'S1':
        res = np.array([[1, np.sqrt(2)], [np.sqrt(2), 2]], dtype=complex) / 3
    if label == 'S2':
        res = np.array([[1, np.exp(np.pi * 2j / 3) * np.sqrt(2)],
                        [np.exp(-np.pi * 2j / 3) * np.sqrt(2), 2]]) / 3
    if label == 'S3':
        res = np.array([[1, np.exp(-np.pi * 2j / 3) * np.sqrt(2)],
                        [np.exp(np.pi * 2j / 3) * np.sqrt(2), 2]]) / 3
    return res


###########################################################################
# PauliBasis Object
###########################################################################

SICBasis = TomographyBasis('SIC', measurement=None,
                           preparation=(('S0', 'S1', 'S2', 'S3'),
                                        sicpovm_preparation_circuit,
                                        sicpovm_preparation_matrix))
