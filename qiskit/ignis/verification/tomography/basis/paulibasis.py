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

"""Pauli tomography preparation and measurement basis
"""

# Needed for functions
import numpy as np

# Import QISKit classes
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from .tomographybasis import TomographyBasis


###########################################################################
# Built-in circuit functions
###########################################################################

def pauli_measurement_circuit(
        op: str,
        qubit: QuantumRegister,
        clbit: ClassicalRegister
) -> QuantumCircuit:
    """Return a qubit Pauli operator measurement circuit.

    Params:
        op: Pauli operator 'X', 'Y', 'Z'.
        qubit: qubit to be measured.
        clbit: clbit for measurement outcome.

    Returns:
        The measurement circuit for the given Pauli.
    """

    circ = QuantumCircuit(qubit.register, clbit.register)
    if op == 'X':
        circ.h(qubit)
        circ.measure(qubit, clbit)
    if op == 'Y':
        circ.sdg(qubit)
        circ.h(qubit)
        circ.measure(qubit, clbit)
    if op == 'Z':
        circ.measure(qubit, clbit)
    return circ


def pauli_preparation_circuit(
        op: str,
        qubit: QuantumRegister
) -> QuantumCircuit:
    """Return a qubit Pauli eigenstate preparation circuit.

    This circuit assumes the qubit is initialized
    in the :math:`Zp` eigenstate :math:`[1, 0]`.

    Params:
        op: Pauli eigenstate 'Zp', 'Zm', 'Xp', 'Xm', 'Yp', or 'Ym'.
        qubit: qubit to be prepared.

    Returns:
        The preparation circuit for the given Pauli eigenstate.
    """

    circ = QuantumCircuit(qubit.register)
    if op == 'Xp':
        circ.h(qubit)
    if op == 'Xm':
        circ.x(qubit)
        circ.h(qubit)
    if op == 'Yp':
        circ.h(qubit)
        circ.s(qubit)
    if op == 'Ym':
        circ.x(qubit)
        circ.h(qubit)
        circ.s(qubit)
    if op == 'Zm':
        circ.x(qubit)
    return circ


###########################################################################
# Matrix functions for built-in bases
###########################################################################

def pauli_preparation_matrix(label: str) -> np.array:
    r"""Return the matrix corresponding to a Pauli eigenstate preparation.

    Args:
        label: single-qubit Pauli eigenstate operator label.

    Returns:
        A Numpy array for the Pauli eigenstate. Allowed inputs
            and corresponding returned matrices are:

            'Xp' : :math:`\frac{1}{2}
            \left(\begin{array}{cc}1 & 1\\1 & 1\end{array}\right)`

            'Xm' : :math:`\frac{1}{2}
            \left(\begin{array}{cc}1 & -1\\1 & -1\end{array}\right)`

            'Yp' : :math:`\frac{1}{2}
            \left(\begin{array}{cc}1 & -i\\i & 1\end{array}\right)`

            'Ym' : :math:`\frac{1}{2}
            \left(\begin{array}{cc}1 & i\\-i & 1\end{array}\right)`

            'Zp' : :math:`\left(\begin{array}
            {cc}1 & 0\\0 & 0\end{array}\right)`

            'Zm' : :math:`\left(\begin{array}
            {cc}01 & 0\\0 & 1\end{array}\right)`
    """
    res = np.array([])
    # Return matrix for allowed label
    if label == 'Xp':
        res = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
    if label == 'Xm':
        res = np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=complex)
    if label == 'Yp':
        res = np.array([[0.5, -0.5j], [0.5j, 0.5]], dtype=complex)
    if label == 'Ym':
        res = np.array([[0.5, 0.5j], [-0.5j, 0.5]], dtype=complex)
    if label == 'Zp':
        res = np.array([[1, 0], [0, 0]], dtype=complex)
    if label == 'Zm':
        res = np.array([[0, 0], [0, 1]], dtype=complex)
    return res


def pauli_measurement_matrix(label: str, outcome: int) -> np.array:
    r"""Return the matrix corresponding to a Pauli measurement outcome.

    Args:
        label: single-qubit Pauli measurement operator label.
        outcome: measurement outcome.

    Returns:
        A Numpy array for measurement outcome operator.
            Allowed inputs and corresponding returned matrices are:

            'X', 0 : :math:`\frac{1}{2}
            \left(\begin{array}{cc}1 & 1\\1 & 1\end{array}\right)`

            'X', 1 : :math:`\frac{1}{2}
            \left(\begin{array}{cc}1 & -1\\1 & -1\end{array}\right)`

            'Y', 0 : :math:`\frac{1}{2}
            \left(\begin{array}{cc}1 & -i\\i & 1\end{array}\right)`

            'Y', 1 : :math:`\frac{1}{2}
            \left(\begin{array}{cc}1 & i\\-i & 1\end{array}\right)`

            'Z', 0 : :math:`\left(\begin{array}
            {cc}1 & 0\\0 & 0\end{array}\right)`

            'Z', 1 : :math:`\left(\begin{array}
            {cc}01 & 0\\0 & 1\end{array}\right)`
    """
    res = np.array([])
    # Return matrix
    if label == 'X':
        if outcome in ['0', 0]:
            res = pauli_preparation_matrix('Xp')
        if outcome in ['1', 1]:
            res = pauli_preparation_matrix('Xm')
    if label == 'Y':
        if outcome in ['0', 0]:
            res = pauli_preparation_matrix('Yp')
        if outcome in ['1', 1]:
            res = pauli_preparation_matrix('Ym')
    if label == 'Z':
        if outcome in ['0', 0]:
            res = pauli_preparation_matrix('Zp')
        if outcome in ['1', 1]:
            res = pauli_preparation_matrix('Zm')
    return res


###########################################################################
# PauliBasis Object
###########################################################################

PauliBasis = TomographyBasis('Pauli',  # pylint: disable=invalid-name
                             measurement=(('X', 'Y', 'Z'),
                                          pauli_measurement_circuit,
                                          pauli_measurement_matrix),
                             preparation=(('Xp', 'Xm', 'Yp', 'Ym', 'Zp', 'Zm'),
                                          pauli_preparation_circuit,
                                          pauli_preparation_matrix))
