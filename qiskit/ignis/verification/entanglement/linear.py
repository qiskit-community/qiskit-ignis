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
The module linear.py provides the linear
preparation analogous of parallelize.py.
"""

from typing import Tuple, List
from qiskit.circuit import ClassicalRegister, QuantumRegister, Parameter
from qiskit.circuit import QuantumCircuit


def get_measurement_circ(n: int,
                         qregname: str,
                         cregname: str,
                         full_measurement: bool = True
                         ) -> QuantumCircuit:
    """
    Creates a measurement circuit that can toggle between
    measuring the first control qubit or measuring all qubits.
    The default is measurement of all qubits.

    Args:
        n: number of qubits
        qregname: The name to use for the quantum register
        cregname: The name to use for the classical register
        full_measurement: Whether to append full measurement, or only on
            the first qubit.

    Returns:
        The measurement suffix for a circuit
    """
    q = QuantumRegister(n, qregname)
    if full_measurement:
        cla = ClassicalRegister(n, cregname)
        meas = QuantumCircuit(q, cla)
        meas.barrier()
        meas.measure(q, cla)
        return meas

    cla = ClassicalRegister(1, cregname)
    meas = QuantumCircuit(q, cla)
    meas.barrier()
    meas.measure(q[0], cla)
    return meas


def get_ghz_simple(n: int,
                   measure: bool = True,
                   full_measurement: bool = True
                   ) -> QuantumCircuit:
    """
    Creates a linear GHZ state with the option of measurement

    Args:
        n: number of qubits
        measure: Whether to add measurement gates
        full_measurement: Whether to append full measurement, or only on
            the first qubit. Relevant only for measure=True

    Returns:
        A linear GHZ Circuit
    """
    q = QuantumRegister(n, 'q')
    circ = QuantumCircuit(q)
    circ.h(q[0])
    for i in range(1, n):
        circ.cx(q[i - 1], q[i])
    if measure:
        meas = get_measurement_circ(n, 'q', 'c', full_measurement)
        circ = circ + meas

    return circ


def get_ghz_mqc(n: int,
                delta: float,
                full_measurement: bool = True
                ) -> QuantumCircuit:
    """
    This function creates an MQC circuit with n qubits,
    where the middle phase rotation around the z axis is by delta

    Args:
        n: number of qubits
        delta: the rotation of the middle phase around the z axis
        full_measurement: Whether to append full measurement, or only
            on the first qubit.

    Returns:
       The MQC circuit
    """
    q = QuantumRegister(n, 'q')
    circ = get_ghz_simple(n, measure=False)
    circinv = circ.inverse()
    circ.barrier()
    circ.u1(delta, q)
    circ.x(q)
    circ.barrier()
    circ += circinv
    meas = get_measurement_circ(n, 'q', 'c', full_measurement)
    circ = circ + meas
    return circ


def get_ghz_mqc_para(n: int,
                     full_measurement: bool = True
                     ) -> Tuple[QuantumCircuit, Parameter]:
    """
    This function creates an MQC circuit with n qubits,
    where the middle phase rotation around the z axis is parameterized

    Args:
        n: number of qubits
        full_measurement: Whether to append full measurement, or only
            on the first qubit.

    Returns:
        An mqc circuit and its Delta parameter
    """
    q = QuantumRegister(n, 'q')
    circ = get_ghz_simple(n, measure=False)
    delta = Parameter('t')
    circinv = circ.inverse()
    circ.barrier()
    circ.u1(delta, q)
    circ.x(q)
    circ.barrier()
    circ += circinv
    meas = get_measurement_circ(n, 'q', 'c', full_measurement)
    circ = circ + meas
    return circ, delta


def get_ghz_po(n: int, delta: float) -> QuantumCircuit:
    """
    This function creates an Parity Oscillation circuit
    with n qubits, where the middle superposition rotation around
    the x and y axes is by delta

    Args:
        n: number of qubits
        delta: the middle superposition rotation

    Returns:
        The Parity Oscillation circuit
    """
    q = QuantumRegister(n, 'q')
    circ = get_ghz_simple(n, measure=False)

    circ.barrier()
    circ.u2(delta, -delta, q)
    circ.barrier()
    meas = get_measurement_circ(n, 'q', 'c', True)
    circ = circ + meas
    return circ


def get_ghz_po_para(n: int) -> Tuple[QuantumCircuit, List[Parameter]]:
    """
    This function creates a Parity Oscillation circuit with n qubits,
    where the middle superposition rotation around

    the x and y axes is by delta

    Args:
        n: number of qubits

    Returns:
        The parity oscillation circuit and its Delta/minus-delta parameters
    """
    q = QuantumRegister(n, 'q')
    delta = Parameter('t')
    deltaneg = Parameter('-t')
    circ = get_ghz_simple(n, measure=False)

    circ.barrier()
    circ.u2(delta, deltaneg, q)
    meas = get_measurement_circ(n, 'q', 'c', True)
    circ = circ + meas
    return circ, [delta, deltaneg]
