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

from qiskit.circuit import ClassicalRegister, QuantumRegister, Parameter
from qiskit.circuit import QuantumCircuit


def get_measurement_circ(n, qregname, cregname, full_measurement=True):
    """
    Creates a measurement circuit that can toggle between
    measuring the first control qubit or measuring all qubits.
    The default is measurement of all qubits.

    Args:
        n (int): number of qubits
        qregname (str): The name to use for the quantum register
        cregname (str): The name to use for the classical register
        full_measurement (bool): Whether to append full measurement, or only on
            the first qubit.

    Returns:
        QuantumCircuit: The measurement suffix for a circuit
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


def get_ghz_simple(n, measure=True, full_measurement=True):
    """
    Creates a linear GHZ state with the option of measurement

    Args:
        n (int): number of qubits
        measure (bool): Whether to add measurement gates
        full_measurement (bool): Whether to append full measurement, or only on
            the first qubit. Relevant only for measure=True

    Returns:
        QuantumCircuit: A linear GHZ Circuit
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


def get_ghz_mqc(n, delta, full_measurement):
    """
    This function creates an MQC circuit with n qubits,
    where the middle phase rotation around the z axis is by delta
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


def get_ghz_mqc_para(n, full_measurement=True):
    """
    This function creates an MQC circuit with n qubits,
    where the middle phase rotation around the z axis is by delta

    Args:
        n (int): number of qubits
        full_measurement (bool): Whether to append full measurement, or only
            on the first qubit.

    Returns:
        tuple: A tuple of type (``QuantumCircuit``, ``Parameter``): An mqc
            circuit and its Delta parameter
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


def get_ghz_po(n, delta):
    """
    This function creates an Parity Oscillation circuit
    with n qubits, where the middle superposition rotation around
    the x and y axes is by delta
    """
    q = QuantumRegister(n, 'q')
    circ = get_ghz_simple(n, measure=False)

    circ.barrier()
    circ.u2(delta, -delta, q)
    circ.barrier()
    meas = get_measurement_circ(n, 'q', 'c', True)
    circ = circ + meas
    return circ


def get_ghz_po_para(n):
    """
    This function creates a Parity Oscillation circuit with n qubits,
    where the middle superposition rotation around

    the x and y axes is by delta

    Args:
        n (int): number of qubits

    Returns:
        tuple: A tuple of type (``QuantumCircuit``, ``list``) containing a
            parity oscillation circuit and its Delta/minus-delta parameters
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
