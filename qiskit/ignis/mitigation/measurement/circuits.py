# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
Measurement calibration circuits. To apply the measurement mitigation
use the fitters to produce a filter.
"""

from qiskit import QuantumRegister, ClassicalRegister, \
    QuantumCircuit, QiskitError
from ...verification.tomography import count_keys


def complete_meas_cal(qubit_list=None, qr=None, cr=None, circlabel=''):
    """
    Return a list of measurement calibration circuits for the full
    Hilbert space.

    Each of the 2**n circuits creates a basis state

    Args:
        qubit_list: A list of qubits to perform the measurement correction on,
        if None and qr is given then assumed to be performed over the entire
        qr. The calibration states will be labelled according to this ordering

        qr (QuantumRegister): A quantum register. If none one is created

        cr (ClassicalRegister): A classical register. If none one is created

        circlabel: A string to add to the front of circuit names for
        unique identification

    Returns:
        A list of QuantumCircuit objects containing the calibration circuits

        A list of calibration state labels

    Additional Information:
        The returned circuits are named circlabel+cal_XXX
        where XXX is the basis state,
        e.g., cal_1001

        Pass the results of these circuits to "MeasurementFitter" constructor
    """

    if qubit_list is None and qr is None:
        raise QiskitError("Must give one of a qubit_list or a qr")

    # Create the registers if not already done
    if qr is None:
        qr = QuantumRegister(max(qubit_list)+1)

    if qubit_list is None:
        qubit_list = range(len(qr))

    cal_circuits = []
    nqubits = len(qubit_list)

    # create classical bit registers
    if cr is None:
        cr = ClassicalRegister(nqubits)

    # labels for 2**n qubit states
    state_labels = count_keys(nqubits)

    for basis_state in state_labels:
        qc_circuit = QuantumCircuit(qr, cr,
                                    name='%scal_%s' % (circlabel, basis_state))
        for qind, _ in enumerate(basis_state):
            if int(basis_state[nqubits-qind-1]):
                # the index labeling of the label is backwards with
                # the list
                qc_circuit.x(qr[qubit_list[qind]])

            # add measurements
            qc_circuit.measure(qr[qubit_list[qind]], cr[qind])

        cal_circuits.append(qc_circuit)

    return cal_circuits, state_labels


def tensored_meas_cal(mit_pattern=None, qr=None, cr=None, circlabel=''):
    """
    Return a list of calibration circuits for the all zeros and all ones basis states.

    Args:
        qubit_list: A list of qubits to perform the measurement correction on,
        if None and qr is given then assumed to be performed over the entire
        qr.

        qr (QuantumRegister): A quantum register. If none one is created

        cr (ClassicalRegister): A classical register. If none one is created

        circlabel: A string to add to the front of circuit names for
        unique identification

    Returns:
        A list of two QuantumCircuit objects containing the calibration circuits

        A list of calibration state labels

    Additional Information:
        The returned circuits are named circlabel+cal_XXX
        where XXX is the basis state,
        i.e., cal_000 and cal_111

        Pass the results of these circuits to "MeasurementFitter" constructor
    """

    if mit_pattern is None and qr is None:
        raise QiskitError("Must give one of mit_pattern or qr")

    qubits_in_pattern = []
    if mit_pattern is not None:
        for qubit_list in mit_pattern:
            for qubit in qubit_list:
                if qubit in qubits_in_pattern:
                    raise QiskitError("mit_pattern cannot contain multiple instances of the same qubit")
                qubits_in_pattern.append(qubit)
                
        # Create the registers if not already done
        if qr is None:
            qr = QuantumRegister(max(qubits_in_pattern)+1)
    else:
        qubits_in_pattern = range(len(qr))
        mit_pattern = [qubits_in_pattern]

    nqubits = len(qubits_in_pattern)

    # create classical bit registers
    if cr is None:
        cr = ClassicalRegister(nqubits)

    size_of_largest_group = max([len(qubit_list) for qubit_list in mit_pattern])
    largest_labels = count_keys(size_of_largest_group)

    state_labels = []
    for largest_state in largest_labels:
        basis_state = ''
        for qubit_list in mit_pattern:
            basis_state += largest_state[:len(qubit_list)]
        state_labels.append(basis_state)
                

    cal_circuits = []
    for basis_state in state_labels:
        qc_circuit = QuantumCircuit(qr, cr,
                                    name='%scal_%s' % (circlabel, basis_state))

        start_index = 0
        for qubit_list in mit_pattern:
            
            end_index = start_index + len(qubit_list)
            substate = basis_state[start_index:end_index]
            start_index = end_index
            
            for qind, _ in enumerate(substate):
                if substate[len(substate)-qind-1] == '1':
                    # the index labeling of the label is backwards with
                    # the list
                    qc_circuit.x(qr[qubit_list[qind]])
                    qc_circuit.t(qr[qubit_list[qind]])

                # add measurements
                qc_circuit.measure(qr[qubit_list[qind]],
                                   cr[end_index-qind-1])

        cal_circuits.append(qc_circuit)

    return cal_circuits, state_labels
