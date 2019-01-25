# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
Measurement correction scripts.
"""

from scipy.optimize import minimize
import scipy.linalg as la
import numpy as np
import qiskit.tools.qcvv.tomography as tomo
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, QiskitError

def measurement_calibration_circuits(qubits):

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
        The returned circuits are named cal_XXX where XXX is the basis state, e.g.,
        cal_1001

        Pass the results of these circuits to "generate_calibration_matrix()"
    """

    # Expand out the registers if not already done
    if isinstance(qubits, list):
        qubits = _format_registers(*qubits)  # Unroll list of registers
    else:
        qubits = _format_registers(qubits)

    qubit_registers = set([q[0] for q in qubits])

    cal_circuits = []
    nqubits = len(qubits)

    #create classical bit registers
    clbits = ClassicalRegister(nqubits)

    #labels for 2**n qubit states
    state_labels = tomo.count_keys(nqubits)

    for basis_state in state_labels:
        qc_circuit = QuantumCircuit(*qubit_registers, clbits, name='cal_%s'%basis_state)
        for qind, _ in enumerate(basis_state):
            if int(basis_state[nqubits-qind-1]):
                #the index labeling of the label is backwards with
                #the list
                qc_circuit.x(qubits[qind])

            #add measurements
            qc_circuit.measure(qubits[qind],clbits[qind])

        cal_circuits.append(qc_circuit)

    return cal_circuits, state_labels

def generate_calibration_matrix(results, state_labels):

    """
    Return a measurement calibration matrix from the results of running the
    measurement calibration circuits

    Args:
        results: the results of running the measurement calibration ciruits

        state_labels: list of calibration state labels. The output matrix will obey
        this ordering


    Returns:
        A 2**n x 2**n matrix that can be used to correct measurement errors


    Additional Information:
        Use this matrix in correct_measurement()
    """

    cal_matrix = np.zeros([len(state_labels),len(state_labels)],dtype=float)

    for stateidx, state in enumerate(state_labels):
        state_cnts = results.get_counts('cal_%s'%state)
        shots = sum(state_cnts.values())
        for stateidx2, state2 in enumerate(state_labels):
            cal_matrix[stateidx,stateidx2] = state_cnts.get(state2,0)/shots

    return cal_matrix.transpose()

def remove_measurement_errors(raw_data, state_labels, cal_matrix, \
                              method=1):

    """
    Apply the calibration matrix to results

    Args:
        raw_data: The data to be corrected. Can be in a number of forms.
        Form1: a counts dictionary from results.get_counts
        Form2: a list of counts of length==len(state_labels)
        Form3: a list of counts of length==M*len(state_labels) where M is an
        integer (e.g. for use with the tomography data)

        state_labels: list of calibration state labels and the ordering of the cal_matrix

        cal_matrix: from `generate_calibration_matrix`

        method: 0: pseudo-inverse, 1: least-squares constrained to have physical probabilities


    Returns:
        The corrected data in the same form as raw_data


    Additional Information:

    """



    #check forms of raw_data
    if type(raw_data) is dict:
        #counts dictionary
        data_format = 0

        #convert to form2
        raw_data2 = [np.zeros(len(state_labels),dtype=float)]

        for stateidx, state in enumerate(state_labels):
            raw_data2[0][stateidx] = raw_data.get(state,0)

    elif type(raw_data) is list:

        size_ratio = len(raw_data)/len(state_labels)

        if len(raw_data)==len(state_labels):
            data_format = 1
            raw_data2 = [raw_data]

        elif int(size_ratio)==size_ratio:
            data_format = 2
            size_ratio = int(size_ratio)
            #make the list into chunks the size of state_labels for easier processing
            raw_data2 = np.zeros([size_ratio,len(state_labels)])

            for i in range(size_ratio):
                raw_data2[i][:] = raw_data[i*size_ratio:(i+1)*size_ratio]

        else:

            raise QiskitError("Data list is not an integer multiple \
                              of the number of calibrated states")

    else:

        raise QiskitError("Unrecognized type for raw_data.")



    #Apply the correction
    for data_idx,_ in enumerate(raw_data2):

        if method == 0:

            raw_data2[data_idx] = \
                np.dot(la.pinv(cal_matrix),raw_data2[data_idx])

        elif method == 1:

            nshots = sum(raw_data2[data_idx])
            fun = lambda x: sum((raw_data2[data_idx] - np.dot(\
                cal_matrix, x))**2)
            x0 = np.random.rand(len(state_labels))
            x0 = x0/sum(x0)
            cons = ({'type': 'eq', 'fun': lambda x: nshots - sum(x)})
            bnds = tuple((0, nshots) for x in x0)
            res = minimize(fun, x0, method='SLSQP',
                           constraints=cons, bounds=bnds, tol=1e-6)
            raw_data2[data_idx] = res.x


    if data_format==2:
        #flatten back out the list
        raw_data2 = raw_data2.flatten()

    elif data_format==0:
        #convert back into a counts dictionary
        new_count_dict = {}
        for stateidx, state in enumerate(state_labels):
            if raw_data2[0][stateidx]!=0:
                new_count_dict[state] = raw_data2[0][stateidx]

        raw_data2 = new_count_dict

    else:
        raw_data2 = raw_data2[0]

    return raw_data2


def _format_registers(*registers):
    """
    Return a list of qubit QuantumRegister tuples.
    """
    if not registers:
        raise Exception('No registers are being measured!')
    qubits = []
    for tuple_element in registers:
        if isinstance(tuple_element, QuantumRegister):
            for j in range(tuple_element.size):
                qubits.append((tuple_element, j))
        else:
            qubits.append(tuple_element)
    # Check registers are unique
    if len(qubits) != len(set(qubits)):
        raise Exception('Qubits to be measured are not unique!')
    return qubits
