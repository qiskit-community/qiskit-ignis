# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
Interface for the full state and process tomography flow
"""

import qiskit
from qiskit import Aer
from qiskit.quantum_info import state_fidelity
from qiskit.tools.qi.qi import outer
from qiskit.ignis.verification.tomography.fitters import fitter_data, \
    state_cvx_fit, process_cvx_fit, state_mle_fit, process_mle_fit
from .basis.circuits import state_tomography_circuits
from .basis.circuits import process_tomography_circuits
from .data import tomography_data


def perform_state_tomography(circuit, measured_qubits,
                             backend=None,
                             fitter='cvx',
                             ideal=True,
                             fidelity=True,
                             shots=5000,
                             ):
    """
       Run the full state tomography flow:
       1) Generate tomography circuits.
       2) Run the circuits on the given backend
       3) Extract from the results a fitter-friendly representation
       4) Run fitter to obtain density operator
       5) (optional) create an ideal result using a statevector simulator

       Args:
           circuit: the circuit generating the state to measure
           measured_qubits: the qubits to be measured
           backend: the backend on which the circuit is ran
           fitter: the fitter used to obtain the density operator
           ideal: whether to compute an ideal density operator
           fidelity: whether to compute the fidelity between the tomography
                     result and the ideal result
           shots: how many shots to perform with the backend
                  on each measurement

       Returns:
            if ideal and fidelity are False, returns the density operator
            otherwise returns a dictionary with keys
                'rho': the density operator
                'ideal_psi': ideal statevector
                'fidelity' the fidelity between rho and ideal_psi
       """

    if backend is None:
        backend = Aer.get_backend('qasm_simulator')
    tomography_circuits = state_tomography_circuits(circuit, measured_qubits)
    job = qiskit.execute(tomography_circuits, backend, shots=shots)
    data_results = tomography_data(job.result(), tomography_circuits)
    data, basis, _ = fitter_data(data_results)
    rho = None
    if fitter == 'cvx':
        try:
            rho = state_cvx_fit(data, basis)
        except RuntimeError as e:
            print("CVX run failed: {}, attempting MLE fit".format(e))
    if fitter == 'mle' or rho is None:
        rho = state_mle_fit(data, basis)

    if ideal is False and fidelity is False:
        return rho

    result = {'rho': rho}
    if ideal:
        job = qiskit.execute(circuit, Aer.get_backend('statevector_simulator'))
        ideal_psi = job.result().get_statevector(circuit)
        result['ideal_psi'] = ideal_psi
        if fidelity:
            fidelity_value = state_fidelity(ideal_psi, rho)
            result['fidelity'] = fidelity_value

    return result


def perform_process_tomography(circuit, measured_qubits,
                               backend=None,
                               fitter='cvx',
                               ideal=True,
                               fidelity=True,
                               shots=4000, ):
    """
       Run the full process tomography flow:
       1) Generate tomography circuits.
       2) Run the circuits on the given backend
       3) Extract from the results a fitter-friendly representation
       4) Run fitter to obtain choi matrix
       5) (optional) create an ideal result using a unitary simulator

       Args:
           circuit: the circuit representing the process to measure
           measured_qubits: the qubits to be measured
           backend: the backend on which the circuit is ran
           fitter: the fitter used to obtain the choi matrix
           ideal: whether to compute an ideal choi matrix
           fidelity: whether to compute the fidelity between the tomography
                     result and the ideal result
           shots: how many shots to perform with the backend
                  on each measurement

       Returns:
            if ideal and fidelity are False, returns the density operator
            otherwise returns a dictionary with keys
                'rho': the density operator
                'ideal_psi': ideal statevector
                'fidelity' the fidelity between rho and ideal_psi
       """

    if backend is None:
        backend = Aer.get_backend('qasm_simulator')
    circuits = process_tomography_circuits(circuit, measured_qubits)
    job = qiskit.execute(circuits, backend, shots=shots)
    tomography_data_results = tomography_data(job.result(), circuits)
    data, basis, _ = fitter_data(tomography_data_results)
    choi = None
    if fitter == 'cvx':
        try:
            choi = process_cvx_fit(data, basis)
        except RuntimeError as e:
            print("CVX run failed: {}, attempting MLE fit".format(e))
    if fitter == 'mle' or choi is None:
        choi = process_mle_fit(data, basis)

    if ideal is False and fidelity is False:
        return choi

    result = {'choi': choi}
    if ideal:
        job = qiskit.execute(circuit, Aer.get_backend('unitary_simulator'))
        ideal_unitary = job.result().get_unitary(circuit)
        ideal_choi = outer(ideal_unitary.ravel(order='F'))
        result['ideal_choi'] = ideal_choi
        if fidelity:
            n = len(measured_qubits)
            fidelity_value = state_fidelity(ideal_choi / 2 ** n, choi / 2 ** n)
            result['fidelity'] = fidelity_value
    return result
