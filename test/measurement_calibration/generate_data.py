# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Generate data for measurement calibration fitter tests
"""

import os
import json
from test.utils import convert_ndarray_to_list_in_data
import qiskit
from qiskit.ignis.mitigation.measurement import (CompleteMeasFitter, TensoredMeasFitter,
                                                 complete_meas_cal, tensored_meas_cal,
                                                 MeasurementFilter)
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import pauli_error

# fixed seed for simulations
SEED = 42


def meas_calibration_circ_execution(nqunits: int, shots: int, seed: int):
    """
    create measurement calibration circuits and simulates them with noise
    Args:
        nqunits (int): number of qubits to run the measurement calibration on
        shots (int): number of shots per simulation
        seed (int): the seed to use in the simulations

    Returns:
        list: list of Results of the measurement calibration simulations
        list: list of all the possible states with this amount of qubits
        dict: dictionary of results counts of bell circuit simulation with measurement errors
    """
    # define the circuits
    qr = qiskit.QuantumRegister(nqunits)
    meas_calibs, state_labels = complete_meas_cal(qr=qr, circlabel='test')
    # define noise
    prob = 0.2
    error_meas = pauli_error([('X', prob), ('I', 1 - prob)])
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas, "measure")

    # run the circuits multiple times
    backend = qiskit.Aer.get_backend('qasm_simulator')
    cal_results = qiskit.execute(meas_calibs, backend=backend, shots=shots, noise_model=noise_model,
                                 seed_simulator=seed).result()

    # create bell state and get it's results
    qc = qiskit.QuantumCircuit(nqunits, nqunits)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.measure(qc.qregs[0], qc.cregs[0])

    bell_results = qiskit.execute(qc, backend=backend, shots=shots, noise_model=noise_model,
                                  seed_simulator=seed).result().get_counts()

    return cal_results, state_labels, bell_results


def tensored_calib_circ_execution(shots: int, seed: int):
    """
    create tensored measurement calibration circuits and simulates them with noise
    Args:
        shots (int): number of shots per simulation
        seed (int): the seed to use in the simulations

    Returns:
        list: list of Results of the measurement calibration simulations
        list: the mitigation pattern
        dict: dictionary of results counts of bell circuit simulation with measurement errors
    """
    # define the circuits
    qr = qiskit.QuantumRegister(5)
    mit_pattern = [[2], [4, 1]]
    meas_calibs, mit_pattern = tensored_meas_cal(mit_pattern=mit_pattern, qr=qr, circlabel='test')
    # define noise
    prob = 0.2
    error_meas = pauli_error([('X', prob), ('I', 1 - prob)])
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas, "measure")

    # run the circuits multiple times
    backend = qiskit.Aer.get_backend('qasm_simulator')
    cal_results = qiskit.execute(meas_calibs, backend=backend, shots=shots, noise_model=noise_model,
                                 seed_simulator=seed).result()

    # create bell state and get it's results
    cr = qiskit.ClassicalRegister(3)
    qc = qiskit.QuantumCircuit(qr, cr)
    qc.h(qr[2])
    qc.cx(qr[2], qr[4])
    qc.cx(qr[2], qr[1])
    qc.measure(qr[2], cr[0])
    qc.measure(qr[4], cr[1])
    qc.measure(qr[1], cr[2])

    bell_results = qiskit.execute(qc, backend=backend, shots=shots, noise_model=noise_model,
                                  seed_simulator=seed).result()

    return cal_results, mit_pattern, bell_results


def generate_meas_calibration(results_file_path: str, runs: int):
    """
    run the measurement calibration circuits, calculates the fitter matrix in few methods
    and saves the results
    The simulation results files will contain a list of dictionaries with the keys:
        cal_matrix - the matrix used to calculate the ideal measurement
        fidelity - the calculated fidelity of using this matrix
        results - results of a bell state circuit with noise
        results_pseudo_inverse - the result of using the psedo-inverse method on the bell state
        results_least_square - the result of using the least-squares method on the bell state

    Args:
        results_file_path: path of the json file of the results file
        runs: the number of different runs to save
    """
    results = []
    for run in range(runs):
        cal_results, state_labels, circuit_results = \
            meas_calibration_circ_execution(3, 1000, SEED + run)

        meas_cal = CompleteMeasFitter(cal_results, state_labels, circlabel='test')
        meas_filter = MeasurementFilter(meas_cal.cal_matrix, state_labels)

        # Calculate the results after mitigation
        results_pseudo_inverse = meas_filter.apply(
            circuit_results, method='pseudo_inverse')
        results_least_square = meas_filter.apply(
            circuit_results, method='least_squares')
        results.append({"cal_matrix": convert_ndarray_to_list_in_data(meas_cal.cal_matrix),
                        "fidelity": meas_cal.readout_fidelity(),
                        "results": circuit_results,
                        "results_pseudo_inverse": results_pseudo_inverse,
                        "results_least_square": results_least_square})

    with open(results_file_path, "w") as results_file:
        json.dump(results, results_file)


def generate_tensormeas_calibration(results_file_path: str):
    """
    run the tensored measurement calibration circuits, calculates the fitter in few methods
    and saves the results
    The simulation results files will contain a list of dictionaries with the keys:
        cal_results - the results of the measurement calibration circuit
        results - results of a bell state circuit with noise
        mit_pattern - the mitigation pattern
        fidelity - the calculated fidelity of using this matrix
        results_pseudo_inverse - the result of using the psedo-inverse method on the bell state
        results_least_square - the result of using the least-squares method on the bell state

    Args:
        results_file_path: path of the json file of the results file
    """
    cal_results, mit_pattern, circuit_results = \
        tensored_calib_circ_execution(1000, SEED)

    meas_cal = TensoredMeasFitter(cal_results, mit_pattern=mit_pattern)
    meas_filter = meas_cal.filter

    # Calculate the results after mitigation
    results_pseudo_inverse = meas_filter.apply(
        circuit_results.get_counts(), method='pseudo_inverse')
    results_least_square = meas_filter.apply(
        circuit_results.get_counts(), method='least_squares')
    results = {"cal_results": cal_results.to_dict(),
               "results": circuit_results.to_dict(),
               "mit_pattern": mit_pattern,
               "fidelity": meas_cal.readout_fidelity(),
               "results_pseudo_inverse": results_pseudo_inverse,
               "results_least_square": results_least_square}

    with open(results_file_path, "w") as results_file:
        json.dump(results, results_file)

if __name__ == '__main__':
    DIRNAME = os.path.dirname(os.path.abspath(__file__))
    generate_meas_calibration(os.path.join(DIRNAME, 'test_meas_results.json'), 3)
    generate_tensormeas_calibration(os.path.join(DIRNAME, 'test_tensored_meas_results.json'))
