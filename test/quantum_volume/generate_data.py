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
Generate data for quantum volume fitter tests
"""

import os
from test.utils import save_results_as_json
import qiskit
import qiskit.ignis.verification.quantum_volume as qv
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import depolarizing_error

# fixed seed for simulations
SEED = 42


def qv_circuit_execution(qubit_lists: list, ntrials: int, shots: int):
    """
    create quantum volume circuits, simulates the ideal state and run a noisy simulation
    Args:
        qubit_lists (list): list of lists of qubits to apply qv circuits to
        ntrials (int): number of iterations
        shots (int): number of shots per simulation

    Returns:
        tuple: a tuple of 2 lists:
            list of Results of the ideal statevector simulations
            list of Results of the noisy circuits simulations

    """
    # create the qv circuit
    qv_circs, qv_circs_nomeas = qv.qv_circuits(qubit_lists, ntrials)
    # get the ideal state
    statevector_backend = qiskit.Aer.get_backend('statevector_simulator')
    ideal_results = []
    for trial in range(ntrials):
        ideal_results.append(qiskit.execute(qv_circs_nomeas[trial],
                                            backend=statevector_backend).result())

    # define noise_model
    noise_model = NoiseModel()
    p1q = 0.002
    p2q = 0.02
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p1q, 1), 'u2')
    noise_model.add_all_qubit_quantum_error(depolarizing_error(2 * p1q, 1), 'u3')
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p2q, 2), 'cx')

    # get the noisy results
    backend = qiskit.Aer.get_backend('qasm_simulator')
    basis_gates = ['u1', 'u2', 'u3', 'cx']  # use U,CX for now
    exp_results = []
    for trial in range(ntrials):
        print('Running trial %d' % trial)
        exp_results.append(
            qiskit.execute(qv_circs[trial], basis_gates=basis_gates, backend=backend,
                           noise_model=noise_model, shots=shots,
                           seed_simulator=SEED,
                           backend_options={'max_parallel_experiments': 0}).result())

    return ideal_results, exp_results


def generate_qv_fitter_data(ideal_results_file_path: str, exp_results_file_path: str):
    """
    run the quantum volume circuits and saves the results
    The simulation results files will contain a list of Result objects in a dictionary format.

    Args:
        ideal_results_file_path: path of the json file of the ideal simulation results file
        exp_results_file_path: path of the json file of the noisy simulation results file
    """
    # parameters
    qubit_lists = [[0, 1, 3], [0, 1, 3, 5], [0, 1, 3, 5, 7],
                   [0, 1, 3, 5, 7, 10]]
    ntrials = 5
    # execute the circuit
    ideal_results, exp_results = qv_circuit_execution(qubit_lists, ntrials, shots=1024)
    # save the results
    save_results_as_json(ideal_results, ideal_results_file_path)
    save_results_as_json(exp_results, exp_results_file_path)


if __name__ == '__main__':
    DIRNAME = os.path.dirname(os.path.abspath(__file__))
    generate_qv_fitter_data(os.path.join(DIRNAME, 'qv_ideal_results.json'),
                            os.path.join(DIRNAME, 'qv_exp_results.json'))
