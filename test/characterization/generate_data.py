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
Generate data for characterization fitters tests
"""

import os
import json
import numpy as np

import qiskit

from qiskit.providers.aer.noise.errors.standard_errors import \
                                     thermal_relaxation_error

from qiskit.providers.aer.noise import NoiseModel

from qiskit.ignis.characterization.coherence import t1_circuits

from qiskit.ignis.characterization.coherence.fitters import T1Fitter, \
                                                            T2Fitter, \
                                                            T2StarFitter

from qiskit.ignis.characterization.hamiltonian.fitters import ZZFitter

# Fix seed for simulations
SEED = 9000


def t1_circuit_execution():
    """
    Create T1 circuits and simulate them.
    
    Return:
    - Backend result.
    - xdata.
    - Qubits for the T1 measurement.
    - T1 that was used in the circuits creation.
    """
    
    # 15 numbers ranging from 1 to 200, linearly spaced
    num_of_gates = (np.linspace(1, 200, 15)).astype(int)
    gate_time = 0.11
    qubits = [0]

    circs, xdata = t1_circuits(num_of_gates, gate_time, qubits)

    t1 = 10
    error = thermal_relaxation_error(t1, 2*t1, gate_time)
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error, 'id')
    # TODO: Include SPAM errors

    backend = qiskit.Aer.get_backend('qasm_simulator')
    shots = 100
    backend_result = qiskit.execute(
        circs, backend,
        shots=shots,
        seed_simulator=SEED,
        backend_options={'max_parallel_experiments': 0},
        noise_model=noise_model,
        optimization_level=0).result()

    return backend_result, xdata, qubits, t1

def generate_data_t1(filename):
    """
    Create T1 circuits and simulate them, then write the results in a json file.
    The file will contain a dictionary with the following keys:
    - 'backend_result', value is stored in the form of a dictionary.
    - 'xdata', value is stored as a list (and not as a numpy array).
    - 'qubits', these are the qubits for the T1 measurement.
    - 't1'

    Args:
       filename - name of the json file. 
    """

    backend_result, xdata, qubits, t1 = t1_circuit_execution()

    data = {
        'backend_result': backend_result.to_dict(),
        'xdata': xdata.tolist(),
        'qubits': qubits,
        't1': t1
        }

    with open(filename, 'w') as handle:
        json.dump(data, handle)

