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
     (thermal_relaxation_error,
      coherent_unitary_error)

from qiskit.providers.aer.noise import NoiseModel

from qiskit.ignis.characterization.coherence import (t1_circuits,
                                                     t2_circuits,
                                                     t2star_circuits)

from qiskit.ignis.characterization.coherence.fitters import (T1Fitter,
                                                            T2Fitter,
                                                            T2StarFitter)

from qiskit.ignis.characterization.hamiltonian import zz_circuits, ZZFitter

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

def t2_circuit_execution():
    """
    Create T2 circuits and simulate them.
    
    Return:
    - Backend result.
    - xdata.
    - Qubits for the T2 measurement.
    - T2 that was used in the circuits creation.
    """

    num_of_gates = (np.linspace(1, 30, 10)).astype(int)
    gate_time = 0.11
    qubits = [0]
    n_echos = 5
    alt_phase_echo = True

    circs, xdata = t2_circuits(num_of_gates, gate_time, qubits,
                               n_echos, alt_phase_echo)

    t2 = 20
    error = thermal_relaxation_error(np.inf, t2, gate_time, 0.5)
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

    return backend_result, xdata, qubits, t2

def generate_data_t2(filename):
    """
    Create T2 circuits and simulate them, then write the results in a json file.
    The file will contain a dictionary with the following keys:
    - 'backend_result', value is stored in the form of a dictionary.
    - 'xdata', value is stored as a list (and not as a numpy array).
    - 'qubits', these are the qubits for the T2 measurement.
    - 't2'

    Args:
       filename - name of the json file. 
    """

    backend_result, xdata, qubits, t2 = t2_circuit_execution()

    data = {
        'backend_result': backend_result.to_dict(),
        'xdata': xdata.tolist(),
        'qubits': qubits,
        't2': t2
        }

    with open(filename, 'w') as handle:
        json.dump(data, handle)

def t2star_circuit_execution():
    """
    Create T2* circuits and simulate them.
    
    Return:
    - Backend result.
    - xdata.
    - Qubits for the T2* measurement.
    - T2* that was used in the circuits creation.
    - Frequency.
    """

    # Setting parameters

    num_of_gates = np.append(
        (np.linspace(10, 150, 10)).astype(int),
        (np.linspace(160, 450, 5)).astype(int))
    gate_time = 0.1
    qubits = [0]

    t2 = 10
    error = thermal_relaxation_error(np.inf, t2, gate_time, 0.5)
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error, 'id')

    backend = qiskit.Aer.get_backend('qasm_simulator')
    shots = 200

    # Estimate T2* via an oscilliator function
    circs_osc, xdata, omega = t2star_circuits(num_of_gates, gate_time,
                                              qubits, 5)

    backend_result = qiskit.execute(
            circs_osc, backend,
            shots=shots,
            seed_simulator=SEED,
            backend_options={'max_parallel_experiments': 0},
            noise_model=noise_model,
            optimization_level=0).result()

    return backend_result, xdata, qubits, t2, omega

def generate_data_t2star(filename):
    """
    Create T2* circuits and simulate them, then write the results in a json file.
    The file will contain a dictionary with the following keys:
    - 'backend_result', value is stored in the form of a dictionary.
    - 'xdata', value is stored as a list (and not as a numpy array).
    - 'qubits', these are the qubits for the T2 measurement.
    - 't2'
    - 'omega'

    Args:
       filename - name of the json file. 
    """

    backend_result, xdata, qubits, t2, omega = t2star_circuit_execution()

    data = {
        'backend_result': backend_result.to_dict(),
        'xdata': xdata.tolist(),
        'qubits': qubits,
        't2': t2,
        'omega': omega
        }

    with open(filename, 'w') as handle:
        json.dump(data, handle)

def zz_circuit_execution():
    """
    Create ZZ circuits and simulate them.
    
    Return:
    - Backend result.
    - xdata.
    - Qubits for the ZZ measurement.
    - Spectators.
    - ZZ parameter that used in the circuit creation
    - Frequency.
    """

    num_of_gates = np.arange(0, 60, 10)
    gate_time = 0.1
    qubits = [0]
    spectators = [1]

    # Generate experiments
    circs, xdata, omega = zz_circuits(num_of_gates,
                                      gate_time, qubits,
                                      spectators, nosc=2)

    # Set the simulator with ZZ
    zz = 0.1
    zz_unitary = np.eye(4, dtype=complex)
    zz_unitary[3, 3] = np.exp(1j*2*np.pi*zz*gate_time)
    error = coherent_unitary_error(zz_unitary)
    noise_model = NoiseModel()
    noise_model.add_nonlocal_quantum_error(error, 'id', [0], [0, 1])

    # Run the simulator
    backend = qiskit.Aer.get_backend('qasm_simulator')
    shots = 100

    backend_result = qiskit.execute(circs, backend,
                                    shots=shots,
                                    seed_simulator=SEED,
                                    noise_model=noise_model,
                                    optimization_level=0).result()

    return backend_result, xdata, qubits, spectators, zz, omega

def generate_data_zz(filename):
    """
    Create ZZ circuits and simulate them, then write the results in a json file.
    The file will contain a dictionary with the following keys:
    - 'backend_result', value is stored in the form of a dictionary.
    - 'xdata', value is stored as a list (and not as a numpy array).
    - 'qubits', these are the qubits for the ZZ measurement.
    - 'spectators'
    - 'zz'
    - 'omega'

    Args:
       filename - name of the json file. 
    """

    backend_result, xdata, qubits, spectators, zz, omega = zz_circuit_execution()

    data = {
        'backend_result': backend_result.to_dict(),
        'xdata': xdata.tolist(),
        'qubits': qubits,
        'spectators': spectators,
        'zz': zz,
        'omega': omega
        }

    with open(filename, 'w') as handle:
        json.dump(data, handle)

generate_data_zz('zz_data.json')
