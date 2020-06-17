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

# pylint: disable=no-member,invalid-name

"""
Generate data for accreditation tests
"""

# Import general libraries (needed for functions)
import json
import qiskit


# Import Qiskit classes
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import depolarizing_error

# Import the accreditation functions.
from qiskit.ignis.verification.accreditation import AccreditationFitter, AccreditationCircuits


def make_accred_system(seed=None):
    """generate accred circuits"""
    # Create a Quantum Register with n_qb qubits.
    q_reg = QuantumRegister(4, 'q')
    # Create a Classical Register with n_qb bits.
    c_reg = ClassicalRegister(4, 's')
    # Create a Quantum Circuit acting on the q register
    target_circuit = QuantumCircuit(q_reg, c_reg)
    target_circuit.h(0)
    target_circuit.h(1)
    target_circuit.h(2)
    target_circuit.h(3)
    target_circuit.cx(0, 1)
    target_circuit.cx(0, 2)
    target_circuit.h(1)
    target_circuit.h(2)
    target_circuit.cx(0, 3)
    target_circuit.cx(1, 2)
    target_circuit.h(1)
    target_circuit.h(2)
    target_circuit.h(3)
    target_circuit.measure(q_reg, c_reg)

    return AccreditationCircuits(target_circuit, seed=seed)


def generate_data_ideal():
    """generaete a ideal data and save it as json"""
    seed_accreditation = 134780132

    accsys = make_accred_system(seed=seed_accreditation)
    simulator = qiskit.Aer.get_backend('qasm_simulator')

    # Number of runs
    d = 20
    v = 10

    all_results = []
    all_postp_list = []
    all_v_zero = []
    for run in range(d):
        print([run, d])
        # Create target and trap circuits with random Pauli gates
        circuit_list, postp_list, v_zero = accsys.generate_circuits(v)

        job = execute(circuit_list,
                      simulator,
                      shots=1)
        all_results.append(job.result())
        all_postp_list.append([postp.tolist() for postp in postp_list])
        all_v_zero.append(v_zero)
    outputdict = {'all_results': [result.to_dict() for result in all_results],
                  'all_postp_list': all_postp_list,
                  'all_v_zero': all_v_zero}
    with open('accred_ideal_results.json', "w") as results_file:
        json.dump(outputdict, results_file)


def generate_data_noisy():
    """generaete a noisy data and save it as json"""
    seed_accreditation = 1435754
    seed_simulator = 877924554

    accsys = make_accred_system(seed=seed_accreditation)
    simulator = qiskit.Aer.get_backend('qasm_simulator')
    noise_model = NoiseModel()
    p1q = 0.002
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p1q, 1), 'u1')
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p1q, 1), 'u2')
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p1q, 1), 'u3')
    p2q = 0.02
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p2q, 2), 'cx')

    basis_gates = ['u1', 'u2', 'u3', 'cx']
    # Number of runs
    d = 20
    v = 10

    test_3 = AccreditationFitter()
    all_results = []
    all_postp_list = []
    all_v_zero = []
    all_acc = []
    for run in range(d):
        print([run, d])
        # Create target and trap circuits with random Pauli gates
        circuit_list, postp_list, v_zero = accsys.generate_circuits(v)
        # Implement all these circuits with noise
        job = execute(circuit_list, simulator,
                      noise_model=noise_model, basis_gates=basis_gates,
                      shots=1, seed_simulator=seed_simulator + run)
        all_results.append(job.result())
        all_postp_list.append([postp.tolist() for postp in postp_list])
        all_v_zero.append(v_zero)
        # Post-process the outputs and see if the protocol accepts
        test_3.single_protocol_run(job.result(), postp_list, v_zero)
        all_acc.append(test_3.flag)

    theta = 5/100
    test_3.bound_variation_distance(theta)
    bound = test_3.bound
    outputdict = {'all_results': [result.to_dict() for result in all_results],
                  'all_postp_list': all_postp_list,
                  'all_v_zero': all_v_zero,
                  'all_acc': all_acc,
                  'theta': theta,
                  'bound': bound}
    with open('accred_noisy_results.json', "w") as results_file:
        json.dump(outputdict, results_file)


if __name__ == '__main__':
    generate_data_ideal()
    generate_data_noisy()
