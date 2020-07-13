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
Generate data for rb fitters tests
"""

import os
import sys
import json
from test.utils import save_results_as_json, convert_ndarray_to_list_in_data
import numpy as np
import qiskit
import qiskit.ignis.verification.randomized_benchmarking as rb
from qiskit.ignis.verification.randomized_benchmarking import \
    RBFitter, InterleavedRBFitter, PurityRBFitter, CNOTDihedralRBFitter
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import depolarizing_error,\
    thermal_relaxation_error, coherent_unitary_error

# fixed seed for simulations and for rb random seed
SEED = 42


def create_depolarizing_noise_model():
    """
    create noise model of depolarizing error

    Returns:
        NoiseModel: depolarizing error noise model

    """
    noise_model = NoiseModel()
    p1q = 0.002
    p2q = 0.01
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p1q, 1), 'u2')
    noise_model.add_all_qubit_quantum_error(depolarizing_error(2 * p1q, 1), 'u3')
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p2q, 2), 'cx')
    return noise_model


def rb_circuit_execution(rb_opts: dict, shots: int):
    """
    Create rb circuits with depolarizing error and simulate them

    Args:
        rb_opts: the options for the rb circuits
        shots: number of shots for each circuit simulation

    Returns:
        list: list of Results of the circuits simulations
        list: the xdata of the rb circuit

    """
    # Load simulator
    backend = qiskit.Aer.get_backend('qasm_simulator')
    basis_gates = ['u1', 'u2', 'u3', 'cx']

    rb_circs, xdata = rb.randomized_benchmarking_seq(**rb_opts)

    noise_model = create_depolarizing_noise_model()

    results = []
    for circuit in rb_circs:
        results.append(qiskit.execute(circuit, backend=backend,
                                      basis_gates=basis_gates,
                                      shots=shots,
                                      noise_model=noise_model,
                                      seed_simulator=SEED).result())
    return results, xdata


def rb_circuit_execution_2(rb_opts: dict, shots: int):
    """
        Create rb circuits with T1 and T2 errors and simulate them

        Args:
            rb_opts: the options for the rb circuits
            shots: number of shots for each circuit simulation

        Returns:
            list: list of Results of the circuits simulations
            list: the xdata of the rb circuit

        """
    # Load simulator
    backend = qiskit.Aer.get_backend('qasm_simulator')
    basis_gates = ['u1', 'u2', 'u3', 'cx']

    rb_circs, xdata = rb.randomized_benchmarking_seq(**rb_opts)

    noise_model = NoiseModel()

    # Add T1/T2 noise to the simulation
    t_1 = 100.
    t_2 = 80.
    gate1q = 0.1
    gate2q = 0.5
    noise_model.add_all_qubit_quantum_error(thermal_relaxation_error(t_1, t_2, gate1q), 'u2')
    noise_model.add_all_qubit_quantum_error(thermal_relaxation_error(t_1, t_2, 2 * gate1q), 'u3')
    noise_model.add_all_qubit_quantum_error(
        thermal_relaxation_error(t_1, t_2, gate2q).tensor(
            thermal_relaxation_error(t_1, t_2, gate2q)), 'cx')

    results = []
    for circuit in rb_circs:
        results.append(qiskit.execute(circuit, backend=backend,
                                      basis_gates=basis_gates,
                                      shots=shots,
                                      noise_model=noise_model,
                                      seed_simulator=SEED).result())
    return results, xdata


def rb_interleaved_execution(rb_opts: dict, shots: int):
    """
        Create interleaved rb circuits with depolarizing error and simulate them
        Args:
            rb_opts: the options for the rb circuits
            shots: number of shots for each circuit simulation

        Returns:
            list: list of Results of the original circuits simulations
            list: the xdata of the rb circuit
            list: list of Results of the interleaved circuits simulations

        """
    # Load simulator
    backend = qiskit.Aer.get_backend('qasm_simulator')
    basis_gates = ['u1', 'u2', 'u3', 'cx']

    rb_original_circs, xdata, rb_interleaved_circs = rb.randomized_benchmarking_seq(**rb_opts)

    noise_model = create_depolarizing_noise_model()

    results = []
    for circuit in rb_original_circs:
        results.append(qiskit.execute(circuit, backend=backend,
                                      basis_gates=basis_gates,
                                      shots=shots,
                                      noise_model=noise_model,
                                      seed_simulator=SEED).result())
    interleaved_results = []
    for circuit in rb_interleaved_circs:
        interleaved_results.append(qiskit.execute(circuit, backend=backend,
                                                  basis_gates=basis_gates,
                                                  shots=shots, noise_model=noise_model,
                                                  seed_simulator=SEED).result())
    return results, xdata, interleaved_results


def rb_cnotdihedral_execution(rb_opts: dict, shots: int):
    """
        Create cnot-dihedral rb circuits with depolarizing errors and simulate them

        Args:
            rb_opts: the options for the rb circuits
            shots: number of shots for each circuit simulation

        Returns:
            list: list of Results of the cnot-dihedral circuits simulations in the x plane
            list: the xdata of the rb circuit
            list: list of Results of the cnot-dihedral circuits simulations in the z plane

        """
    # Load simulator
    backend = qiskit.Aer.get_backend('qasm_simulator')
    basis_gates = ['u1', 'u2', 'u3', 'cx']

    rb_cnotdihedral_z_circs, xdata, rb_cnotdihedral_x_circs = \
        rb.randomized_benchmarking_seq(**rb_opts)

    noise_model = create_depolarizing_noise_model()

    cnotdihedral_x_results = []
    for circuit in rb_cnotdihedral_x_circs:
        cnotdihedral_x_results.append(qiskit.execute(circuit, backend=backend,
                                                     basis_gates=basis_gates,
                                                     shots=shots,
                                                     noise_model=noise_model,
                                                     seed_simulator=SEED).result())
    cnotdihedral_z_results = []
    for circuit in rb_cnotdihedral_z_circs:
        cnotdihedral_z_results.append(qiskit.execute(circuit, backend=backend,
                                                     basis_gates=basis_gates,
                                                     shots=shots, noise_model=noise_model,
                                                     seed_simulator=SEED).result())
    return cnotdihedral_x_results, xdata, cnotdihedral_z_results


def rb_purity_circuit_execution(rb_opts: dict, shots: int):
    """
        Create purity rb circuits with depolarizing errors and simulate them

        Args:
            rb_opts: the options for the rb circuits
            shots: number of shots for each circuit simulation

        Returns:
            list: list of Results of the purity circuits simulations
            list: the xdata of the rb circuit
            int: npurity (3^(number of qubits))
            list: list of Results of the coherent circuits simulations

        """
    # Load simulator
    backend = qiskit.Aer.get_backend('qasm_simulator')
    basis_gates = ['u1', 'u2', 'u3', 'cx']

    rb_purity_circs, xdata, npurity = rb.randomized_benchmarking_seq(**rb_opts)

    noise_model = create_depolarizing_noise_model()

    # coherent noise
    err_unitary = np.zeros([2, 2], dtype=complex)
    angle_err = 0.1
    for i in range(2):
        err_unitary[i, i] = np.cos(angle_err)
        err_unitary[i, (i + 1) % 2] = np.sin(angle_err)
    err_unitary[0, 1] *= -1.0

    error = coherent_unitary_error(err_unitary)
    coherent_noise_model = NoiseModel()
    coherent_noise_model.add_all_qubit_quantum_error(error, 'u3')

    purity_results = []
    coherent_results = []
    for circuit_list in rb_purity_circs:
        for purity_num in range(npurity):
            current_circ = circuit_list[purity_num]
            # non-coherent purity results
            purity_results.append(qiskit.execute(current_circ, backend=backend,
                                                 basis_gates=basis_gates,
                                                 shots=shots,
                                                 noise_model=noise_model,
                                                 seed_simulator=SEED).result())
            # coherent purity results
            # THE FITTER IS NOT TESTED YET
            coherent_results.append(qiskit.execute(current_circ, backend=backend,
                                                   basis_gates=basis_gates,
                                                   shots=shots,
                                                   noise_model=coherent_noise_model,
                                                   seed_simulator=SEED).result())

    return purity_results, xdata, npurity, coherent_results


def generate_fitter_data_1(results_file_path: str, expected_results_file_path: str):
    """
    Create rb circuits with depolarizing error and simulate them,
    then write the results in a json file.
    also creates fitter for the data results,
    and also write the fitter results in another json file.

    The simulation results file will contain a list of Result objects in a dictionary format.
    The fitter data file will contain dictionary with the following keys:
    - 'ydata', value is stored in the form of list of dictionaries with keys of:
        - mean, list of integers
        - std, list of integers
    - fit, value is stored in the form of list of dictionaries with keys of:
        - params, list of integers
        - params_err, list of integers
        - epc, list of integers
        - epc_err, list of integers

    Args:
        results_file_path: path of the json file of the simulation results file
        expected_results_file_path: path of the json file of the fitter results file
    """

    rb_opts = {}
    shots = 1024
    rb_opts['nseeds'] = 5
    rb_opts['rb_pattern'] = [[0, 1], [2]]
    rb_opts['length_multiplier'] = [1, 2]
    rb_opts['length_vector'] = np.arange(1, 200, 20)
    rb_opts['rand_seed'] = SEED

    rb_results, xdata = rb_circuit_execution(rb_opts, shots)
    save_results_as_json(rb_results, results_file_path)

    # generate also the expected results of the fitter
    rb_fit = RBFitter(rb_results, xdata, rb_opts['rb_pattern'])
    ydata = rb_fit.ydata
    fit = rb_fit.fit
    # convert ndarray to list
    ydata = convert_ndarray_to_list_in_data(ydata)
    fit = convert_ndarray_to_list_in_data(fit)

    expected_result = {"ydata": ydata, "fit": fit}
    with open(expected_results_file_path, "w") as expected_results_file:
        json.dump(expected_result, expected_results_file)


def generate_fitter_data_2(results_file_path: str, expected_results_file_path: str):
    """
        Create rb circuits with T1 and T2 errors and simulate them,
        then write the results in a json file.
        also creates fitter for the data results,
        and also write the fitter results in another json file.

        The simulation results file will contain a list of Result objects in a dictionary format.
        The fitter data file will contain dictionary with the following keys:
        - 'ydata', value is stored in the form of list of dictionaries with keys of:
            - mean, list of integers
            - std, list of integers
        - fit, value is stored in the form of list of dictionaries with keys of:
            - params, list of integers
            - params_err, list of integers
            - epc, list of integers
            - epc_err, list of integers

        Args:
            results_file_path: path of the json file of the simulation results file
            expected_results_file_path: path of the json file of the fitter results file
        """
    rb_opts = {}
    shots = 1024
    rb_opts['nseeds'] = 5
    rb_opts['rb_pattern'] = [[0]]
    rb_opts['length_vector'] = np.arange(1, 200, 20)
    rb_opts['rand_seed'] = SEED

    rb_results, xdata = rb_circuit_execution_2(rb_opts, shots)
    save_results_as_json(rb_results, results_file_path)

    # generate also the expected results of the fitter
    rb_fit = RBFitter(rb_results, xdata, rb_opts['rb_pattern'])
    ydata = rb_fit.ydata
    fit = rb_fit.fit
    # convert ndarray to list
    ydata = convert_ndarray_to_list_in_data(ydata)
    fit = convert_ndarray_to_list_in_data(fit)

    expected_result = {"ydata": ydata, "fit": fit}
    with open(expected_results_file_path, "w") as expected_results_file:
        json.dump(expected_result, expected_results_file)


def generate_interleaved_data(results_file_path_original: str,
                              results_file_path_interleaved: str,
                              expected_results_file_path: str):
    """
        Create interleaved rb circuits with depolarizing error and simulate them,
         then write the results in a json file.
        also creates fitter for the data results,
        and also write the fitter results in another json file.

        The simulation results file will contain a list of Result objects in a dictionary format.
        The fitter data file will contain dictionary with the following keys:
        - 'original_ydata', value is stored in the form of list of dictionaries with keys of:
            - mean, list of integers
            - std, list of integers
        - 'interleaved_ydata', value is stored in the form of list of dictionaries with keys of:
            - mean, list of integers
            - std, list of integers
        - joint_fit, value is stored in the form of list of dictionaries with keys of:
            - alpha, list of integers
            - alpha_err, list of integers
            - alpha_c, list of integers
            - alpha_c_err, list of integers
            - epc_est, list of integers
            - epc_est_err, list of integers
            - systematic_err, list of integers
            - systematic_err_L, list of integers
            - systematic_err_R, list of integers

        Args:
            results_file_path_original: path of the json file of the simulation results file
                of the original circuits
            results_file_path_interleaved: path of the json file of the simulation results file
                of the interleaved circuits
            expected_results_file_path: path of the json file of the fitter results file
        """

    rb_opts = {}
    shots = 200
    rb_opts['nseeds'] = 2
    rb_opts['rb_pattern'] = [[0, 2], [1]]
    rb_opts['length_vector'] = np.arange(1, 100, 10)
    rb_opts['length_multiplier'] = [1, 3]
    # create interleaved elem of the form [['x 0', 'x 1', 'cx 0 1'], ['x 0']]
    qc1 = qiskit.QuantumCircuit(2)
    qc1.x(0)
    qc1.x(1)
    qc1.cx(0, 1)
    qc2 = qiskit.QuantumCircuit(1)
    qc2.x(0)
    rb_opts["interleaved_elem"] = [qc1, qc2]
    rb_opts['rand_seed'] = SEED

    results, xdata, interleaved_results = rb_interleaved_execution(rb_opts, shots)
    save_results_as_json(results, results_file_path_original)
    save_results_as_json(interleaved_results, results_file_path_interleaved)

    # generate also the expected results of the fitter
    joint_rb_fit = InterleavedRBFitter(results, interleaved_results, xdata,
                                       rb_opts['rb_pattern'])
    joint_fit = joint_rb_fit.fit_int
    original_ydata = joint_rb_fit.ydata[0]
    interleaved_ydata = joint_rb_fit.ydata[1]
    # convert ndarray to list
    original_ydata = convert_ndarray_to_list_in_data(original_ydata)
    interleaved_ydata = convert_ndarray_to_list_in_data(interleaved_ydata)
    joint_fit = convert_ndarray_to_list_in_data(joint_fit)

    expected_result = {"original_ydata": original_ydata,
                       "interleaved_ydata": interleaved_ydata, "joint_fit": joint_fit}
    with open(expected_results_file_path, "w") as expected_results_file:
        json.dump(expected_result, expected_results_file)


def generate_cnotdihedral_data(results_file_path_cnotdihedral_x: str,
                               results_file_path_cnotdihedral_z: str,
                               expected_results_file_path: str):
    """
        Create cnot-dihedral rb circuits with depolarizing error and simulate them,
         then write the results in a json file (separate files for x and z).
        also creates fitter for the data results,
        and also write the fitter results in another json file.

        The simulation results file will contain a list of Result objects in a dictionary format.
        The fitter data file will contain dictionary with the following keys:
        - 'cnotdihedral_Z_ydata', value is stored in the form of list of dictionaries with keys of:
            - mean, list of integers
            - std, list of integers
        - 'cnotdihedral_X_ydata', value is stored in the form of list of dictionaries with keys of:
            - mean, list of integers
            - std, list of integers
        - joint_fit, value is stored in the form of list of dictionaries with keys of:
            - alpha, list of integers
            - alpha_err, list of integers
            - epc_est, list of integers
            - epc_est_err, list of integers

        Args:
            results_file_path_cnotdihedral_z: path of the json file of the simulation results file
                of the cnot-dihedral in the z plane circuits
            results_file_path_cnotdihedral_x: path of the json file of the simulation results file
                of the cnot-dihedral in the x plane circuits
            expected_results_file_path: path of the json file of the fitter results file
        """

    rb_opts = {}
    shots = 200
    rb_opts['nseeds'] = 5
    rb_opts['rb_pattern'] = [[0, 2], [1]]
    rb_opts['length_vector'] = np.arange(1, 200, 20)
    rb_opts['length_multiplier'] = [1, 3]
    rb_opts['group_gates'] = 'CNOT-Dihedral'
    rb_opts['rand_seed'] = SEED

    cnotdihedral_x_results, xdata, cnotdihedral_z_results = \
        rb_cnotdihedral_execution(rb_opts, shots)
    save_results_as_json(cnotdihedral_x_results, results_file_path_cnotdihedral_x)
    save_results_as_json(cnotdihedral_z_results, results_file_path_cnotdihedral_z)

    # generate also the expected results of the fitter
    joint_rb_fit = CNOTDihedralRBFitter(cnotdihedral_z_results, cnotdihedral_x_results, xdata,
                                        rb_opts['rb_pattern'])

    joint_fit = joint_rb_fit.fit_cnotdihedral
    cnotdihedral_z_ydata = joint_rb_fit.ydata[0]
    cnotdihedral_x_ydata = joint_rb_fit.ydata[1]
    # convert ndarray to list
    cnotdihedral_x_ydata = convert_ndarray_to_list_in_data(cnotdihedral_x_ydata)
    cnotdihedral_z_ydata = convert_ndarray_to_list_in_data(cnotdihedral_z_ydata)
    joint_fit = convert_ndarray_to_list_in_data(joint_fit)

    expected_result = {"cnotdihedral_Z_ydata": cnotdihedral_z_ydata,
                       "cnotdihedral_X_ydata": cnotdihedral_x_ydata, "joint_fit": joint_fit}
    with open(expected_results_file_path, "w") as expected_results_file:
        json.dump(expected_result, expected_results_file)


def generate_purity_data(results_file_path_purity: str,
                         results_file_path_coherent: str,
                         expected_results_file_path_purity: str,
                         expected_results_file_path_coherent: str):
    """
        Create purity rb circuits with depolarizing error and simulate them,
         then write the results in a json file (separate files for purity and coherent).
        also creates fitter for the data results,
        and also write the fitter results in another json file (separate files for purity
         and coherent).

        The simulation results file will contain a list of Result objects in a dictionary format.
        The fitter data file will contain dictionary with the following keys:
        - 'ydata', value is stored in the form of list of dictionaries with keys of:
            - mean, list of integers
            - std, list of integers
        - fit, value is stored in the form of list of dictionaries with keys of:
            - params, list of integers
            - params_err, list of integers
            - epc, list of integers
            - epc_err, list of integers
            - pepc, list of integers
            - pepc_err, list of integers

        Args:
            results_file_path_purity: path of the json file of the purity simulation results file
            results_file_path_coherent: path of the json file of the coherent simulation results
            file
            expected_results_file_path_purity: path of the json file of the purity fitter
            results file
            expected_results_file_path_coherent: path of the json file of the coherent fitter
            results file
        """
    rb_opts = {}
    shots = 200
    rb_opts['nseeds'] = 3
    rb_opts['rb_pattern'] = [[0, 1]]
    rb_opts['length_vector'] = np.arange(1, 200, 20)
    rb_opts['is_purity'] = True
    rb_opts['rand_seed'] = SEED

    rb_purity_results, xdata, npurity, rb_coherent_results = \
        rb_purity_circuit_execution(rb_opts, shots)

    # save the results
    save_results_as_json(rb_purity_results, results_file_path_purity)
    # COHERENT FITTER IS NOT TESTED YET
    save_results_as_json(rb_coherent_results, results_file_path_coherent)

    # generate also the expected results of the fitter
    rbfit_purity = PurityRBFitter(rb_purity_results, npurity, xdata, rb_opts['rb_pattern'])

    fit = rbfit_purity.fit
    ydata = rbfit_purity.ydata

    # convert ndarray to list
    ydata = convert_ndarray_to_list_in_data(ydata)
    fit = convert_ndarray_to_list_in_data(fit)

    expected_result = {"ydata": ydata,
                       "fit": fit}
    with open(expected_results_file_path_purity, "w") as expected_results_file:
        json.dump(expected_result, expected_results_file)

    # generate also the expected results of the fitter for coherent
    rbfit_coherent = rb.PurityRBFitter(rb_coherent_results, npurity, xdata, rb_opts['rb_pattern'])

    coherent_fit = rbfit_coherent.fit
    coherent_ydata = rbfit_coherent.ydata

    # convert ndarray to list
    coherent_ydata = convert_ndarray_to_list_in_data(coherent_ydata)
    coherent_fit = convert_ndarray_to_list_in_data(coherent_fit)

    coherent_expected_result = {"ydata": coherent_ydata, "fit": coherent_fit}
    # COHERENT FITTER IS NOT TESTED YET
    with open(expected_results_file_path_coherent, "w") as expected_results_file:
        json.dump(coherent_expected_result, expected_results_file)


if __name__ == '__main__':
    DIRNAME = os.path.dirname(os.path.abspath(__file__))
    for rb_type in sys.argv[1:]:
        if rb_type == 'standard':
            generate_fitter_data_1(os.path.join(DIRNAME, 'test_fitter_results_1.json'),
                                   os.path.join(DIRNAME, 'test_fitter_expected_results_1.json'))
            generate_fitter_data_2(os.path.join(DIRNAME, 'test_fitter_results_2.json'),
                                   os.path.join(DIRNAME, 'test_fitter_expected_results_2.json'))
        elif rb_type == 'interleaved':
            generate_interleaved_data(os.path.join(DIRNAME, 'test_fitter_original_results.json'),
                                      os.path.join(DIRNAME,
                                                   'test_fitter_interleaved_results.json'),
                                      os.path.join(DIRNAME,
                                                   'test_fitter_interleaved_'
                                                   'expected_results.json'))
        elif rb_type == 'cnotdihedral':
            generate_cnotdihedral_data(os.path.join(DIRNAME,
                                                    'test_fitter_cnotdihedral_X_results.json'),
                                       os.path.join(DIRNAME,
                                                    'test_fitter_cnotdihedral_Z_results.json'),
                                       os.path.join(DIRNAME,
                                                    'test_fitter_cnotdihedral_'
                                                    'expected_results.json'))
        elif rb_type == 'purity':
            generate_purity_data(os.path.join(DIRNAME, 'test_fitter_purity_results.json'),
                                 os.path.join(DIRNAME, 'test_fitter_coherent_purity_results.json'),
                                 os.path.join(DIRNAME, 'test_fitter_purity_expected_results.json'),
                                 os.path.join(DIRNAME,
                                              'test_fitter_coherent_purity_expected_results.json'))
        else:
            print('Skipping unknown argument ' + rb_type)
