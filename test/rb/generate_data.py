from test.utils import *
import numpy as np
import qiskit
import qiskit.ignis.verification.randomized_benchmarking as rb
from qiskit.ignis.verification.randomized_benchmarking import \
    RBFitter, InterleavedRBFitter, PurityRBFitter, CNOTDihedralRBFitter
import matplotlib.pyplot as plt
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import depolarizing_error,\
    thermal_relaxation_error

# fixed seed for simulations
SEED = 42

def rb_circuit_execution(rb_opts, shots):
    """
    Create rb circuits with depolarizing error and simulates them

    Args:
        rb_opts: the options for the rb circuits
        shots: number of shots for each circuit simulation

    Returns:
        list of Results of the circuits simulations
        the xdata of the rb circuit

    """
    # Load simulator
    backend = qiskit.Aer.get_backend('qasm_simulator')
    basis_gates = ['u1', 'u2', 'u3', 'cx']

    rb_circs, xdata = rb.randomized_benchmarking_seq(**rb_opts)
    rb_result = rb.randomized_benchmarking_seq(**rb_opts)

    noise_model = NoiseModel()
    p1Q = 0.002
    p2Q = 0.01
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p1Q, 1), 'u2')
    noise_model.add_all_qubit_quantum_error(depolarizing_error(2 * p1Q, 1), 'u3')
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p2Q, 2), 'cx')

    results = []
    for circuit in rb_circs:
        results.append(qiskit.execute(circuit, backend=backend,
                                      basis_gates=basis_gates,
                                      shots=shots,
                                      noise_model=noise_model,
                                      seed_simulator=SEED).result())
    return results, xdata

def rb_circuit_execution_2(rb_opts, shots):
    """
        Create rb circuits with T1 and T2 errors and simulates them

        Args:
            rb_opts: the options for the rb circuits
            shots: number of shots for each circuit simulation

        Returns:
            list of Results of the circuits simulations
            the xdata of the rb circuit

        """
    # Load simulator
    backend = qiskit.Aer.get_backend('qasm_simulator')
    basis_gates = ['u1', 'u2', 'u3', 'cx']

    rb_circs, xdata = rb.randomized_benchmarking_seq(**rb_opts)
    rb_result = rb.randomized_benchmarking_seq(**rb_opts)

    noise_model = NoiseModel()

    # Add T1/T2 noise to the simulation
    t1 = 100.
    t2 = 80.
    gate1Q = 0.1
    gate2Q = 0.5
    noise_model.add_all_qubit_quantum_error(thermal_relaxation_error(t1, t2, gate1Q), 'u2')
    noise_model.add_all_qubit_quantum_error(thermal_relaxation_error(t1, t2, 2 * gate1Q), 'u3')
    noise_model.add_all_qubit_quantum_error(
        thermal_relaxation_error(t1, t2, gate2Q).tensor(
            thermal_relaxation_error(t1, t2, gate2Q)), 'cx')

    results = []
    for circuit in rb_circs:
        results.append(qiskit.execute(circuit, backend=backend,
                                      basis_gates=basis_gates,
                                      shots=shots,
                                      noise_model=noise_model,
                                      seed_simulator=SEED).result())
    return results, xdata

def rb_interleaved_circuit_execution(rb_opts, shots):
    """
        Create interleaved rb circuits with depolarizing error and simulates them
        Args:
            rb_opts: the options for the rb circuits
            shots: number of shots for each circuit simulation

        Returns:
            list of Results of the original circuits simulations
            the xdata of the rb circuit
            list of Results of the interleaved circuits simulations

        """
    # Load simulator
    backend = qiskit.Aer.get_backend('qasm_simulator')
    basis_gates = ['u1', 'u2', 'u3', 'cx']

    rb_original_circs, xdata, rb_interleaved_circs = rb.randomized_benchmarking_seq(**rb_opts)
    rb_result = rb.randomized_benchmarking_seq(**rb_opts)

    noise_model = NoiseModel()
    p1Q = 0.002
    p2Q = 0.01
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p1Q, 1), 'u2')
    noise_model.add_all_qubit_quantum_error(depolarizing_error(2 * p1Q, 1), 'u3')
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p2Q, 2), 'cx')

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
                                           shots=shots, noise_model=noise_model).result())
    return results, xdata, interleaved_results

def generate_fitter_data_1(results_file_path, expected_results_file_path):
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
    rb_opts['length_vector'] = np.arange(1,200,20)

    rb_results, xdata = rb_circuit_execution(rb_opts, shots)
    save_results_as_json(results, results_file_path)

    # generate also the expected results of the fitter
    rb_fit = RBFitter(results_1_list, xdata, rb_opts['rb_pattern'])
    ydata = rb_fit.ydata
    fit = rb_fit.fit
    # convert ndarray to list
    ydata = convert_ndarray_to_list_in_data(ydata)
    fit = convert_ndarray_to_list_in_data(fit)

    expected_result = {"ydata": ydata, "fit": fit}
    with open(expected_results_file_path, "w") as expected_results_file:
        json.dump(expected_result, expected_results_file)


def generate_fitter_data_2(results_file_path, expected_results_file_path):
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
    rb_opts['length_vector'] = np.arange(1,200,20)

    rb_results, xdata = rb_circuit_execution_2(rb_opts, shots)
    save_results_as_json(results, results_file_path)

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

def generate_interleaved_fitter_data(results_file_path_original, results_file_path_interleaved,
                                     expected_results_file_path):
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
    rb_opts['length_vector'] = np.arange(1,100,10)
    rb_opts['length_multiplier'] = [1, 3]
    rb_opts['interleaved_gates'] = [['x 0', 'x 1', 'cx 0 1'], ['x 0']]

    results, xdata, interleaved_results = rb_interleaved_circuit_execution(rb_opts, shots)
    save_results_as_json(results, results_file_path_original)
    save_results_as_json(interleaved_results, results_file_path_interleaved)

    # generate also the expected results of the fitter
    joint_rb_fit = rb.InterleavedRBFitter(results, intr_results, xdata, rb_opts['rb_pattern'])
    joint_fit = joint_rb_fit.fit_int
    original_ydata = joint_rb_fit.ydata[0]
    interleaved_ydata = joint_rb_fit.ydata[1]
    # convert ndarray to list
    original_ydata = convert_ndarray_to_list_in_data(original_ydata)
    interleaved_ydata = convert_ndarray_to_list_in_data(interleaved_ydata)
    joint_fit = convert_ndarray_to_list_in_data(joint_fit)

    expected_result = {"original_ydata": original_ydata, "interleaved_ydata": interleaved_ydata,  "joint_fit": joint_fit}
    with open(expected_results_file_path, "w") as expected_results_file:
        json.dump(expected_result, expected_results_file)

def generate_purity_fitter_data(results_file_path_purity, results_file_path_coherent):
    rb_opts = {}
    shots = 200
    rb_opts['nseeds'] = 3
    rb_opts['rb_pattern'] = [[0, 1], [2, 3]]
    rb_opts['length_vector'] = [1, 21, 41, 61, 81, 101, 121, 141, 161, 181]
    rb_opts['is_purity'] = True
    # purity rb
    generate_data(rb_opts, results_file_path_purity, shots)

    # coherent purity rb
    generate_data(rb_opts, results_file_path_coherent, shots)

def generate_cnotdihedral_fitter_data(results_file_path_cnotdihedral_X,
                                      results_file_path_cnotdihedral_Z):
    rb_opts = {}
    shots = 200
    rb_opts['nseeds'] = 2
    rb_opts['rb_pattern'] = [[0, 2], [1]]
    rb_opts['length_vector'] = [1, 21, 41, 61, 81, 101, 121, 141, 161, 181]
    rb_opts['length_multiplier'] = [1, 3]
    rb_opts['group_gates'] = 'CNOT-Dihedral'
    # cnotdihedral_X rb
    generate_data(rb_opts, results_file_path_cnotdihedral_X, shots)

    # cnotdihedral_Z rb
    generate_data(rb_opts, results_file_path_cnotdihedral_Z, shots)

def convert_ndarray_to_list_in_data(data):
    new_data = []
    for item in data:
        new_item = {}
        for key, value in item.items():
            new_item[key] = value.tolist()
        new_data.append(new_item)

    return new_data

def generate_fitter_expected_data(results_1_file_path, results_2_file_path,
                                  save_1_path, save_2_path):
    # first test
    # adjust the parameters
    xdata = np.array([[1, 21, 41, 61, 81, 101, 121, 141, 161, 181],
                       [2, 42, 82, 122, 162, 202, 242, 282, 322, 362]])
    rb_pattern = [[0, 1], [2]]

    # create the results
    results_1_list = load_results_from_json(results_1_file_path)
    rb_fit_1 = RBFitter(results_1_list, xdata, rb_pattern)
    ydata_1 = rb_fit_1.ydata
    fit_1 = rb_fit_1.fit
    # convert ndarray to list
    ydata_1 = convert_ndarray_to_list_in_data(ydata_1)
    fit_1 = convert_ndarray_to_list_in_data(fit_1)

    expected_result_1 = {"ydata": ydata_1, "fit": fit_1}
    with open(save_1_path, "w") as expected_results_1_file:
        json.dump(expected_result_1, expected_results_1_file)

    # second test
    # adjust the parameters
    xdata = np.array([[1, 21, 41, 61, 81, 101, 121, 141, 161, 181]])
    rb_pattern = [[0]]

    # create the results
    results_2_list = load_results_from_json(results_2_file_path)
    rb_fit_2 = RBFitter(results_2_list, xdata, rb_pattern)
    ydata_2 = rb_fit_2.ydata
    fit_2 = rb_fit_2.fit
    # convert ndarray to list
    ydata_2 = convert_ndarray_to_list_in_data(ydata_2)
    fit_2 = convert_ndarray_to_list_in_data(fit_2)

    expected_result_2 = {"ydata": ydata_2, "fit": fit_2}
    with open(save_2_path, "w") as expected_results_2_file:
        json.dump(expected_result_2, expected_results_2_file)


if __name__ == '__main__':
    # generate_fitter_data_1("./data_test_1.json", "./data_test_1_expected_results.json")
    # generate_fitter_data_2("./data_test_2.json", "./data_test_2_expected_results.json")
    # generate_interleaved_fitter_data("./data_original_test.json",
    #                                  "./data_interleaved_test.json",
    #                                  "interleaved_expected_results.json")
    # generate_purity_fitter_data("./data_purity_test.json","./data_purity_coherent_test.json")
    # generate_cnotdihedral_fitter_data("./data_cnotdihedral_X_test.json",
    #                                   "./data_cnotdihedral_Z_test.json")
    # generate_fitter_expected_data("test_fitter_results_1.json",
    #                               "test_fitter_results_2.json", "./test_1.json", "./test_2.json")
