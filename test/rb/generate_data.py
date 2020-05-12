from test.utils import *
import numpy as np
import qiskit
import qiskit.ignis.verification.randomized_benchmarking as rb
from qiskit.ignis.verification.randomized_benchmarking import \
    RBFitter, InterleavedRBFitter, PurityRBFitter, CNOTDihedralRBFitter

def generate_data(rb_opts, results_file_path, shots):
    # Load simulator
    backend = qiskit.Aer.get_backend('qasm_simulator')
    basis_gates = ['u1', 'u2', 'u3', 'cx']

    rb_result = rb.randomized_benchmarking_seq(**rb_opts)
    results = []
    for circuit in rb_result[0][0]:
        results.append(qiskit.execute(circuit, backend=backend,
                       basis_gates=basis_gates,
                       shots=shots).result())

    save_results_as_json(results, results_file_path)

def generate_fitter_data_1(results_file_path):
    rb_opts = {}
    shots = 1024
    rb_opts['nseeds'] = 5
    rb_opts['rb_pattern'] = [[0, 1], [2]]
    rb_opts['length_multiplier'] = [1, 2]
    rb_opts['length_vector'] = [1, 21, 41, 61, 81, 101, 121, 141, 161, 181]
    generate_data(rb_opts, results_file_path, shots)

def generate_fitter_data_2(results_file_path):
    rb_opts = {}
    shots = 1024
    rb_opts['nseeds'] = 5
    rb_opts['rb_pattern'] = [[0]]
    rb_opts['length_vector'] = [1, 21, 41, 61, 81, 101, 121, 141, 161, 181]
    generate_data(rb_opts, results_file_path, shots)

def generate_interleaved_fitter_data(results_file_path_original, results_file_path_interleaved):
    rb_opts = {}
    shots = 200
    rb_opts['nseeds'] = 2
    rb_opts['rb_pattern'] = [[0, 2], [1]]
    rb_opts['length_vector'] = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91]
    rb_opts['length_multiplier'] = [1, 3]
    # original rb
    generate_data(rb_opts, results_file_path_original, shots)

    # interleaved rb
    rb_opts['interleaved_gates'] = [['x 0', 'x 1', 'cx 0 1'], ['x 0']]
    generate_data(rb_opts, results_file_path_interleaved, shots)

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
    # generate_fitter_data_1("./data_test_1.json")
    # generate_fitter_data_2("./data_test_2.json")
    # generate_interleaved_fitter_data("./data_original_test.json","./data_interleaved_test.json")
    # generate_purity_fitter_data("./data_purity_test.json","./data_purity_coherent_test.json")
    generate_cnotdihedral_fitter_data("./data_cnotdihedral_X_test.json",
                                      "./data_cnotdihedral_Z_test.json")
    # generate_fitter_expected_data("test_fitter_results_1.json",
    #                               "test_fitter_results_2.json", "./test_1.json", "./test_2.json")
