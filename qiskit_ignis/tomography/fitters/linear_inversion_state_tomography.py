import itertools
import numpy

X = numpy.array([[0, 1], [1, 0]])
Y = numpy.array([[0, -1j], [1j, 0]])
Z = numpy.array([[1, 0], [0, -1]])

PauliBasis = {'X': X, 'Y': Y, 'Z': Z}

def compute_coefficient(v, op_names, probs):
    # measurements with 'Id' are redundant; we can replace with any other operator
    replacement_op_name = op_names[-1]
    measurement_name = tuple([op_names[i] if i != 0 else replacement_op_name for i in v])
    measurement = probs[measurement_name]
    total_weighted_sum = 0.0
    w = [1 if i != 0 else 0 for i in v]
    for (measurement_result_bits, measurement_result_prob) in measurement.items():
        measurement_bits_vector = numpy.array([int(x) for x in measurement_result_bits])
        weight = (-1) ** sum(w * measurement_bits_vector)
        total_weighted_sum += weight * measurement_result_prob

    return total_weighted_sum

def experimental_results_to_probabilities(results):
    probs = {}
    for key, measurements in results.items():
        probs[key] = {}
        total_measurement_count = sum(measurements.values())
        for measurement_key, measurement_count in measurements.items():
            probs[key][measurement_key] = measurement_count / total_measurement_count
    return probs

def get_matrix(v, matrices):
    """The tensor product of the matrices indexed by v"""
    matrix = matrices[v[0]]
    for i in v[1:]:
        matrix = numpy.kron(matrix, matrices[i])
    return matrix


def linear_inversion_state_tomography(results, ops ='pauli'):
    """Performs state tomography for operators that constitute an orthonormal basis"""
    if isinstance(ops, str):
        if ops == 'pauli':
            ops = PauliBasis
    Id = numpy.eye(2)
    qubits_num = len(list(results.keys())[0])
    rho = numpy.zeros((2 ** qubits_num, 2 ** qubits_num), dtype='complex128')
    op_names, op_matrices = zip(*ops.items())
    op_names = ['Id'] + list(op_names)
    op_matrices = [Id] + list(op_matrices)
    n = len(op_names)
    probs = experimental_results_to_probabilities(results)
    for v in itertools.product(range(n), repeat=qubits_num):
        coeff = compute_coefficient(v, op_names, probs)
        matrix = get_matrix(v, op_matrices)
        rho += coeff * matrix

    return rho / 2 ** qubits_num