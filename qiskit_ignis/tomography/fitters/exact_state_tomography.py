import itertools
import numpy

X = numpy.array([[0, 1], [1, 0]])
Y = numpy.array([[0, -1j], [1j, 0]])
Z = numpy.array([[1, 0], [0, -1]])

PauliBasis = {'X': X, 'Y': Y, 'Z': Z}

def compute_coefficient(v, op_names, results):
    # measurements with 'Id' are redundant; we can replace with any other operator
    replacement_op_name = op_names[-1]
    measurement_name = tuple([op_names[i] if i != 0 else replacement_op_name for i in v])
    measurement = results[measurement_name]
    total_weighted_sum = 0
    w = [1 if i != 0 else 0 for i in v]
    for (measurement_result_bits, measurement_result_count) in measurement.items():
        measurement_bits_vector = numpy.array([int(x) for x in measurement_result_bits])
        weight = (-1) ** sum(w * measurement_bits_vector)
        total_weighted_sum += weight * measurement_result_count

    measurement_count = sum([m[1] for m in measurement.items()])
    return (float(total_weighted_sum) / measurement_count)


def get_matrix(v, matrices):
    """The tensor product of the matrices indexed by v"""
    matrix = matrices[v[0]]
    for i in v[1:]:
        matrix = numpy.kron(matrix, matrices[i])
    return matrix


def exact_state_tomography(results, ops = 'pauli'):
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

    for v in itertools.product(range(n), repeat=qubits_num):
        coeff = compute_coefficient(v, op_names, results)
        matrix = get_matrix(v, op_matrices)
        rho += coeff * matrix

    return rho / 2 ** qubits_num