# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
State tomography fitter based on linear inversion
"""

import itertools
import numpy

X = numpy.array([[0, 1], [1, 0]])
Y = numpy.array([[0, -1j], [1j, 0]])
Z = numpy.array([[1, 0], [0, -1]])

PauliBasis = {'X': X, 'Y': Y, 'Z': Z}

def linear_inversion_state_tomography(results, op_indices ='pauli'):
    """
    Performs state tomography for operators that constitute an orthonormal basis

    Args:
        results (dictionary like): dictionary of tomography results of the form
            ('Op1', 'Op2', ..., 'Opn'): {'00..0': count1, '00...1': count2, ..., '11...1': count2^n}
        op_indices (string like or dictionary like): a string or a dictionary of the tomography operators

    Returns:
        a matrix obtained from a linear combination of ops which attempts to reconstruct the
        density operator of the quantum state measured by the results dictionary

    Additional Information:
        1-qubit state tomography can be achieved by noting that X,Y,Z along with the identity I
        constitute an orthonormal basis for the space of 2x2 matrices. So for a given
        density operator rho, Parseval's identity shows that
        rho = <rho, I>I + <rho, X>X + <rho, Y>Y + <rho, Z>Z
        the inner product being Hilbert-Schmidt, <A,B> = 1/2 tr(B^dagger A)
        so we have rho = 1/2[tr(rho)I + tr(X*rho)X + tr(Y*rho)Y + tr(Z*rho) Z]
        the observable values tr(A*rho) are obtained via tomography.
        Note that tr(rho) = 1 always so it needs no tomography data

        The same method works for multiple qubits, with operators obtained as
        tensor products of the base operator set. for tensor products involving I
        we can infer the trace from the rest of the tomography data, so no measurements
        involving I are required
    """

    if isinstance(op_indices, str):
        if op_indices == 'pauli':
            op_indices = PauliBasis

    qubits_num = len(list(results.keys())[0])

    #since we construct rho as a linear combination, start with the zero matrix
    rho = numpy.zeros((2 ** qubits_num, 2 ** qubits_num), dtype='complex128')

    op_names, op_matrices = zip(*op_indices.items())
    #we assume the id matrix is not present in the operator list and add it ourself
    op_names = ['Id'] + list(op_names)
    op_matrices = [numpy.eye(2)] + list(op_matrices)

    #n is the number of different 1-qubit operators used in the measurements
    n = len(op_names)

    #this computes an average; it does not verify the results make sense
    probs = experimental_results_to_probabilities(results)

    for op_indices in itertools.product(range(n), repeat=qubits_num):
        coeff = compute_coefficient(op_indices, op_names, probs)
        matrix = get_matrix(op_indices, op_matrices)
        rho += coeff * matrix

    return rho / 2 ** qubits_num

def compute_coefficient(op_indices, op_names, probs):
    """
    Compute tr(A*rho) from the tomography results for the operator A.

    Args:
        op_indices (vector like): Sequence of integers indicating the 1-qubit operators A is built from
        op_names (vector like): The names (strings) of the 1-qubit operators used in the tomography
        probs: (dictionary like): The full list of tomography results, normalized as probabilities

    Returns:
        The coefficient of A in the linear combination giving rho; equals to tr(A*rho) on exact data

    Additional Information:
        Every measurement for A is given a value of 1 or -1 and summed with weight
        corresponding to its probability.
        The value is decided by counting the number of |0> and |1> outcomes
        for every 1-qubit operator comprising A except the identity operator which is ignored
        since we don't have explicit measurements for operators A that are built using the identity
        we use another operator which is identical in all non-identity operatos

        We look at an example for the 3-qubit case with Pauli basis op_names = ['I', 'X', 'Y', 'Z']
        if ops_indices = [3,1,2] this is the operator A = Z tensor X tensor Y
        We search prob[('Z', 'X', 'Y')] and obtain the probs dictionary
        measurement = {'000': 0.2, '010': 0.5, '110': 0.3}
        The '000' and '110' correspond to value 1 since there is an even number of |1> outcomes
        the '010' corresponds to -1 since there is an odd number of |1> outcomes
        The total weighted sum is therefore 0.2 + 0.3 - 0.5 = 0

        if ops_indices = [0,1,2] this is the operator A = Id tensor X tensor Y
        We don't have a measurement for ('I', 'X', 'Y') but we can use ('Z', 'X', 'Y') instead
        We simply ignore the first bit (corresponding to the Id operator) when deciding on value
        so '110' has value -1 (because of the middle qubit) and the sum is 0.2 - 0.3 - 0.5 = -0.6
    """
    # measurements with 'Id' are redundant; we can replace with any other operator
    replacement_op_name = op_names[-1]
    measurement_name = tuple([op_names[i] if i != 0 else replacement_op_name for i in op_indices])
    measurement = probs[measurement_name]
    total_weighted_sum = 0.0

    # w is used to "disable" qubits corresponding to Id when deciding on value (1 or -1)
    w = [1 if i != 0 else 0 for i in op_indices]
    for (measurement_result_bits, measurement_result_prob) in measurement.items():
        measurement_bits_vector = numpy.array([int(x) for x in measurement_result_bits])
        weight = (-1) ** sum(w * measurement_bits_vector)
        total_weighted_sum += weight * measurement_result_prob

    return total_weighted_sum

def experimental_results_to_probabilities(results):
    """
    Normalizes the results to represent probabilities

    Args:
        results (dictionary like): Dictionary of tomography results

    Returns:
        A results dictionary with the numerical values normalized a probabilities (between 0 and 1)

    Additional Information:
        As an example, a typical item in results may look like this:
        {'010': 2473, '101': 2527}
        Which indicates two possible outcomes of 5,000 experiments, with probabilities close to 0.5 each
        This will be normalized to
        {'010': 0.4946, '101': 0.5054}
    """
    probs = {}
    for key, measurements in results.items():
        probs[key] = {}
        total_measurement_count = sum(measurements.values())
        for measurement_key, measurement_count in measurements.items():
            probs[key][measurement_key] = measurement_count / total_measurement_count
    return probs

def get_matrix(op_indices, op_matrices):
    """
    Builds the tensor product A of the matrices indexed by op_indices

    Args:
        op_indices (vector like): Sequence of integers indicating the 1-qubit operators A is built from
        op_matrices (vector like): The matrices of the 1-qubit operators used in the tomography

    Returns:
        The tensor product matrices[op_indices[0]] tensor ... tensor [op_indices[n]]
    """

    matrix = op_matrices[op_indices[0]]
    for i in op_indices[1:]:
        matrix = numpy.kron(matrix, op_matrices[i])
    return matrix