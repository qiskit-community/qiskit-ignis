from qiskit.ignis.verification.tomography.data import marginal_counts, count_keys
from typing import List, Dict, Any, Tuple, Callable
import numpy as np
import itertools as it

def _fitter_data(counts: List[np.array],
                 metadata: List[Dict[str, Any]],
                 measurement: np.array = None,
                 preparation: np.array = None,
                 standard_weights: bool = True,
                 beta: float = 0.5):
    """Generate tomography fitter data from a tomography data dictionary.

    Args:
        measurement: Measurement matrix for the tomography basis
        preparation: Preparation matrix for the tomography basis
        standard_weight: Apply weights to basis matrix
            and data based on count probability
        beta: hedging parameter for 0, 1 probabilities

    Returns:
        tuple: (data, basis_matrix, weights) where `data`
        is a vector of the
        probability values, and `basis_matrix`
        is a matrix of the preparation
        and measurement operator, and `weights`
        is a vector of weights for the
        given probabilities.

    Additional Information
    ----------------------
    standard_weights:
        Weights are calculated from from binomial distribution standard
        deviation
    """
    data = []
    basis_blocks = []
    if standard_weights:
        weights = []
    else:
        weights = None

    for meta, cts in zip(metadata, counts):
        # Get probabilities
        shots = np.sum(cts)
        probs = np.array(cts) / shots
        data += list(probs)

        # Compute binomial weights
        if standard_weights is True:
            wts = _binomial_weights(cts, beta)
            weights += list(wts)

        # Get reconstruction basis operators
        prep_op = _preparation_op(meta['prep_label'], preparation)
        meas_ops = _measurement_ops(meta['meas_label'], measurement)
        block = _basis_operator_matrix(
            [np.kron(prep_op.T, mop) for mop in meas_ops])
        basis_blocks.append(block)

    return data, np.vstack(basis_blocks), weights


def _binomial_weights(counts: Dict[str, int],
                      beta: float = 0.5
                      ) -> np.array:
    """
    Compute binomial weights for list or dictionary of counts.

    Args:
        counts: A set of measurement counts for
            all outcomes of a given measurement configuration.
        beta: (default: 0.5) A nonnegative hedging parameter used to bias
        probabilities computed from input counts away from 0 or 1.

    Returns:
        The binomial weights for the input counts and beta parameter.
    Raises:
        ValueError: In case beta is negative.
    Additional Information:

        The weights are determined by
            w[i] = sqrt(shots / p[i] * (1 - p[i]))
            p[i] = (counts[i] + beta) / (shots + K * beta)
        where
            `shots` is the sum of all counts in the input
            `p` is the hedged probability computed for a count
            `K` is the total number of possible measurement outcomes.
    """

    # Sort counts if input is a dictionary
    if isinstance(counts, dict):
        mcts = marginal_counts(counts, pad_zeros=True)
        ordered_keys = sorted(list(mcts))
        counts = np.array([mcts[k] for k in ordered_keys])
    # Assume counts are already sorted if a list
    else:
        counts = np.array(counts)
    shots = np.sum(counts)

    # If beta is 0 check if we would be dividing by zero
    # If so change beta value and log warning.

    if beta < 0:
        raise ValueError('beta = {} must be non-negative.'.format(beta))
    if beta == 0 and (shots in counts or 0 in counts):
        beta = 0.5
        msg = ("Counts result in probabilities of 0 or 1 "
               "in binomial weights "
               "calculation. Setting hedging "
               "parameter beta={} to prevent "
               "dividing by zero.".format(beta))
        logger.warning(msg)

    outcomes_num = len(counts)
    # Compute hedged frequencies which are shifted to never be 0 or 1.
    freqs_hedged = (counts + beta) / (shots + outcomes_num * beta)

    # Return gaussian weights for 2-outcome measurements.
    return np.sqrt(shots / (freqs_hedged * (1 - freqs_hedged)))


def _basis_operator_matrix(basis: List[np.array]) -> np.array:
    """Return a basis measurement matrix of the input basis.

    Args:
        basis: a list of basis matrices.

    Returns:
        A numpy array of shape (n, col * row) where n is the number
        of operators of shape (row, col) in `basis`.
    """
    # Dimensions
    num_ops = len(basis)
    nrows, ncols = basis[0].shape
    size = nrows * ncols

    ret = np.zeros((num_ops, size), dtype=complex)
    for j, b in enumerate(basis):
        ret[j] = np.array(b).reshape((1, size), order='F').conj()
    return ret


def _preparation_op(label: Tuple[str],
                    prep_matrix_fn: Callable[[str], np.array]
                    ) -> np.array:
    """
    Return the multi-qubit matrix for a state preparation label.

    Args:
        label: a preparation configuration label for a
            tomography circuit.
        prep_matrix_fn: a function that returns the matrix
            corresponding to a single qubit preparation label.
            The functions should have signature:
                ``prep_matrix_fn(str) -> np.array``
    Returns:
        A Numpy array for the multi-qubit preparation operator specified
        by label.

    Additional Information:
        See the Pauli and SIC-POVM preparation functions
        `pauli_preparation_matrix` and `sicpovm_preparation_matrix` for
        examples.
    """

    # Trivial case
    if label is None:
        return np.eye(1, dtype=complex)

    # Construct preparation matrix
    op = np.eye(1, dtype=complex)
    for label_inst in label:
        op = np.kron(prep_matrix_fn(label_inst), op)
    return op


def _measurement_ops(label: Tuple[str],
                     meas_matrix_fn: Callable[[str, int], np.array]
                     ) -> List[np.array]:
    """
    Return a list multi-qubit matrices for a measurement label.

    Args:
        label: a measurement configuration label for a
            tomography circuit.
        meas_matrix_fn: a function that returns the matrix
            corresponding to a single qubit measurement label
            for a given outcome. The functions should have
            signature meas_matrix_fn(str, int) -> np.array

    Returns:
        A list of Numpy array for the multi-qubit measurement operators
        for all measurement outcomes for the measurement basis specified
        by the label. These are ordered in increasing binary order. Eg for
        2-qubits the returned matrices correspond to outcomes
        [00, 01, 10, 11]

    Additional Information:
        See the Pauli measurement function `pauli_measurement_matrix`
        for an example.
    """
    num_qubits = len(label)
    meas_ops = []

    # Construct measurement POVM for all measurement outcomes for a given
    # measurement label. This will be a list of 2 ** n operators.

    for outcomes in sorted(it.product((0, 1), repeat=num_qubits)):
        op = np.eye(1, dtype=complex)
        # Reverse label to correspond to QISKit bit ordering
        for m, outcome in zip(reversed(label), outcomes):
            op = np.kron(op, meas_matrix_fn(m, outcome))
        meas_ops.append(op)
    return meas_ops