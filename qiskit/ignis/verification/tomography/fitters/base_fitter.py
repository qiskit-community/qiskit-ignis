# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""
Maximum-Likelihood estimation quantum tomography fitter
"""

import logging
import itertools as it
from ast import literal_eval
import numpy as np

from qiskit import QiskitError
from qiskit import QuantumCircuit
from ..basis import TomographyBasis, default_basis
from ..data import marginal_counts, combine_counts, count_keys
from .cvx_fit import cvxpy, cvx_fit
from .lstsq_fit import lstsq_fit

# Create logger
logger = logging.getLogger(__name__)


class TomographyFitter:
    """Basse maximum-likelihood estimate tomography fitter class"""

    def __init__(self,
                 result,
                 circuits,
                 meas_basis='Pauli',
                 prep_basis='Pauli'):
        """Initialize tomography fitter with experimental data.

        Args:
            result (Result): a Qiskit Result object obtained from executing
                            tomography circuits.
            circuits (list): a list of circuits or circuit names to extract
                            count information from the result object.
            meas_basis (TomographyBasis, str): A function to return
                        measurement operators corresponding to measurement
                        outcomes. See Additional Information
                        (default: 'Pauli')
            prep_basis (TomographyBasis, str): A function to return
                        preparation operators. See Additional
                        Information (default: 'Pauli')
        """

        # Set the measure and prep basis
        self._meas_basis = None
        self._prep_basis = None
        self.set_measure_basis(meas_basis)
        self.set_preparation_basis(prep_basis)

        # Add initial data
        self._data = {}
        self.add_data(result, circuits)

    def set_measure_basis(self, basis):
        """Set the measurement basis

        Args:
            basis (TomographyBasis or str): measurement basis
        """
        self._meas_basis = default_basis(basis)
        if isinstance(self._meas_basis, TomographyBasis):
            if self._meas_basis.measurement is not True:
                raise QiskitError("Invalid measurement basis")

    def set_preparation_basis(self, basis):
        """Set the prepearation basis function

        Args:
            basis (TomographyBasis or str): preparation basis
        """
        self._prep_basis = default_basis(basis)
        if isinstance(self._prep_basis, TomographyBasis):
            if self._prep_basis.preparation is not True:
                raise QiskitError("Invalid preparation basis")

    @property
    def measure_basis(self):
        """Return the tomography measurement basis."""
        return self._meas_basis

    @property
    def preparation_basis(self):
        """Return the tomography preperation basis."""
        return self._prep_basis

    def fit(self, method='auto', standard_weights=True, beta=0.5, **kwargs):
        """
        Reconstruct a quantum state using CVXPY convex optimization.

        Args:
            method (str): The fitter method 'auto', 'cvx' or 'lstsq'.
            standard_weights (bool, optional): Apply weights to
                                            tomography data
                                            based on count probability
                                            (default: True)
            beta (float): hedging parameter for converting counts
                        to probabilities
                        (default: 0.5)
            PSD (bool, optional): Enforced the fitted matrix to be positive
                                semidefinite (default: True)
            trace (int, optional): trace constraint for the fitted matrix
                                (default: None).
            trace_preserving (bool, optional): Enforce the fitted matrix to be
                trace preserving when fitting a Choi-matrix in quantum process
                tomography. Note this method does not apply for 'lstsq' fitter
                method (default: False).
            **kwargs (optional): kwargs for fitter method.

        Returns:
            The fitted matrix rho that minimizes
            ||basis_matrix * vec(rho) - data||_2.

        Additional Information:

            Fitter method
            -------------
            The 'cvx' fitter method used CVXPY convex optimization package.
            The 'lstsq' method uses least-squares fitting (linear inversion).
            The 'auto' method will use 'cvx' if the CVXPY package is found on
            the system, otherwise it will default to 'lstsq'.

            Objective function
            ------------------
            This fitter solves the constrained least-squares minimization:

                minimize: ||a * x - b ||_2
                subject to: x >> 0 (PSD, optional)
                            trace(x) = t (trace, optional)
                            partial_trace(x) = identity (trace_preserving,
                                                        optional)

            where:
                a is the matrix of measurement operators a[i] = vec(M_i).H
                b is the vector of expectation value data for each projector
                b[i] ~ Tr[M_i.H * x] = (a * x)[i]
                x is the vectorized density matrix (or Choi-matrix)
                to be fitted

            PSD constraint
            --------------
            The PSD keyword constrains the fitted matrix to be
            postive-semidefinite.
            For the 'lstsq' fitter method the fitted matrix is rescaled
            using the method proposed in Reference [1].
            For the 'cvx' fitter method the convex constraint makes the
            optimization problem a SDP. If PSD=False the fitted matrix
            will still
            be constrained to be Hermitian, but not PSD. In this case the
            optimization problem becomes a SOCP.

            Trace constraint
            ----------------
            The trace keyword constrains the trace of the fitted matrix. If
            trace=None there will be no trace constraint on the fitted matrix.
            This constraint should not be used for process tomography and the
            trace preserving constraint should be used instead.

            Trace preserving (TP) constraint
            --------------------------------
            The trace_preserving keyword constrains the fitted matrix
            to be TP. This should only be used for process tomography,
            not state tomography.
            Note that the TP constraint implicitly enforces
            the trace of the fitted
            matrix to be equal to the square-root of the matrix dimension.
            If a trace constraint is also specified that
            differs from this value the fit
            will likely fail. Note that this can only be used
            for the CVX method.

            CVXPY Solvers:
            -------
            Various solvers can be called in CVXPY using the `solver` keyword
            argument. Solvers included in CVXPY are:
                'CVXOPT': SDP and SOCP (default solver)
                'SCS'   : SDP and SOCP
                'ECOS'  : SOCP only
            See the documentation on CVXPY for more information on solvers.

            References:
            [1] J Smolin, JM Gambetta, G Smith, Phys. Rev. Lett. 108, 070502
                (2012). Open access: arXiv:1106.5458 [quant-ph].
        """
        # Get fitter data
        data, basis_matrix, weights = self._fitter_data(standard_weights,
                                                        beta)
        # Choose automatic method
        if method == 'auto':
            if cvxpy is None:
                method = 'lstsq'
            else:
                method = 'cvx'
        if method == 'lstsq':
            return lstsq_fit(data, basis_matrix, weights=weights, **kwargs)

        if method == 'cvx':
            return cvx_fit(data, basis_matrix, weights=weights, **kwargs)

        raise QiskitError('Unrecognised fit method {}'.format(method))

    @property
    def data(self):
        """
        Return tomography data
        """
        return self._data

    def add_data(self, result, circuits):
        """Add tomography data from a Qiskit Result object.

        Args:
            result (Result): a Qiskit Result object obtained from executing
                            tomography circuits.
            circuits (list): a list of circuits or circuit names to extract
                            count information from the result object.
        """
        if len(circuits[0].cregs) == 1:
            marginalize = False
        else:
            marginalize = True

        # Process measurement counts into probabilities
        for circ in circuits:
            counts = result.get_counts(circ)
            if isinstance(circ, str):
                tup = literal_eval(circ)
            elif isinstance(circ, QuantumCircuit):
                tup = literal_eval(circ.name)
            else:
                tup = circ
            if marginalize:
                counts = marginal_counts(counts, range(len(tup[0])))
            if tup in self._data:
                self._data[tup] = combine_counts(self._data[tup], counts)
            else:
                self._data[tup] = counts

    def _fitter_data(self, standard_weights, beta):
        """Generate tomography fitter data from a tomography data dictionary.

        Args:
            standard_weights (bool, optional): Apply weights to basis matrix
                            and data based on count probability
                            (default: True)
            beta (float): hedging parameter for 0, 1
            probabilities (default: 0.5)

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
        # Get basis matrix functions
        if self._meas_basis:
            measurement = self._meas_basis.measurement_matrix
        else:
            measurement = None
        if self._prep_basis:
            preparation = self._prep_basis.preparation_matrix
        else:
            preparation = None

        data = []
        basis_blocks = []
        if standard_weights:
            weights = []
        else:
            weights = None

        # Check if input data is state or process tomography data based
        # on the label tuples
        label = next(iter(self._data))
        is_qpt = (isinstance(label, tuple) and len(label) == 2 and
                  isinstance(label[0], tuple) and isinstance(label[1], tuple))
        # Generate counts keys for converting to np array
        if is_qpt:
            ctkeys = count_keys(len(label[1]))
        else:
            ctkeys = count_keys(len(label))
        for label, cts in self._data.items():

            # Convert counts dict to numpy array
            if isinstance(cts, dict):
                cts = np.array([cts.get(key, 0) for key in ctkeys])

            # Get probabilities
            shots = np.sum(cts)
            probs = np.array(cts) / shots
            data += list(probs)

            # Compute binomial weights
            if standard_weights is True:
                wts = self._binomial_weights(cts, beta)
                weights += list(wts)

            # Get reconstruction basis operators
            if is_qpt:
                prep_label = label[0]
                meas_label = label[1]
            else:
                prep_label = None
                meas_label = label
            prep_op = self._preparation_op(prep_label, preparation)
            meas_ops = self._measurement_ops(meas_label, measurement)
            block = self._basis_operator_matrix(
                [np.kron(prep_op.T, mop) for mop in meas_ops])
            basis_blocks.append(block)

        return data, np.vstack(basis_blocks), weights

    def _binomial_weights(self, counts, beta=0.5):
        """
        Compute binomial weights for list or dictionary of counts.

        Args:
            counts (dict, vector): A set of measurement counts for
                                all outcomes of a given measurement
                                configuration.
            beta (float >= 0): A hedging parameter used to bias probabilities
                            computed from input counts away from 0 or 1.

        Returns:
            A numpy array of binomial weights for the input counts and beta
            parameter.

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

        K = len(counts)  # Number of possible outcomes.
        # Compute hedged frequencies which are shifted to never be 0 or 1.
        freqs_hedged = (counts + beta) / (shots + K * beta)

        # Return gaussian weights for 2-outcome measurements.
        return np.sqrt(shots / (freqs_hedged * (1 - freqs_hedged)))

    def _basis_operator_matrix(self, basis):
        """
        Return a basis measurement matrix of the input basis.

        Args:
            basis (list (array like)): a list of basis matrices.

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

    def _preparation_op(self, label, prep_matrix_fn):
        """
        Return the multi-qubit matrix for a state preparation label.

        Args:
            label (tuple(str)): a preparation configuration label for a
                                tomography circuit.
            prep_matrix_fn (function): a function that returns the matrix
                            corresponding to a single qubit preparation label.
                            The functions should have signature
                                prep_matrix_fn(str) -> np.array
        Returns:
            A Numpy array for the multi-qubit prepration operator specified
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
        for l in label:
            op = np.kron(prep_matrix_fn(l), op)
        return op

    def _measurement_ops(self, label, meas_matrix_fn):
        """
        Return a list multi-qubit matrices for a measurement label.

        Args:
            label (tuple(str)): a measurement configuration label for a
                                tomography circuit.
            meas_matrix_fn (function): a function that returns the matrix
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

        for l in sorted(it.product((0, 1), repeat=num_qubits)):
            op = np.eye(1, dtype=complex)
            # Reverse label to correspond to QISKit bit ordering
            for m, outcome in zip(reversed(label), l):
                op = np.kron(op, meas_matrix_fn(m, outcome))
            meas_ops.append(op)
        return meas_ops
