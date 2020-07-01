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
from typing import List, Union, Optional, Dict, Tuple, Callable
from ast import literal_eval
import numpy as np


from qiskit import QiskitError
from qiskit import QuantumCircuit
from qiskit.result import Result
from ..basis import TomographyBasis, default_basis
from ..data import marginal_counts, combine_counts, count_keys
from .lstsq_fit import lstsq_fit
from .cvx_fit import cvx_fit, _HAS_CVX

# Create logger
logger = logging.getLogger(__name__)


class TomographyFitter:
    """Base maximum-likelihood estimate tomography fitter class"""

    _HAS_SDP_SOLVER = None

    def __init__(self,
                 result: Union[Result, List[Result]],
                 circuits: Union[List[QuantumCircuit], List[str]],
                 meas_basis: Union[TomographyBasis, str] = 'Pauli',
                 prep_basis: Union[TomographyBasis, str] = 'Pauli'):
        """Initialize tomography fitter with experimental data.

        Args:
            result: a Qiskit Result object obtained from executing
                tomography circuits.
            circuits: a list of circuits or circuit names to extract
                count information from the result object.
            meas_basis: (default: 'Pauli') A function to return
                measurement operators corresponding to measurement
                outcomes. See Additional Information.
            prep_basis: (default: 'Pauli') A function to return
                preparation operators. See Additional
                Information
        """

        # Set the measure and prep basis
        self._meas_basis = None
        self._prep_basis = None
        self.set_measure_basis(meas_basis)
        self.set_preparation_basis(prep_basis)

        # Add initial data
        self._data = {}
        if isinstance(result, Result):
            result = [result]  # unify results handling
        self.add_data(result, circuits)

    def set_measure_basis(self, basis: Union[TomographyBasis, str]):
        """Set the measurement basis

        Args:
            basis: measurement basis

        Raises:
            QiskitError: In case of invalid measurement or preparation basis.
        """
        self._meas_basis = default_basis(basis)
        if isinstance(self._meas_basis, TomographyBasis):
            if self._meas_basis.measurement is not True:
                raise QiskitError("Invalid measurement basis")

    def set_preparation_basis(self, basis: Union[TomographyBasis, str]):
        """Set the preparation basis function

        Args:
            basis: preparation basis
        Raises:
            QiskitError: in case the basis has no preperation data
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
        """Return the tomography preparation basis."""
        return self._prep_basis

    def fit(self,
            method: str = 'auto',
            standard_weights: bool = True,
            beta: float = 0.5,
            psd: bool = True,
            trace: Optional[int] = None,
            trace_preserving: bool = False,
            **kwargs) -> np.array:
        r"""Reconstruct a quantum state using CVXPY convex optimization.

                **Fitter method**

        The ``cvx`` fitter method used CVXPY convex optimization package.
        The ``lstsq`` method uses least-squares fitting (linear inversion).
        The ``auto`` method will use 'cvx' if the CVXPY package is found on
        the system, otherwise it will default to 'lstsq'.

        **Objective function**

        This fitter solves the constrained least-squares minimization:
        minimize: :math:`||a \cdot x - b ||_2`

        subject to:

         * :math:`x \succeq 0`
         * :math:`\text{trace}(x) = 1`

        where:

         * a is the matrix of measurement operators
           :math:`a[i] = \text{vec}(M_i).H`
         * b is the vector of expectation value data for each projector
           :math:`b[i] \sim \text{Tr}[M_i.H \cdot x] = (a \cdot x)[i]`
         * x is the vectorized density matrix to be fitted

        **PSD constraint**

        The PSD keyword constrains the fitted matrix to be
        postive-semidefinite. For the ``lstsq`` fitter method the fitted matrix
        is rescaled using the method proposed in Reference [1]. For the ``cvx``
        fitter method the convex constraint makes the optimization problem a
        SDP. If PSD=False the fitted matrix will still be constrained to be
        Hermitian, but not PSD. In this case the optimization problem becomes
        a SOCP.

        **Trace constraint**

        The trace keyword constrains the trace of the fitted matrix. If
        trace=None there will be no trace constraint on the fitted matrix.
        This constraint should not be used for process tomography and the
        trace preserving constraint should be used instead.

        **CVXPY Solvers:**

        Various solvers can be called in CVXPY using the `solver` keyword
        argument. See the `CVXPY documentation
        <https://www.cvxpy.org/tutorial/advanced/index.html#solve-method-options>`_
        for more information on solvers.

        References:

        [1] J Smolin, JM Gambetta, G Smith, Phys. Rev. Lett. 108, 070502
            (2012). Open access:
            `arXiv:1106.5458 <https://arxiv.org/abs/1106.5458>`_ [quant-ph].

        Args:
            method: The fitter method 'auto', 'cvx' or 'lstsq'.
            standard_weights: (default: True) Apply weights to
                tomography data based on count probability
            beta: hedging parameter for converting counts
                to probabilities
            psd: Enforced the fitted matrix to be positive semidefinite.
            trace: trace constraint for the fitted matrix.
            trace_preserving: Enforce the fitted matrix to be
                trace preserving when fitting a Choi-matrix in quantum process
                tomography. Note this method does not apply for 'lstsq' fitter
                method.
            **kwargs: kwargs for fitter method.
        Raises:
            QiskitError: In case the fitting method is unrecognized.
        Returns:
            The fitted matrix rho that minimizes
            :math:`||\text{basis_matrix} * \text{vec(rho)} - \text{data}||_2`.
        """
        # Get fitter data
        data, basis_matrix, weights = self._fitter_data(standard_weights,
                                                        beta)
        # Choose automatic method
        if method == 'auto':
            self._check_for_sdp_solver()
            if self._HAS_SDP_SOLVER:
                method = 'cvx'
            else:
                method = 'lstsq'
        if method == 'lstsq':
            return lstsq_fit(data, basis_matrix,
                             weights=weights,
                             psd=psd,
                             trace=trace,
                             **kwargs)

        if method == 'cvx':
            return cvx_fit(data, basis_matrix,
                           weights=weights,
                           psd=psd,
                           trace=trace,
                           trace_preserving=trace_preserving,
                           **kwargs)

        raise QiskitError('Unrecognized fit method {}'.format(method))

    @property
    def data(self):
        """
        Return tomography data
        """
        return self._data

    def add_data(self,
                 results: List[Result],
                 circuits: List[Union[QuantumCircuit, str]]
                 ):
        """Add tomography data from a Qiskit Result object.

        Args:
            results: The results obtained from executing tomography circuits.
            circuits: circuits or circuit names to extract
                count information from the result object.

        Raises:
            QiskitError: In case some of the tomography data is not found
                in the results
        """
        if len(circuits[0].cregs) == 1:
            marginalize = False
        else:
            marginalize = True

        # Process measurement counts into probabilities
        for circ in circuits:
            counts = None
            for result in results:
                try:
                    counts = result.get_counts(circ)
                except QiskitError:
                    pass
            if counts is None:
                raise QiskitError("Result for {} not found".format(circ.name))
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
                and data based on count probability (default: True)
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

    def _binomial_weights(self, counts: Dict[str, int],
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

    def _basis_operator_matrix(self, basis: List[np.array]) -> np.array:
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

    def _preparation_op(self,
                        label: Tuple[str],
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

    def _measurement_ops(self,
                         label: Tuple[str],
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

    @classmethod
    def _check_for_sdp_solver(cls):
        """Check if CVXPY solver is available"""
        if cls._HAS_SDP_SOLVER is None:
            if _HAS_CVX:
                # pylint:disable=import-error
                import cvxpy
                solvers = cvxpy.installed_solvers()
                if 'CVXOPT' in solvers:
                    cls._HAS_SDP_SOLVER = True
                    return
                if 'SCS' in solvers:
                    # Try example problem to see if built with BLAS
                    # SCS solver cannot solver larger than 2x2 matrix
                    # problems without BLAS
                    try:
                        var = cvxpy.Variable((4, 4), PSD=True)
                        obj = cvxpy.Minimize(cvxpy.norm(var))
                        cvxpy.Problem(obj).solve(solver='SCS')
                        cls._HAS_SDP_SOLVER = True
                        return
                    except cvxpy.error.SolverError:
                        pass
            cls._HAS_SDP_SOLVER = False
