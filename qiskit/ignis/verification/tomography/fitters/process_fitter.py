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
Maximum-Likelihood estimation quantum process tomography fitter
"""

import numpy as np
from qiskit import QiskitError
from qiskit.quantum_info.operators import Choi
from .base_fitter import TomographyFitter
from .cvx_fit import cvxpy, cvx_fit
from .lstsq_fit import lstsq_fit


class ProcessTomographyFitter(TomographyFitter):
    """Maximum-Likelihood estimation process tomography fitter."""

    def fit(self, method='auto', standard_weights=True, beta=0.5, **kwargs):
        """
        Reconstruct a quantum channel using CVXPY convex optimization.

        Args:
            method (str): The fitter method 'auto', 'cvx' or 'lstsq'.
            standard_weights (bool, optional): Apply weights
                                            to tomography data
                                            based on count probability
                                            (default: True)
            beta (float): hedging parameter for converting counts
                        to probabilities (default: 0.5)
            **kwargs (optional): kwargs for fitter method.

        Returns:
            Choi: The fitted Choi-matrix J for the channel that maximizes
            ||basis_matrix * vec(J) - data||_2. The Numpy matrix can be
            obtained from `Choi.data`.

        Additional Information:

            Choi matrix
            -----------
            The Choi matrix object is a QuantumChannel representation which
            may be converted to other representations using the classes
            `SuperOp`, `Kraus`, `Stinespring`, `PTM`, `Chi` from the module
            `qiskit.quantum_info.operators`. The raw matrix data for the
            representation may be obtained by `channel.data`.

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
                subject to: x >> 0 (PSD)
                            trace(x) = dim (trace)
                            partial_trace(x) = identity (trace_preserving)

            where:
                a is the matrix of measurement operators a[i] = vec(M_i).H
                b is the vector of expectation value data for each projector
                b[i] ~ Tr[M_i.H * x] = (a * x)[i]
                x is the vectorized Choi-matrix to be fitted

            PSD constraint
            --------------
            The PSD keyword constrains the fitted matrix to be
            postive-semidefinite.
            For the 'lstsq' fitter method the fitted matrix is
            rescaled using the
            method proposed in Reference [1].
            For the 'cvx' fitter method the convex constraint makes the
            optimization problem a SDP. If PSD=False the
            fitted matrix will still
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
            The trace_preserving keyword constrains the fitted
            matrix to be TP.
            This should only be used for process tomography,
            not state tomography.
            Note that the TP constraint implicitly enforces
            the trace of the fitted
            matrix to be equal to the square-root of the
            matrix dimension. If a
            trace constraint is also specified that differs f
            rom this value the fit
            will likely fail. Note that this can
            only be used for the CVX method.

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

        # Calculate trace of Choi-matrix from projector length
        _, cols = np.shape(basis_matrix)
        dim = int(np.sqrt(np.sqrt(cols)))
        if dim ** 4 != cols:
            raise ValueError("Input data does not correspond "
                             "to a process matrix.")
        # Choose automatic method
        if method == 'auto':
            if cvxpy is None:
                method = 'lstsq'
            else:
                method = 'cvx'
        if method == 'lstsq':
            return Choi(lstsq_fit(data, basis_matrix, weights=weights,
                                  trace=dim, **kwargs))
        if method == 'cvx':
            return Choi(cvx_fit(data, basis_matrix, weights=weights, trace=dim,
                                trace_preserving=True, **kwargs))
        raise QiskitError('Unrecognised fit method {}'.format(method))
