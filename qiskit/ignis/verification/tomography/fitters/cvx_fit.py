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
CVXPY convex optimization quantum tomography fitter
"""

import numpy as np
from scipy import sparse as sps

# Check if CVXPY package is installed
try:
    import cvxpy
except ImportError:
    cvxpy = None


def cvx_fit(data, basis_matrix, weights=None, PSD=True, trace=None,
            trace_preserving=False, **kwargs):
    """
    Reconstruct a quantum state using CVXPY convex optimization.

    Args:
        data (vector like): vector of expectation values
        basis_matrix (matrix like): matrix of measurement operators
        weights (vector like, optional): vector of weights to apply to the
                                         objective function (default: None)
        PSD (bool, optional): Enforced the fitted matrix to be positive
                              semidefinite (default: True)
        trace (int, optional): trace constraint for the fitted matrix
                               (default: None).
        trace_preserving (bool, optional): Enforce the fitted matrix to be
            trace preserving when fitting a Choi-matrix in quantum process
            tomography (default: False).
        **kwargs (optional): kwargs for cvxpy solver.

    Returns:
        The fitted matrix rho that minimizes
        ||basis_matrix * vec(rho) - data||_2.

    Additional Information:

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
            x is the vectorized density matrix (or Choi-matrix) to be fitted

        PSD constraint
        --------------
        The PSD keyword constrains the fitted matrix to be
        postive-semidefinite, which makes the optimization problem a SDP. If
        PSD=False the fitted matrix will still be constrained to be Hermitian,
        but not PSD. In this case the optimization problem becomes a SOCP.

        Trace constraint
        ----------------
        The trace keyword constrains the trace of the fitted matrix. If
        trace=None there will be no trace constraint on the fitted matrix.
        This constraint should not be used for process tomography and the
        trace preserving constraint should be used instead.

        Trace preserving (TP) constraint
        --------------------------------
        The trace_preserving keyword constrains the fitted matrix to be TP.
        This should only be used for process tomography, not state tomography.
        Note that the TP constraint implicitly enforces the trace of the fitted
        matrix to be equal to the square-root of the matrix dimension. If a
        trace constraint is also specified that differs from this value the fit
        will likely fail.

        CVXPY Solvers:
        -------
        Various solvers can be called in CVXPY using the `solver` keyword
        argument. Solvers included in CVXPY are:
            'CVXOPT': SDP and SOCP (default solver)
            'SCS'   : SDP and SOCP
            'ECOS'  : SOCP only
        See the documentation on CVXPY for more information on solvers.
    """

    # Check if CVXPY package is installed
    if cvxpy is None:
        raise Exception('CVXPY is not installed. Use `mle_fit` instead.')
    # Check CVXPY version
    version = cvxpy.__version__
    if not (version[0] == '1' or version[:3] == '0.4'):
        raise Exception('Incompatible CVXPY version. Install 1.0 or 0.4')

    # SDP VARIABLES

    # Since CVXPY only works with real variables we must specify the real
    # and imaginary parts of rho seperately: rho = rho_r + 1j * rho_i

    dim = int(np.sqrt(basis_matrix.shape[1]))
    if version[:3] == '0.4':
        # Compatibility with legacy 0.4
        rho_r = cvxpy.Variable(dim, dim)
        rho_i = cvxpy.Variable(dim, dim)
    else:
        rho_r = cvxpy.Variable((dim, dim))
        rho_i = cvxpy.Variable((dim, dim))

    # CONSTRAINTS

    # The constraint that rho is Hermitian (rho.H = rho)
    # transforms to the two constraints
    #   1. rho_r.T = rho_r.T  (real part is symmetric)
    #   2. rho_i.T = -rho_i.T  (imaginary part is anti-symmetric)

    cons = [rho_r == rho_r.T, rho_i == -rho_i.T]

    # Trace constraint: note this should not be used at the same
    # time as the trace preserving constraint.

    if trace is not None:
        cons.append(cvxpy.trace(rho_r) == trace)

    # Since we can only work with real matrices in CVXPY we can specify
    # a complex PSD constraint as
    #   rho >> 0 iff [[rho_r, -rho_i], [rho_i, rho_r]] >> 0

    if PSD is True:
        rho = cvxpy.bmat([[rho_r, -rho_i], [rho_i, rho_r]])
        cons.append(rho >> 0)

    # Trace preserving constraint when fiting Choi-matrices for
    # quantum process tomography. Note that this adds an implicity
    # trace constraint of trace(rho) = sqrt(len(rho)) = dim
    # if a different trace constraint is specified above this will
    # cause the fitter to fail.

    if trace_preserving is True:
        sdim = int(np.sqrt(dim))
        ptr = partial_trace_super(sdim, sdim)
        cons.append(ptr * cvxpy.vec(rho_r) == np.identity(sdim).ravel())

    # Rescale input data and matrix by weights if they are provided
    if weights is not None:
        w = np.array(weights)
        w = w / np.sqrt(sum(w**2))
        basis_matrix = w[:, None] * basis_matrix
        data = w * data

    # OBJECTIVE FUNCTION

    # The function we wish to minimize is || arg ||_2 where
    #   arg =  bm * vec(rho) - data
    # Since we are working with real matrices in CVXPY we expand this as
    #   bm * vec(rho) = (bm_r + 1j * bm_i) * vec(rho_r + 1j * rho_i)
    #                 = bm_r * vec(rho_r) - bm_i * vec(rho_i)
    #                   + 1j * (bm_r * vec(rho_i) + bm_i * vec(rho_r))
    #                 = bm_r * vec(rho_r) - bm_i * vec(rho_i)
    # where we drop the imaginary part since the expectation value is real

    bm_r = np.real(basis_matrix)
    bm_i = np.imag(basis_matrix)

    # CVXPY doesn't seem to handle sparse matrices very well so we convert
    # sparse matrices to Numpy arrays.

    if isinstance(basis_matrix, sps.spmatrix):
        bm_r = bm_r.todense()
        bm_i = bm_i.todense()

    arg = bm_r * cvxpy.vec(rho_r) - bm_i * cvxpy.vec(rho_i) - np.array(data)

    # SDP objective function
    obj = cvxpy.Minimize(cvxpy.norm(arg, p=2))

    # Solve SDP
    prob = cvxpy.Problem(obj, cons)
    iters = 5000
    max_iters = kwargs.get('max_iters', 20000)

    # Set the default solver to 'CVXOPT'
    if 'solver' not in kwargs:
        kwargs['solver'] = 'CVXOPT'

    problem_solved = False
    while not problem_solved:
        kwargs['max_iters'] = iters
        prob.solve(**kwargs)
        if prob.status in ["optimal_inaccurate", "optimal"]:
            problem_solved = True
        elif prob.status == "unbounded_inaccurate":
            if iters < max_iters:
                iters *= 2
            else:
                raise RuntimeError(
                    "CVX fit failed, probably not enough iterations for the "
                    "solver")
        elif prob.status in ["infeasible", "unbounded"]:
            raise RuntimeError(
                "CVX fit failed, problem status {} which should not "
                "happen".format(prob.status))
        else:
            raise RuntimeError("CVX fit failed, reason unknown")
    rho_fit = rho_r.value + 1j * rho_i.value
    return rho_fit


###########################################################################
# Helper Functions
###########################################################################


def partial_trace_super(d1, d2):
    """
    Return the partial trace superoperator in the column-major basis.

    This returns the superoperator S_TrB such that:
        S_TrB * vec(rho_AB) = vec(rho_A)
    for rho_AB = kron(rho_A, rho_B)

    Args:
        d1 (int): the dimension of the system not being traced
        d2 (int): the diemsnion of the system being traced over

    Returns:
        A Numpy array of the partial trace superoperator S_TrB.
    """

    iden = sps.identity(d1)
    ptr = sps.csr_matrix((d1 * d1, d1 * d2 * d1 * d2))

    for j in range(d2):
        vj = sps.coo_matrix(([1], ([0], [j])), shape=(1, d2))
        tmp = sps.kron(iden, vj.tocsr())
        ptr += sps.kron(tmp, tmp)

    return ptr
