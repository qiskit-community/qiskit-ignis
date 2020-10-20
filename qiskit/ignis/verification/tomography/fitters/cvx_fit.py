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

from typing import Optional
import numpy as np
from scipy import sparse as sps

# Check if CVXPY package is installed
try:
    import cvxpy
    _HAS_CVX = True
except ImportError:
    _HAS_CVX = False


def cvx_fit(data: np.array,
            basis_matrix: np.array,
            weights: Optional[np.array] = None,
            psd: bool = True,
            trace: Optional[int] = None,
            trace_preserving: bool = False,
            **kwargs
            ) -> np.array:
    r"""
    Reconstruct a quantum state using CVXPY convex optimization.

    **Objective function**

    This fitter solves the constrained least-squares minimization:
    :math:`minimize: ||a * x - b ||_2`

    subject to:

    * :math:`x >> 0` (PSD, optional)
    * :math:`\text{trace}(x) = t` (trace, optional)
    * :math:`\text{partial_trace}(x)` = identity (trace_preserving, optional)

    where:
    * a is the matrix of measurement operators :math:`a[i] = vec(M_i).H`
    * b is the vector of expectation value data for each projector
      :math:`b[i] ~ \text{Tr}[M_i.H * x] = (a * x)[i]`
    * x is the vectorized density matrix (or Choi-matrix) to be fitted

    **PSD constraint**

    The PSD keyword constrains the fitted matrix to be
    postive-semidefinite, which makes the optimization problem a SDP. If
    PSD=False the fitted matrix will still be constrained to be Hermitian,
    but not PSD. In this case the optimization problem becomes a SOCP.

    **Trace constraint**

    The trace keyword constrains the trace of the fitted matrix. If
    trace=None there will be no trace constraint on the fitted matrix.
    This constraint should not be used for process tomography and the
    trace preserving constraint should be used instead.

    **Trace preserving (TP) constraint**

    The trace_preserving keyword constrains the fitted matrix to be TP.
    This should only be used for process tomography, not state tomography.
    Note that the TP constraint implicitly enforces the trace of the fitted
    matrix to be equal to the square-root of the matrix dimension. If a
    trace constraint is also specified that differs from this value the fit
    will likely fail.

    **CVXPY Solvers**

    Various solvers can be called in CVXPY using the `solver` keyword
    argument. See the `CVXPY documentation
    <https://www.cvxpy.org/tutorial/advanced/index.html#solve-method-options>`_
    for more information on solvers.

    Args:
        data: (vector like) vector of expectation values
        basis_matrix: (matrix like) measurement operators
        weights: (vector like) weights to apply to the
            objective function (default: None)
        psd: (default: True) enforces the fitted matrix to be positive
            semidefinite (default: True)
        trace: trace constraint for the fitted matrix
            (default: None).
        trace_preserving: (default: False) Enforce the fitted matrix to be
            trace preserving when fitting a Choi-matrix in quantum process
            tomography (default: False).
        **kwargs: kwargs for cvxpy solver.
    Raises:
        ImportError: if cvxpy is not present
        RuntimeError: In case cvx fitting failes
    Returns:
        The fitted matrix rho that minimizes
            :math:`||basis_matrix * vec(rho) - data||_2`.
    """

    # Check if CVXPY package is installed
    if not _HAS_CVX:
        raise ImportError("The CVXPY package is required to use the cvx_fit() "
                          "function. You can install it with 'pip install "
                          "cvxpy' or use a `lstsq` fitter instead of cvx_fit.")
    # SDP VARIABLES

    # Since CVXPY only works with real variables we must specify the real
    # and imaginary parts of rho seperately: rho = rho_r + 1j * rho_i

    dim = int(np.sqrt(basis_matrix.shape[1]))
    rho_r = cvxpy.Variable((dim, dim), symmetric=True)
    rho_i = cvxpy.Variable((dim, dim))

    # CONSTRAINTS

    # The constraint that rho is Hermitian (rho.H = rho)
    # transforms to the two constraints
    #   1. rho_r.T = rho_r.T  (real part is symmetric)
    #   2. rho_i.T = -rho_i.T  (imaginary part is anti-symmetric)

    cons = [rho_i == -rho_i.T]

    # Trace constraint: note this should not be used at the same
    # time as the trace preserving constraint.

    if trace is not None:
        cons.append(cvxpy.trace(rho_r) == trace)

    # Since we can only work with real matrices in CVXPY we can specify
    # a complex PSD constraint as
    #   rho >> 0 iff [[rho_r, -rho_i], [rho_i, rho_r]] >> 0

    if psd is True:
        rho = cvxpy.bmat([[rho_r, -rho_i], [rho_i, rho_r]])
        cons.append(rho >> 0)

    # Trace preserving constraint when fitting Choi-matrices for
    # quantum process tomography. Note that this adds an implicity
    # trace constraint of trace(rho) = sqrt(len(rho)) = dim
    # if a different trace constraint is specified above this will
    # cause the fitter to fail.

    if trace_preserving is True:
        sdim = int(np.sqrt(dim))
        ptr = partial_trace_super(sdim, sdim)
        cons.append(ptr @ cvxpy.vec(rho_r) == np.identity(sdim).ravel())
        cons.append(ptr @ cvxpy.vec(rho_i) == np.zeros(sdim*sdim))

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

    arg = bm_r @ cvxpy.vec(rho_r) - bm_i @ cvxpy.vec(rho_i) - np.array(data)

    # SDP objective function
    obj = cvxpy.Minimize(cvxpy.norm(arg, p=2))

    # Solve SDP
    prob = cvxpy.Problem(obj, cons)
    iters = 5000
    max_iters = kwargs.get('max_iters', 20000)
    # Set default solver if none is specified
    if 'solver' not in kwargs:
        if 'CVXOPT' in cvxpy.installed_solvers():
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


def partial_trace_super(dim1: int, dim2: int) -> np.array:
    """
    Return the partial trace superoperator in the column-major basis.

    This returns the superoperator S_TrB such that:
        S_TrB * vec(rho_AB) = vec(rho_A)
    for rho_AB = kron(rho_A, rho_B)

    Args:
        dim1: the dimension of the system not being traced
        dim2: the dimension of the system being traced over

    Returns:
        A Numpy array of the partial trace superoperator S_TrB.
    """

    iden = sps.identity(dim1)
    ptr = sps.csr_matrix((dim1 * dim1, dim1 * dim2 * dim1 * dim2))

    for j in range(dim2):
        v_j = sps.coo_matrix(([1], ([0], [j])), shape=(1, dim2))
        tmp = sps.kron(iden, v_j.tocsr())
        ptr += sps.kron(tmp, tmp)

    return ptr
