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
Maximum-Likelihood estimation quantum state tomography fitter
"""

from .base_fitter import TomographyFitter


class StateTomographyFitter(TomographyFitter):
    """Maximum-Likelihood estimation state tomography fitter."""

    def __init__(self,
                 result,
                 circuits,
                 meas_basis='Pauli'):
        """Initialize state tomography fitter with experimental data.

        Args:
            result (Result): a Qiskit Result object obtained from executing
                            tomography circuits.
            circuits (list): a list of circuits or circuit names to extract
                            count information from the result object.
            meas_basis (TomographyBasis, str): A function to return measurement
                        operators corresponding to measurement outcomes. See
                        Additional Information (default: 'Pauli')
        """
        super().__init__(result, circuits, meas_basis, None)

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
                subject to: x >> 0
                            trace(x) = 1

            where:
                a is the matrix of measurement operators a[i] = vec(M_i).H
                b is the vector of expectation value data for each projector
                b[i] ~ Tr[M_i.H * x] = (a * x)[i]
                x is the vectorized density matrix to be fitted

            PSD constraint
            --------------
            The PSD keyword constrains the fitted matrix to be
            postive-semidefinite.
            For the 'lstsq' fitter method the fitted matrix
            is rescaled using the
            method proposed in Reference [1].
            For the 'cvx' fitter method the convex constraint makes the
            optimization problem a SDP. If PSD=False the fitted
            matrix will still
            be constrained to be Hermitian, but not PSD. In this case the
            optimization problem becomes a SOCP.

            Trace constraint
            ----------------
            The trace keyword constrains the trace of the fitted matrix. If
            trace=None there will be no trace constraint on the fitted matrix.
            This constraint should not be used for process tomography and the
            trace preserving constraint should be used instead.

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
        return super().fit(method, standard_weights, beta,
                           trace=1, PSD=True, **kwargs)
