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


"""Maximum-Likelihood estimation quantum state tomography fitter
"""
from typing import List, Union
import numpy as np
from qiskit.result import Result
from qiskit import QuantumCircuit
from ..basis import TomographyBasis
from .base_fitter import TomographyFitter


class StateTomographyFitter(TomographyFitter):
    """Maximum-Likelihood estimation state tomography fitter."""

    def __init__(self,
                 result: Result,
                 circuits: List[QuantumCircuit],
                 meas_basis: Union[TomographyBasis, str] = 'Pauli'
                 ):
        """Initialize state tomography fitter with experimental data.

        Args:
            result: a Qiskit Result object obtained from executing
                tomography circuits.
            circuits: a list of circuits or circuit names to extract
                count information from the result object.
            meas_basis: (default: 'Pauli') A function to return measurement
                operators corresponding to measurement outcomes. See
                Additional Information (default: 'Pauli')
        """
        super().__init__(result, circuits, meas_basis, None)

    def fit(self,  # pylint: disable=arguments-differ
            method: str = 'auto',
            standard_weights: bool = True,
            beta: float = 0.5,
            **kwargs) -> np.array:
        r"""Reconstruct a quantum state using CVXPY convex optimization.

        **Fitter method**

        The ``cvx`` fitter method used CVXPY convex optimization package.
        The ``lstsq`` method uses least-squares fitting (linear inversion).
        The ``auto`` method will use 'cvx' if the CVXPY package is found on
        the system, otherwise it will default to 'lstsq'.

        **Objective function**

        This fitter solves the constrained least-squares minimization:
        :math:`minimize: ||a \cdot x - b ||_2`

        subject to:

         * :math:`x >> 0`
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
            (2012). Open access: arXiv:1106.5458 [quant-ph].

        Args:
            method: The fitter method 'auto', 'cvx' or 'lstsq'.
            standard_weights: (default: True) Apply weights to
                tomography data based on count probability
            beta: (default: 0.5) hedging parameter for converting counts
                to probabilities
            **kwargs: kwargs for fitter method.
        Raises:
            QiskitError: In case the fitting method is unrecognized.
        Returns:
            The fitted matrix rho that minimizes
            :math:`||\text{basis_matrix} \cdot
            \text{vec}(\text{rho}) - \text{data}||_2`.
        """
        return super().fit(method, standard_weights, beta,
                           trace=1, psd=True, **kwargs)
