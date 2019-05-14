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
Functions used for the analysis of interleaved
randomized benchmarking results.
"""

import numpy as np
from .fitters import RBFitter


class InterleavedRBFitter():
    """
        Class for fitters for interleaved RB
        Derived from RBFitter class
    """

    def __init__(self, original_result, interleaved_result,
                 cliff_lengths, rb_pattern=None):
        """
        Args:
            original_result: list of results of the
            original RB sequence (qiskit.Result).
            intelreaved_result: list of results of the
            interleaved RB sequence (qiskit.Result).
            cliff_lengths: the Clifford lengths, 2D list i x j where i is the
                number of patterns, j is the number of cliffords lengths
            rb_pattern: the pattern for the rb sequences.
        """
        self.rbfit_original = RBFitter(
            original_result, cliff_lengths, rb_pattern)
        self.rbfit_interleaved = RBFitter(
            interleaved_result, 2 * cliff_lengths, rb_pattern)
        self._fit_interleaved = []

    def fit_interleaved_data(self):
        """
        Fit the interleaved RB results
        Fit each of the patterns

        According to the paper: "Efficient measurement of quantum gate
        error by interleaved randomized benchmarking" (arXiv:1203.4550)
        Equations (4) and (5)

        Puts the results into a list of fit dictionaries:
            where each dictionary corresponds to a pattern and has fields:
            'epc_est' - the estimated error per the interleaved Clifford
            'params_err' - error bound of epc_est
            'params_err_L' = epc_est - params_err (left error bound)
            'params_err_R' = epc_est + params_err (right error bound)
        """
        self.rbfit_original.fit_data()
        self.rbfit_interleaved.fit_data()
        self._fit_interleaved = []

        for patt_ind, (_, qubits) in enumerate(zip(self._cliff_lengths,
                                                   self._rb_pattern)):
            # calculate nrb=d=2^n:
            nrb = 2 ** len(qubits)

            # Calculate alpha (=p) and alpha_c (=p_c):
            alpha = self.rbfit_original.fit[patt_ind]['params'][1]
            alpha_c = self.rbfit_interleaved.fit[patt_ind]['params'][1]

            # Calculate epc_est (=r_c^est) - Eq. (4):
            epc_est = (nrb - 1) * (1 - alpha_c / alpha) / nrb

            # Calculate the error bounds - Eq. (5):
            params_err_1 = (nrb - 1) * (abs(alpha - alpha_c / alpha)
                                        + (1 - alpha)) / nrb
            params_err_2 = 2 * (nrb * nrb - 1) * (1 - alpha) / \
                (alpha * nrb * nrb) + 4 * (np.sqrt(1 - alpha)) * \
                (np.sqrt(nrb * nrb - 1)) / alpha
            params_err = min(params_err_1, params_err_2)
            params_err_L = epc_est - params_err
            params_err_R = epc_est + params_err

            self._fit_interleaved.append({'epc_est': epc_est,
                                          'params_err': params_err,
                                          'params_err_L': params_err_L,
                                          'params_err_R': params_err_R})
