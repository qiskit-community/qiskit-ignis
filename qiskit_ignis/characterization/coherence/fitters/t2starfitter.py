# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
T2* fitter
"""

import numpy as np
from .basefitter import BaseCoherenceFitter

class T2StarExpFitter(BaseCoherenceFitter):
    """
    T2* fitter
    """

    def __init__(self, backend_result, shots, xdata,
                 num_of_qubits, measured_qubit,
                 fit_p0, fit_bounds):

        BaseCoherenceFitter.__init__(self, '$T_2^*$ exp',
                                     backend_result, shots, xdata,
                                     num_of_qubits, measured_qubit,
                                     BaseCoherenceFitter._exp_fit_fun,
                                     fit_p0, fit_bounds)

        self._time = self.params[1]
        self._time_err = self.params_err[1]


class T2StarOscFitter(BaseCoherenceFitter):
    """
    T2* fitter
    """

    def __init__(self, backend_result, shots, xdata,
                 num_of_qubits, measured_qubit,
                 fit_p0, fit_bounds):

        BaseCoherenceFitter.__init__(self, '$T_2^*$',
                                     backend_result, shots, xdata,
                                     num_of_qubits, measured_qubit,
                                     T2StarOscFitter._osc_fit_fun,
                                     fit_p0, fit_bounds)

        self._time = self.params[1]
        self._time_err = self.params_err[1]


    @staticmethod
    def _osc_fit_fun(x, a, tau, f, phi, c):
        """
        Function used to fit the decay cosine
        """

        return a * np.exp(-x / tau) * np.cos(2 * np.pi * f * x + phi) + c
