# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
T1 fitter
"""

from .basefitter import BaseCoherenceFitter

class T1Fitter(BaseCoherenceFitter):
    """
    T1 fitter
    """

    def __init__(self, backend_result, shots, xdata,
                 num_of_qubits, measured_qubit,
                 fit_p0, fit_bounds):

        BaseCoherenceFitter.__init__(self, '$T_1$',
                                     backend_result, shots, xdata,
                                     num_of_qubits, measured_qubit,
                                     BaseCoherenceFitter.exp_fit_fun,
                                     fit_p0, fit_bounds)

        self.time = self.params[1]
        self.time_err = self.params_err[1]
