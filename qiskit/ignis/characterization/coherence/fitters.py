# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Fitters of characteristic times
"""

from .. import BaseCoherenceFitter


class T1Fitter(BaseCoherenceFitter):
    """
    T1 fitter
    """

    def __init__(self, backend_result, xdata,
                 qubits,
                 fit_p0, fit_bounds):

        circuit_names = []
        for cind, _ in enumerate(xdata):
            circuit_names.append('t1circuit_%d_' % cind)

        BaseCoherenceFitter.__init__(self, '$T_1$',
                                     backend_result, xdata,
                                     qubits,
                                     BaseCoherenceFitter._exp_fit_fun,
                                     fit_p0, fit_bounds,
                                     circuit_names, expected_state='1',
                                     time_index=1)

    def plot(self, qind, series='0', ax=None, show_plot=False):

        ax = BaseCoherenceFitter.plot(self, qind, series, ax,
                                      show_plot)
        ax.set_ylabel("Excited State Population")

        return ax


class T2Fitter(BaseCoherenceFitter):
    """
    T2 fitter
    """

    def __init__(self, backend_result, xdata,
                 qubits, fit_p0, fit_bounds, circbasename='t2'):

        circuit_names = []
        for cind, _ in enumerate(xdata):
            circuit_names.append('%scircuit_%d_' % (circbasename, cind))

        BaseCoherenceFitter.__init__(self, '$T_2$',
                                     backend_result,
                                     xdata, qubits,
                                     BaseCoherenceFitter._exp_fit_fun,
                                     fit_p0, fit_bounds,
                                     circuit_names, expected_state='0',
                                     time_index=1)

    def plot(self, qind, series='0', ax=None, show_plot=False):

        ax = BaseCoherenceFitter.plot(self, qind, series,
                                      ax, show_plot)
        ax.set_ylabel("Ground State Population")

        return ax


class T2StarFitter(BaseCoherenceFitter):
    """
    T2* fitter
    """

    def __init__(self, backend_result, xdata,
                 qubits, fit_p0, fit_bounds):

        circuit_names = []
        for cind, _ in enumerate(xdata):
            circuit_names.append('t2starcircuit_%d_' % cind)

        BaseCoherenceFitter.__init__(self, '$T_2^*$',
                                     backend_result,
                                     xdata, qubits,
                                     T2StarFitter._osc_fit_fun,
                                     fit_p0, fit_bounds,
                                     circuit_names, expected_state='0',
                                     time_index=1)

    def plot(self, qind, series='0', ax=None, show_plot=False):

        ax = BaseCoherenceFitter.plot(self, qind, series,
                                      ax, show_plot)
        ax.set_ylabel("Ground State Population")

        return ax
