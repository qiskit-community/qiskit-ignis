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
Fitters of characteristic times
"""

import numpy as np
from ..fitters import BaseCoherenceFitter


class T1Fitter(BaseCoherenceFitter):
    """
    T1 fitter
    """

    def __init__(self, backend_result, xdata,
                 qubits,
                 fit_p0, fit_bounds,
                 time_unit='micro-seconds'):

        circuit_names = []
        for cind, _ in enumerate(xdata):
            circuit_names.append('t1circuit_%d_' % cind)

        BaseCoherenceFitter.__init__(self, '$T_1$',
                                     backend_result, xdata,
                                     qubits,
                                     self._exp_fit_fun,
                                     fit_p0, fit_bounds,
                                     circuit_names, expected_state='1',
                                     time_index=1, time_unit=time_unit)

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
                 qubits, fit_p0, fit_bounds, circbasename='t2',
                 time_unit='micro-seconds'):

        circuit_names = []
        for cind, _ in enumerate(xdata):
            circuit_names.append('%scircuit_%d_' % (circbasename, cind))

        BaseCoherenceFitter.__init__(self, '$T_2$',
                                     backend_result,
                                     xdata, qubits,
                                     self._exp_fit_fun,
                                     fit_p0, fit_bounds,
                                     circuit_names, expected_state='0',
                                     time_index=1, time_unit=time_unit)

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
                 qubits, fit_p0, fit_bounds,
                 time_unit='micro-seconds'):

        circuit_names = []
        for cind, _ in enumerate(xdata):
            circuit_names.append('t2starcircuit_%d_' % cind)

        BaseCoherenceFitter.__init__(self, '$T_2^*$',
                                     backend_result,
                                     xdata, qubits,
                                     self._osc_fit_fun,
                                     fit_p0, fit_bounds,
                                     circuit_names, expected_state='0',
                                     time_index=1, time_unit=time_unit)

    def guess_params(self, qind=0):
        """
        Guess fit parameters for oscillation data

        Args:
            qind: qubit index to guess fit parameters for

        Returns:
            List of fit guess parameters
        """

        a = np.max(self.ydata['0'][qind]['mean'])
        c = np.mean(self.ydata['0'][qind]['mean'])

        fft_data = np.fft.fft(self.ydata['0'][qind]['mean'])

        # find the max
        fft_freqs = np.fft.fftfreq(len(self.xdata),
                                   self.xdata[1]-self.xdata[0])

        # main freq
        main_freq = np.argmax(np.abs(fft_data)[1:])
        f_guess = fft_freqs[1:][main_freq]

        return [a, self.xdata[-1]*10, f_guess,
                np.angle(fft_data[1:][main_freq]), c]

    def plot(self, qind, series='0', ax=None, show_plot=False):

        ax = BaseCoherenceFitter.plot(self, qind, series,
                                      ax, show_plot)
        ax.set_ylabel("Ground State Population")

        return ax
