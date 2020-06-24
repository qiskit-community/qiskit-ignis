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

# pylint: disable=invalid-name

"""
Fitters for calibration routines
"""

import numpy as np
from ..fitters import IQFitter


class RabiFitter(IQFitter):
    """
    Rabi Experiment fitter
    """

    def __init__(self, backend_result, xdata,
                 qubits,
                 fit_p0, fit_bounds=None):

        """
            See BaseCalibrationFitter __init__

            fit_po is [amp, freq, phase, offset]
        """

        schedule_names = []
        for cind, _ in enumerate(xdata):
            schedule_names.append('rabisched_%d_' % cind)

        IQFitter.__init__(self, '$Rabi$',
                          backend_result, xdata,
                          qubits,
                          self._osc_nodecay_fit_fun,
                          fit_p0, fit_bounds,
                          schedule_names)

    def guess_params(self, qind=0):
        """
        Guess fit parameters for rabi oscillation data

        Args:
            qind (int): qubit index to guess fit parameters for

        Returns:
            list: List of fit guess parameters
                [amp, freq, phase, offset]
        """

        a_out = np.max(self.ydata['0'][qind]['mean'])
        c_out = np.mean(self.ydata['0'][qind]['mean'])

        fft_data = np.fft.fft(self.ydata['0'][qind]['mean'])

        # find the max
        fft_freqs = np.fft.fftfreq(len(self.xdata),
                                   self.xdata[1]-self.xdata[0])

        # main freq
        main_freq = np.argmax(np.abs(fft_data))
        f_guess = fft_freqs[main_freq]

        return [a_out, f_guess, np.angle(fft_data[main_freq]), c_out]

    def pi2_amplitude(self, qind=-1):
        r"""
        Return the pi/2 amplitude from the fit

        Args:
            qind (int): qubit index

        Returns:
            float: :math:`\frac{\pi}{2}` amp
        """

        return self.pi_amplitude(qind)/2

    def pi_amplitude(self, qind=-1):
        r"""
        Return the pi amplitude from the fit

        Args:
            qind (int): qubit index

        Returns:
            float: :math:`\pi` amp
        """

        piamp_list = self._get_param(1, -1)

        piamp_list = 1/np.array(piamp_list)/2.

        if qind == -1:
            return piamp_list

        return piamp_list[qind]

    def plot(self, qind, series='0', ax=None, show_plot=False):
        """
        Plot the data and fit

        Args:
            qind (int): qubit index
            series (str): data series to plot (for rabi data always '0')
            ax (Axes): matploblib axes (if none created)
            show_plot (bool): do plot.show

        Returns:
            Axes: Plot axes
        """

        ax = IQFitter.plot(self, qind, series, ax,
                           show_plot)

        ax.set_ylabel("IQ Signal")
        ax.set_xlabel("Drive Amplitude")
        ax.axvline(self.pi_amplitude(qind), color='black',
                   linestyle='dashed')

        return ax


class DragFitter(IQFitter):
    """
    Drag Experiment fitter
    """

    def __init__(self, backend_result, xdata,
                 qubits,
                 fit_p0, fit_bounds=None):

        """
            See IQFitter __init__

            fit_p0 is [a, x0, c] where the fit is a*(x-x0)^2+c
        """

        schedule_names = []
        for cind, _ in enumerate(xdata):
            schedule_names.append('dragsched_%d_' % cind)

        if fit_bounds is None:
            fit_bounds = ([-np.inf for e in range(len(fit_p0))],
                          [np.inf for e in range(len(fit_p0))])

        IQFitter.__init__(self, '$DRAG$',
                          backend_result, xdata,
                          qubits,
                          self._quadratic,
                          fit_p0, fit_bounds,
                          schedule_names)

    def guess_params(self, qind=0):
        """
        Guess parameters for the drag fit

        Args:
            qind (int): qubit index

        Returns:
            list: guess parameters
                [a, x0, c] where the fit is
                :math:`a * (x - x0)^{2+c}`
        """

        a = np.max(self.ydata['0'][qind]['mean']) - \
            np.min(self.ydata['0'][qind]['mean'])

        i1 = np.argmax(self.ydata['0'][qind]['mean'])
        i2 = np.argmin(self.ydata['0'][qind]['mean'])

        if np.abs(self.xdata[i1]) < np.abs(self.xdata[i2]):
            x0 = self.xdata[i1]
            a *= -1
            c = self.ydata['0'][qind]['mean'][i1]
        else:
            x0 = self.xdata[i2]
            c = self.ydata['0'][qind]['mean'][i2]

        return [a/np.max(self.xdata)**2, x0, c]

    def drag_amplitude(self, qind=-1):
        """
        Return the drag amplitude from the fit

        Args:
            qind (int): qubit index

        Returns:
            float: drag amp
        """

        return self._get_param(1, qind)

    def plot(self, qind, series='0', ax=None, show_plot=False):
        """
        Plot the data and fit

        Args:
            qind (int): qubit index
            series (str): data series to plot (for rabi data always '0')
            ax (Axes): matploblib axes (if none created)
            show_plot (bool): do plot.show

        Returns:
            Axes: Plot axes
        """

        ax = IQFitter.plot(self, qind, series, ax,
                           show_plot)

        ax.set_ylabel("IQ Signal")
        ax.set_xlabel("DRAG Amplitude")

        return ax
