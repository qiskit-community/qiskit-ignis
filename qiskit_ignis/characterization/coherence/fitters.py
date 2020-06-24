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

from typing import Union, List, Tuple
import numpy as np
import qiskit
from ..fitters import BaseCoherenceFitter


class T1Fitter(BaseCoherenceFitter):
    r"""
    Estimate T\ :sub:`1`\ , based on experiments outcomes,

    The experiments were created by `t1_circuits`, and executed on the device.

    The probabilities of measuring 1 is assumed to be of the form

    .. math::
        f(t) = A\mathrm{e}^{-t/T_1}+B,

    for unknown parameters `A`, `B`, and T\ :sub:`1`\ .

    Args:
        backend_result: result of execution of `t1_circuits` on the backend.
        xdata: delay times of the T\ :sub:`1` circuits.
        qubits:  indices of the qubits whose T\ :sub:`1`\ 's are to be measured.
        fit_p0: initial values to the fit parameters,
                where the order is :math:`(A, T_1, B)`.
        fit_bounds: bounds on the parameters to fit.
                    The first tuple is the lower bounds, in the order :math:`(A, T_1, B)`.
                    The second tuple is the upper bounds.
        time_unit: unit of delay times in `xdata`.
    """

    def __init__(self,
                 backend_result: qiskit.result.Result,
                 xdata: Union[List[float], np.array],
                 qubits: List[int],
                 fit_p0: List[float],  # any way to enforce length 3?
                 fit_bounds: Tuple[List[float], List[float]],
                 time_unit: str = 'micro-seconds'):

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
    r"""
    Estimate T\ :sub:`2`\ , based on experiments outcomes.

    The experiments were created by `t2_circuits`, and executed on the device.

    The probabilities of measuring 0 is assumed to be of the form

    .. math::
        f(t) = A\mathrm{e}^{-t/T_2}+B,

    for unknown parameters `A`, `B`, and T\ :sub:`2`\ .

    Args:
        backend_result: result of execution of `t2_circuits` on the backend.
        xdata: delay times of the T\ :sub:`2` circuits.
        qubits:  indices of the qubits whose T\ :sub:`2`\ 's are to be measured.
        fit_p0: initial values to the fit parameters,
                where the order is :math:`(A, T_2, B)`.
        fit_bounds: bounds on the parameters to fit.
                    The first tuple is the lower bounds, in the order :math:`(A, T_2, B)`.
                    The second tuple is the upper bounds.
        circbasename: prefix to all circuit names.
        time_unit: unit of delay times in `xdata`.
    """

    def __init__(self,
                 backend_result: qiskit.result.Result,
                 xdata: Union[List[float], np.array],
                 qubits: List[int],
                 fit_p0: List[float],  # any way to enforce length 3?
                 fit_bounds: Tuple[List[float], List[float]],
                 circbasename: str = 't2',
                 time_unit: str = 'micro-seconds'):

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
    r"""
    Estimate T\ :sub:`2`:sup:`*`\ , based on experiments outcomes.

    The experiments were created by `t2star_circuits`, and executed on the device.

    The probabilities of measuring 0 is assumed to be of the form

    .. math::
        f(t) = A\mathrm{e}^{-t / T_2^*}\cos(2\pi ft + \phi) + B

    for unknown parameters :math:`A, B, f, \phi, T_2^*`.

    Args:
        backend_result: result of execution of `t2star_circuits` on the backend.
        xdata: delay times of the T\ :sub:`2`:sup:`*` circuits.
        qubits:  indices of the qubits whose T\ :sub:`2`\:sup:`*` 's are to be measured.
        fit_p0: initial values to the fit parameters,
                where the order is :math:`(A, T_2^*, f, \phi, B)`.
        fit_bounds: bounds on the parameters to fit.
                    The first tuple is the lower bounds,
                    in the order :math:`(A, T_2^*, f, \phi, B)`.
                    The second tuple is the upper bounds.
        time_unit: unit of delay times in `xdata`.
    """

    def __init__(self,
                 backend_result: qiskit.result.Result,
                 xdata: Union[List[float], np.array],
                 qubits: List[int],
                 fit_p0: List[float],  # any way to enforce length 5?
                 fit_bounds: Tuple[List[float], List[float]],
                 time_unit: str = 'micro-seconds'):

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

    def guess_params(self,
                     qind: int = 0) -> List[float]:
        """
        Guess fit parameters for oscillation data

        Args:
            qind: qubit index to guess fit parameters for

        Returns:
            Fit guessed parameters
        """

        a_val = np.max(self.ydata['0'][qind]['mean'])
        c = np.mean(self.ydata['0'][qind]['mean'])

        fft_data = np.fft.fft(self.ydata['0'][qind]['mean'])

        # find the max
        fft_freqs = np.fft.fftfreq(len(self.xdata),
                                   self.xdata[1]-self.xdata[0])

        # main freq
        main_freq = np.argmax(np.abs(fft_data)[1:])
        f_guess = fft_freqs[1:][main_freq]

        return [a_val, self.xdata[-1]*10, f_guess,
                np.angle(fft_data[1:][main_freq]), c]

    def plot(self, qind, series='0', ax=None, show_plot=False):

        ax = BaseCoherenceFitter.plot(self, qind, series,
                                      ax, show_plot)
        ax.set_ylabel("Ground State Population")

        return ax
