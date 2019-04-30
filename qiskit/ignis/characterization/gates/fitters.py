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
Fitters for hamiltonian parameters
"""

import numpy as np
from .. import BaseGateFitter


class AmpCalFitter(BaseGateFitter):
    """
    Amplitude error fitter
    """

    def __init__(self, backend_result, xdata,
                 qubits, fit_p0, fit_bounds):

        circuit_names = []
        for cind, _ in enumerate(xdata):
            circuit_names.append('ampcal1Qcircuit_%d_' % cind)

        # theoretically
        # curve is 0.5-0.5*cos((x+1)*2*(pi/4+dphi))
        # cos(pi/2*x + 2*x*dphi + pi/2+2*dphi)

        BaseGateFitter.__init__(self, '$AmpCal1Q$',
                                backend_result, xdata,
                                qubits, self._amp_cal_fit, fit_p0,
                                fit_bounds, circuit_names,
                                expected_state='1')

    @staticmethod
    def _amp_cal_fit(x, thetaerr, c):
        return AmpCalFitter._cal_fit_fun(x, -0.5,
                                         thetaerr, thetaerr,
                                         np.pi/2, np.pi/2, c)

    def angle_err(self, qind=-1):

        """
        Return the gate angle error

        Args:
            qind: qubit index to return (-1 return all)

        return a list of errors
        """

        fitparam = self._get_param(0, qind, series='0', err=False)

        return np.array(fitparam)/2.

    def plot(self, qind, series='0', ax=None, show_plot=False):

        ax = BaseGateFitter.plot(self, qind, series, ax,
                                 show_plot)
        return ax


class AngleCalFitter(BaseGateFitter):
    """
    Amplitude error fitter
    """

    def __init__(self, backend_result, xdata,
                 qubits, fit_p0, fit_bounds):

        circuit_names = []
        for cind, _ in enumerate(xdata):
            circuit_names.append('anglecal1Qcircuit_%d_' % cind)

        # fit function is  0.5-0.5*sin(pi/2*x+delta*x+delta+pi/2)

        BaseGateFitter.__init__(self, '$AngleCal1Q$',
                                backend_result, xdata,
                                qubits, self._angle_cal_fit, fit_p0,
                                fit_bounds, circuit_names,
                                expected_state='1')

    @staticmethod
    def _angle_cal_fit(x, thetaerr, c):
        return AngleCalFitter._cal_fit_fun(x, -0.5,
                                           thetaerr, thetaerr,
                                           np.pi/2, np.pi/2, c)

    def angle_err(self, qind=-1):

        """
        Return the gate angle error

        Args:
            qind: qubit index to return (-1 return all)

        return a list of errors
        """

        fitparam = self._get_param(0, qind, series='0', err=False)

        return np.array(fitparam)/2

    def plot(self, qind, series='0', ax=None, show_plot=False):

        ax = BaseGateFitter.plot(self, qind, series, ax,
                                 show_plot)

        return ax


class AmpCalCXFitter(BaseGateFitter):
    """
    Amplitude error fitter
    """

    def __init__(self, backend_result, xdata,
                 qubits, fit_p0, fit_bounds):

        circuit_names = []
        for cind, _ in enumerate(xdata):
            circuit_names.append('ampcalcxcircuit_%d_' % cind)

        # theoretically
        # curve is 0.5-0.5*cos((x+1)*2*(pi/4+dphi))
        # cos(pi/2*x + 2*x*dphi + pi/2+2*dphi)

        BaseGateFitter.__init__(self, '$AmpCalCX$',
                                backend_result, xdata,
                                qubits, self._amp_calcx_fit, fit_p0,
                                fit_bounds, circuit_names,
                                expected_state='1')

    @staticmethod
    def _amp_calcx_fit(x, thetaerr, c):
        return AmpCalCXFitter._cal_fit_fun(x, -0.5,
                                           thetaerr, 0,
                                           np.pi, np.pi/2, c)

    def angle_err(self, qind=-1):

        """
        Return the gate angle error

        Args:
            qind: qubit index to return (-1 return all)

        return a list of errors
        """

        fitparam = self._get_param(0, qind, series='0', err=False)

        return np.array(fitparam)/2.

    def plot(self, qind, series='0', ax=None, show_plot=False):

        ax = BaseGateFitter.plot(self, qind, series, ax,
                                 show_plot)

        return ax


class AngleCalCXFitter(BaseGateFitter):
    """
    Amplitude error fitter
    """

    def __init__(self, backend_result, xdata,
                 qubits, fit_p0, fit_bounds):

        circuit_names = []
        for cind, _ in enumerate(xdata):
            circuit_names.append('anglecalcxcircuit_%d_' % cind)

        # fit function is  0.5-0.5*sin(pi/2*x+delta*x+delta+pi/2)

        BaseGateFitter.__init__(self, '$AngleCalCX$',
                                backend_result, xdata,
                                qubits, self._angle_calcx_fit, fit_p0,
                                fit_bounds, circuit_names,
                                expected_state='1')

    @staticmethod
    def _angle_calcx_fit(x, thetaerr, c):
        return AngleCalCXFitter._cal_fit_fun(x, -0.5,
                                             thetaerr, 0.0,
                                             np.pi, np.pi/2, c)

    def angle_err(self, qind=-1):

        """
        Return the gate angle error

        Args:
            qind: qubit index to return (-1 return all)

        return a list of errors
        """

        fitparam = self._get_param(0, qind, series='0', err=False)

        return np.array(fitparam)/2

    def plot(self, qind, series='0', ax=None, show_plot=False):

        ax = BaseGateFitter.plot(self, qind, series, ax,
                                 show_plot)

        return ax
