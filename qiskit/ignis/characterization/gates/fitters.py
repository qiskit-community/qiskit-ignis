# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Fitters for hamiltonian parameters
"""

import numpy as np
from .. import BaseCoherenceFitter


class AmpCalFitter(BaseCoherenceFitter):
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

        BaseCoherenceFitter.__init__(self, '$AmpCal1Q$',
                                     backend_result, xdata,
                                     qubits, self._amp_cal_fit, fit_p0,
                                     fit_bounds, circuit_names,
                                     expected_state='1',
                                     time_index=1)

    @staticmethod
    def _amp_cal_fit(x, thetaerr, c):
        return BaseCoherenceFitter._cal_fit_fun(x, -0.5,
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

    def plot_err(self, qind, ax=None, show_plot=False):

        """
        Plot err data.

        Args:
            qind: qubit index to plot
            ax: plot axes
            show_plot: call plt.show()

        return the axes object
        """

        from matplotlib import pyplot as plt

        if ax is None:
            plt.figure()
            ax = plt.gca()

        ax.errorbar(self._xdata, self._ydata['0'][qind]['mean'],
                    self._ydata['0'][qind]['std'],
                    marker='.', markersize=9,
                    c='b', linestyle='')
        ax.plot(self._xdata, self._fit_fun(self._xdata,
                                           *self._params['0'][qind]),
                c='r', linestyle='--',
                label='Q%d' % (self._qubits[qind]))

        ax.tick_params(axis='x', labelsize=14, labelrotation=70)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_xlabel('time [micro-seconds]', fontsize=16)
        ax.set_ylabel('Excited state population', fontsize=16)
        ax.set_title(self._description + ' for qubit ' +
                     str(self._qubits[qind]), fontsize=18)
        ax.legend(fontsize=12)
        ax.grid(True)
        if show_plot:
            plt.show()

        return ax


class AngleCalFitter(BaseCoherenceFitter):
    """
    Amplitude error fitter
    """

    def __init__(self, backend_result, xdata,
                 qubits, fit_p0, fit_bounds):

        circuit_names = []
        for cind, _ in enumerate(xdata):
            circuit_names.append('anglecal1Qcircuit_%d_' % cind)

        # fit function is  0.5-0.5*sin(pi/2*x+delta*x+delta+pi/2)

        BaseCoherenceFitter.__init__(self, '$AngleCal1Q$',
                                     backend_result, xdata,
                                     qubits, self._angle_cal_fit, fit_p0,
                                     fit_bounds, circuit_names,
                                     expected_state='1',
                                     time_index=1)

    @staticmethod
    def _angle_cal_fit(x, thetaerr, c):
        return BaseCoherenceFitter._cal_fit_fun(x, -0.5,
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

    def plot_err(self, qind, ax=None, show_plot=False):

        """
        Plot err data.

        Args:
            qind: qubit index to plot
            ax: plot axes
            show_plot: call plt.show()

        return the axes object
        """

        from matplotlib import pyplot as plt

        if ax is None:
            plt.figure()
            ax = plt.gca()

        ax.errorbar(self._xdata, self._ydata['0'][qind]['mean'],
                    self._ydata['0'][qind]['std'],
                    marker='.', markersize=9,
                    c='b', linestyle='')
        ax.plot(self._xdata, self._fit_fun(self._xdata,
                                           *self._params['0'][qind]),
                c='r', linestyle='--',
                label='Q%d' % (self._qubits[qind]))

        ax.tick_params(axis='x', labelsize=14, labelrotation=70)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_xlabel('time [micro-seconds]', fontsize=16)
        ax.set_ylabel('Excited state population', fontsize=16)
        ax.set_title(self._description + ' for qubit ' +
                     str(self._qubits[qind]), fontsize=18)
        ax.legend(fontsize=12)
        ax.grid(True)
        if show_plot:
            plt.show()

        return ax


class AmpCalCXFitter(BaseCoherenceFitter):
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

        BaseCoherenceFitter.__init__(self, '$AmpCalCX$',
                                     backend_result, xdata,
                                     qubits, self._amp_calcx_fit, fit_p0,
                                     fit_bounds, circuit_names,
                                     expected_state='1',
                                     time_index=1)

    @staticmethod
    def _amp_calcx_fit(x, thetaerr, c):
        return BaseCoherenceFitter._cal_fit_fun(x, -0.5,
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

    def plot_err(self, qind, ax=None, show_plot=False):

        """
        Plot err data.

        Args:
            qind: qubit index to plot
            ax: plot axes
            show_plot: call plt.show()

        return the axes object
        """

        from matplotlib import pyplot as plt

        if ax is None:
            plt.figure()
            ax = plt.gca()

        ax.errorbar(self._xdata, self._ydata['0'][qind]['mean'],
                    self._ydata['0'][qind]['std'],
                    marker='.', markersize=9,
                    c='b', linestyle='')
        ax.plot(self._xdata, self._fit_fun(self._xdata,
                                           *self._params['0'][qind]),
                c='r', linestyle='--',
                label='Q%d' % (self._qubits[qind]))

        ax.tick_params(axis='x', labelsize=14, labelrotation=70)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_xlabel('time [micro-seconds]', fontsize=16)
        ax.set_ylabel('Excited state population', fontsize=16)
        ax.set_title(self._description + ' for qubit ' +
                     str(self._qubits[qind]), fontsize=18)
        ax.legend(fontsize=12)
        ax.grid(True)
        if show_plot:
            plt.show()

        return ax


class AngleCalCXFitter(BaseCoherenceFitter):
    """
    Amplitude error fitter
    """

    def __init__(self, backend_result, xdata,
                 qubits, fit_p0, fit_bounds):

        circuit_names = []
        for cind, _ in enumerate(xdata):
            circuit_names.append('anglecalcxcircuit_%d_' % cind)

        # fit function is  0.5-0.5*sin(pi/2*x+delta*x+delta+pi/2)

        BaseCoherenceFitter.__init__(self, '$AngleCalCX$',
                                     backend_result, xdata,
                                     qubits, self._angle_calcx_fit, fit_p0,
                                     fit_bounds, circuit_names,
                                     expected_state='1',
                                     time_index=1)

    @staticmethod
    def _angle_calcx_fit(x, thetaerr, c):
        return BaseCoherenceFitter._cal_fit_fun(x, -0.5,
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

    def plot_err(self, qind, ax=None, show_plot=False):

        """
        Plot err data.

        Args:
            qind: qubit index to plot
            ax: plot axes
            show_plot: call plt.show()

        return the axes object
        """

        from matplotlib import pyplot as plt

        if ax is None:
            plt.figure()
            ax = plt.gca()

        ax.errorbar(self._xdata, self._ydata['0'][qind]['mean'],
                    self._ydata['0'][qind]['std'],
                    marker='.', markersize=9,
                    c='b', linestyle='')
        ax.plot(self._xdata, self._fit_fun(self._xdata,
                                           *self._params['0'][qind]),
                c='r', linestyle='--',
                label='Q%d' % (self._qubits[qind]))

        ax.tick_params(axis='x', labelsize=14, labelrotation=70)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_xlabel('time [micro-seconds]', fontsize=16)
        ax.set_ylabel('Excited state population', fontsize=16)
        ax.set_title(self._description + ' for qubit ' +
                     str(self._qubits[qind]), fontsize=18)
        ax.legend(fontsize=12)
        ax.grid(True)
        if show_plot:
            plt.show()

        return ax
