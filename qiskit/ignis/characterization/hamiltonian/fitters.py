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


class ZZFitter(BaseCoherenceFitter):
    """
    ZZ fitter
    """

    def __init__(self, backend_result, xdata,
                 qubits, spectators,
                 fit_p0, fit_bounds,
                 time_unit='micro-seconds'):

        circuit_names = []
        for cind, _ in enumerate(xdata):
            circuit_names.append('zzcircuit_%d_' % cind)

        self._spectators = spectators

        BaseCoherenceFitter.__init__(self, '$ZZ$',
                                     backend_result, xdata,
                                     qubits,
                                     self._osc_nodecay_fit_fun,
                                     fit_p0, fit_bounds, circuit_names,
                                     series=['0', '1'], expected_state='0',
                                     time_index=1, time_unit=time_unit)

    def ZZ_rate(self, qind=-1):

        """
        Return the ZZ rate from the fit of the two curves

        Args:
            qind: qubit index to return (-1 return all)

        return a list of zz_rates
        """

        freq0 = self._get_param(1, qind, series='0', err=False)
        freq1 = self._get_param(1, qind, series='1', err=False)

        return np.array(freq1)-np.array(freq0)

    def plot_ZZ(self, qind, ax=None, show_plot=False):

        """
        Plot ZZ data. Will plot both traces on the plot.

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

        pltc = ['b', 'g']
        linec = ['r', 'black']

        for seriesind, series in enumerate(['0', '1']):

            ax.errorbar(self._xdata, self._ydata[series][qind]['mean'],
                        self._ydata[series][qind]['std'],
                        marker='.', markersize=9,
                        c=pltc[seriesind], linestyle='')
            ax.plot(self._xdata, self._fit_fun(self._xdata,
                                               *self._params[series][qind]),
                    c=linec[seriesind], linestyle='--',
                    label='Q%d in state %s' %
                    (self._spectators[qind], series))

        ax.tick_params(axis='x', labelsize=14, labelrotation=70)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_xlabel('Time [' + self._time_unit + ']', fontsize=16)
        ax.set_ylabel('Ground state population', fontsize=16)
        ax.set_title(self._description + ' for qubit ' +
                     str(self._qubits[qind]), fontsize=18)
        ax.legend(fontsize=12)
        ax.grid(True)
        if show_plot:
            plt.show()

        return ax
