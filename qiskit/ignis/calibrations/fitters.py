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
Fitters for calibration routines
"""

import numpy as np
from qiskit import QiskitError
from ..characterization.fitters import BaseFitter


class BaseCalibrationFitter(BaseFitter):
    """
    Base Fitter Class for experiments with Level 1 results
    """

    def __init__(self, description, backend_result, xdata,
                 qubits, fit_fun, fit_p0,
                 fit_bounds, circuit_names,
                 series=None):

        """
        See BaseFitter __init__
        """

        BaseFitter.__init__(self, description,
                            backend_result, xdata,
                            qubits, fit_fun,
                            fit_p0, fit_bounds, circuit_names,
                            series)

    def _build_iq_list(self):
        """
        From a list of results, calculate the mean
        """

        iq_list = {}
        iq_list_mean = {}
        iq_list_var = {}
        shots_list = {}
        meas_ret = ''

        for single_result in self._backend_result_list:
            # go through each of the schedules in this run
            for result in single_result.results:
                sname = result.header.name
                mem_slots = result.header.memory_slots

                if meas_ret == '':
                    # update
                    if result.meas_return == 'avg':
                        meas_ret = 'avg'
                    else:
                        meas_ret = 'single'

                if result.meas_level != 1:
                    raise QiskitError('This fitter works with IQ data')

                if meas_ret != result.meas_return:
                    raise QiskitError('All data must be single shot or '
                                      'averaged')

                shots_list[sname] = (result.shots +
                                     shots_list.get(sname, 0))

                if meas_ret == 'avg':

                    # data is averaged already, but if we need to
                    # further average over different runs
                    # we need to take into account the full number of shots

                    iq_list[sname] = (result.shots *
                                      single_result.get_memory(sname) +
                                      iq_list.get(sname, np.zeros(mem_slots)))

                else:

                    # data is in single shot IQ mode
                    iq_list[sname] = (result.shots *
                                      single_result.get_memory(sname) +
                                      iq_list.get(sname,
                                                  np.zeros([result.shots,
                                                            mem_slots])))

        for sname in iq_list:

            if meas_ret == 'avg':
                iq_list_mean[sname] = iq_list[sname]/shots_list[sname]
                iq_list_var[sname] = np.ones(len(iq_list_mean[sname]))*1e-4
            else:
                iq_list_mean[sname] = np.sum(iq_list[sname])/shots_list[sname]
                iq_list_var[sname] = np.std(iq_list[sname])

        return iq_list_mean, iq_list_var

    def _calc_data(self):
        """
        Calculate the IQ signal from the results

        Overloaded from Base Fitter class which assumed shots
        """

        iq_list_mean, iq_list_var = self._build_iq_list()

        self._ydata = {}
        for _, serieslbl in enumerate(self._series):
            self._ydata[serieslbl] = []
            for qind, _ in enumerate(self._qubits):
                self._ydata[serieslbl].append({'mean': [], 'std': []})
                mean_list = self._ydata[serieslbl][-1]['mean']
                var_list = self._ydata[serieslbl][-1]['std']
                for circ, _ in enumerate(self._xdata):
                    circname = self._circuit_names[circ] + serieslbl
                    mean_list.append(iq_list_mean[circname][qind])
                    var_list.append(iq_list_var[circname][qind])
                    # problem for the fitter if one of the std points is
                    # exactly zero
                    if var_list[-1] == 0:
                        var_list[-1] = 1e-4

                # project the data onto a line
                # mean over all the experiment
                q_exp_mean = np.mean(mean_list)
                mean_list -= q_exp_mean
                real_ext = np.mean(np.abs(np.real(mean_list)))
                imag_ext = np.mean(np.abs(np.imag(mean_list)))
                crot = real_ext/(real_ext**2+imag_ext**2)**0.5
                srot = imag_ext/(real_ext**2+imag_ext**2)**0.5
                mean_list = crot*np.real(mean_list)+srot*np.imag(mean_list)
                self._ydata[serieslbl][-1]['mean'] = mean_list

    def plot(self, qind, series='0', ax=None, show_plot=True):
        """
        Plot calibration data.

        Args:
            qind: qubit index to plot
            ax: plot axes
            show_plot: call plt.show()

        Returns:
            The axes object
        """

        from matplotlib import pyplot as plt

        if ax is None:
            plt.figure()
            ax = plt.gca()

        ax.errorbar(self._xdata, self._ydata[series][qind]['mean'],
                    self._ydata[series][qind]['std'],
                    marker='.', markersize=9,
                    c='b', linestyle='')
        ax.plot(self._xdata, self._fit_fun(self._xdata,
                                           *self._params[series][qind]),
                c='r', linestyle='--',
                label='Q%d' % (self._qubits[qind]))

        ax.tick_params(axis='x', labelsize=14, labelrotation=70)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_xlabel('X', fontsize=16)
        ax.set_ylabel('IQ Signal', fontsize=16)
        ax.set_title(self._description + ' for qubit ' +
                     str(self._qubits[qind]), fontsize=18)
        ax.legend(fontsize=12)
        ax.grid(True)
        if show_plot:
            plt.show()

        return ax


class RabiFitter(BaseCalibrationFitter):
    """
    Rabi Experiment fitter
    """

    def __init__(self, backend_result, xdata,
                 qubits,
                 fit_p0, fit_bounds=None):

        """
            See BaseCalibrationFitter __init__
        """

        schedule_names = []
        for cind, _ in enumerate(xdata):
            schedule_names.append('rabicircuit_%d_' % cind)

        BaseCalibrationFitter.__init__(self, '$Rabi$',
                                       backend_result, xdata,
                                       qubits,
                                       self._osc_nodecay_fit_fun,
                                       fit_p0, fit_bounds,
                                       schedule_names)

    def guess_params(self, qind=0):
        """
        Guess fit parameters for rabi oscillation data

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
        main_freq = np.argmax(np.abs(fft_data))
        f_guess = fft_freqs[main_freq]

        return [a, f_guess, np.angle(fft_data[main_freq]), c]

    def pi2amp(self, qind=-1):
        """
        Return the pi/2 amplitude from the fit

        Args:
            qind: qubit index

        Returns:
            Pi/2 amp (float)
        """

        return self.piamp(qind)/2

    def piamp(self, qind=-1):
        """
        Return the pi amplitude from the fit

        Args:
            qind: qubit index

        Returns:
            Pi amp (float)
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
            qind: qubit index
            series: data series to plot (for rabi data always '0')
            ax: matploblib axes (if none created)
            show_plot: do plot.show

        Returns:
            Plot axes
        """

        ax = BaseCalibrationFitter.plot(self, qind, series, ax,
                                        show_plot)
        ax.set_ylabel("IQ Signal")
        ax.set_xlabel("Drive Amplitude")
        ax.axvline(self.piamp(qind), color='black', linestyle='dashed')

        return ax


class DragFitter(BaseCalibrationFitter):
    """
    Drag Experiment fitter
    """

    def __init__(self, backend_result, xdata,
                 qubits,
                 fit_p0, fit_bounds=None):

        """
            See BaseCalibrationFitter __init__
        """

        schedule_names = []
        for cind, _ in enumerate(xdata):
            schedule_names.append('dragcircuit_%d_' % cind)

        if fit_bounds is None:
            fit_bounds = ([-np.inf for e in range(len(fit_p0))],
                          [np.inf for e in range(len(fit_p0))])

        BaseCalibrationFitter.__init__(self, '$DRAG$',
                                       backend_result, xdata,
                                       qubits,
                                       self._quadratic,
                                       fit_p0, fit_bounds,
                                       schedule_names)

    def guess_params(self, qind=0):
        """
        Guess parameters for the drag fit

        Args:
            qind: qubit index

        Returns:
            guess parameters (list)
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

    def dragamp(self, qind=-1):
        """
        Return the drag amplitude from the fit

        Args:
            qind: qubit index

        Returns:
            drag amp (float)
        """

        return self._get_param(1, qind)

    def plot(self, qind, series='0', ax=None, show_plot=False):
        """
        Plot the data and fit

        Args:
            qind: qubit index
            series: data series to plot (for rabi data always '0')
            ax: matploblib axes (if none created)
            show_plot: do plot.show

        Returns:
            Plot axes
        """

        ax = BaseCalibrationFitter.plot(self, qind, series, ax,
                                        show_plot)
        ax.set_ylabel("IQ Signal")
        ax.set_xlabel("DRAG Amplitude")

        return ax
