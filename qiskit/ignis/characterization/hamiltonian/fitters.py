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

from typing import List, Optional, Tuple, Dict

import numpy as np
from qiskit.ignis.characterization import CharacterizationError
from qiskit.result import Result
from scipy.linalg import expm, norm
from scipy.optimize import minimize

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

        Returns:
            the axes object
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


class CRFitter:
    """
    CR Fitter.
    """
    def __init__(self,
                 backend_result: Result,
                 t_qubit: int,
                 cr_times: List[float]):
        """
        Create new CR Fitter.

        [1] Sheldon, S., Magesan, E., Chow, J. M. & Gambetta, J. M.
        Procedure for systematically tuning up cross-talk in the cross-resonance gate.
        Phys. Rev. A 93, 060302 (2016).

        Args:
            backend_result: experimental result to analyze.
            t_qubit: index of target qubit.
            cr_times: list of CR pulse duration in sec for Rabi experiments.
        """
        self.t_qubit = t_qubit
        self.cr_times = cr_times

        # get expectation values in each measurement basis
        self._expvs_x = self._get_meas_basis(backend_result, 'x')
        self._expvs_y = self._get_meas_basis(backend_result, 'y')
        self._expvs_z = self._get_meas_basis(backend_result, 'z')

        # rotation generators
        self._vec0 = None
        self._vec1 = None

    def fit(self,
            fit0: Optional[np.ndarray] = None,
            fit1: Optional[np.ndarray] = None,
            **fitter_kwargs) -> None:
        """
        Fit Rabi oscillation by Bloch equation to extract CR Hamiltonian.

        Args:
            fit0: initial fit parameters (Omega_x, Omega_y, Delta) for control qubit = 0.
            fit1: initial fit parameters (Omega_x, Omega_y, Delta) for control qubit = 1.
            **fitter_kwargs: see options of `scipy.optimize.minimize`.
        """
        res_x0, res_x1 = self._expvs_x
        res_y0, res_y1 = self._expvs_y
        res_z0, res_z1 = self._expvs_z

        if fit0 is None:
            fit0 = np.array([0, 0, 0])
        if fit1 is None:
            fit1 = np.array([0, 0, 0])

        def fit_func(params, *args):
            """ Fitting function for Bloch equation.
            """
            omega_x, omega_y, delta = params
            xs, ys, zs, ts = args

            # initial bloch vector of target qubit |0>
            vec_r0 = np.matrix([0, 0, 1]).T

            # see ref [1] eq. 5
            mat_a = np.matrix([[0, delta, omega_y],
                               [-delta, 0, -omega_x],
                               [-omega_y, omega_x, 0]])

            # fit equation, see ref [1] eq. 4
            residuals = np.zeros_like(ts)
            for ii, (x, y, z, t) in enumerate(zip(xs, ys, zs, ts)):
                vec_rt = np.matrix([x, y, z]).T
                residuals[ii] = norm(vec_rt - expm(mat_a * t) * vec_r0)

            return np.sum(residuals)

        res0 = minimize(fun=fit_func, x0=fit0,
                        args=(res_x0, res_y0, res_z0, self.cr_times),
                        **fitter_kwargs)
        res1 = minimize(fun=fit_func, x0=fit1,
                        args=(res_x1, res_y1, res_z1, self.cr_times),
                        **fitter_kwargs)

        self._vec0 = res0.x
        self._vec1 = res1.x

    def hamiltonian(self) -> Dict[str, float]:
        """
        Return CR Hamiltonian in dictionary format.
        """
        if self._vec0 is None or self._vec1 is None:
            raise CharacterizationError('No fitting results. Run `.fit()` first.')

        cr_terms = {
            'IX': (self.vec0[0] + self.vec1[0]) / 2,
            'IY': (self.vec0[1] + self.vec1[1]) / 2,
            'IZ': (self.vec0[2] + self.vec1[2]) / 2,
            'ZX': (self.vec0[0] - self.vec1[0]) / 2,
            'ZY': (self.vec0[1] - self.vec1[1]) / 2,
            'ZZ': (self.vec0[2] - self.vec1[2]) / 2
        }

        return cr_terms

    def plot_rabi_oscillation(self,
                              resolution: Optional[int] = 1000,
                              show_plot: Optional[bool] = True):
        """
        Visualize Rabi oscillation.

        Args:
            resolution: number of points for curve fitting.
            show_plot: call plt.show()
        """
        try:
            from matplotlib import pyplot as plt
        except ImportError:
            raise CharacterizationError('matplotlib is not installed.')

        fig = plt.figure(figsize=(15, 3))

        ax_x = fig.add_subplot(131)
        ax_y = fig.add_subplot(132)
        ax_z = fig.add_subplot(133)

        cr_times = np.array(self.cr_times) * 1e9

        ax_x.scatter(cr_times, self._expvs_x[0], color='b', label='|00>')
        ax_y.scatter(cr_times, self._expvs_y[0], color='b', label='|00>')
        ax_z.scatter(cr_times, self._expvs_z[0], color='b', label='|00>')
        ax_x.scatter(cr_times, self._expvs_x[1], color='r', label='|10>')
        ax_y.scatter(cr_times, self._expvs_y[1], color='r', label='|10>')
        ax_z.scatter(cr_times, self._expvs_z[1], color='r', label='|10>')

        # add fitting curve if fit is done
        if self._vec0 is not None and self._vec1 is not None:
            ts = np.linspace(0, max(self.cr_times), resolution)
            vec_r0 = np.matrix([0, 0, 1]).T

            xs0, ys0, zs0 = [np.zeros_like(ts, dtype=float) for _ in ('x', 'y', 'z')]
            xs1, ys1, zs1 = [np.zeros_like(ts, dtype=float) for _ in ('x', 'y', 'z')]

            omega_x0, omega_y0, delta0 = self._vec0
            omega_x1, omega_y1, delta1 = self._vec1

            mat_a0 = np.matrix([[0, delta0, omega_y0],
                                [-delta0, 0, -omega_x0],
                                [-omega_y0, omega_x0, 0]])

            mat_a1 = np.matrix([[0, delta1, omega_y1],
                                [-delta1, 0, -omega_x1],
                                [-omega_y1, omega_x1, 0]])

            for ii, t in enumerate(ts):
                vec_rt0 = expm(mat_a0 * t) * vec_r0
                vec_rt1 = expm(mat_a1 * t) * vec_r0

                xs0[ii], ys0[ii], zs0[ii] = np.array(vec_rt0)
                xs1[ii], ys1[ii], zs1[ii] = np.array(vec_rt1)

            ts *= 1e9

            ax_x.plot(ts, xs0, 'b:')
            ax_y.plot(ts, ys0, 'b:')
            ax_z.plot(ts, zs0, 'b:')
            ax_x.plot(ts, xs1, 'r:')
            ax_y.plot(ts, ys1, 'r:')
            ax_z.plot(ts, zs1, 'r:')

        ax_x.set_xlim(0, max(cr_times))
        ax_y.set_xlim(0, max(cr_times))
        ax_z.set_xlim(0, max(cr_times))

        ax_x.set_ylim(-1, 1)
        ax_y.set_ylim(-1, 1)
        ax_z.set_ylim(-1, 1)

        ax_x.legend()
        ax_y.legend()
        ax_z.legend()

        if show_plot:
            plt.show()
            return None

        return fig

    def plot_bloch_vec(self,
                       show_plot: Optional[bool] = True):
        """
        Visualize Bloch vector trajectory.

        Args:
            show_plot: call plt.show()
        """
        try:
            from matplotlib import pyplot as plt
        except ImportError:
            raise CharacterizationError('matplotlib is not installed.')

        from qiskit.visualization.bloch import Bloch

        fig = plt.figure(figsize=(10, 5))

        ax_state0 = fig.add_subplot(121, projection='3d')
        ax_state1 = fig.add_subplot(122, projection='3d')

        pb0 = Bloch(axes=ax_state0)
        pb0.add_points((self._expvs_x[0], self._expvs_y[0], self._expvs_z[0]), meth='l')
        pb0.add_points((self._expvs_x[0], self._expvs_y[0], self._expvs_z[0]), meth='s')
        pb0.render(title='Initial state |00>')

        pb1 = Bloch(axes=ax_state1)
        pb1.add_points((self._expvs_x[1], self._expvs_y[1], self._expvs_z[1]), meth='l')
        pb1.add_points((self._expvs_x[1], self._expvs_y[1], self._expvs_z[1]), meth='s')
        pb1.render(title='Initial state |10>')

        if show_plot:
            plt.show()
            return None

        return fig

    @property
    def vec0(self):
        """
        Return generator when control qubit is |0>.
        """
        return self._vec0

    @property
    def vec1(self):
        """
        Return generator when control qubit is |1>.
        """
        return self._vec1

    def _get_meas_basis(self,
                        backend_result: Result,
                        axis: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract expected value from experimental result.

        Args:
            backend_result: experimental results.
            axis: basis of measurement.
        """
        expvs_c0 = np.zeros_like(self.cr_times, dtype=float)
        expvs_c1 = np.zeros_like(self.cr_times, dtype=float)

        for index in range(len(self.cr_times)):
            for c_state in (0, 1):
                experiment_name = '%s,%s,%s' % (index, axis, c_state)
                counts = backend_result.get_counts(experiment_name)
                # convert population into eigenvalue
                expv = 0
                for key, val in counts.items():
                    if key[::-1][self.t_qubit] == '1':
                        expv -= val
                    else:
                        expv += val
                expv /= sum(counts.values())

                if c_state == 0:
                    expvs_c0[index] = expv
                if c_state == 1:
                    expvs_c1[index] = expv

        return expvs_c0, expvs_c1

