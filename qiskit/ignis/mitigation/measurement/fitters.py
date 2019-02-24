# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=cell-var-from-loop


"""
Measurement correction fitters.
"""

from scipy.optimize import minimize
import scipy.linalg as la
import numpy as np
from qiskit import QiskitError

try:
    from matplotlib import pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class MeasurementFitter():
    """Measurement correction fitter"""

    def __init__(self, results, state_labels, circlabel=''):
        """
        Initialize a measurement calibration matrix from the results of running
        the circuits returned by `measurement_calibration_circuits`

        Args:
            results: the results of running the measurement calibration c
                ciruits. If this is None the user will set a call matrix later

            state_labels: list of calibration state labels
                returned from `measurement_calibration_circuits`. The output matrix
                will obey this ordering.
        """

        self._results = results
        self._state_labels = state_labels
        self._cal_matrix = None
        self._circlabel = circlabel

        if self._results is not None:
            self._build_calibration_matrix()

    @property
    def cal_matrix(self):
        """Return cal_matrix."""
        return self._cal_matrix

    @cal_matrix.setter
    def cal_matrix(self, new_cal_matrix):
        """Set cal_matrix."""
        self._cal_matrix = new_cal_matrix

    def readout_fidelity(self, label_list=None):
        """
        Based on the results output the readout fidelity which is the
        trace of the calibration matrix

        Args:
            label_list: If none returns the average assignment fidelity
            of a single state. Otherwise it returns the assignment fidelity
            to be in any one of these states averaged over the second index.

        Returns:
            readout fidelity (assignment fidelity)


        Additional Information:
            The on-diagonal elements of the calibration matrix are the
            probabilities of measuring state 'x' given preparation of state
            'x' and so the trace is the average assignment fidelity
        """

        if self._cal_matrix is None:
            raise QiskitError("Cal matrix has not been set")

        fidelity_label_list = []
        if label_list is None:
            fidelity_label_list = [[i] for i in range(len(self._cal_matrix))]
        else:
            for fid_sublist in label_list:
                fidelity_label_list.append([])
                for fid_statelabl in fid_sublist:
                    for label_idx, label in enumerate(self._state_labels):
                        if fid_statelabl == label:
                            fidelity_label_list[-1].append(label_idx)
                            continue

        # fidelity_label_list is a 2D list of indices in the
        # cal_matrix, we find the assignment fidelity of each
        # row and average over the list
        assign_fid_list = []

        for fid_label_sublist in fidelity_label_list:
            assign_fid_list.append(0)
            for state_idx_i in fid_label_sublist:
                for state_idx_j in fid_label_sublist:
                    assign_fid_list[-1] += self._cal_matrix[state_idx_i,
                                                            state_idx_j]
            assign_fid_list[-1] /= len(fid_label_sublist)

        return np.mean(assign_fid_list)

    def _build_calibration_matrix(self):
        """
        Build the measurement calibration matrix from the results of running
        the circuits returned by `measurement_calibration`

        Creates a 2**n x 2**n matrix that can be used to correct measurement
        errors
        """

        cal_matrix = np.zeros(
            [len(self._state_labels), len(self._state_labels)], dtype=float)

        for stateidx, state in enumerate(self._state_labels):
            state_cnts = self._results.get_counts('%scal_%s' %
                                                  (self._circlabel, state))
            shots = sum(state_cnts.values())
            for stateidx2, state2 in enumerate(self._state_labels):
                cal_matrix[stateidx, stateidx2] = state_cnts.get(
                    state2, 0) / shots

        self._cal_matrix = cal_matrix.transpose()

    def apply(self, raw_data, method='least_squares'):
        """
        Apply the calibration matrix to results

        Args:
            raw_data: The data to be corrected. Can be in a number of forms.
                Form1: a counts dictionary from results.get_counts
                Form2: a list of counts of length==len(state_labels)
                Form3: a list of counts of length==M*len(state_labels) where M
                    is an integer (e.g. for use with the tomography data)

            method (str): fitting method. If None, then least_squares is used.
                'pseudo_inverse': direct inversion of the A matrix
                'least_squares': constrained to have physical probabilities

        Returns:
            The corrected data in the same form as raw_data

        Additional Information:

            e.g.
            calcircuits, state_labels = measurement_calibration(
                qiskit.QuantumRegister(5))
            job = qiskit.execute(calcircuits)
            meas_fitter = MeasurementFitter(job.results(),
                                            state_labels)

            job2 = qiskit.execute(my_circuits)
            result2 = job2.results()

            error_mitigated_counts = meas_fitter.apply(
                result2.get_counts('circ1'))

        """
        # check forms of raw_data
        if isinstance(raw_data, dict):
            # counts dictionary
            data_format = 0
            # convert to form2
            raw_data2 = [np.zeros(len(self._state_labels), dtype=float)]
            for stateidx, state in enumerate(self._state_labels):
                raw_data2[0][stateidx] = raw_data.get(state, 0)

        elif isinstance(raw_data, list):
            size_ratio = len(raw_data)/len(self._state_labels)
            if len(raw_data) == len(self._state_labels):
                data_format = 1
                raw_data2 = [raw_data]
            elif int(size_ratio) == size_ratio:
                data_format = 2
                size_ratio = int(size_ratio)
                # make the list into chunks the size of state_labels for easier
                # processing
                raw_data2 = np.zeros([size_ratio, len(self._state_labels)])
                for i in range(size_ratio):
                    raw_data2[i][:] = raw_data[
                        i * len(self._state_labels):(i + 1)*len(
                            self._state_labels)]
            else:
                raise QiskitError("Data list is not an integer multiple "
                                  "of the number of calibrated states")

        else:
            raise QiskitError("Unrecognized type for raw_data.")

        # Apply the correction
        for data_idx, _ in enumerate(raw_data2):

            if method == 'pseudo_inverse':
                raw_data2[data_idx] = np.dot(
                    la.pinv(self._cal_matrix), raw_data2[data_idx])

            elif method == 'least_squares':
                nshots = sum(raw_data2[data_idx])

                def fun(x):
                    return sum(
                        (raw_data2[data_idx] - np.dot(self._cal_matrix, x))**2)
                x0 = np.random.rand(len(self._state_labels))
                x0 = x0 / sum(x0)
                cons = ({'type': 'eq', 'fun': lambda x: nshots - sum(x)})
                bnds = tuple((0, nshots) for x in x0)
                res = minimize(fun, x0, method='SLSQP',
                               constraints=cons, bounds=bnds, tol=1e-6)
                raw_data2[data_idx] = res.x

            else:
                raise QiskitError("Unrecognized method.")

        if data_format == 2:
            # flatten back out the list
            raw_data2 = raw_data2.flatten()

        elif data_format == 0:
            # convert back into a counts dictionary
            new_count_dict = {}
            for stateidx, state in enumerate(self._state_labels):
                if raw_data2[0][stateidx] != 0:
                    new_count_dict[state] = raw_data2[0][stateidx]

            raw_data2 = new_count_dict
        else:
            raw_data2 = raw_data2[0]
        return raw_data2

    def plot_calibration(self, ax=None, show_plot=True):
        """
        Plot the calibration matrix (2D color grid plot)

        Args:
            show_plot (bool): call plt.show()

        """

        if self._cal_matrix is None:
            raise QiskitError("Cal matrix has not been set")

        if not HAS_MATPLOTLIB:
            raise ImportError('The function plot_rb_data needs matplotlib. '
                              'Run "pip install matplotlib" before.')

        if ax is None:
            plt.figure()
            ax = plt.gca()

        axim = ax.matshow(self._cal_matrix, cmap=plt.cm.binary, clim=[0, 1])
        ax.figure.colorbar(axim)

        if show_plot:
            plt.show()
