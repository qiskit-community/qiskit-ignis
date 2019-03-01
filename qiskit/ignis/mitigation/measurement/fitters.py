# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=cell-var-from-loop


"""
Measurement correction fitters.
"""

import numpy as np
from qiskit import QiskitError
from .filters import MeasurementFilter

try:
    from matplotlib import pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class CompleteMeasFitter():
    """
    Measurement correction fitter for a full calibration
    """

    def __init__(self, results, state_labels, circlabel=''):
        """
        Initialize a measurement calibration matrix from the results of running
        the circuits returned by `measurement_calibration_circuits`

        Args:
            results: the results of running the measurement calibration
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
        """set cal_matrix."""
        self._cal_matrix = new_cal_matrix

    @property
    def filter(self):
        """return a measurement filter using the cal matrix"""
        return MeasurementFilter(self._cal_matrix, self._state_labels)

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
                    assign_fid_list[-1] += \
                        self._cal_matrix[state_idx_i][state_idx_j]
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
        ax.set_xlabel('Prepared State')
        ax.xaxis.set_label_position('top')
        ax.set_ylabel('Measured State')
        ax.set_xticks(np.arange(len(self._state_labels)))
        ax.set_yticks(np.arange(len(self._state_labels)))
        ax.set_xticklabels(self._state_labels)
        ax.set_yticklabels(self._state_labels)

        if show_plot:
            plt.show()
