# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=cell-var-from-loop


"""
Measurement correction fitters.
"""

import copy
import numpy as np
from qiskit import QiskitError
from .filters import MeasurementFilter, TensoredFilter

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
            ciruits. If this is None the user will set a calibrarion matrix
            later

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
        normalized trace of the calibration matrix

        Args:
            label_list: If none returns the average assignment fidelity
            of a single state. Otherwise it returns the assignment fidelity
            to be in any one of these states averaged over the second index.

        Returns:
            readout fidelity (assignment fidelity)


        Additional Information:
            The on-diagonal elements of the calibration matrix are the
            probabilities of measuring state 'x' given preparation of state
            'x' and so the normalized trace is the average assignment fidelity
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


class TensoredMeasFitter():
    """
    Measurement correction fitter for a tensored calibration
    """

    def __init__(self, results, state_labels, mit_pattern=None, circlabel=''):
        """
        Initialize a measurement calibration matrix from the results of running
        the circuits returned by `measurement_calibration_circuits`

        Args:
            results: the results of running the measurement calibration
            circuits. If this is None the user will set calibration matrices
            later

            state_labels: list of calibration state labels
            returned from `measurement_calibration_circuits`

            mit_pattern: see tensored_meas_cal in circuits.py
        """

        self._results = results
        self._state_labels = state_labels
        self._cal_matrices = None
        self._circlabel = circlabel

        if mit_pattern is None:
            self._qubit_list_sizes = [len(state_labels[0])]
        else:
            self._qubit_list_sizes = \
                [len(qubit_list) for qubit_list in mit_pattern]
            if self.nqubits != len(state_labels[0]):
                raise ValueError("mit_pattern does not match state_labels")

        if self._results is not None:
            self._build_calibration_matrices()

    @property
    def cal_matrices(self):
        """Return cal_matrices."""
        return self._cal_matrices

    @cal_matrices.setter
    def cal_matrices(self, new_cal_matrices):
        """set cal_matrices."""
        self._cal_matrices = copy.deepcopy(new_cal_matrices)

    @property
    def filter(self):
        """return a measurement filter using the cal matrices"""
        return TensoredFilter(self._cal_matrices, self._qubit_list_sizes)

    @property
    def nqubits(self):
        """Return _qubit_list_sizes"""
        return sum(self._qubit_list_sizes)

    def readout_fidelity(self, label_list=None):
        """
        Based on the results output the readout fidelity, which is the average
        of the diagonal entries in the calibration matrices

        Args:
            label_list (list of lists on states):
            Returns the average fidelity over of the groups of states.
            If None then each state used in the construction of the
            calibration matrices forms a group of size 1

        Returns:
            readout fidelity (assignment fidelity)


        Additional Information:
            The on-diagonal elements of the calibration matrices are the
            probabilities of measuring state 'x' given preparation of state
            'x'
        """

        if self._cal_matrices is None:
            raise QiskitError("Cal matrix has not been set")

        if label_list is None:
            # TODO: consider changing the default label_list,
            # probably to all the states
            label_list = [[label] for label in self._state_labels]

        assign_fid_list = []
        for state_list in label_list:
            state_list_prob = 0.
            for state1 in state_list:
                for state2 in state_list:
                    end_index = self.nqubits
                    state1_state2_prob = 1.
                    for list_size, cal_mat \
                            in zip(self._qubit_list_sizes,
                                   self._cal_matrices):
                        start_index = end_index - list_size
                        substate1_as_int = \
                            int(state1[start_index:end_index], 2)
                        substate2_as_int = \
                            int(state2[start_index:end_index], 2)
                        end_index = start_index

                        state1_state2_prob *= \
                            cal_mat[substate1_as_int][substate2_as_int]

                    state_list_prob += state1_state2_prob

            assign_fid_list.append(state_list_prob)

        return np.mean(assign_fid_list)

    def _build_calibration_matrices(self):
        """
        Build the measurement calibration matrices from the results of running
        the circuits returned by `measurement_calibration`
        """

        self._cal_matrices = []
        for list_size in self._qubit_list_sizes:
            self._cal_matrices.append(np.zeros([2**list_size, 2**list_size],
                                               dtype=float))

        for state in self._state_labels:
            state_cnts = self._results.get_counts('%scal_%s' %
                                                  (self._circlabel, state))
            for measured_state, counts in state_cnts.items():
                end_index = self.nqubits
                for list_size, cal_mat \
                        in zip(self._qubit_list_sizes,
                               self._cal_matrices):
                    start_index = end_index - list_size
                    substate_as_int = int(state[start_index:end_index], 2)
                    measured_substate_as_int = \
                        int(measured_state[start_index:end_index], 2)
                    end_index = start_index

                    cal_mat[measured_substate_as_int][substate_as_int] += \
                        counts

        for mat_index, _ in enumerate(self._cal_matrices):
            sums_of_columns = np.sum(self._cal_matrices[mat_index], axis=0)
            # pylint: disable=assignment-from-no-return
            self._cal_matrices[mat_index] = np.divide(
                self._cal_matrices[mat_index], sums_of_columns,
                out=np.zeros_like(self._cal_matrices[mat_index]),
                where=sums_of_columns != 0)
