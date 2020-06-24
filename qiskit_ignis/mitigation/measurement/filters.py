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

# pylint: disable=cell-var-from-loop,invalid-name


"""
Measurement correction filters.

"""
from copy import deepcopy
from scipy.optimize import minimize
import scipy.linalg as la
import numpy as np
import qiskit
from qiskit import QiskitError
from qiskit.tools import parallel_map
from ...verification.tomography import count_keys


class MeasurementFilter():
    """
    Measurement error mitigation filter.

    Produced from a measurement calibration fitter and can be applied
    to data.

    """

    def __init__(self,
                 cal_matrix: np.matrix,
                 state_labels: list):
        """
        Initialize a measurement error mitigation filter using the cal_matrix
        from a measurement calibration fitter.

        Args:
            cal_matrix: the calibration matrix for applying the correction
            state_labels: the states for the ordering of the cal matrix
        """

        self._cal_matrix = cal_matrix
        self._state_labels = state_labels

    @property
    def cal_matrix(self):
        """Return cal_matrix."""
        return self._cal_matrix

    @property
    def state_labels(self):
        """return the state label ordering of the cal matrix"""
        return self._state_labels

    @state_labels.setter
    def state_labels(self, new_state_labels):
        """set the state label ordering of the cal matrix"""
        self._state_labels = new_state_labels

    @cal_matrix.setter
    def cal_matrix(self, new_cal_matrix):
        """Set cal_matrix."""
        self._cal_matrix = new_cal_matrix

    def apply(self,
              raw_data,
              method='least_squares'):
        """Apply the calibration matrix to results.

        Args:
            raw_data (dict or list): The data to be corrected. Can be in a number of forms:

                 Form 1: a counts dictionary from results.get_counts

                 Form 2: a list of counts of `length==len(state_labels)`

                 Form 3: a list of counts of `length==M*len(state_labels)` where M is an
                 integer (e.g. for use with the tomography data)

                 Form 4: a qiskit Result

            method (str): fitting method. If `None`, then least_squares is used.

                ``pseudo_inverse``: direct inversion of the A matrix

                ``least_squares``: constrained to have physical probabilities

        Returns:
            dict or list: The corrected data in the same form as `raw_data`

        Raises:
            QiskitError: if `raw_data` is not an integer multiple
                of the number of calibrated states.

        """

        # check forms of raw_data
        if isinstance(raw_data, dict):
            # counts dictionary
            for data_label in raw_data.keys():
                if data_label not in self._state_labels:
                    raise QiskitError("Unexpected state label '" + data_label +
                                      "', verify the fitter's state labels "
                                      "correpsond to the input data")
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

        elif isinstance(raw_data, qiskit.result.result.Result):

            # extract out all the counts, re-call the function with the
            # counts and push back into the new result
            new_result = deepcopy(raw_data)

            new_counts_list = parallel_map(
                self._apply_correction,
                [resultidx for resultidx, _ in enumerate(raw_data.results)],
                task_args=(raw_data, method))

            for resultidx, new_counts in new_counts_list:
                new_result.results[resultidx].data.counts = new_counts

            return new_result

        else:
            raise QiskitError("Unrecognized type for raw_data.")

        if method == 'pseudo_inverse':
            pinv_cal_mat = la.pinv(self._cal_matrix)

        # Apply the correction
        for data_idx, _ in enumerate(raw_data2):

            if method == 'pseudo_inverse':
                raw_data2[data_idx] = np.dot(
                    pinv_cal_mat, raw_data2[data_idx])

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
            # TODO: should probably change to:
            # raw_data2 = raw_data2[0].tolist()
            raw_data2 = raw_data2[0]
        return raw_data2

    def _apply_correction(self, resultidx, raw_data, method):
        """Wrapper to call apply with a counts dictionary."""
        new_counts = self.apply(
            raw_data.get_counts(resultidx), method=method)
        return resultidx, new_counts


class TensoredFilter():
    """
    Tensored measurement error mitigation filter.

    Produced from a tensored measurement calibration fitter and can be applied
    to data.
    """

    def __init__(self,
                 cal_matrices: np.matrix,
                 substate_labels_list: list):
        """
        Initialize a tensored measurement error mitigation filter using
        the cal_matrices from a tensored measurement calibration fitter.

        Args:
            cal_matrices: the calibration matrices for applying the correction.
            substate_labels_list: for each calibration matrix
                a list of the states (as strings, states in the subspace)
        """

        self._cal_matrices = cal_matrices
        self._qubit_list_sizes = []
        self._indices_list = []
        self._substate_labels_list = []
        self.substate_labels_list = substate_labels_list

    @property
    def cal_matrices(self):
        """Return cal_matrices."""
        return self._cal_matrices

    @cal_matrices.setter
    def cal_matrices(self, new_cal_matrices):
        """Set cal_matrices."""
        self._cal_matrices = deepcopy(new_cal_matrices)

    @property
    def substate_labels_list(self):
        """Return _substate_labels_list"""
        return self._substate_labels_list

    @substate_labels_list.setter
    def substate_labels_list(self, new_substate_labels_list):
        """Return _substate_labels_list"""
        self._substate_labels_list = new_substate_labels_list

        # get the number of qubits in each subspace
        self._qubit_list_sizes = []
        for _, substate_label_list in enumerate(self._substate_labels_list):
            self._qubit_list_sizes.append(
                int(np.log2(len(substate_label_list))))

        # get the indices in the calibration matrix
        self._indices_list = []
        for _, sub_labels in enumerate(self._substate_labels_list):

            self._indices_list.append(
                {lab: ind for ind, lab in enumerate(sub_labels)})

    @property
    def qubit_list_sizes(self):
        """Return _qubit_list_sizes."""
        return self._qubit_list_sizes

    @property
    def nqubits(self):
        """Return the number of qubits. See also MeasurementFilter.apply() """
        return sum(self._qubit_list_sizes)

    def apply(self, raw_data, method='least_squares'):
        """
        Apply the calibration matrices to results.

        Args:
            raw_data (dict or Result): The data to be corrected. Can be in one of two forms:

                * A counts dictionary from results.get_counts

                * A Qiskit Result

            method (str): fitting method. The following methods are supported:

                * 'pseudo_inverse': direct inversion of the cal matrices.

                * 'least_squares': constrained to have physical probabilities.

                * If `None`, 'least_squares' is used.

        Returns:
            dict or Result: The corrected data in the same form as raw_data

        Raises:
            QiskitError: if raw_data is not in a one of the defined forms.
        """

        all_states = count_keys(self.nqubits)
        num_of_states = 2**self.nqubits

        # check forms of raw_data
        if isinstance(raw_data, dict):
            # counts dictionary
            # convert to list
            raw_data2 = [np.zeros(num_of_states, dtype=float)]
            for state, count in raw_data.items():
                stateidx = int(state, 2)
                raw_data2[0][stateidx] = count

        elif isinstance(raw_data, qiskit.result.result.Result):

            # extract out all the counts, re-call the function with the
            # counts and push back into the new result
            new_result = deepcopy(raw_data)

            new_counts_list = parallel_map(
                self._apply_correction,
                [resultidx for resultidx, _ in enumerate(raw_data.results)],
                task_args=(raw_data, method))

            for resultidx, new_counts in new_counts_list:
                new_result.results[resultidx].data.counts = new_counts

            return new_result

        else:
            raise QiskitError("Unrecognized type for raw_data.")

        if method == 'pseudo_inverse':
            pinv_cal_matrices = []
            for cal_mat in self._cal_matrices:
                pinv_cal_matrices.append(la.pinv(cal_mat))

        # Apply the correction
        for data_idx, _ in enumerate(raw_data2):

            if method == 'pseudo_inverse':
                inv_mat_dot_raw = np.zeros([num_of_states], dtype=float)
                for state1_idx, state1 in enumerate(all_states):
                    for state2_idx, state2 in enumerate(all_states):
                        if raw_data2[data_idx][state2_idx] == 0:
                            continue

                        product = 1.
                        end_index = self.nqubits
                        for p_ind, pinv_mat in enumerate(pinv_cal_matrices):

                            start_index = end_index - \
                                self._qubit_list_sizes[p_ind]

                            state1_as_int = \
                                self._indices_list[p_ind][
                                    state1[start_index:end_index]]

                            state2_as_int = \
                                self._indices_list[p_ind][
                                    state2[start_index:end_index]]

                            end_index = start_index
                            product *= \
                                pinv_mat[state1_as_int][state2_as_int]
                            if product == 0:
                                break
                        inv_mat_dot_raw[state1_idx] += \
                            (product * raw_data2[data_idx][state2_idx])
                raw_data2[data_idx] = inv_mat_dot_raw

            elif method == 'least_squares':

                def fun(x):
                    mat_dot_x = np.zeros([num_of_states], dtype=float)
                    for state1_idx, state1 in enumerate(all_states):
                        mat_dot_x[state1_idx] = 0.
                        for state2_idx, state2 in enumerate(all_states):
                            if x[state2_idx] != 0:
                                product = 1.
                                end_index = self.nqubits
                                for c_ind, cal_mat in \
                                        enumerate(self._cal_matrices):

                                    start_index = end_index - \
                                        self._qubit_list_sizes[c_ind]

                                    state1_as_int = \
                                        self._indices_list[c_ind][
                                            state1[start_index:end_index]]

                                    state2_as_int = \
                                        self._indices_list[c_ind][
                                            state2[start_index:end_index]]

                                    end_index = start_index
                                    product *= \
                                        cal_mat[state1_as_int][state2_as_int]
                                    if product == 0:
                                        break
                                mat_dot_x[state1_idx] += \
                                    (product * x[state2_idx])
                    return sum(
                        (raw_data2[data_idx] - mat_dot_x)**2)

                x0 = np.random.rand(num_of_states)
                x0 = x0 / sum(x0)
                nshots = sum(raw_data2[data_idx])
                cons = ({'type': 'eq', 'fun': lambda x: nshots - sum(x)})
                bnds = tuple((0, nshots) for x in x0)
                res = minimize(fun, x0, method='SLSQP',
                               constraints=cons, bounds=bnds, tol=1e-6)
                raw_data2[data_idx] = res.x

            else:
                raise QiskitError("Unrecognized method.")

        # convert back into a counts dictionary
        new_count_dict = {}
        for state_idx, state in enumerate(all_states):
            if raw_data2[0][state_idx] != 0:
                new_count_dict[state] = raw_data2[0][state_idx]

        return new_count_dict

    def _apply_correction(self, resultidx, raw_data, method):
        """Wrapper to call apply with a counts dictionary."""
        new_counts = self.apply(
            raw_data.get_counts(resultidx), method=method)
        return resultidx, new_counts
