# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=cell-var-from-loop


"""
Measurement correction filters.

"""
from copy import deepcopy
from scipy.optimize import minimize
import scipy.linalg as la
import numpy as np
import qiskit
from qiskit import QiskitError


class MeasurementFilter():
    """
    Measurement error mitigation filter

    Produced from a measurement calibration fitter and can be applied
    to data

    """

    def __init__(self, cal_matrix, state_labels):
        """
        Initialize a measurement error mitigation filter using the cal_matrix
        from a measurement calibration fitter

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

    @cal_matrix.setter
    def cal_matrix(self, new_cal_matrix):
        """Set cal_matrix."""
        self._cal_matrix = new_cal_matrix

    def apply(self, raw_data, method='least_squares'):
        """
        Apply the calibration matrix to results

        Args:
            raw_data: The data to be corrected. Can be in a number of forms.
                Form1: a counts dictionary from results.get_counts
                Form2: a list of counts of length==len(state_labels)
                Form3: a list of counts of length==M*len(state_labels) where M
                    is an integer (e.g. for use with the tomography data)
                Form4: a qiskit Result

            method (str): fitting method. If None, then least_squares is used.
                'pseudo_inverse': direct inversion of the A matrix
                'least_squares': constrained to have physical probabilities

        Returns:
            The corrected data in the same form as raw_data

        Additional Information:

            e.g.
            calcircuits, state_labels = complete_measurement_calibration(
                qiskit.QuantumRegister(5))
            job = qiskit.execute(calcircuits)
            meas_fitter = CompleteMeasFitter(job.results(),
                                            state_labels)
            meas_filter = MeasurementFilter(meas_fitter.cal_matrix)

            job2 = qiskit.execute(my_circuits)
            result2 = job2.results()

            error_mitigated_counts = meas_filter.apply(
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

        elif isinstance(raw_data, qiskit.result.result.Result):

            # extract out all the counts, re-call the function with the
            # counts and push back into the new result
            new_result = deepcopy(raw_data)

            for resultidx, _ in enumerate(raw_data.results):
                new_counts = self.apply(
                    raw_data.get_counts(resultidx), method=method)
                new_result.results[resultidx].data.counts = \
                    new_result.results[resultidx]. \
                    data.counts.from_dict(new_counts)

            return new_result

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
