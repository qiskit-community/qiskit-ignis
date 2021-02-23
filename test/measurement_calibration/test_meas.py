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

# pylint: disable=invalid-name

"""
Test of measurement calibration:
1) Preparation of the basis states, generating the calibration circuits
(without noise), computing the calibration matrices,
and validating that they equal
to the identity matrices
2) Generating ideal (equally distributed) results, computing
the calibration output (without noise),
and validating that it is equally distributed
3) Testing the the measurement calibration on a circuit
(without noise), verifying that it is close to the
expected (equally distributed) result
4) Testing the fitters on pre-generated data with noise
"""

import unittest
import os
import json
from test.measurement_calibration.generate_data \
    import tensored_calib_circ_creation, meas_calib_circ_creation
import numpy as np
import qiskit
from qiskit.result.result import Result
from qiskit import Aer
from qiskit.ignis.mitigation.measurement \
     import (CompleteMeasFitter, TensoredMeasFitter,
             complete_meas_cal, tensored_meas_cal,
             MeasurementFilter)
from qiskit.ignis.verification.tomography import count_keys

# fixed seed for tests - for both simulator and transpiler
SEED = 42


class TestMeasCal(unittest.TestCase):
    # TODO: after terra 0.8, derive test case like this
    # class TestMeasCal(QiskitTestCase):
    """The test class."""

    def setUp(self):
        """setUp and global parameters"""
        self.nq_list = [1, 2, 3, 4, 5]  # Test up to 5 qubits
        self.shots = 1024  # Number of shots (should be a power of 2)

    @staticmethod
    def choose_calibration(nq, pattern_type):
        """
        Generate a calibration circuit

        Args:
            nq (int): number of qubits
            pattern_type (int): a pattern in range(1, 2**nq)

        Returns:
            qubits: a list of qubits according to the given pattern
            weight: the weight of the pattern_type,
                    equals to the number of qubits

        Additional Information:
            qr[i] exists if and only if the i-th bit in the binary
            expression of
            pattern_type equals 1
        """
        qubits = []
        weight = 0
        for i in range(nq):
            pattern_bit = pattern_type & 1
            pattern_type = pattern_type >> 1
            if pattern_bit == 1:
                qubits.append(i)
                weight += 1
        return qubits, weight

    def generate_ideal_results(self, state_labels, weight):
        """
        Generate ideal equally distributed results

        Args:
            state_labels (list): a list of calibration state labels
            weight (int): the number of qubits

        Returns:
            results_dict: a dictionary of equally distributed results
            results_list: a list of equally distributed results

        Additional Information:
            for each state in state_labels:
            result_dict[state] = #shots/len(state_labels)
        """
        results_dict = {}
        results_list = [0]*(2 ** weight)
        state_num = len(state_labels)
        for state in state_labels:
            shots_per_state = self.shots/state_num
            results_dict[state] = shots_per_state
            # converting state (binary) to an integer
            place = int(state, 2)
            results_list[place] = shots_per_state
        return results_dict, results_list

    def test_ideal_meas_cal(self):
        """Test ideal execution, without noise."""
        for nq in self.nq_list:
            print("Testing %d qubit measurement calibration" % nq)

            for pattern_type in range(1, 2 ** nq):

                # Generate the quantum register according to the pattern
                qubits, weight = self.choose_calibration(nq, pattern_type)

                # Generate the calibration circuits
                meas_calibs, state_labels = \
                    complete_meas_cal(qubit_list=qubits,
                                      circlabel='test')

                # Perform an ideal execution on the generated circuits
                backend = Aer.get_backend('qasm_simulator')
                job = qiskit.execute(meas_calibs, backend=backend,
                                     shots=self.shots)
                cal_results = job.result()

                # Make a calibration matrix
                meas_cal = CompleteMeasFitter(cal_results, state_labels,
                                              circlabel='test')

                # Assert that the calibration matrix is equal to identity
                IdentityMatrix = np.identity(2 ** weight)
                self.assertListEqual(meas_cal.cal_matrix.tolist(),
                                     IdentityMatrix.tolist(),
                                     'Error: the calibration matrix is \
                                     not equal to identity')

                # Assert that the readout fidelity is equal to 1
                self.assertEqual(meas_cal.readout_fidelity(), 1.0,
                                 'Error: the average fidelity  \
                                 is not equal to 1')

                # Generate ideal (equally distributed) results
                results_dict, results_list = \
                    self.generate_ideal_results(state_labels, weight)

                # Output the filter
                meas_filter = meas_cal.filter

                # Apply the calibration matrix to results
                # in list and dict forms using different methods
                results_dict_1 = meas_filter.apply(results_dict,
                                                   method='least_squares')
                results_dict_0 = meas_filter.apply(results_dict,
                                                   method='pseudo_inverse')
                results_list_1 = meas_filter.apply(results_list,
                                                   method='least_squares')
                results_list_0 = meas_filter.apply(results_list,
                                                   method='pseudo_inverse')

                # Assert that the results are equally distributed
                self.assertListEqual(results_list, results_list_0.tolist())
                self.assertListEqual(results_list,
                                     np.round(results_list_1).tolist())
                self.assertDictEqual(results_dict, results_dict_0)
                round_results = {}
                for key, val in results_dict_1.items():
                    round_results[key] = np.round(val)
                self.assertDictEqual(results_dict, round_results)

    def test_meas_cal_on_circuit(self):
        """Test an execution on a circuit."""
        print("Testing measurement calibration on a circuit")

        # Generate the calibration circuits
        meas_calibs, state_labels, ghz = meas_calib_circ_creation()

        # Run the calibration circuits
        backend = Aer.get_backend('qasm_simulator')
        job = qiskit.execute(meas_calibs, backend=backend,
                             shots=self.shots,
                             seed_simulator=SEED,
                             seed_transpiler=SEED)
        cal_results = job.result()

        # Make a calibration matrix
        meas_cal = CompleteMeasFitter(cal_results, state_labels)
        # Calculate the fidelity
        fidelity = meas_cal.readout_fidelity()

        job = qiskit.execute([ghz], backend=backend,
                             shots=self.shots,
                             seed_simulator=SEED,
                             seed_transpiler=SEED)
        results = job.result()

        # Predicted equally distributed results
        predicted_results = {'000': 0.5,
                             '111': 0.5}

        meas_filter = meas_cal.filter

        # Calculate the results after mitigation
        output_results_pseudo_inverse = meas_filter.apply(
            results, method='pseudo_inverse').get_counts(0)
        output_results_least_square = meas_filter.apply(
            results, method='least_squares').get_counts(0)

        # Compare with expected fidelity and expected results
        self.assertAlmostEqual(fidelity, 1.0)
        self.assertAlmostEqual(
            output_results_pseudo_inverse['000']/self.shots,
            predicted_results['000'],
            places=1)

        self.assertAlmostEqual(
            output_results_least_square['000']/self.shots,
            predicted_results['000'],
            places=1)

        self.assertAlmostEqual(
            output_results_pseudo_inverse['111']/self.shots,
            predicted_results['111'],
            places=1)

        self.assertAlmostEqual(
            output_results_least_square['111']/self.shots,
            predicted_results['111'],
            places=1)

    def test_meas_fitter_with_noise(self):
        """Test the MeasurementFitter with noise."""
        print("Testing MeasurementFitter with noise")

        # pre-generated results with noise
        # load from json file
        with open(os.path.join(
                os.path.dirname(__file__), 'test_meas_results.json'), "r") as saved_file:
            tests = json.load(saved_file)

        # Set the state labels
        state_labels = ['000', '001', '010', '011',
                        '100', '101', '110', '111']
        meas_cal = CompleteMeasFitter(None, state_labels,
                                      circlabel='test')

        for tst_index, _ in enumerate(tests):
            # Set the calibration matrix
            meas_cal.cal_matrix = tests[tst_index]['cal_matrix']
            # Calculate the fidelity
            fidelity = meas_cal.readout_fidelity()

            meas_filter = MeasurementFilter(tests[tst_index]['cal_matrix'],
                                            state_labels)

            # Calculate the results after mitigation
            output_results_pseudo_inverse = meas_filter.apply(
                tests[tst_index]['results'], method='pseudo_inverse')
            output_results_least_square = meas_filter.apply(
                tests[tst_index]['results'], method='least_squares')

            # Compare with expected fidelity and expected results
            self.assertAlmostEqual(fidelity,
                                   tests[tst_index]['fidelity'],
                                   places=0)
            self.assertAlmostEqual(
                output_results_pseudo_inverse['000'],
                tests[tst_index]['results_pseudo_inverse']['000'], places=0)

            self.assertAlmostEqual(
                output_results_least_square['000'],
                tests[tst_index]['results_least_square']['000'], places=0)

            self.assertAlmostEqual(
                output_results_pseudo_inverse['111'],
                tests[tst_index]['results_pseudo_inverse']['111'], places=0)

            self.assertAlmostEqual(
                output_results_least_square['111'],
                tests[tst_index]['results_least_square']['111'], places=0)

    def test_ideal_tensored_meas_cal(self):
        """Test ideal execution, without noise."""

        mit_pattern = [[1, 2], [3, 4, 5], [6]]
        meas_layout = [1, 2, 3, 4, 5, 6]

        # Generate the calibration circuits
        meas_calibs, _ = tensored_meas_cal(mit_pattern=mit_pattern)

        # Perform an ideal execution on the generated circuits
        backend = Aer.get_backend('qasm_simulator')
        cal_results = qiskit.execute(
            meas_calibs, backend=backend, shots=self.shots).result()

        # Make calibration matrices
        meas_cal = TensoredMeasFitter(cal_results, mit_pattern=mit_pattern)

        # Assert that the calibration matrices are equal to identity
        cal_matrices = meas_cal.cal_matrices
        self.assertEqual(len(mit_pattern), len(cal_matrices),
                         'Wrong number of calibration matrices')
        for qubit_list, cal_mat in zip(mit_pattern, cal_matrices):
            IdentityMatrix = np.identity(2**len(qubit_list))
            self.assertListEqual(cal_mat.tolist(),
                                 IdentityMatrix.tolist(),
                                 'Error: the calibration matrix is \
                                 not equal to identity')

        # Assert that the readout fidelity is equal to 1
        self.assertEqual(meas_cal.readout_fidelity(), 1.0,
                         'Error: the average fidelity  \
                         is not equal to 1')

        # Generate ideal (equally distributed) results
        results_dict, _ = \
            self.generate_ideal_results(count_keys(6), 6)

        # Output the filter
        meas_filter = meas_cal.filter

        # Apply the calibration matrix to results
        # in list and dict forms using different methods
        results_dict_1 = meas_filter.apply(results_dict,
                                           method='least_squares',
                                           meas_layout=meas_layout)
        results_dict_0 = meas_filter.apply(results_dict,
                                           method='pseudo_inverse',
                                           meas_layout=meas_layout)

        # Assert that the results are equally distributed
        self.assertDictEqual(results_dict, results_dict_0)
        round_results = {}
        for key, val in results_dict_1.items():
            round_results[key] = np.round(val)
        self.assertDictEqual(results_dict, round_results)

    def test_tensored_meas_cal_on_circuit(self):
        """Test an execution on a circuit."""

        # Generate the calibration circuits
        meas_calibs, mit_pattern, ghz, meas_layout = tensored_calib_circ_creation()

        # Run the calibration circuits
        backend = Aer.get_backend('qasm_simulator')
        cal_results = qiskit.execute(meas_calibs, backend=backend,
                                     shots=self.shots,
                                     seed_simulator=SEED,
                                     seed_transpiler=SEED).result()

        # Make a calibration matrix
        meas_cal = TensoredMeasFitter(cal_results,
                                      mit_pattern=mit_pattern)
        # Calculate the fidelity
        fidelity = meas_cal.readout_fidelity(0)*meas_cal.readout_fidelity(1)

        results = qiskit.execute([ghz], backend=backend,
                                 shots=self.shots,
                                 seed_simulator=SEED,
                                 seed_transpiler=SEED).result()

        # Predicted equally distributed results
        predicted_results = {'000': 0.5,
                             '111': 0.5}

        meas_filter = meas_cal.filter

        # Calculate the results after mitigation
        output_results_pseudo_inverse = meas_filter.apply(
            results, method='pseudo_inverse', meas_layout=meas_layout).get_counts(0)
        output_results_least_square = meas_filter.apply(
            results, method='least_squares', meas_layout=meas_layout).get_counts(0)

        # Compare with expected fidelity and expected results
        self.assertAlmostEqual(fidelity, 1.0)
        self.assertAlmostEqual(
            output_results_pseudo_inverse['000']/self.shots,
            predicted_results['000'],
            places=1)

        self.assertAlmostEqual(
            output_results_least_square['000']/self.shots,
            predicted_results['000'],
            places=1)

        self.assertAlmostEqual(
            output_results_pseudo_inverse['111']/self.shots,
            predicted_results['111'],
            places=1)

        self.assertAlmostEqual(
            output_results_least_square['111']/self.shots,
            predicted_results['111'],
            places=1)

    def test_tensored_meas_fitter_with_noise(self):
        """Test the TensoredFitter with noise."""

        # pre-generated results with noise
        # load from json file
        with open(os.path.join(
                os.path.dirname(__file__), 'test_tensored_meas_results.json'), "r") as saved_file:
            saved_info = json.load(saved_file)
        saved_info['cal_results'] = Result.from_dict(saved_info['cal_results'])
        saved_info['results'] = Result.from_dict(saved_info['results'])

        meas_cal = TensoredMeasFitter(
            saved_info['cal_results'],
            mit_pattern=saved_info['mit_pattern'])

        # Calculate the fidelity
        fidelity = meas_cal.readout_fidelity(0)*meas_cal.readout_fidelity(1)
        # Compare with expected fidelity and expected results
        self.assertAlmostEqual(fidelity,
                               saved_info['fidelity'],
                               places=0)

        meas_filter = meas_cal.filter

        # Calculate the results after mitigation
        output_results_pseudo_inverse = meas_filter.apply(
            saved_info['results'].get_counts(0),
            method='pseudo_inverse',
            meas_layout=saved_info['meas_layout'])
        output_results_least_square = meas_filter.apply(
            saved_info['results'], method='least_squares', meas_layout=saved_info['meas_layout'])

        self.assertAlmostEqual(
            output_results_pseudo_inverse['000'],
            saved_info['results_pseudo_inverse']['000'], places=0)

        self.assertAlmostEqual(
            output_results_least_square.get_counts(0)['000'],
            saved_info['results_least_square']['000'], places=0)

        self.assertAlmostEqual(
            output_results_pseudo_inverse['111'],
            saved_info['results_pseudo_inverse']['111'], places=0)

        self.assertAlmostEqual(
            output_results_least_square.get_counts(0)['111'],
            saved_info['results_least_square']['111'], places=0)

        substates_list = []
        for qubit_list in saved_info['mit_pattern']:
            substates_list.append(count_keys(len(qubit_list))[::-1])

        fitter_other_order = TensoredMeasFitter(
            saved_info['cal_results'],
            substate_labels_list=substates_list,
            mit_pattern=saved_info['mit_pattern'])

        fidelity = fitter_other_order.readout_fidelity(0) * \
            meas_cal.readout_fidelity(1)

        self.assertAlmostEqual(fidelity,
                               saved_info['fidelity'],
                               places=0)

        meas_filter = fitter_other_order.filter

        # Calculate the results after mitigation
        output_results_pseudo_inverse = meas_filter.apply(
            saved_info['results'].get_counts(0),
            method='pseudo_inverse',
            meas_layout=saved_info['meas_layout'])
        output_results_least_square = meas_filter.apply(
            saved_info['results'], method='least_squares', meas_layout=saved_info['meas_layout'])

        self.assertAlmostEqual(
            output_results_pseudo_inverse['000'],
            saved_info['results_pseudo_inverse']['000'], places=0)

        self.assertAlmostEqual(
            output_results_least_square.get_counts(0)['000'],
            saved_info['results_least_square']['000'], places=0)

        self.assertAlmostEqual(
            output_results_pseudo_inverse['111'],
            saved_info['results_pseudo_inverse']['111'], places=0)

        self.assertAlmostEqual(
            output_results_least_square.get_counts(0)['111'],
            saved_info['results_least_square']['111'], places=0)


if __name__ == '__main__':
    unittest.main()
