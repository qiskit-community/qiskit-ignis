# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Test of measurement calibration:
1) Preparation of all 2 ** n basis states, generating the calibration circuits
(without noise), computing the calibration matrices,
and validating that they equal
to the identity matrices
2) Generating ideal (equally distributed) results, computing
the calibration output (without noise),
and validating that it is equally distributed
3) Testing the the measurement calibration on a circuit
(without noise) verifying that it is close to the
expected (equally distributed) result
4) Testing MeasurementFitter on pre-generated data with noise
"""

import unittest
import pickle
import numpy as np
import qiskit
from qiskit import QuantumCircuit, ClassicalRegister, Aer
from qiskit.ignis.mitigation.measurement import MeasurementFitter
from qiskit.ignis.mitigation.measurement import measurement_calibration


class TestMeasCal(unittest.TestCase):
    # TODO: after terra 0.8, derive test case like this
    # class TestMeasCal(QiskitTestCase):
    """ The test class """

    def setUp(self):
        """
            setUp and global parameters
        """
        self.nq_list = [1, 2, 3, 4, 5]  # Test up to 5 qubits
        self.shots = 1024  # Number of shots (should be a power of 2)

    @staticmethod
    def choose_calibration(nq, pattern_type):
        """
            Generate a calibration circuit

            Args:
                nq: number of qubits
                pattern_type: a pattern in range(1,2**nq)

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
                state_labels: a list of calibration state labels
                weight: the number of qubits

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
        """
            Test ideal execution, without noise
        """
        for nq in self.nq_list:
            print("Testing %d qubit measurement calibration" % nq)

            for pattern_type in range(1, 2 ** nq):
                # Generate the quantum register according to the pattern
                qubits, weight = self.choose_calibration(nq, pattern_type)
                # Generate the calibration circuits
                meas_calibs, state_labels = \
                    measurement_calibration(qubit_list=qubits,
                                            circlabel='test')

                # Perform an ideal execution on the generated circuits
                backend = Aer.get_backend('qasm_simulator')
                qobj = qiskit.compile(meas_calibs, backend=backend,
                                      shots=self.shots)
                job = backend.run(qobj)
                cal_results = job.result()

                # Make a calibration matrix
                meas_cal = MeasurementFitter(cal_results, state_labels,
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

                # Apply the calibration matrix to results
                # in list and dict forms using different methods
                results_dict_1 = meas_cal.apply(results_dict,
                                                method='least_squares')
                results_dict_0 = meas_cal.apply(results_dict,
                                                method='pseudo_inverse')
                results_list_1 = meas_cal.apply(results_list,
                                                method='least_squares')
                results_list_0 = meas_cal.apply(results_list,
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
        """
            Test an execution on a circuit
        """
        print("Testing measurement calibration on a circuit")

        # Choose all triples from 5 qubits
        for q1 in range(5):
            for q2 in range(q1+1, 5):
                for q3 in range(q2+1, 5):
                    # Generate the quantum register according to the pattern
                    qr = qiskit.QuantumRegister(5)
                    # Generate the calibration circuits
                    meas_calibs, state_labels = \
                        measurement_calibration(qubit_list=[1, 2, 3], qr=qr)

                    # Run the calibration circuits
                    backend = Aer.get_backend('qasm_simulator')
                    qobj = qiskit.compile(meas_calibs, backend=backend,
                                          shots=self.shots)
                    job = backend.run(qobj)
                    cal_results = job.result()

                    # Make a calibration matrix
                    meas_cal = MeasurementFitter(cal_results, state_labels)
                    # Calculate the fidelity
                    fidelity = meas_cal.readout_fidelity()

                    # Make a 3Q GHZ state
                    cr = ClassicalRegister(3)
                    ghz = QuantumCircuit(qr, cr)
                    ghz.h(qr[q1])
                    ghz.cx(qr[q1], qr[q2])
                    ghz.cx(qr[q2], qr[q3])
                    ghz.measure(qr[q1], cr[0])
                    ghz.measure(qr[q2], cr[1])
                    ghz.measure(qr[q3], cr[2])

                    qobj = qiskit.compile([ghz], backend=backend,
                                          shots=self.shots)
                    job = backend.run(qobj)
                    results = job.result()

                    # Predicted equally distributed results
                    predicted_results = {'000': 0.5,
                                         '111': 0.5}

                    # Calculate the results after mitigation
                    output_results_pseudo_inverse = meas_cal.apply(
                        results, method='pseudo_inverse').get_counts(0)
                    output_results_least_square = meas_cal.apply(
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
        """
            Test the MeasurementFitter with noise
        """
        print("Testing MeasurementFitter with noise")

        # pre-generated results with noise
        # load from pickled file
        fo = open('test_meas_results.pkl', 'rb')
        tests = pickle.load(fo)
        fo.close()

        # Set the state labels
        state_labels = ['000', '001', '010', '011',
                        '100', '101', '110', '111']
        meas_cal = MeasurementFitter(None, state_labels,
                                     circlabel='test')

        for tst_index, _ in enumerate(tests):
            # Set the calibration matrix
            meas_cal.cal_matrix = tests[tst_index]['cal_matrix']
            # Calculate the fidelity
            fidelity = meas_cal.readout_fidelity()

            # Calculate the results after mitigation
            output_results_pseudo_inverse = meas_cal.apply(
                tests[tst_index]['results'], method='pseudo_inverse')
            output_results_least_square = meas_cal.apply(
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


if __name__ == '__main__':
    unittest.main()
