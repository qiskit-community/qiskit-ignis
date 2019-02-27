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
                    self.assertAlmostEqual(output_results_pseudo_inverse['000']/
                                           self.shots,
                                           predicted_results['000'],
                                           places=1)
                    self.assertAlmostEqual(output_results_least_square['000']/
                                           self.shots,
                                           predicted_results['000'],
                                           places=1)
                    self.assertAlmostEqual(output_results_pseudo_inverse['111']/
                                           self.shots,
                                           predicted_results['111'],
                                           places=1)
                    self.assertAlmostEqual(output_results_least_square['111']/
                                           self.shots,
                                           predicted_results['111'],
                                           places=1)

                    # Asserting the corrected result is close to ideal expected
                    # TODO: replace the entire block below with these
                    # lines after terra 0.8
                    # predicted_results = {'000': self.shots/2, '111': self.shots/2}
                    # delta = 0.1 * self.shots
                    # self.assertDictAlmostEqual(output_results_0,
                    #   predicted_results, delta=delta)
                    # self.assertDictAlmostEqual(output_results_1,
                    #   predicted_results, delta=delta)

    def test_meas_fitter_with_noise(self):
        """
            Test the MeasurementFitter with noise
        """
        print("Testing MeasurementFitter with noise")

        # pre-generated results with noise
        tests = [
            {
                'cal_matrix': [[0.44238281, 0.05566406, 0.06152344, 0.00976562,
                                0.05078125, 0.00878906, 0.00683594, 0.00195312],
                               [0.16015625, 0.50195312, 0.0234375, 0.07910156,
                                0.01464844, 0.07324219, 0.00390625, 0.00585938],
                               [0.12597656, 0.02050781, 0.50390625, 0.05566406,
                                0.02148438, 0.00097656, 0.06445312, 0.00488281],
                               [0.0390625, 0.1875, 0.16015625, 0.62011719,
                                0.01171875, 0.02539062, 0.02441406, 0.07324219],
                               [0.140625, 0.02050781, 0.01367188, 0.00195312,
                                0.50195312, 0.06933594, 0.06640625, 0.015625],
                               [0.04199219, 0.15820312, 0.00292969, 0.02441406,
                                0.15332031, 0.61425781, 0.02050781, 0.078125],
                               [0.0390625, 0.00683594, 0.17675781, 0.0234375,
                                0.1875, 0.02539062, 0.61523438, 0.08886719],
                               [0.01074219, 0.04882812, 0.05761719, 0.18554688,
                                0.05859375, 0.18261719, 0.19824219, 0.73144531]],
                'fidelity': 0.5626220703125,
                'results': {'001': 75, '010': 75, '100': 73, '000': 209,
                            '110': 63, '101': 73, '111': 399, '011': 57},
                'results_pseudo_inverse': {'000': 467.50917920701283,
                                           '001': -9.531737085939437,
                                           '010': 29.558556028438662,
                                           '011': -6.008851307135412,
                                           '100': -4.093789013503745,
                                           '101': 22.63568794725104,
                                           '110': -12.630370153454054,
                                           '111': 536.5613240573772},
                'results_least_square': {'000': 459.020911256551,
                                         '001': 7.341523222681445e-14,
                                         '010': 18.895659397949345,
                                         '011': 2.4304993781476547e-14,
                                         '100': 6.01205283473627e-14,
                                         '101': 16.439452980690653,
                                         '110': 6.432126966465779e-14,
                                         '111': 529.6439763648087}
            },
            {'cal_matrix': [[0.4052734375, 0.0634765625, 0.0615234375, 0.009765625,
                             0.0517578125, 0.0087890625, 0.0087890625, 0.0],
                            [0.12890625, 0.4873046875, 0.017578125, 0.0673828125,
                             0.021484375, 0.056640625, 0.0009765625, 0.0068359375],
                            [0.146484375, 0.0244140625, 0.5234375, 0.0771484375,
                             0.01953125, 0.0009765625, 0.07421875, 0.0068359375],
                            [0.041015625, 0.17578125, 0.1484375, 0.59765625,
                             0.00390625, 0.025390625, 0.0263671875, 0.08203125],
                            [0.1748046875, 0.0185546875, 0.0224609375, 0.001953125,
                             0.5078125, 0.0751953125, 0.0703125, 0.0107421875],
                            [0.0498046875, 0.162109375, 0.005859375, 0.0224609375,
                             0.16796875, 0.6259765625, 0.025390625, 0.0947265625],
                            [0.0380859375, 0.0126953125, 0.16015625, 0.025390625,
                             0.1640625, 0.01953125, 0.578125, 0.078125],
                            [0.015625, 0.0556640625, 0.060546875, 0.1982421875,
                             0.0634765625, 0.1875, 0.2158203125, 0.720703125]],
             'fidelity': 0.5557861328125,
             'results': {'001': 71, '011': 60, '111': 393, '110': 64,
                         '101': 69, '000': 217, '100': 74, '010': 76},
             'results_pseudo_inverse': {'011': -6.8815256891790195,
                                        '110': 22.619248458685863,
                                        '111': 534.6299647735469,
                                        '101': 0.9243073704424183,
                                        '010': -14.241362831443489,
                                        '001': -2.1230898660093067,
                                        '100': -55.62171263469475,
                                        '000': 544.6941704186509},
             'results_least_square': {'111': 526.9381028692694,
                                      '101': 5.535363833963913e-13,
                                      '010': 3.6472908027107565e-13,
                                      '011': 1.9366452885805074e-14,
                                      '001': 1.4224732503009818e-13,
                                      '100': 1.485506162524075e-12,
                                      '000': 497.0618971307282}
            },
            {'cal_matrix': [[0.419921875, 0.0595703125, 0.05859375, 0.00390625,
                             0.05859375, 0.0078125, 0.0107421875, 0.001953125],
                            [0.123046875, 0.48828125, 0.0205078125, 0.06640625,
                             0.017578125, 0.068359375, 0.00390625, 0.0048828125],
                            [0.1337890625, 0.0224609375, 0.46484375, 0.0673828125,
                             0.0224609375, 0.0029296875, 0.05078125, 0.0146484375],
                            [0.048828125, 0.189453125, 0.1865234375, 0.6005859375,
                             0.005859375, 0.0234375, 0.02734375, 0.08984375],
                            [0.1630859375, 0.017578125, 0.0185546875, 0.0029296875,
                             0.5009765625, 0.07421875, 0.06640625, 0.005859375],
                            [0.048828125, 0.1640625, 0.00390625, 0.01953125,
                             0.1689453125, 0.603515625, 0.021484375, 0.0849609375],
                            [0.048828125, 0.0078125, 0.17578125, 0.0263671875,
                             0.1630859375, 0.01953125, 0.62890625, 0.0732421875],
                            [0.013671875, 0.05078125, 0.0712890625, 0.212890625,
                             0.0625, 0.2001953125, 0.1904296875, 0.724609375]],
             'fidelity': 0.553955078125,
             'results': {'110': 56, '011': 61, '101': 72, '000': 214,
                         '100': 85, '010': 73, '111': 389, '001': 74},
             'results_pseudo_inverse': {'101': -2.238556155943683,
                                        '100': 0.3942157002026464,
                                        '000': 504.5427395146546,
                                        '001': 23.025360373240567,
                                        '111': 537.1157356019106,
                                        '110': -11.579410856888408,
                                        '010': -1.169412176840766,
                                        '011': -26.090672000335474},
             'results_least_square': {'101': 1.6470194662123887,
                                      '100': 3.704068302079477e-15,
                                      '000': 495.51971581347107,
                                      '001': 6.66022993369183,
                                      '111': 520.1730347866247,
                                      '110': 2.3160726808635346e-15,
                                      '010': 5.309554879096012e-15,
                                      '011': 1.8772526935717515e-14}
            }
        ]

        # Set the state labels
        state_labels = ['000', '001', '010', '011', '100', '101', '110', '111']
        meas_cal = MeasurementFitter(None, state_labels,
                                     circlabel='test')

        for tst_index, tst in enumerate(tests):
            # Set the calibration matrix
            meas_cal._cal_matrix = tests[tst_index]['cal_matrix']
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
            self.assertAlmostEqual(output_results_pseudo_inverse['000'],
                                   tests[tst_index]['results_pseudo_inverse']['000'],
                                   places=0)
            self.assertAlmostEqual(output_results_least_square['000'],
                                   tests[tst_index]['results_least_square']['000'],
                                   places=0)
            self.assertAlmostEqual(output_results_pseudo_inverse['111'],
                                   tests[tst_index]['results_pseudo_inverse']['111'],
                                   places=0)
            self.assertAlmostEqual(output_results_least_square['111'],
                                   tests[tst_index]['results_least_square']['111'],
                                   places=0)

if __name__ == '__main__':
    unittest.main()
