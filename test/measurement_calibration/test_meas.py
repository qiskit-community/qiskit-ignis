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
    the calibration output
(withut noise), and validating that it is equally distributed
3) Testing the the measurement calibration with noise, verifying
    that it is close in the
L1-norm to the expected (equally distributed) result
"""

import unittest
import qiskit
import numpy as np
import qiskit.ignis.error_mitigation.measurement as meas_cal
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.providers.aer import noise


class TestMeasCal(unittest.TestCase):
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
            if (pattern_bit == 1):
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
                    meas_cal.measurement_calibration(qubit_list=qubits,
                                                     circlabel='test')

                # Perform an ideal execution on the generated circuits
                backend = qiskit.Aer.get_backend('qasm_simulator')
                qobj = qiskit.compile(meas_calibs, backend=backend,
                                      shots=self.shots)
                job = backend.run(qobj)
                cal_results = job.result()

                # Make a calibration matrix
                MeasCal = meas_cal.MeasurementFitter(cal_results,
                                                     state_labels,
                                                     circlabel='test')

                # Assert that the calibration matrix is equal to identity
                IdentityMatrix = np.identity(2 ** weight)
                self.assertListEqual(MeasCal.cal_matrix.tolist(),
                                     IdentityMatrix.tolist(),
                                     'Error: the calibration matrix is \
                                     not equal to identity')

                # Generate ideal (equally distributed) results
                results_dict, results_list = \
                    self.generate_ideal_results(state_labels, weight)

                # Apply the calibration matrix to results
                # in list and dict forms and using methods 0 and 1
                results_dict_1 = MeasCal.calibrate(results_dict, method=1)
                results_dict_0 = MeasCal.calibrate(results_dict, method=0)
                results_list_1 = MeasCal.calibrate(results_list, method=1)
                results_list_0 = MeasCal.calibrate(results_list, method=0)

                # Assert that the results are equally distributed
                self.assertListEqual(results_list, results_list_0.tolist(),
                                     'Error: calibration output (list format) \
                                     is not the equal (method 0)')
                self.assertListEqual(results_list,
                                     np.round(results_list_1).tolist(),
                                     'Error: calibration output (list format) \
                                     is not the equal (method 1)')
                self.assertDictEqual(results_dict, results_dict_0,
                                     'Error: calibration output (dict format) \
                                     is not the equal (method 0)')
                round_results = {}
                for key, val in results_dict_1.items():
                    round_results[key] = np.round(val)
                self.assertDictEqual(results_dict, round_results,
                                     'Error: calibration output (dict format) \
                                     is not the equal (method 0)')

    def test_meas_cal_with_noise(self):
        """
            Test an execution with a noise model
        """
        print("Testing measurement calibration with noise")

        # Generate a noise model for the qubits
        noise_model = noise.NoiseModel()
        for qi in range(5):
            read_err = noise.errors.readout_error.ReadoutError([[0.75, 0.25],
                                                                [0.1, 0.9]])
            noise_model.add_readout_error(read_err, [qi])

        # Choose all triples from 5 qubits
        for q1 in range(5):
            for q2 in range(q1+1, 5):
                for q3 in range(q2+1, 5):
                    # Generate the quantum register according to the pattern
                    qr = qiskit.QuantumRegister(5)
                    # Generate the calibration circuits
                    meas_calibs, state_labels = \
                        meas_cal.measurement_calibration(
                                qubit_list=[1, 2, 3], qr=qr)

                    # Run the calibration circuits
                    backend = qiskit.Aer.get_backend('qasm_simulator')
                    qobj = qiskit.compile(meas_calibs, backend=backend,
                                          shots=self.shots)
                    job = backend.run(qobj, noise_model=noise_model)
                    cal_results = job.result()

                    # Make a calibration matrix
                    MeasCal = meas_cal.MeasurementFitter(cal_results,
                                                         state_labels)

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
                    job = backend.run(qobj, noise_model=noise_model)
                    results = job.result()

                    # Predicted equally distributed results
                    predicted_results = [self.shots/2, 0, 0, 0, 0, 0, 0,
                                         self.shots/2]
                    # Output results with calibration using methods 0 and 1
                    output_results_0 = \
                        MeasCal.calibrate(results.get_counts(0), method=0)
                    output_results_1 = \
                        MeasCal.calibrate(results.get_counts(0), method=0)
                    output_resuls_0_array = \
                        np.asarray(list(output_results_0.values()))
                    output_resuls_1_array = \
                        np.asarray(list(output_results_1.values()))

                    # Asserting that the output result
                    # is close in the L1-norm to the predicted result
                    self.assertTrue(np.linalg.norm(predicted_results -
                                    output_resuls_0_array,
                                    1) < self.shots/2,
                                    'Error: The output results are too \
                                    far from the predicted \
                                    results (method 0)')
                    self.assertTrue(np.linalg.norm(predicted_results -
                                    output_resuls_1_array, 1) <
                                    self.shots/2,
                                    'Error: The output results are too \
                                    far from the predicted \
                                    results (method 1)')


if __name__ == '__main__':
    unittest.main()
