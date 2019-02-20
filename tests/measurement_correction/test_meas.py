# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Test of measurement correction:
Preparation of all 2 ** n basis states, generating the calibration circuits,
and comouting the callibration matrices
"""

import unittest
import qiskit
import numpy as np
import qiskit.ignis.measurement_correction as meas_corr

class TestMeasCorr(unittest.TestCase):
    """ The test class """

    @staticmethod
    def choose_calibration(nq,qr,pattern_type):
        qubits = []
        weight = 0
        for i in range (nq):
            pattern_bit = pattern_type&1
            pattern_type = pattern_type>>1
            if (pattern_bit == 1):
                qubits.append(qr[i])
                weight += 1
        return qubits, weight


    def test_meas_corr(self):
        # Test up to 5 qubits
        nq_list = [1, 2, 3, 4, 5]

        for nq in nq_list:

            print("Testing %d qubit measurement_correction" % nq)
            qr = qiskit.QuantumRegister(nq)

            for pattern_type in range(1, 2 ** nq):
                # Choose patterns
                print (pattern_type, bin(pattern_type))
                qubits, weight = self.choose_calibration(nq,qr,pattern_type)
                print (qubits)
                # Generate the calibration circuits
                meas_calibs, state_labels = meas_corr.measurement_calibration(qubits)
                print (state_labels)

                # Perform an ideal execution on the generated circuits
                backend = qiskit.Aer.get_backend('qasm_simulator')
                qobj = qiskit.compile(meas_calibs, backend=backend, shots=1000)
                job = backend.run(qobj)
                cal_results = job.result()

                # make a calibration matrix
                MeasCal = meas_corr.MeasurementFitter(cal_results, state_labels)

                #verify that the calibration matrix is equal to identity
                print(MeasCal.cal_matrix.tolist())
                self.assertSequenceEqual(MeasCal.cal_matrix.tolist(), (np.identity(2 ** weight)).tolist(),
                                 'Error: the calibration matrix is not equal to identity')

if __name__ == '__main__':
    unittest.main()

