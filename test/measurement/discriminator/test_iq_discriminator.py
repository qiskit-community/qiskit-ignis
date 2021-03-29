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

# pylint: disable=no-name-in-module,invalid-name

"""
Test IQ discrimination fitters.
"""

import unittest

import test.utils as utils

try:
    from sklearn.svm import SVC
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

import qiskit
from qiskit import Aer
from qiskit.exceptions import QiskitError
from qiskit.ignis.measurement.discriminator.filters import DiscriminationFilter
from qiskit.ignis.measurement.discriminator.iq_discriminators import \
    LinearIQDiscriminator, SklearnIQDiscriminator
from qiskit.ignis.mitigation.measurement import tensored_meas_cal
from qiskit.result.models import ExperimentResultData


@unittest.skipUnless(HAS_SKLEARN, 'scikit-learn is required to run these tests')
class BaseTestIQDiscriminator(unittest.TestCase):
    """
    Base class for IQ discriminator test cases.
    """

    def setUp(self):
        """
        Setup internal variables and a fake simulation. Aer is used to get the
        structure of the qiskit.Result. The IQ data is generated using gaussian
        random number generators.
        """
        self.shots = 52
        self.qubits = [0, 1]

        meas_cal, _ = tensored_meas_cal([[0], [1]])

        backend = Aer.get_backend('qasm_simulator')
        job = qiskit.execute(meas_cal, backend=backend, shots=self.shots,
                             meas_level=1)

        self.cal_results = job.result()

        i0, q0, i1, q1 = 0., -1., 0., 1.
        ground = utils.create_shots(i0, q0, 0.1, 0.1, self.shots, self.qubits)
        excited = utils.create_shots(i1, q1, 0.1, 0.1, self.shots, self.qubits)

        self.cal_results.results[0].meas_level = 1
        self.cal_results.results[1].meas_level = 1
        self.cal_results.results[0].data = ExperimentResultData(memory=ground)
        self.cal_results.results[1].data = ExperimentResultData(memory=excited)


class TestLinearIQDiscriminator(BaseTestIQDiscriminator):
    """
    Test methods of the IQ discriminators.
    """

    def test_get_xdata(self):
        """
        Tests that the discriminator properly retrieves the x data from the
        Qiskit result.
        """
        discriminator = LinearIQDiscriminator(self.cal_results,
                                              self.qubits,
                                              ['00', '11'])

        xdata = discriminator.get_xdata(self.cal_results, 0)

        self.assertEqual(len(xdata), self.shots*2)
        self.assertEqual(len(xdata[0]), len(self.qubits) * 2)

        xdata = discriminator.get_xdata(self.cal_results, 0, ['cal_00'])

        self.assertEqual(len(xdata), self.shots)
        self.assertEqual(len(xdata[0]), 4)

        xdata = discriminator.get_xdata(self.cal_results, 1)

        self.assertEqual(len(xdata), 0)

    def test_get_ydata(self):
        """
        Tests that the discriminator properly retrieves the y data from the
        Qiskit calibration results.
        """
        discriminator = LinearIQDiscriminator(self.cal_results,
                                              self.qubits,
                                              ['00', '11'])

        xdata = discriminator.get_xdata(self.cal_results, 0)
        ydata = discriminator.get_ydata(self.cal_results, 0)

        self.assertEqual(len(xdata), len(ydata))

        ydata = discriminator.get_ydata(self.cal_results, 0, ['cal_00'])

        self.assertEqual(len(ydata), self.shots)
        self.assertEqual(ydata[0], '00')

        ydata = discriminator.get_ydata(self.cal_results, 1)

        self.assertEqual(len(ydata), 0)

    def test_discrimination(self):
        """
        Test that the discriminator can be trained on the simulated data and
        that it can properly discriminate between ground and excited sates.
        """
        i0, q0, i1, q1 = 0., -1., 0., 1.
        discriminator = LinearIQDiscriminator(self.cal_results,
                                              self.qubits,
                                              ['00', '11'])

        excited_predicted = discriminator.discriminate([[i1, q1, i1, q1]])
        ground_predicted = discriminator.discriminate([[i0, q0, i0, q0]])

        self.assertEqual(excited_predicted[0], '11')
        self.assertEqual(ground_predicted[0], '00')

    def filter_and_discriminate(self):
        """
        Test the process of discriminating and then applying the discriminator
        using a filter.
        """
        i0, q0, i1, q1 = 0., -1., 0., 1.

        discriminator = LinearIQDiscriminator(self.cal_results,
                                              self.qubits,
                                              ['00', '11'])

        iq_filter = DiscriminationFilter(discriminator)

        new_result = iq_filter.apply(self.cal_results)

        for nr in new_result.results:
            self.assertEqual(nr.meas_level, 2)

        for state in new_result.results[0].data.counts.to_dict():
            self.assertEqual(state, '0x0')

        for state in new_result.results[1].data.counts.to_dict():
            self.assertEqual(state, '0x3')

        self.assertEqual(len(new_result.get_memory(0)), self.shots)

        self.qubits = [0]

        discriminator = LinearIQDiscriminator(self.cal_results,
                                              self.qubits,
                                              ['0', '1'])

        self.assertEqual(discriminator.discriminate([[i0, q0]])[0], '0')
        self.assertEqual(discriminator.discriminate([[i1, q1]])[0], '1')

    def test_is_calibration(self):
        """
        Test is the discriminator can properly recognize calibration names.
        """
        discriminator = LinearIQDiscriminator([], [])

        self.assertTrue(discriminator.is_calibration('cal_01101'))
        self.assertTrue(discriminator.is_calibration('cal_2121'))
        self.assertFalse(discriminator.is_calibration('cal_01101b'))
        self.assertFalse(discriminator.is_calibration('cal01101b'))
        self.assertFalse(discriminator.is_calibration('test'))
        self.assertFalse(discriminator.is_calibration('_cal_2121'))


class ClassifierWithoutFit:
    """ A dummy classifier without a .fit() method. """
    def predict(self):
        """ A dummy predict method. """


class ClassifierWithoutPredict:
    """ A dummy classifier withou a .predict() method. """
    def fit(self):
        """ A dummy fit method. """


class TestSklearnIQDiscriminator(BaseTestIQDiscriminator):
    """
    Test methods of the sklearn IQ discriminators.
    """

    def test_classifier_type_check(self):
        """
        Test that the discriminator correctly checks that its classifier
        has fit and predict methods.
        """
        with self.assertRaisesRegex(
                QiskitError,
                r'^\'Classifier of type "ClassifierWithoutFit" does not have a'
                r' callable "fit" method\.\'$'
                ):
            SklearnIQDiscriminator(
                ClassifierWithoutFit(), self.cal_results, self.qubits,
                ['00', '11'])

        with self.assertRaisesRegex(
                QiskitError,
                r'^\'Classifier of type "ClassifierWithoutPredict" does not'
                r' have a callable "predict" method\.\'$'
                ):
            SklearnIQDiscriminator(
                ClassifierWithoutPredict(), self.cal_results, self.qubits,
                ['00', '11'])

        # check that a valid classifier does not raise an error
        svc = SVC(C=10., kernel="rbf", gamma="scale")
        SklearnIQDiscriminator(
            svc, self.cal_results, self.qubits, ['00', '11'])

    def test_discrimination(self):
        """
        Test that the discriminator can be trained on the simulated data and
        that it can properly discriminate between ground and excited sates.
        """
        i0, q0, i1, q1 = 0., -1., 0., 1.

        svc = SVC(C=10., kernel="rbf", gamma="scale")
        discriminator = SklearnIQDiscriminator(
            svc, self.cal_results, self.qubits, ['00', '11'])

        excited_predicted = discriminator.discriminate([[i1, q1, i1, q1]])
        ground_predicted = discriminator.discriminate([[i0, q0, i0, q0]])

        self.assertEqual(excited_predicted[0], '11')
        self.assertEqual(ground_predicted[0], '00')

    def filter_and_discriminate(self):
        """
        Test the process of discriminating and then applying the discriminator
        using a filter.
        """
        i0, q0, i1, q1 = 0., -1., 0., 1.

        svc = SVC(C=10., kernel="rbf", gamma="scale")
        discriminator = SklearnIQDiscriminator(
            svc, self.cal_results, self.qubits, ['00', '11'])

        iq_filter = DiscriminationFilter(discriminator)

        new_result = iq_filter.apply(self.cal_results)

        for nr in new_result.results:
            self.assertEqual(nr.meas_level, 2)

        for state in new_result.results[0].data.counts.to_dict():
            self.assertEqual(state, '0x0')

        for state in new_result.results[1].data.counts.to_dict():
            self.assertEqual(state, '0x3')

        self.assertEqual(len(new_result.get_memory(0)), self.shots)

        self.qubits = [0]

        svc = SVC(C=10., kernel="rbf", gamma="scale")
        discriminator = SklearnIQDiscriminator(
            svc, self.cal_results, self.qubits, ['0', '1'])

        self.assertEqual(discriminator.discriminate([[i0, q0]])[0], '0')
        self.assertEqual(discriminator.discriminate([[i1, q1]])[0], '1')
