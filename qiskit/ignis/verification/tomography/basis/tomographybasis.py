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

"""
TomographyBasis class
"""

from qiskit import QiskitError
from qiskit.circuit import Qubit, Clbit


class TomographyBasis:
    """
    Tomography basis class.
    """

    def __init__(self, name, measurement=None, preparation=None):

        # TODO: check that measurement and preparation are both tuples
        # (labels, circuit_fn, matrix_fn)
        # Also check functions have correct signature and are return valid
        # outputs for all specified labels

        self._name = name
        self._measurement = False
        self._preparation = False

        if measurement is not None and len(measurement) == 3:
            self._measurement = True
            self._measurement_labels = measurement[0]
            self._measurement_circuit = measurement[1]
            self._measurement_matrix = measurement[2]

        if preparation is not None and len(preparation) == 3:
            self._preparation = True
            self._preparation_labels = preparation[0]
            self._preparation_circuit = preparation[1]
            self._preparation_matrix = preparation[2]

    @property
    def name(self):
        """The name of the tomography basis."""
        return self._name

    @property
    def measurement(self):
        """The measurement of the tomography basis."""
        return self._measurement

    @property
    def preparation(self):
        """The preparation of the tomography basis."""
        return self._preparation

    @property
    def measurement_labels(self):
        """The measurement labels of the tomography basis."""
        if self.measurement is True:
            return self._measurement_labels
        return None

    @property
    def preparation_labels(self):
        """The preparation labels of the tomography basis."""
        if self.preparation is True:
            return self._preparation_labels
        return None

    def measurement_circuit(self, op, qubit, clbit):
        """Return the measurement circuits."""
        # Error Checking
        if self.measurement is False:
            raise QiskitError(
                "{} is not a measurement basis".format(self._name))

        if not isinstance(qubit, Qubit):
            raise QiskitError('Input must be a qubit in a QuantumRegister')

        if not isinstance(clbit, Clbit):
            raise QiskitError('Input must be a bit in a ClassicalRegister')

        if op not in self._measurement_labels:
            msg = "Invalid {0} measurement operator label".format(self._name)
            error = "'{0}' != {1}".format(op, self._measurement_labels)
            raise ValueError("{0}: {1}".format(msg, error))

        # Return QuantumCircuit function output
        return self._measurement_circuit(op, qubit, clbit)

    def preparation_circuit(self, op, qubit):
        """Return the preparation circuits."""
        # Error Checking
        if self.preparation is False:
            raise QiskitError("{} is not a preparation basis".format(
                self._name))

        if not isinstance(qubit, Qubit):
            raise QiskitError('Input must be a qubit in a QuantumRegister')

        if op not in self._preparation_labels:
            msg = "Invalid {0} preparation operator label".format(self._name)
            error = "'{}' not in {}".format(op, self._preparation_labels)
            raise ValueError("{0}: {1}".format(msg, error))

        return self._preparation_circuit(op, qubit)

    def measurement_matrix(self, label, outcome):
        """Return the measurement matrix."""
        if self.measurement is False:
            raise QiskitError("{} is not a measurement basis".format(
                self._name))

        # Check input is valid for this basis
        if label not in self._measurement_labels:
            msg = "Invalid {0} measurement operator label".format(self._name)
            error = "'{}' not in {}".format(label, self._measurement_labels)
            raise ValueError("{0}: {1}".format(msg, error))

        # Check outcome is valid for this measurement
        allowed_outcomes = [0, 1, '0', '1']
        if outcome not in allowed_outcomes:
            error = "'{}' not in {}".format(outcome, allowed_outcomes)
            raise ValueError('Invalid measurement outcome: {}'.format(error))

        return self._measurement_matrix(label, outcome)

    def preparation_matrix(self, label):
        """Return the preparation matrix."""

        if self.preparation is False:
            raise QiskitError("{} is not a preparation basis".format(
                self._name))

        # Check input is valid for this basis
        if label not in self._preparation_labels:
            msg = "Invalid {0} preparation operator label".format(self._name)
            error = "'{}' not in {}".format(label, self._preparation_labels)
            raise ValueError("{0}: {1}".format(msg, error))

        return self._preparation_matrix(label)
