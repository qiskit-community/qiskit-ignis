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

"""TomographyBasis class
"""
from typing import Optional, Tuple, Union
import numpy as np
from qiskit import QiskitError
from qiskit.circuit import Qubit, Clbit, QuantumCircuit


class TomographyBasis:
    """Tomography basis class.
    """

    def __init__(self, name: str,
                 measurement: Optional[Tuple] = None,
                 preparation: Optional[Tuple] = None
                 ):
        """Initializes the tomography basis with given basis data

        For both measurement and preparation bases, we excpet the same
        input: a tuple (names, circuit_fn, matrix_fn) containing

        * **names** - the names (strings) of the elements of the basis.
        * **circuit_fn** a function taking a pair (str, QuantumRegister)
          and returns a circuit on the given qubits that corresponds to
          the basis element denoted by the given string.
        * **matrix_fx** - a function taking a basis
          element and returning the matrix describing it.
          In the case of measurement matrix, the function should
          accept a pair (**name**, **outcome**) of the basis element name
          and the expected outcome (0 or 1); in the case of preparation
          matrix only **name** is passed.

        Args:
            name: Name for the whole basis (e.g. "Pauli")
            measurement: measurement data (as described above)
            preparation: preparation data (as described above)
        """
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

    def measurement_circuit(self, op: str,
                            qubit: Qubit,
                            clbit: Clbit
                            ) -> QuantumCircuit:
        """Return the measurement circuit on the given qubit and clbit

        Args:
            op: The name of the measurement operator
            qubit: The qubit on which to apply the measurement operator
            clbit: The classical bit that will hold the measurement result
        Raises:
            QiskitError: In case no measurement data is present in the basis
                or **qubit**/**clbit** are not Qubit/Clbit
            ValueError: if **op** is not a n name of a measurement operator
                in the basis
        Returns:
            The measurement circuit
        """
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

        return self._measurement_circuit(op, qubit, clbit)

    def preparation_circuit(self, op: str, qubit: Qubit) -> QuantumCircuit:
        """Return the preparation circuit on the given qubit

        Args:
            op: The name of the preparation operator
            qubit: The qubit on which to apply the preparation operator
        Raises:
            QiskitError: In case no preparation data is present in the basis
                or **qubit** is not Qubit
            ValueError: if **op** is not a n name of a preparation operator
                in the basis
        Returns:
            The preparation circuit
        """
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

    def measurement_matrix(self, label: str,
                           outcome: Union[str, int]
                           ) -> np.array:
        """Return the measurement matrix for a given measurement operator
        and expected outcome.

        Args:
            label: Name of the measurement element.
            outcome: The expected outcome: 0 or 1 or '0' or '1'.
        Raises:
            QiskitError: If the TomographyBasis has no measurement data
            ValueError: if **label** does not describe an element of the
                measurement data, or if **outcome** is not a valid outcome.
        Returns:
            The measurement matrix.
        """
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

    def preparation_matrix(self, label: str) -> np.array:
        """Return the preparation matrix for a given preparation operator.

        Args:
            label: Name of the preparation element.
        Raises:
            QiskitError: If the TomographyBasis has no preparation data
            ValueError: if **label** does not describe an element of the
                preparation data.
        Returns:
            The preparation matrix.
        """

        if self.preparation is False:
            raise QiskitError("{} is not a preparation basis".format(
                self._name))

        # Check input is valid for this basis
        if label not in self._preparation_labels:
            msg = "Invalid {0} preparation operator label".format(self._name)
            error = "'{}' not in {}".format(label, self._preparation_labels)
            raise ValueError("{0}: {1}".format(msg, error))

        return self._preparation_matrix(label)
