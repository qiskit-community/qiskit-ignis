# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Expectation value measurement error migitation generator.
"""

from typing import Optional, Tuple, List

from qiskit import QuantumCircuit, transpile
from qiskit.exceptions import QiskitError
from qiskit.providers import BaseBackend


def expval_meas_mitigator_circuits(num_qubits: int,
                                   method: str = 'CTMP',
                                   labels: Optional[List[str]] = None):
    """Generate measurement error mitigator calibration circuits.

    Circuits is a list of circuits for the experiments, metadata is a list
    of metadata for the experiment that is required by the fitter to
    interpreting results.

    Args:
        num_qubits: the number of qubits to calibrate.
        method: the mitigation method 'complete', 'tensored', or 'CTMP'.
        labels: custom labels to run for calibration.

    Returns:
        tuple: (circuits, metadata)
    """
    generator = ExpvalMeasMitigatorCircuits(num_qubits, method, labels)
    return generator.generate_circuits()


class ExpvalMeasMitigatorCircuits:
    """Expecation value measurement error mitigator calibration circuits."""

    # pylint: disable=arguments-differ
    def __init__(self,
                 num_qubits: int,
                 method: str = 'CTMP',
                 labels: Optional[List[str]] = None):
        """Initialize measurement mitigator calibration generator.

        Args:
            num_qubits: the number of qubits to calibrate.
            method: the mitigation method 'complete', 'tensored', or 'CTMP'.
            labels: custom labels to run for calibration.
        """
        self._num_qubits = num_qubits
        self._circuits = []
        self._metadata = []
        if labels is None:
            labels = self._method_labels(method)
        for label in labels:
            self._metadata.append({
                'experiment': 'meas_mit',
                'cal': label,
            })
            self._circuits.append(self._calibration_circuit(num_qubits, label))

    def _method_labels(self, method):
        """Generate labels for initilizing via a standard method."""

        if method == 'tensored':
            return [self._num_qubits * '0', self._num_qubits * '1']

        if method in ['CTMP', 'ctmp']:
            labels = [self._num_qubits * '0', self._num_qubits * '1']
            for i in range(self._num_qubits):
                labels.append(((self._num_qubits - i - 1) * '0') + '1' +
                              (i * '0'))
            return labels

        if method == 'complete':
            labels = []
            for i in range(2**self._num_qubits):
                bits = bin(i)[2:]
                label = (self._num_qubits - len(bits)) * '0' + bits
                labels.append(label)
            return labels

        raise QiskitError("Unrecognized method {}".format(method))

    def generate_circuits(self) -> Tuple[List[QuantumCircuit], List[dict]]:
        """Return experiment payload data.
​
        Circuits is a list of circuits for the experiments, metadata is a list of metadata
        for the experiment that is required by the fitter to interpreting results.
​
        Returns:
            tuple: circuits, metadata
        """
        return self._circuits, self._metadata

    @staticmethod
    def _calibration_circuit(num_qubits: int, label: str) -> QuantumCircuit:
        """Return a calibration circuit.

        This is an N-qubit circuit where N is the length of the label.
        The circuit consists of X-gates on qubits with label bits equal to 1,
        and measurements of all qubits.
        """
        circ = QuantumCircuit(num_qubits, name='meas_mit_cal_' + label)
        for i, val in enumerate(reversed(label)):
            if val == '1':
                circ.x(i)
        circ.measure_all()
        return circ
