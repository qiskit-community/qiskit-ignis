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

from math import ceil, log2
from typing import Optional, Tuple, List, Dict

from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError


def expval_meas_mitigator_circuits(num_qubits: int,
                                   method: Optional[str] = 'CTMP',
                                   labels: Optional[List[str]] = None) -> Tuple[
                                       List[QuantumCircuit], List[Dict[str, any]]
                                   ]:
    """Generate measurement error mitigator circuits and metadata.

    Use the :class:`~qiskit.ignis.mitigation.ExpvalMeasMitigatorFitter`
    class to fit the execution results to construct a calibrated expectation
    value measurement error mitigator.

    Args:
        num_qubits: the number of qubits to calibrate.
        method: the mitigation method ``'complete'``, ``'tensored'``, or ``'CTMP'``.
        labels: Optional, custom labels to run for calibration. If None
                the method will determine the default label values.

    Returns:
        tuple: (circuits, metadata) the measurement error characterization
               circuits, and metadata for the fitter.

    Mitigation Method:
        * The ``'complete'`` method will generate all :math:`2^n` computational
          basis states measurement circuits and fitting will return a
          :class:`~qiskit.ignis.mitigation.CompleteExpvalMeasMitigator`. This
          method should only be used for small numbers of qubits.
        * The ``'tensored'`` method will generate two input state circuits of
          the all 0 and all 1 states on number of qubits unless custom labels
          are specified. Ftting will return a
          :class:`~qiskit.ignis.mitigation.TensoredExpvalMeasMitigator`. This
          method assumes measurement errors are uncorrelated between qubits.
        * The ``'CTMP'`` method will generate input state circuits,
          unless custom labels are specified. The default input states must
          obey the following cirterion: for every pair of qubits, projection
          of the input states on the two qubits contains all four possible
          assignments to the qubits (`00, 01, 10, 11`). For `n<7`, these
          would be the all 1 state and the :math:`n` states with a
          single qubit in the 1 state and all others in the 0 state (also
          the all 0 state, if n<3). For `n>=7`, these would be the all 0
          state, the all 1 state, and 2*ceil(log2(n)) states resulting from
          the following procedure: For each qubits, write its index in binary
          form, horizontically. For example: with 8 qubits `0, 1, 2,..., 7`,
          for qubit 7 we write:
          1
          1
          1
          And for all 8 qubits, we obtain `ceil(log2(n))=3` lines:
          00001111
          00110011
          01010101
          One can see that the every column is the binary form of the column
          number.
          Then write again, the same lines, negated:
          11110000
          11001100
          10101010
          The all 0 and all 1 states guarantee that each pair of qubits has
          input states with projections `00` and `11`. The other lines
          guarantee the projections `01` and `10` (since the qubits are
          different, when written in binary form, there must be a digit
          in which they differ).
          Fitting will return a
          :class:`~qiskit.ignis.mitigation.CTMPExpvalMeasMitigator`.

    Example:

        The following example shows calibrating a 5-qubit expectation value
        measurement error mitigator using the ``'tensored'`` method.

        .. jupyter-execute::

            from qiskit import execute
            from qiskit.test.mock import FakeVigo
            import qiskit.ignis.mitigation as mit

            backend = FakeVigo()
            num_qubits = backend.configuration().num_qubits

            # Generate calibration circuits
            circuits, metadata = mit.expval_meas_mitigator_circuits(
                num_qubits, method='tensored')
            result = execute(circuits, backend, shots=8192).result()

            # Fit mitigator
            mitigator = mit.ExpvalMeasMitigatorFitter(result, metadata).fit()

            # Plot fitted N-qubit assignment matrix
            mitigator.plot_assignment_matrix()

        The following shows how to use the above mitigator to apply measurement
        error mitigation to expectation value computations

        .. jupyter-execute::

            from qiskit import QuantumCircuit

            # Test Circuit with expectation value -1.
            qc = QuantumCircuit(num_qubits)
            qc.x(range(num_qubits))
            qc.measure_all()

            # Execute
            shots = 8192
            seed_simulator = 1999
            result = execute(qc, backend, shots=8192, seed_simulator=1999).result()
            counts = result.get_counts(0)

            # Expectation value of Z^N without mitigation
            expval_nomit, error_nomit = mit.expectation_value(counts)
            print('Expval (no mitigation): {:.2f} \u00B1 {:.2f}'.format(
                expval_nomit, error_nomit))

            # Expectation value of Z^N with mitigation
            expval_mit, error_mit = mit.expectation_value(counts,
                meas_mitigator=mitigator)
            print('Expval (with mitigation): {:.2f} \u00B1 {:.2f}'.format(
                expval_mit, error_mit))

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
                'method': method,
            })
            self._circuits.append(self._calibration_circuit(num_qubits, label))

    def _method_labels(self, method):
        """Generate labels for initializing via a standard method."""

        if method == 'tensored':
            return [self._num_qubits * '0', self._num_qubits * '1']

        if method in ['CTMP', 'ctmp']:
            # See details at the docstring of expval_meas_mitigator_circuits
            if self._num_qubits >= 7:
                length = ceil(log2(self._num_qubits))
                labels = ['']*2*length
                labels.extend([self._num_qubits * '0', self._num_qubits * '1'])
                for i in range(self._num_qubits):
                    bits = bin(i)[2:]
                    bits = (length - len(bits)) * '0' + bits
                    for j, b in enumerate(bits):
                        labels[j] = labels[j] + b
                        if b == '0':
                            labels[j+length] = labels[j+length] + '1'
                        else:
                            labels[j+length] = labels[j+length] + '0'
            else:
                labels = [self._num_qubits * '1']
                if self._num_qubits < 3:
                    labels.append(self._num_qubits * '0')
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
