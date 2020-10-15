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
Expectation Value Experiment.
"""

from typing import Optional, Dict, Union, List, Callable

from qiskit import QuantumCircuit
from qiskit.result import Counts, Result
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info import Statevector

from qiskit.ignis.experiments.base import Experiment, Analysis
from .pauli_expval_method import pauli_qst_generator, pauli_qst_analyze


class StateTomography(Experiment):
    """Quantum state tomography experiment."""

    # pylint: disable=arguments-differ
    def __init__(self,
                 num_qubits: int,
                 meas_labels: Optional[List[str]] = None,
                 initial_state: Optional[Union[QuantumCircuit, Statevector]] = None,
                 qubits: Optional[List[int]] = None):
        """Initialize expectation value experiment.

        Args:
            num_qubits: the number of qubits for tomography.
            meas_labels: labels for Pauli measurement bases.
            initial_state: Optional, the initial state quantum circuit. If
                           a Statevector or array is passed in it will be
                           converted to a circuit using the `initialize`
                           instruction. Can be set after initialization.
            qubits: Optional, the qubits to contract the operator on if the
                    initial state circuit is larger than the operator.

        Additional Information:
            Custom operator decomposition methods can be used by passing in a
            callable for the method. The signature of this callable should be:
            ``method(observable)`` where ``observable`` is an operator object.

            The default method ``"Pauli"`` will convert the input operator
            into a ``SparsePauliOp`` and perform a Pauli basis measurement
            for each Pauli component of the operator.
        """

        self._meas_circuits = []
        self._metadata = []
        self._num_clbits = num_qubits

        self._meas_circuits, metadata = pauli_qst_generator(
            num_qubits, labels=meas_labels)
        super().__init__(self._meas_circuits[0].num_qubits)

        # Add metadata to base metadata
        base_meta = {'experiment': 'qst',
                     'meas_basis': 'Pauli',
                     'qubits': None}
        for meta in metadata:
            new_meta = base_meta.copy()
            for key, val in meta.items():
                new_meta[key] = val
            self._metadata.append(new_meta)

        # Set optional initial circuit
        # This can also be set later after initialization
        self._qubits = None
        self._initial_circuit = None
        if initial_state is not None:
            self.set_initial_state(initial_state, qubits=qubits)

    def set_initial_state(self,
                          initial_state: Union[QuantumCircuit, Statevector],
                          qubits: Optional[List[int]] = None):
        """Set initial state for the expectation value.

        Args:
            initial_state: Optional, the initial state quantum circuit. If
                           a Statevector or array is passed in it will be
                           converted to a circuit using the `initialize`
                           instruction. Can be set after initialization.
            qubits: Optional, the qubits to contract the operator on if the
                    initial state circuit is larger than the operator.

        Raises:
            QiskitError: if the initial state is invalid.
            QiskitError: if the number of qubits does not match the observable.
        """
        if initial_state is None:
            self._initial_circuit = None
            self._num_qubits = self._num_clbits
        elif isinstance(initial_state, QuantumCircuit):
            self._initial_circuit = initial_state
            self._num_qubits = self._initial_circuit.num_qubits
        else:
            initial_state = Statevector(initial_state)
            num_qubits = initial_state.num_qubits
            self._initial_circuit = QuantumCircuit(num_qubits)
            self._initial_circuit.initialize(initial_state.data, list(range(num_qubits)))
            self._num_qubits = self._initial_circuit.num_qubits
        self.set_meas_qubits(qubits)

    def set_meas_qubits(self, qubits: List[int]):
        """Set qubits to contract the operator on.

        Args:
            qubits: Optional, the qubits to contract the operator on if the
                    initial state circuit is larger than the operator.

        Raises:
            QiskitError: if the number of qubits does not match the observable."""
        if qubits is not None and len(qubits) != self._num_clbits:
            raise QiskitError('Number of qubits does not match operator '
                              '{} != {}'.format(len(qubits), self._num_clbits))
        self._qubits = qubits
        for meta in self._metadata:
            meta['qubits'] = qubits

    def circuits(self) -> List[QuantumCircuit]:
        """Generate a list of experiment circuits."""
        expval_circuits = []
        num_qubits = self.num_qubits
        for meas_circ in self._meas_circuits:
            num_clbits = meas_circ.num_qubits
            circ = QuantumCircuit(num_qubits, num_clbits)
            if self._initial_circuit is not None:
                circ.compose(self._initial_circuit, inplace=True)
            circ.compose(meas_circ, qubits=self._qubits, inplace=True)
            expval_circuits.append(circ)
        return expval_circuits

    def metadata(self) -> List[dict]:
        """Generate a list of experiment circuits metadata."""
        return self._metadata


class StateTomographyAnalysis(Analysis):
    """Quantum state tomography experiment analysis."""

    # pylint: disable=arguments-differ
    def __init__(self,
                 data: Optional[any] = None,
                 metadata: Optional[Dict[str, any]] = None,
                 method: str = 'linear_inversion',
                 mitigator: Optional = None,
                 psd: bool = True):
        """Initialize expectation value experiment.

        Args:
            data: Optional, result data to initialize with.
            metadata: Optional, result metadata to initialize with.
            method: Optional, the analysis method. See additional information.
            mitigator: Optional, measurement error mitigator object to apply
                       mitigation.
        """
        # Measurement Error Mitigation
        self._mitigator = mitigator
        self._psd_constraint = psd

        # Set analyze function for method
        if method != 'linear_inversion':
            raise QiskitError("Unrecognized method: {}".format(method))

        # Base Experiment Result class
        super().__init__('qst', data=data, metadata=metadata)

    def _analyze(self,
                 data: List[Counts],
                 metadata: List[Dict[str, any]],
                 mitigator: Optional = None,
                 psd: Optional = None):
        """Fit and return the Mitigator object from the calibration data."""
        if mitigator is None:
            mitigator = self._mitigator
        if psd is None:
            psd = self._psd_constraint

        return pauli_qst_analyze(data, metadata, mitigator=mitigator, psd=psd)
