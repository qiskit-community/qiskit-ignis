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

from typing import Optional, Dict, Union, List

from qiskit import QuantumCircuit
from qiskit.result import Result
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info import Statevector

from qiskit.ignis.experiments.base import Experiment, Generator, Analysis
from .pauli_method import PauliExpvalGenerator, pauli_analysis_fn
from .snapshot_method import SnapshotExpvalGenerator, snapshot_analysis_fn


class ExpectationValueExperiment(Experiment):
    """Expectation value experiment."""

    # pylint: disable=arguments-differ
    def __init__(self,
                 observable: Optional[BaseOperator] = None,
                 method: str = 'Pauli',
                 initial_state: Optional[Union[QuantumCircuit, Statevector]] = None,
                 qubits: Optional[List[int]] = None,
                 mitigator: Optional = None,
                 job: Optional = None):
        """Initialize expectation value experiment.

        Args:
            observable: an operator object for obserable.
            method: the measurement circuit conversion method for the observable.
                    See additional information.
            initial_state: Optional, the initial state quantum circuit. If
                           a Statevector or array is passed in it will be
                           converted to a circuit using the `initialize`
                           instruction. Can be set after initialization.
            qubits: Optional, the qubits to contract the operator on if the
                    initial state circuit is larger than the operator.
            mitigator: Optional, measurement mitigator object.
            job: Optional, job result object.

        Raises:
            QiskitError: if input is not valid.

        Additional Information:
            Custom operator decomposition methods can be used by passing in a
            callable for the method. The signature of this callable should be:
            ``method(observable)`` where ``observable`` is an operator object.

            The default method ``"Pauli"`` will convert the input operator
            into a ``SparsePauliOp`` and perform a Pauli basis measurement
            for each Pauli component of the operator.
        """
        analysis = ExpectationValueAnalysis(mitigator=mitigator)
        if observable is not None:
            generator = ExpectationValueGenerator(
                observable=observable,
                method=method,
                initial_state=initial_state,
                qubits=qubits)
        else:
            generator = None

        super().__init__(generator=generator, analysis=analysis, job=job)


class ExpectationValueGenerator(Generator):
    """Expectation value experiment generator."""

    # pylint: disable=arguments-differ
    def __init__(self,
                 observable: BaseOperator,
                 method: Union[str, Generator] = 'Pauli',
                 initial_state: Optional[Union[QuantumCircuit, Statevector]] = None,
                 qubits: Optional[List[int]] = None):
        """Initialize expectation value experiment.

        Args:
            observable: an operator object for obserable.
            method: the measurement circuit conversion method for the observable.
                    See additional information.
            initial_state: Optional, the initial state quantum circuit. If
                           a Statevector or array is passed in it will be
                           converted to a circuit using the `initialize`
                           instruction. Can be set after initialization.
            qubits: Optional, the qubits to contract the operator on if the
                    initial state circuit is larger than the operator.

        Raises:
            QiskitError: if input parameters are not valid.

        Additional Information:
            Custom operator decomposition methods can be used by passing in a
            callable for the method. The signature of this callable should be:
            ``method(observable)`` where ``observable`` is an operator object.

            The default method ``"Pauli"`` will convert the input operator
            into a ``SparsePauliOp`` and perform a Pauli basis measurement
            for each Pauli component of the operator.
        """
        # Attributes
        self._op = observable

        # Measurement circuit Generator
        if isinstance(method, Generator):
            self._meas_generator = method(observable, qubits=qubits)
        elif method == 'Pauli':
            self._meas_generator = PauliExpvalGenerator(self._op, qubits=qubits)
        elif method == 'snapshot':
            self._meas_generator = SnapshotExpvalGenerator(self._op, qubits=qubits)
        else:
            raise QiskitError("Unrecognized ExpectationValue method: {}".format(method))

        super().__init__('expval', self._meas_generator.qubits)

        # Initial state circuit
        # This can also be set later after initialization
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
        # TODO: double check qubits for meas circuits
        if isinstance(initial_state, QuantumCircuit):
            self._initial_circuit = initial_state
        else:
            initial_state = Statevector(initial_state)
            num_qubits = initial_state.num_qubits
            self._initial_circuit = QuantumCircuit(num_qubits)
            self._initial_circuit.initialize(initial_state.data, list(range(num_qubits)))

        self._num_qubits = self._initial_circuit.num_qubits
        self.observable_qubits(qubits)

    def observable_qubits(self, qubits: List[int]):
        """Set qubits to contract the operator on.

        Args:
            qubits: Optional, the qubits to contract the operator on if the
                    initial state circuit is larger than the operator.

        Raises:
            QiskitError: if the number of qubits does not match the observable."""
        if qubits is not None:
            if len(qubits) != self._op.num_qubits:
                raise QiskitError('Number of qubits does not match operator '
                                  '{} != {}'.format(len(qubits), self._op.num_qubits))
            self._meas_generator.qubits = qubits

    def circuits(self) -> List[QuantumCircuit]:
        """Generate a list of experiment circuits."""
        expval_circuits = []
        num_qubits = self._initial_circuit.num_qubits
        for meas_circ in self._meas_generator.circuits():
            num_clbits = meas_circ.num_qubits
            circ = QuantumCircuit(num_qubits, num_clbits)
            circ.compose(self._initial_circuit, inplace=True)
            circ.compose(meas_circ, qubits=self._meas_generator.qubits, inplace=True)
            expval_circuits.append(circ)
        return expval_circuits

    def _extra_metadata(self) -> List[dict]:
        """Generate a list of experiment circuits metadata."""
        # TODO: Add bit mask if initial circuit has clbits
        return self._meas_generator.metadata()


class ExpectationValueAnalysis(Analysis):
    """Expectation value experiment analysis."""

    # pylint: disable=arguments-differ
    def __init__(self,
                 data: Optional[any] = None,
                 metadata: Optional[Dict[str, any]] = None,
                 method: Optional[Analysis] = None,
                 mitigator: Optional = None,
                 exp_id: Optional[str] = None):
        """Initialize expectation value experiment.

        Args:
            data: Optional, result data to initialize with.
            metadata: Optional, result metadata to initialize with.
            method: Optional, the analysis method. See additional information.
            mitigator: Optional, measurement error mitigator object to apply
                       mitigation.
            exp_id: Optional, experiment id string.

        Raises:
            QiskitError: if inputs are invalid.

        Additional Information:
            Custom analysis methods can be supplied using a callable for the
            ``method`` kwarg. The signature of this callable should be:
            ``method(data, metadata, mitigator)`` where ``data`` is a list of
            :class:`Counts` objects, ``metadata`` is a list of metadata dicts,
            and ``mitigator`` is either a measurement error mitigator object,
            or ``None`` for no mitigation.

            The default method ``"Pauli"`` assumes all counts correspond to
            Pauli basis measurements. The measurement basis and coefficient
            should be stored in the metadata under the fields ``"basis"`` and
            ``"coeff"`` respectively.

            * If the basis field is not present a default basis of measuring in
              the Z-basis on all qubits is used.
            * If the coefficient field is not present it is assumed to be 1.
        """
        # Measurement Error Mitigation
        self._mitigator = mitigator

        # Set analyze function for method
        if isinstance(method, str) and method not in ['Pauli', 'snapshot']:
            raise QiskitError("Unrecognized ExpectationValue method: {}".format(method))
        self._method = method

        super().__init__(data=data,
                         metadata=metadata,
                         name='expval',
                         exp_id=exp_id)

    def run(self, mitigator: Optional = None):
        """Analyze the stored expectation value data.

        Returns:
            any: the output of the analysis,
        """
        if self._method is None:
            method = self._exp_metadata[0].get('method')
        else:
            method = self._method
        self._set_analysis_fn(method)
        if mitigator is None:
            mitigator = self._mitigator
        return super().run(mitigator=mitigator)

    def _set_analysis_fn(self, method):
        if method == 'Pauli':
            self._analysis_fn = pauli_analysis_fn
        elif method == 'snapshot':
            self._analysis_fn = snapshot_analysis_fn
        elif isinstance(method, str):
            raise QiskitError('Unrecognized expectation value method.')
        else:
            self._analysis_fn = method

    def _format_data(self,
                     data: Result,
                     metadata: Dict[str, any],
                     index: int):
        """Filter the required data from a Result.data dict"""
        if self._method is None:
            method = metadata.get('method')
        else:
            method = self._method

        # For snapshots we don't use counts
        if method == 'snapshot':
            snapshots = data.data(index).get('snapshots', {}).get('expectation_value', {})
            snapshots['shots'] = data.results[index].shots
            return snapshots

        # Otherwise we return counts
        return super()._format_data(data, metadata, index)
