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
Snapshot Expectation Value Experiment.
"""

import logging
from typing import Optional, Dict, Union, List

import numpy as np

from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import SparsePauliOp, Operator

from qiskit.ignis.experiments.base import ConstantGenerator

logger = logging.getLogger(__name__)


class SnapshotExpvalGenerator(ConstantGenerator):
    """Snapshot expectation value Generator"""

    def __init__(self,
                 observable: Union[SparsePauliOp, Operator],
                 qubits: Optional[List[int]] = None):
        """Initialize generator"""
        if isinstance(observable, SparsePauliOp):
            self._op = observable
        else:
            self._op = SparsePauliOp.from_operator(observable)

        # Get snapshot params for operator
        params = [[coeff, pauli] for pauli, coeff in self._op.label_iter()]

        # Get snapshot params for operator ** 2
        pauli_op_sq = self._op.dot(self._op).simplify()
        params_sq = [[coeff, pauli] for pauli, coeff in pauli_op_sq.label_iter()]

        num_qubits = self._op.num_qubits
        snapshot_qubits = list(range(num_qubits))

        circuit = QuantumCircuit(num_qubits)
        circuit.snapshot('expval', snapshot_type='expectation_value_pauli',
                         qubits=snapshot_qubits, params=params)
        circuit.snapshot('sq_expval', snapshot_type='expectation_value_pauli',
                         qubits=snapshot_qubits, params=params_sq)

        super().__init__('expval', [circuit], [{'method': 'snapshot'}], qubits=qubits)


def snapshot_analysis_fn(data: List[Dict],
                         metadata: List[Dict[str, any]],
                         mitigator: Optional = None):
    """Fit expectation value from snapshots."""
    if mitigator is not None:
        logger.warning('Error mitigation cannot be used with the snapshot'
                       ' expectation value method.')

    if len(data) != 1:
        raise QiskitError("Invalid list of data")

    snapshots = data[0]
    meta = metadata[0]

    if 'expval' not in snapshots or 'sq_expval' not in snapshots:
        raise QiskitError("Snapshot keys missing from snapshot dict.")

    expval = snapshots['expval'][0]['value']
    sq_expval = snapshots['sq_expval'][0]['value']

    # Convert to real if imaginary part is zero
    if np.isclose(expval.imag, 0):
        expval = expval.real
    if np.isclose(sq_expval.imag, 0):
        sq_expval = sq_expval.real

    # Compute variance and standard error
    variance = sq_expval - expval ** 2

    # Get shots
    if 'shots' in meta:
        shots = meta['shots']
    else:
        shots = snapshots.get('shots', 1)

    stderror = np.sqrt(variance / shots)

    return expval, stderror
