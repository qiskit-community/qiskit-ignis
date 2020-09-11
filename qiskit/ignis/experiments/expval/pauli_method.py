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
Pauli Measurement Expectation Value Experiment.
"""

from typing import Optional, Dict, Union, List

import numpy as np

from qiskit import QuantumCircuit
from qiskit.result import Counts
from qiskit.quantum_info import SparsePauliOp, Operator

from qiskit.ignis.experiments.base import ConstantGenerator
from .expval_utils import _expval_with_variance, pauli_diagonal


class PauliExpvalGenerator(ConstantGenerator):
    """Pauli basis measurement expectation value Generator"""

    def __init__(self,
                 observable: Union[SparsePauliOp, Operator],
                 qubits: Optional[List[int]] = None):
        """Initialize generator"""
        if isinstance(observable, SparsePauliOp):
            self._op = observable
        else:
            self._op = SparsePauliOp.from_operator(observable)

        circuits = []
        metadata = []
        for pauli, coeff in self._op.label_iter():
            # Create measurement circuit
            circuit = QuantumCircuit(len(pauli))
            for i, val in enumerate(reversed(pauli)):
                if val == 'Y':
                    circuit.sdg(i)
                if val in ['Y', 'X']:
                    circuit.h(i)
            circuit.measure_all()
            circuits.append(circuit)
            # Create metadata dict
            metadata.append({'method': 'Pauli',
                             'basis': pauli,
                             'coeff': coeff})
        super().__init__('expval', circuits, metadata, qubits=qubits)


def pauli_analysis_fn(data: List[Counts],
                      metadata: List[Dict[str, any]],
                      mitigator: Optional = None):
    """Fit expectation value from weighted sum of Pauli operators."""
    combined_expval = 0.0
    combined_variance = 0.0

    for dat, meta in zip(data, metadata):
        basis = meta.get('basis', None)
        diagonal = pauli_diagonal(basis) if basis is not None else None
        coeff = _format_coeff(meta.get('coeff', 1))
        qubits = meta.get('qubits', None)

        # Compute expval component
        expval, var = _expval_with_variance(dat,
                                            diagonal=diagonal,
                                            mitigator=mitigator,
                                            mitigator_qubits=qubits)
        # Accumulate
        combined_expval += expval * coeff
        combined_variance += var * abs(coeff) ** 2
    combined_stddev = np.sqrt(combined_variance)
    return combined_expval, combined_stddev


def _format_coeff(coeff: Union[List, float, complex]) -> Union[float, complex]:
    """Format coefficient from Result header."""
    # Result objects convert complex coefficients to a list [real, imag]
    if isinstance(coeff, list):
        if len(coeff) == 1 or np.isclose(coeff[1], 0):
            return coeff[0]
        return coeff[0] + 1j * coeff[1]
    return coeff
