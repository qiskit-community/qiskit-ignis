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
Fixed circuit experiment Generator class.
"""

from typing import List, Dict, Optional

from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from .generator import Generator


class ConstantGenerator(Generator):
    """A generator class for a static list of circuits"""

    def __init__(self,
                 name: str,
                 circuits: List[QuantumCircuit],
                 metadata: Optional[List[Dict[str, any]]] = None,
                 qubits: Optional[List[int]] = None):

        # Format circuits
        if not isinstance(circuits, list):
            circuits = [circuits]
        if not circuits:
            raise QiskitError("Input circuit list is empty")
        self._circuits = circuits

        # Format metadata
        if metadata is None:
            metadata = len(circuits) * [{}]
        if not isinstance(metadata, list):
            metadata = [metadata]
        self._metadata = metadata

        # Get qubits
        if qubits is None:
            qubits = list(range(circuits[0].num_qubits))
        super().__init__(name, qubits)

        # Validation
        if len(self._circuits) != len(metadata):
            raise QiskitError("Input circuits and metadata lists are not the same length.")
        for circ in self._circuits:
            if circ.num_qubits != len(self.qubits):
                raise QiskitError("Input circuits have different numbers of qubits.")

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of experiment circuits."""
        return self._circuits

    def _extra_metadata(self) -> List[Dict[str, any]]:
        """Generate a list of experiment metadata dicts."""
        return self._metadata
