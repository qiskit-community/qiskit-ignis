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
Base Experiment Generator class.
"""

from abc import ABC, abstractmethod
from typing import Union, List, Dict
from numpy import integer

from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError


class Generator(ABC):
    """Base generator class."""

    def __init__(self, name: str, qubits: Union[int, List[int]]):
        """Initialize an experiment.

        Args:
            name: experiment name
            qubits: the number of qubits, or list of physical qubits for the experiment.
        """
        # Circuit generation parameters
        self._name = str(name)
        if isinstance(qubits, (int, integer)):
            self._num_qubits = int(qubits)
            self._qubits = list(range(self._num_qubits))
        else:
            self._qubits = list(qubits)
            self._num_qubits = len(self._qubits)

    @property
    def name(self) -> str:
        """Return experiment name"""
        return self._name

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits for this experiment."""
        return self._num_qubits

    @property
    def qubits(self) -> List[int]:
        """Return the qubits for this experiment."""
        return self._qubits

    @qubits.setter
    def qubits(self, value):
        """Set the qubits for this experiment."""
        if len(value) != self._num_qubits:
            raise QiskitError("Length of qubits does not match Generator qubit number.")
        self._qubits = list(value)

    @abstractmethod
    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of experiment circuits."""

    def metadata(self) -> List[Dict[str, any]]:
        """Generate a list of experiment metadata dicts."""
        metadata = self._extra_metadata()
        for meta in metadata:
            meta['name'] = self.name
            meta['qubits'] = self.qubits
        return metadata

    @abstractmethod
    def _extra_metadata(self) -> List[Dict[str, any]]:
        """Generate a list of experiment metadata dicts."""
        # TODO: Add metadata to QuantumCircuit objects in Terra
