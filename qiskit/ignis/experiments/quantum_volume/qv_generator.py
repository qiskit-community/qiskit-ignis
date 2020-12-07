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
Quantum volume Experiment Generator.
"""
from typing import Optional, Union, List
from numpy import random
from qiskit.circuit.library import QuantumVolume
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.ignis.experiments.base import Generator


class QuantumVolumeGenerator(Generator):
    """Quantum volume experiment generator."""

    # pylint: disable=arguments-differ
    def __init__(self, qubits: Union[int, List[int]],
                 seed: Optional[Union[int, Generator]] = None):
        self._seed = seed
        self._trial_number = 0
        if isinstance(qubits, int):
            qubits = [range(qubits)]
        # for circuit generation, the pysical qubits numbers are not important
        qubits = [len(qubit) for qubit in qubits]
        super().__init__('qv_depth_%d_to_depth_%d' % (qubits[0], qubits[-1]),
                         qubits)

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of quantum volume circuits (circuit for each depth).
        the circuits returned by this method do not contain measurements"""
        qv_circuits = []

        # Initialize RNG
        if isinstance(self._seed, int):
            rng = random.default_rng(self._seed)
        else:
            _seed = self._seed

        for depth in self.qubits:
            if isinstance(self._seed, int):
                _seed = rng.integers(1000)
            qv_circ = QuantumVolume(depth, depth, seed=_seed)
            qv_circ.name = 'qv_depth_%d_trial_%d' % (depth, self._trial_number)
            qv_circuits.append(qv_circ)
        self._trial_number += 1
        return qv_circuits

    def _extra_metadata(self) -> List[dict]:
        """Generate a list of experiment circuits metadata."""
        return [{"depth": qubit,
                 "trail_number": self._trial_number,
                 "seed": self._seed,
                 "circ_name": 'qv_depth_%d_trial_%d' % (qubit, self._trial_number)}
                for qubit in self.qubits]
