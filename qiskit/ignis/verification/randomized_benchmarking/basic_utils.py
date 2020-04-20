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

"""
A basic utils class for different groups for randomized benchmarking.
"""

from abc import ABC, abstractmethod


class BasicUtils(ABC):
    """
        Abstract base class (ABS) for utils for
        various groups and sets of gates for
        randomized benchmarking.
    """

    @abstractmethod
    def num_qubits(self):
        """Return the number of qubits."""
        return

    @abstractmethod
    def group_tables(self):
        """Return the group tables."""
        return

    @abstractmethod
    def elmnt(self):
        """Return a group element."""
        return

    @abstractmethod
    def elmnt_key(self):
        """Return a key of a group element in the table."""
        return

    @abstractmethod
    def gatelist(self):
        """Return a list of gates corresponding to a group element."""
        return

    @abstractmethod
    def load_tables(self):
        """Load pickled group tables,
        or generate them if they do not exist."""
        return

    @abstractmethod
    def compose_gates(self):
        """Compose group elements."""
        return

    @abstractmethod
    def random_gates(self):
        """Pick a random group element."""
        return

    @abstractmethod
    def find_inverse_gates(self):
        """Compute an inverse of a group element."""
        return

    @abstractmethod
    def find_key(self):
        """Return a key to the group element."""
        return
