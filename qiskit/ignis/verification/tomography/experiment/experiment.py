# -*- coding: utf-8 -*-
#
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

# pylint: disable=no-name-in-module,import-error

"""
Quantum tomography experiment class
"""


from typing import List, Optional
from qiskit.ignis.experiments.base import Experiment, Analysis, Generator
from qiskit.providers import BaseJob


class TomographyExperiment(Experiment):
    """
    Tomography experiment base class
    """
    def __init__(self,
                 generator: Optional[Generator] = None,
                 analysis: Optional[Analysis] = None,
                 job: Optional[BaseJob] = None):
        super().__init__(generator=generator, analysis=analysis, job=job)

    def set_target_qubits(self, qubits: List[int]):
        """
        Sets the qubits on which tomography will be performed
        """
        self.analysis.set_target_qubits(qubits)
        return self
