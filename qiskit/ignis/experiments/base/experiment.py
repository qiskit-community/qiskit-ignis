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
Experiment class.
"""

from typing import Optional
import uuid

from qiskit import transpile, assemble
from qiskit.providers import BaseJob
from qiskit.exceptions import QiskitError
from qiskit.providers import BaseBackend

from .generator import Generator
from .analysis import Analysis


class Experiment:
    """Basic experiment class."""

    # pylint: disable=arguments-differ
    def __init__(self,
                 generator: Optional[Generator] = None,
                 analysis: Optional[Analysis] = None,
                 job: Optional[BaseJob] = None):
        """Initialize an experiment."""

        self._generator = generator
        self._analysis = analysis
        self._job = job

    def run(self, backend: BaseBackend, **kwargs) -> any:
        """Run an experiment and perform analysis"""
        self.execute(backend, **kwargs)
        return self.run_analysis()

    def execute(self, backend: BaseBackend, **kwargs) -> BaseJob:
        """Execute the experiment on a backend.
​
        TODO: Add transpiler options

        Args:
            backend: backend to run experiment on.
            kwargs: kwargs for assemble method.
​
        Returns:
            BaseJob: the experiment job.
        """
        # Get circuits and metadata
        exp_id = str(uuid.uuid4())
        circuits = transpile(self.generator.circuits(),
                             backend=backend,
                             initial_layout=self.generator.qubits)
        metadata = self.generator.metadata()

        for meta in metadata:
            meta['name'] = self.generator.name
            meta['exp_id'] = exp_id
            meta['qubits'] = self.generator.qubits

        # Assemble qobj and submit to backend
        qobj = assemble(circuits,
                        backend=backend,
                        qobj_header={'metadata': metadata},
                        **kwargs)
        self._job = backend.run(qobj)
        return self

    @property
    def job(self):
        """Return experiment job"""
        return self._job

    @job.setter
    def job(self, value):
        """Set experiment job"""
        if not isinstance(value, BaseJob):
            raise QiskitError("Invalid Job object.")
        self._job = value

    @property
    def generator(self):
        """Return experiment generator object."""
        if self._generator is None:
            raise QiskitError("No Generator object provided for Experiment.")
        return self._generator

    @generator.setter
    def generator(self, value):
        """Set experiment generator object."""
        if not isinstance(value, Generator):
            raise QiskitError("Invalid Generator object")
        self._generator = value

    @property
    def analysis(self):
        """Return experiment Analysis object."""
        if self._analysis is None:
            raise QiskitError("No Analysis object provided for Experiment.")
        return self._analysis

    @analysis.setter
    def analysis(self, value):
        """Set experiment Analysis object."""
        if not isinstance(value, Analysis):
            raise QiskitError("Invalid Analysis object")
        self._analysis = value

    def run_analysis(self, **params):
        """Analyze the stored data.

        Returns:
            any: the output of the analysis,
        """
        self.analysis.clear_data()
        self.analysis.add_data(self._job)
        result = self.analysis.run(**params)
        return result

    # Generator Methods

    # @property
    # def name(self) -> str:
    #     """Return experiment name"""
    #     return self.generator.name

    # @property
    # def num_qubits(self) -> int:
    #     """Return the number of qubits for this experiment."""
    #     return self.generator.num_qubits

    # @property
    # def qubits(self) -> List[int]:
    #     """Return the qubits for this experiment."""
    #     return self.generator.qubits

    # @qubits.setter
    # def qubits(self, value):
    #     """Set the qubits for this experiment."""
    #     self.generator.qubits = value

    # def circuits(self) -> List[QuantumCircuit]:
    #     """Return a list of experiment circuits."""
    #     return self.generator.circuits()

    # def metadata(self) -> List[QuantumCircuit]:
    #     """Return a list of experiment metadata."""
    #     return self.generator.metadata()

    # Analysis Methods

    # def plot(self, *args, **kwargs) -> Union[None, any, List[any]]:
    #     """Generate a plot of analysis result.

    #     Args:
    #         args: Optional plot arguments
    #         kwargs: Optional plot kwargs.

    #     Additional Information:
    #         This is a base class method that should be overridden by any
    #         experiment Analysis subclasses that generate plots.
    #     """
    #     return self.analysis.plot(*args, **kwargs)

    # @property
    # def result(self) -> any:
    #     """Return the analysis result"""
    #     return self.analysis.result
