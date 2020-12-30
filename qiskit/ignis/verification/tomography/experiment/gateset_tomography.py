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

"""
Gate set tomography experiment classes
"""

from ast import literal_eval
from typing import List, Union, Dict, Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.gate import Gate
from qiskit.ignis.experiments.base import (Generator, Analysis)
from qiskit import QiskitError
from qiskit.providers import BaseJob
from qiskit.result import Result, Counts
from .experiment import TomographyExperiment
from .gateset_utils import (linear_inversion, _ideal_gateset, GaugeOptimize, GST_Optimize)
from ..basis import GateSetBasis
from ..basis import gateset_tomography_circuits
from ..basis import default_gateset_basis
from ..data import marginal_counts


class GatesetTomographyExperiment(TomographyExperiment):
    """
    Gate set tomography experiment class
    """
    # pylint: disable=arguments-differ
    def __init__(self,
                 gate: Gate,
                 gateset_basis: Optional[GateSetBasis] = None,
                 meas_qubits: Optional[Union[int, List[int]]] = None):

        if meas_qubits is None:
            meas_qubits = list(range(gate.num_qubits))

        self._gateset_basis = gateset_basis \
            if gateset_basis is not None \
            else default_gateset_basis()

        self._gateset_basis.add_gate(gate)

        analysis = GatesetTomographyAnalysis(self._gateset_basis)
        generator = GatesetTomographyGenerator(self._gateset_basis, meas_qubits)

        super().__init__(generator=generator, analysis=analysis)

    def basis(self):
        """
        Returns the gateset basis
        """
        return self._gateset_basis


class GatesetTomographyGenerator(Generator):
    """
    Gate set tomography experiment generator class
    """
    def __init__(self,
                 basis: GateSetBasis,
                 meas_qubits: Optional[Union[int, List[int]]] = None
                 ):
        if meas_qubits is None:
            meas_qubits = [0]

        if len(meas_qubits) > 1:
            raise QiskitError("Only 1-qubit gate set tomography "
                              "is currently supported")
        num_qubits = 1 + max(meas_qubits)

        self._meas_qubits = meas_qubits
        super().__init__("gate set tomography", num_qubits)

        self._circuits = gateset_tomography_circuits(self._meas_qubits, basis)

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of experiment circuits."""
        return self._circuits

    def _extra_metadata(self):
        return [{
            'circuit_name': circ.name,
            'meas_qubits': self._meas_qubits,
        }
                for circ in self._circuits]


class GatesetTomographyAnalysis(Analysis):
    """
    Gate set tomography experiment analysis class
    """
    def __init__(self,
                 gateset_basis: GateSetBasis,
                 data: Optional[Union[BaseJob, Result, List[any], any]] = None,
                 metadata: Optional[Union[List[Dict[str, any]], Dict[str, any]]] = None,
                 name: Optional[str] = None,
                 exp_id: Optional[str] = None):
        super().__init__(data, metadata, name, exp_id)
        self._analysis_fn = self.fit
        self._gateset_basis = gateset_basis

    def _format_data(self, data: Result,
                     metadata: Dict[str, any],
                     index: int) -> Counts:
        meas_qubits = metadata['meas_qubits']
        counts = data.get_counts(index)
        counts = marginal_counts(counts, meas_qubits)
        counts_num = sum(counts.values())
        prob = counts.get('0', 0) / counts_num
        return prob

    def fit(self, data, metadata) -> np.array:
        """
        Reconstruct a gate set from measurement data using optimization.

        Returns:
           For each gate in the gateset: its approximation found using the
           optimization process.

        Additional Information:
            The gateset optimization process con/.sists of three phases:
            1) Use linear inversion to obtain an initial approximation.
            2) Use gauge optimization to ensure the linear inversion results
            are close enough to the expected optimization outcome to serve
            as a suitable starting point
            3) Use MLE optimization to obtain the final outcome
        """
        circuit_names = [literal_eval(m['circuit_name']) for m in metadata]
        probs = dict(zip(circuit_names, data))
        linear_inversion_results = linear_inversion(self._gateset_basis, probs)
        n = len(self._gateset_basis.spam_labels)
        gauge_opt = GaugeOptimize(_ideal_gateset(n, self._gateset_basis),
                                  linear_inversion_results,
                                  self._gateset_basis)
        past_gauge_gateset = gauge_opt.optimize()
        optimizer = GST_Optimize(self._gateset_basis.gate_labels,
                                 self._gateset_basis.spam_labels,
                                 self._gateset_basis.spam_spec,
                                 probs)
        optimizer.set_initial_value(past_gauge_gateset)
        optimization_results = optimizer.optimize()
        return optimization_results
