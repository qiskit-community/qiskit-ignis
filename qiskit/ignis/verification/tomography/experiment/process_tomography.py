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
Quantum process tomography experiment class
"""

from ast import literal_eval
from typing import List, Union, Tuple, Optional
from qiskit import QuantumCircuit
from .experiment import TomographyExperiment
from .generator import TomographyGenerator
from .analysis import TomographyAnalysis
from ..basis import process_tomography_circuits

class ProcessTomographyExperiment(TomographyExperiment):
    """
    Process tomography experiment class
    """
    # pylint: disable=arguments-differ
    def __init__(self,
                 circuit: QuantumCircuit,
                 qubits: Union[int, List[int]] = None,
                 meas_basis: str = 'Pauli',
                 meas_labels: Union[str, Tuple[str], List[Tuple[str]]] = 'Pauli',
                 prep_basis: str = 'Pauli',
                 prep_labels: Union[str, Tuple[str], List[Tuple[str]]] = 'Pauli',
                 method: str = 'auto',
                 job: Optional = None):

        analysis = TomographyAnalysis(method=method,
                                      meas_basis=meas_basis,
                                      prep_basis=prep_basis
                                      )
        generator = ProcessTomographyGenerator(circuit, qubits,
                                               meas_basis=meas_basis,
                                               meas_labels=meas_labels,
                                               prep_basis=prep_basis,
                                               prep_labels=prep_labels
                                               )

        super().__init__(generator=generator, analysis=analysis, job=job)


class ProcessTomographyGenerator(TomographyGenerator):
    """
    Process tomography experiment generator class
    """
    def __init__(self,
                 circuit: QuantumCircuit,
                 qubits: Union[int, List[int]] = None,
                 meas_basis: str = 'Pauli',
                 meas_labels: Union[str, Tuple[str], List[Tuple[str]]] = 'Pauli',
                 prep_basis: str = 'Pauli',
                 prep_labels: Union[str, Tuple[str], List[Tuple[str]]] = 'Pauli',
                 ):
        super().__init__("process tomography", circuit, qubits)
        self._prep_basis = prep_basis
        self._meas_basis = meas_basis
        self._circuits = process_tomography_circuits(circuit,
                                                     self._meas_qubits,
                                                     meas_basis=self._meas_basis,
                                                     meas_labels=meas_labels,
                                                     prep_basis=self._prep_basis,
                                                     prep_labels=prep_labels
                                                     )

    def _extra_metadata(self):
        metadata_list = super()._extra_metadata()
        for metadata in metadata_list:
            circuit_labels = literal_eval(metadata['circuit_name'])
            metadata['prep_label'] = circuit_labels[0]
            metadata['prep_basis'] = self._prep_basis
            metadata['meas_label'] = circuit_labels[1]
            metadata['meas_basis'] = self._meas_basis
        return metadata_list
