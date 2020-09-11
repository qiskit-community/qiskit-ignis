# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Measurement error mitigation experiment.
"""

import logging
from typing import Optional, List, Dict, Union

from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.result import Counts

from qiskit.ignis.verification.tomography import combine_counts
from qiskit.ignis.experiments.base import Experiment, Analysis, ConstantGenerator

# TODO: Move mitigator class somewhere else
from qiskit.ignis.experiments.mitigator.meas_mitigator import (
    CompleteMeasMitigator, TensoredMeasMitigator)
from qiskit.ignis.mitigation.expval.utils import assignment_matrix
from qiskit.ignis.mitigation.expval.ctmp_fitter import fit_ctmp_meas_mitigator
from qiskit.ignis.mitigation.expval.ctmp_generator_set import Generator

logger = logging.getLogger(__name__)


class MeasMitigatorExperiment(Experiment):
    """Measurement error mitigator calibration experiment."""

    # pylint: disable=arguments-differ
    def __init__(self,
                 qubits: Union[int, List[int]] = None,
                 labels: Optional[Union[str, List[str]]] = None,
                 method: str = 'CTMP',
                 job: Optional = None):
        """Initialize measurement error mitigator calibration experiment."""
        analysis = MeasMitigatorAnalysis()
        if qubits is not None or labels is not None:
            generator = MeasMitigatorGenerator(qubits, method=method, labels=labels)
        else:
            generator = None
        super().__init__(generator=generator, analysis=analysis, job=job)


class MeasMitigatorGenerator(ConstantGenerator):
    """Measurement error mitigator calibration experiment."""

    # pylint: disable=arguments-differ
    def __init__(self,
                 qubits: Union[int, List[int]] = None,
                 labels: Optional[Union[str, List[str]]] = None,
                 method: str = 'CTMP'):
        """Initialize measurement error mitigator calibration experiment."""
        # Get number of qubits
        if qubits is None:
            if labels is None:
                raise QiskitError("Either qubits or labels must be supplied.")
            else:
                qubits = len(labels[0])

        if isinstance(qubits, int):
            num_qubits = qubits
        else:
            num_qubits = len(qubits)

        # Circuits and metadata
        circuits = []
        metadata = []
        if labels is None:
            labels = self._method_labels(num_qubits, method)
        for label in labels:
            metadata.append({'method': method, 'cal': label})
            circuits.append(self._calibration_circuit(num_qubits, label))
        super().__init__('meas_mit', circuits, metadata, qubits)

    @staticmethod
    def _method_labels(num_qubits, method):
        """Generate labels for initilizing via a standard method."""

        if method == 'tensored':
            return [num_qubits * '0', num_qubits * '1']

        if method in ['CTMP', 'ctmp']:
            labels = [num_qubits * '0', num_qubits * '1']
            for i in range(num_qubits):
                labels.append(((num_qubits - i - 1) * '0') + '1' +
                              (i * '0'))
            return labels

        if method == 'complete':
            labels = []
            for i in range(2**num_qubits):
                bits = bin(i)[2:]
                label = (num_qubits - len(bits)) * '0' + bits
                labels.append(label)
            return labels

        raise QiskitError("Unrecognized method {}".format(method))

    @staticmethod
    def _calibration_circuit(num_qubits: int, label: str) -> QuantumCircuit:
        """Return a calibration circuit.

        This is an N-qubit circuit where N is the length of the label.
        The circuit consists of X-gates on qubits with label bits equal to 1,
        and measurements of all qubits.
        """
        circ = QuantumCircuit(num_qubits, name='meas_mit_cal_' + label)
        for i, val in enumerate(reversed(label)):
            if val == '1':
                circ.x(i)
        circ.measure_all()
        return circ


class MeasMitigatorAnalysis(Analysis):
    """Measurement error mitigator calibration experiment result."""

    # pylint: disable=arguments-differ
    def __init__(self,
                 data: Optional[any] = None,
                 metadata: Optional[Dict[str, any]] = None,
                 method: str = None,
                 exp_id: Optional[str] = None):
        """Initialize measurement error mitigator calibration experiment."""
        # Base Experiment Result class
        super().__init__(data=data,
                         metadata=metadata,
                         name='meas_mit',
                         exp_id=exp_id)

        self._method = method

        # Intermediate representation of results
        self._cal_data = {}
        self._num_qubits = None

    def run(self,
            method: Optional = None,
            generators: Optional = None):
        """Analyze the stored expectation value data.

        Returns:
            any: the output of the analysis.

        Raises:
            QiskitError: if method is not valid.
        """

        cal_data, num_qubits = self._calibration_data(
            self._exp_data, self._exp_metadata)

        if method is None:
            if self._method is None:
                method = self._exp_metadata[0].get('method')
            else:
                method = self._method

        if method == 'complete':
            # Construct A-matrix from calibration data
            amat = assignment_matrix(cal_data, num_qubits)
            self._result = CompleteMeasMitigator(amat)

        elif method == 'tensored':
            # Construct single-qubit A-matrices from calibration data
            amats = []
            for qubit in range(num_qubits):
                amat = assignment_matrix(cal_data, num_qubits, [qubit])
                amats.append(amat)
            self._result = TensoredMeasMitigator(amats)

        elif method == 'CTMP' or method == 'ctmp':
            self._result = fit_ctmp_meas_mitigator(cal_data, num_qubits, generators)

        else:
            raise QiskitError("Invalid analysis method {}".format(method))

        return self._result

    @staticmethod
    def _calibration_data(data: Counts, metadata: Dict[str, any]):
        """Process counts into calibration data"""
        cal_data = {}
        num_qubits = None
        for i, meta in enumerate(metadata):
            if num_qubits is None:
                num_qubits = len(meta['cal'])
            key = int(meta['cal'], 2)
            counts = data[i].int_outcomes()
            if key not in cal_data:
                cal_data[key] = counts
            else:
                cal_data[key] = combine_counts(cal_data[key], counts)
        return cal_data, num_qubits
