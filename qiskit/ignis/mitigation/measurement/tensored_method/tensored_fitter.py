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
Tensor-product matrix measurement error mitigation generator.
"""
from typing import List, Dict
import numpy as np

from qiskit.result import Result
from qiskit.exceptions import QiskitError
from qiskit.ignis.verification.tomography import marginal_counts, combine_counts
from ..meas_mit_utils import counts_probability_vector, filter_calibration_data
from .tensored_mitigator import TensoredMeasMitigator


# NOTE: This is a temporary fitter function and should be replaced with a
# proper fitter class.

def fit_tensored_meas_mitigator(
        result: Result,
        metadata: List[Dict[str, any]]) -> TensoredMeasMitigator:
    """Return TensorMeasureErrorMitigator from result data.

    Args:
        result: Qiskit result object.
        metadata: mitigation generator metadata.

    Raises:
        QiskitError: if input Result object and metadata are not valid.

    Returns:
        Measurement error mitigator object.
    """
    # Filter mitigation calibration data
    cal_data, num_qubits = filter_calibration_data(result, metadata)

    # Construct single-qubit A-matrices from calibration data
    amats = []
    for qubit in range(num_qubits):
        counts0 = {}
        counts1 = {}
        # Marginalize counts
        for label, counts in cal_data.items():
            m_counts = marginal_counts(counts, meas_qubits=[qubit])
            m_label = label[-1 - qubit]
            if m_label == '0':
                counts0 = combine_counts(counts0, m_counts)
            elif m_label == '1':
                counts1 = combine_counts(counts1, m_counts)
            else:
                raise QiskitError('Invalid calibration label')
        amat = np.array([counts_probability_vector(counts0),
                         counts_probability_vector(counts1)]).T
        amats.append(amat)

    return TensoredMeasMitigator(amats)
