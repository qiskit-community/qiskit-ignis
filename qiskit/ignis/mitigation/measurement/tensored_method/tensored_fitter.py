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

from qiskit.result import Result
from ..meas_mit_utils import calibration_data, assignment_matrix
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
    cal_data, num_qubits = calibration_data(result, metadata)

    # Construct single-qubit A-matrices from calibration data
    amats = []
    for qubit in range(num_qubits):
        amat = assignment_matrix(cal_data, num_qubits, [qubit])
        amats.append(amat)

    return TensoredMeasMitigator(amats)
