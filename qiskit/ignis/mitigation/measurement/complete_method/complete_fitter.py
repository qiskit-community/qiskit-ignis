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
Full-matrix measurement error mitigation generator.
"""
from typing import List, Dict
import numpy as np

from qiskit.result import Result
from ..meas_mit_utils import counts_probability_vector, filter_calibration_data
from .complete_mitigator import CompleteMeasMitigator

# NOTE: This is a temporary fitter function and should be replaced with a
# proper fitter class.


def fit_complete_meas_mitigator(
        result: Result,
        metadata: List[Dict[str, any]]) -> CompleteMeasMitigator:
    """Return FullMeasureErrorMitigator from result data.

    Args:
        result: Qiskit result object.
        metadata: mitigation generator metadata.

    Returns:
        Measurement error mitigator object.
    """
    # Filter mitigation calibration data
    cal_data, num_qubits = filter_calibration_data(result, metadata)

    # Construct A-matrix from calibration data
    amat = np.zeros(2 * [2 ** num_qubits], dtype=float)
    for key, counts in cal_data.items():
        vec = counts_probability_vector(counts)
        index = int(key, 2)
        amat[:, index] = vec

    return CompleteMeasMitigator(amat)
