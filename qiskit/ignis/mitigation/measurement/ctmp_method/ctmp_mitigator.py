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
CTMP expectation value measurement error mitigator.
"""
from typing import Dict, Optional, List

from ..base_meas_mitigator import BaseMeasureErrorMitigator
from .calibration import (
    MeasurementCalibrator,
    BaseGeneratorSet,
    BaseCalibrationCircuitSet
)
from .ctmp import mitigated_expectation_value


class CTMPMeasurementErrorMitigator(BaseMeasureErrorMitigator):

    def __init__(self, calibrator: MeasurementCalibrator):
        """Create the mitigator from a given `MeasurementCalibrator` object.

        Args:
            calibrator (MeasurementCalibrator): The input calibrator to use.
        """
        self.cal = calibrator

    @classmethod
    def from_method(cls, num_qubits: int, method: str = 'weight_2'):
        """Construct a mitigator from a given, named calibration circuit set.

        Args:
            num_qubits (int): Number of qubits to calibrate.
            method (str): One of the methods to use for calibration. Currently `'weight_1'` and
                `'weight_2'` are supported.

        Returns:
            ExpectationValueMeasurementErrorMitigator: The resulting mitigator.
        """
        cal = MeasurementCalibrator.standard_construction(num_qubits=num_qubits, method=method)
        res = cls(cal)
        return res

    @classmethod
    def from_cal_sets(
            cls,
            circ_set: BaseCalibrationCircuitSet,
            gen_set: BaseGeneratorSet
    ):
        """Generate a mitigator from a set of circuits and a set of generators.

        Args:
            circ_set (BaseCalibrationCircuitSet): Circuit set to use for calibration.
            gen_set (BaseGeneratorSet): Generator set to use for calibration.

        Returns:
            ExpectationValueMeasurementErrorMitigator: The resulting mitigator.
        """
        if circ_set.num_qubits != gen_set.num_qubits:
            raise ValueError('Circuit set and generator set must act on same number of qubits.')
        cal = MeasurementCalibrator(
            cal_circ_set=circ_set,
            gen_set=gen_set
        )
        res = cls(cal)
        return res

    def expectation_value(self, counts: Dict, clbits: Optional[List[int]] = None,
                          qubits: Optional[List[int]] = None) -> float:
        """Compute mitigated expectation value using CTMP expectation value readout error
        mitigation.

        The ``qubits`` kwarg is used so that count bitstrings correpond to
        measurements of the form ``circuit.measure(qubits, range(num_qubits))``.

        Args:
            counts: counts object
            clbits: Optional, marginalize counts to just these bits.
            qubits: qubits the count bitstrings correspond to.

        Returns:
            float: expval.
        """
        if not self.cal.calibrated:
            raise ValueError('Calibrator has not been calibrated yet.')
        mean = mitigated_expectation_value(cal=self.cal, counts_dict=counts, subset=clbits)
        return mean
