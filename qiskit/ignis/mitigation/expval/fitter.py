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
Expectation value measurement migitator fitter.
"""

from typing import Optional, Dict, List, Union

from qiskit.result import Result
from qiskit.exceptions import QiskitError

from .utils import calibration_data, assignment_matrix
from .complete_mitigator import CompleteExpvalMeasMitigator
from .tensored_mitigator import TensoredExpvalMeasMitigator
from .ctmp_mitigator import CTMPExpvalMeasMitigator
from .ctmp_fitter import fit_ctmp_meas_mitigator
from .ctmp_generator_set import Generator


class ExpvalMeasMitigatorFitter:
    """Expectation value measurement error mitigator calibration fitter.

    See :func:`qiskit.ignis.mitigation.expval_meas_mitigator_circuits` for
    additional documentation.
    """

    def __init__(self,
                 result: Result,
                 metadata: List[Dict[str, any]]):
        """Fit a measurement error mitigator object from experiment data.

        Args:
            result: Qiskit result object.
            metadata: mitigation generator metadata.
        """
        self._num_qubits = None
        self._cal_data = None
        self._mitigator = None
        self._cal_data, self._num_qubits, self._method = calibration_data(
            result, metadata)

    @property
    def mitigator(self):
        """Return the fitted mitigator object"""
        if self._mitigator is None:
            raise QiskitError("Mitigator has not been fitted. Run `fit` first.")
        return self._mitigator

    def fit(self, method: Optional[str] = None,
            generators: Optional[List[Generator]] = None) -> Union[
                CompleteExpvalMeasMitigator,
                TensoredExpvalMeasMitigator,
                CTMPExpvalMeasMitigator]:
        """Fit and return the Mitigator object from the calibration data."""

        if method is None:
            method = self._method

        if method == 'complete':
            # Construct A-matrix from calibration data
            amat = assignment_matrix(self._cal_data, self._num_qubits)
            self._mitigator = CompleteExpvalMeasMitigator(amat)

        elif method == 'tensored':
            # Construct single-qubit A-matrices from calibration data
            amats = []
            for qubit in range(self._num_qubits):
                amat = assignment_matrix(self._cal_data, self._num_qubits, [qubit])
                amats.append(amat)
            self._mitigator = TensoredExpvalMeasMitigator(amats)

        elif method in ['CTMP', 'ctmp']:
            self._mitigator = fit_ctmp_meas_mitigator(
                self._cal_data, self._num_qubits, generators)
        else:
            raise QiskitError(
                "Invalid expval measurement error mitigation method {}".format(method))
        return self._mitigator
