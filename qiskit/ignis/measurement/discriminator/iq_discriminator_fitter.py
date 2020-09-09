# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""
IQ Discriminator module to discriminate date in the IQ Plane.
"""
import numpy as np
from typing import List, Union

from fitters.lda_discriminator_fitter import LDADiscriminator
from fitters.qda_discriminator_fitter import QDADiscriminator
from fitters.pca_discriminator_fitter import PCADiscriminator

from qiskit.exceptions import QiskitError
from qiskit.pulse import PulseError
from qiskit.result import Result
from qiskit.pulse.schedule import Schedule
try:
    from matplotlib import pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class IQDiscriminationFitter:
    """
    Abstract discriminator that implements the data formatting for IQ
    level 1 data.
    """

    def __init__(self, cal_results, qubit_mask, metadata=None):
        """
        Args:
            cal_results (Union[Result, List[Result]]): calibration results,
                Result or list of Result used to fit the discriminator.
            qubit_mask (List[int]): determines which qubit's level 1 data to
                use in the discrimination process.
            expected_states (List[str]): a list that should have the same
                length as schedules. All results in cal_results are used if
                schedules is None. expected_states must have the corresponding
                length.
            standardize (bool): if true the discriminator will standardize the
                xdata using the internal method _scale_data.
            schedules (Union[List[str], List[Schedule]]): The schedules or a
                subset of schedules in cal_results used to train the
                discriminator. The user may also pass the name of the schedules
                instead of the schedules. If schedules is None, then all the
                schedules in cal_results are used.
        """

        self._discriminator = None
        self._cal_results = cal_results
        self._qubit_mask = qubit_mask

    @property
    def discriminator(self):
        """Return the fitted mitigator object"""
        if self._discriminator is None:
            raise QiskitError("Discriminator has not been fitted. Run `fit` first.")
        return self._discriminator

    def fit(self, method):
        """Fits the discriminator using self._xdata and self._ydata."""
        if method == "LDA":
            self._discriminator = LDADiscriminator(self._cal_results, self._qubit_mask)
        elif method == "QDA":
            self._discriminator = QDADiscriminator(self._cal_results, self._qubit_mask)
        elif method == "PCA":
            self._discriminator = PCADiscriminator(self._cal_results, self._qubit_mask)
        else:
            raise ValueError("Invalid method specified.")
        return self._discriminator
