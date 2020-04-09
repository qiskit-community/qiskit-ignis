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

"""
===========================================
Mitigation (:mod:`qiskit.ignis.mitigation`)
===========================================

.. currentmodule:: qiskit.ignis.mitigation

Measurement
===========

The measurement calibration is used to mitigate measurement errors.
The main idea is to prepare all :math:`2^n` basis input states and compute
the probability of measuring counts in the other basis states.
From these calibrations, it is possible to correct the average results
of another experiment of interest.

.. autosummary::
   :toctree: ../stubs/

   complete_meas_cal
   tensored_meas_cal
   MeasurementFilter
   TensoredFilter
   CompleteMeasFitter
   TensoredMeasFitter

"""
from .measurement import (complete_meas_cal, tensored_meas_cal,
                          MeasurementFilter, TensoredFilter,
                          CompleteMeasFitter, TensoredMeasFitter)
