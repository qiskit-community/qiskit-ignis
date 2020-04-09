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
Discriminators (:mod:`qiskit.ignis.measurement.discriminator`)
===========================================

.. currentmodule:: qiskit.ignis.measurement.discriminator

Discriminator
===========

The discriminators are used to to discriminate level one data into level two counts.

.. autosummary::
   :toctree: ../stubs/

   DiscriminationFilter
   IQDiscriminationFitter
   LinearIQDiscriminator
   QuadraticIQDiscriminator
   SklearnIQDiscriminator

"""
from .filters import DiscriminationFilter
from .iq_discriminators import (IQDiscriminationFitter, LinearIQDiscriminator,
                                QuadraticIQDiscriminator, SklearnIQDiscriminator)
