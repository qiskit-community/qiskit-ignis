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


"""Quantum State and Process Tomography module

==========================================================
Base Objects (:mod:`qiskit.ignis.verification.tomography`)
==========================================================

.. currentmodule:: qiskit.ignis.verification.tomography

Base Fitter
===========

.. autosummary::
    :toctree:

   TomographyFitter


Utility functions
=================

.. autosummary::

    marginal_counts
    combine_counts
    expectation_counts
    count_keys

==============================================================
State Tomography (:mod:`qiskit.ignis.verification.tomography`)
==============================================================

.. currentmodule:: qiskit.ignis.verification.tomography

Fitter
======
.. autosummary::

    StateTomographyFitter

Circuits
========
.. autosummary::

    state_tomography_circuits

================================================================
Process Tomography (:mod:`qiskit.ignis.verification.tomography`)
================================================================

.. currentmodule:: qiskit.ignis.verification.tomography

Fitter
======
.. autosummary::

    ProcessTomographyFitter

Circuits
========
.. autosummary::

    process_tomography_circuits
"""

# Tomography circuit generation
from .basis import state_tomography_circuits
from .basis import process_tomography_circuits
from .basis import gateset_tomography_circuits
from . import basis

# Tomography data formatting
from .fitters import StateTomographyFitter
from .fitters import ProcessTomographyFitter
from .fitters import GatesetTomographyFitter
from .fitters import TomographyFitter

# Utility functions TODO: move to qiskit.quantum_info
from .data import marginal_counts     # TODO: move to qiskit.tools
from .data import combine_counts      # TODO: move to qiskit.tools
from .data import expectation_counts  # TODO: move to qiskit.tools
from .data import count_keys  # TODO: move to qiskit.tools
