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


"""
=======================================================
Characterization (:mod:`qiskit.ignis.characterization`)
=======================================================

.. currentmodule:: qiskit.ignis.characterization

Calibrations
============

.. autosummary::
   :toctree: ../stubs/

   rabi_schedules
   drag_schedules
   RabiFitter
   DragFitter
   get_single_q_pulse
   update_u_gates


Coherence
=========

.. autosummary::
   :toctree: ../stubs/

   t1_circuits
   t2_circuits
   t2star_circuits
   T1Fitter
   T2Fitter
   T2StarFitter


Gates
=====

.. autosummary::
   :toctree: ../stubs/

   ampcal_1Q_circuits
   anglecal_1Q_circuits
   ampcal_cx_circuits
   anglecal_cx_circuits
   AmpCalFitter
   AngleCalFitter
   AmpCalCXFitter
   AngleCalCXFitter


Hamiltonian
===========

.. autosummary::
   :toctree: ../stubs/

   zz_circuits
   ZZFitter


Base Fitters
============

.. autosummary::
   :toctree: ../stubs/

   BaseCoherenceFitter
   BaseGateFitter

"""

from .fitters import BaseCoherenceFitter, BaseGateFitter
from .calibrations import (rabi_schedules, drag_schedules,
                           RabiFitter, DragFitter,
                           get_single_q_pulse, update_u_gates)
from .coherence import (t1_circuits, t2_circuits,
                        t2star_circuits,
                        T1Fitter, T2Fitter, T2StarFitter)
from .gates import (ampcal_1Q_circuits, anglecal_1Q_circuits,
                    ampcal_cx_circuits, anglecal_cx_circuits,
                    AmpCalFitter, AngleCalFitter,
                    AmpCalCXFitter, AngleCalCXFitter)
from .hamiltonian import (zz_circuits, ZZFitter)
