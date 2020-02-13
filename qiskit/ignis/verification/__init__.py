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
===============================================
Verification (:mod:`qiskit.ignis.verification`)
===============================================

.. currentmodule:: qiskit.ignis.verification

Quantum Volume
==============

.. autosummary::
   :toctree: ../stubs/

   qv_circuits
   QVFitter


Randomized Benchmarking
=======================

.. autosummary::
   :toctree: ../stubs/

   randomized_benchmarking_seq
   RBFitter
   InterleavedRBFitter
   PurityRBFitter
   CNOTDihedralRBFitter
   BasicUtils
   Clifford
   CliffordUtils
   CNOTDihedral
   DihedralUtils
   count_gates
   gates_per_clifford
   coherence_limit
   twoQ_clifford_error


Tomography
==========

.. autosummary::
   :toctree: ../stubs/

   state_tomography_circuits
   process_tomography_circuits
   basis
   StateTomographyFitter
   ProcessTomographyFitter
   TomographyFitter
   marginal_counts
   combine_counts
   expectation_counts
   count_keys


Topological Codes
=================

.. autosummary::
   :toctree: ../stubs/

   RepetitionCode
   GraphDecoder
   lookuptable_decoding
   postselection_decoding


Accreditation
=============

.. autosummary::
   :toctree: ../stubs/

   accreditation_circuits
   accreditationFitter

"""
from .quantum_volume import qv_circuits, QVFitter
from .randomized_benchmarking import (Clifford, BasicUtils, CliffordUtils,
                                      CNOTDihedral, DihedralUtils,
                                      randomized_benchmarking_seq,
                                      RBFitter, InterleavedRBFitter,
                                      PurityRBFitter, CNOTDihedralRBFitter,
                                      count_gates, gates_per_clifford,
                                      coherence_limit, twoQ_clifford_error)
from .topological_codes import (RepetitionCode, GraphDecoder,
                                lookuptable_decoding,
                                postselection_decoding)
from .tomography import (state_tomography_circuits,
                         process_tomography_circuits, basis,
                         StateTomographyFitter,
                         ProcessTomographyFitter,
                         TomographyFitter,
                         marginal_counts, combine_counts,
                         expectation_counts, count_keys)
from .accreditation import accreditation_circuits, accreditationFitter
