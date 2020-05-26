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

Randomization benchmarking (RB) is a well-known technique to measure average gate performance
by running sequences of random Clifford gates that should return the qubits to the initial state.
Qiskit Ignis has tools to generate one- and two-qubit gate Clifford RB sequences simultaneously,
as well as performing interleaved RB, purity RB and RB on the non-Clifford CNOT-Dihedral group.

.. autosummary::
   :toctree: ../stubs/

   randomized_benchmarking_seq
   RBFitter
   InterleavedRBFitter
   PurityRBFitter
   CNOTDihedralRBFitter
   CNOTDihedral
   count_gates
   gates_per_clifford
   calculate_1q_epg
   calculate_2q_epg
   calculate_1q_epc
   calculate_2q_epc
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

   AccreditationCircuits
   AccreditationFitter
   QOTP
   QOTPCorrectCounts

"""
from .quantum_volume import qv_circuits, QVFitter
from .randomized_benchmarking import (CNOTDihedral,
                                      randomized_benchmarking_seq,
                                      RBFitter, InterleavedRBFitter,
                                      PurityRBFitter, CNOTDihedralRBFitter,
                                      count_gates, gates_per_clifford,
                                      coherence_limit, twoQ_clifford_error,
                                      calculate_1q_epg, calculate_2q_epg,
                                      calculate_1q_epc, calculate_2q_epc)
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
from .accreditation import AccreditationCircuits, AccreditationFitter, QOTP, QOTPCorrectCounts
