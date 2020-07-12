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
Randomized Benchmarking module
"""

# Randomized Benchmarking functions
from .circuits import randomized_benchmarking_seq
from .dihedral import (CNOTDihedral, decompose_cnotdihedral, random_cnotdihedral)
from .fitters import (RBFitter, InterleavedRBFitter, PurityRBFitter,
                      CNOTDihedralRBFitter)
from .rb_utils import (count_gates, gates_per_clifford,
                       coherence_limit, twoQ_clifford_error,
                       calculate_1q_epg, calculate_2q_epg, calculate_1q_epc, calculate_2q_epc)
from .rb_groups import RBgroup
