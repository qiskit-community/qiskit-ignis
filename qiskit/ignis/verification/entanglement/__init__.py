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
In this package, there are three modules, linear.py, parallellize.py,
and analysis.py. Once Terra completes design for circuit repository,
it may be more suitable to move it there. The most important module,
parallellize.py provides methods to parallellize CNOT gates in the
preparation of the GHZ State, which results to the GHZ State having a
much higher fidelity then a normal "linear" CNOT gate preparation of
the GHZ State. Additionally, there are methods within parallelize.py
that configure different characterization tests for the GHZ State,
including Multiple Quantum Coherences (MQC), Parity Oscillations (PO),
and Quantum Tomography.  Nevertheless, the module linear.py provides
the linear preparation analogous of parallelize.py. The module
analysis.py provides several miscellaneous tools for analysis of the
GHZ State (most notably, Fourier Analysis)
"""

# Entanglement functions
from .analysis import ordered_list_generator
from .analysis import composite_pauli_z
from .analysis import composite_pauli_z_expvalue
from .analysis import Plotter
from .analysis import rho_to_fidelity
from .linear import get_measurement_circ
from .linear import get_ghz_simple
from .linear import get_ghz_mqc
from .linear import get_ghz_mqc_para
from .linear import get_ghz_po
from .linear import get_ghz_po_para
from .parallelize import BConfig
