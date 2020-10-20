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

"""Optional support for Numba just-in-time compilation."""

import logging
logger = logging.getLogger(__name__)

try:
    import numba
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    logger.info('Numba not installed, certain functions have improved '
                'performance if Number is installed: '
                'https://pypi.org/project/numba/')


def jit_fallback(func):
    """Decorator to try to apply numba JIT compilation."""
    if _HAS_NUMBA:
        return numba.jit(nopython=True)(func)
    else:
        return func
