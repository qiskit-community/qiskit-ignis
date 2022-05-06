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

"""Qiskit Ignis Root."""

import warnings

from .version import __version__

warnings.warn(
    "The qiskit.ignis package is deprecated and has been supersceded by the "
    "qiskit-experiments project. Refer to the migration guide: "
    "https://github.com/Qiskit/qiskit-ignis#migration-guide on how to migrate "
    "to the new project.",
    DeprecationWarning,
    stacklevel=2)
