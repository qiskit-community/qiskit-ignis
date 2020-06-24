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

"""Utility functions"""


def build_counts_dict_from_list(count_list):
    """
    Add dictionary counts together.

    Parameters:
        count_list (list): List of counts.

    Returns:
        dict: Dict of counts.

    """
    if len(count_list) == 1:
        return count_list[0]

    new_count_dict = {}
    for countdict in count_list:
        for item in countdict:
            new_count_dict[item] = countdict[item]+new_count_dict.get(item, 0)

    return new_count_dict
