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
Quantum tomography data
"""

# Needed for functions
from itertools import combinations
from functools import reduce
from re import match
import numpy as np


###########################################################################
# Data formats for converting from counts to fitter data
#
# TODO: These should be moved to a terra.tools module
###########################################################################

def marginal_counts(counts, meas_qubits=True, pad_zeros=False):
    """
    Compute marginal counts from a counts dictionary.

    Args:
        counts (dict): a counts dictionary.
        meas_qubits (True, list(int)): the qubits to NOT be marinalized over
                                       if this is True meas_qubits will be
                                       all measured qubits (default: True).
        pad_zeros (Bool): Include zero count outcomes in return dict.

    Returns:
        A counts dictionary for the specified qubits. The returned dictionary
        will have any whitespace trimmed from the input counts keys. Thus if
        meas_qubits=True the returned dictionary will have the same values as
        the input dictionary, but with whitespace trimmed from the keys.
    """

    # Extract total number of qubits from first count key
    # We trim the whitespace seperating classical registers
    # and count the number of digits
    num_qubits = len(next(iter(counts)).replace(' ', ''))

    # Check if we do not need to marginalize. In this case we just trim
    # whitespace from count keys
    if num_qubits == len(meas_qubits) or (meas_qubits is True):
        ret = {}
        for key, val in counts.items():
            key = key.replace(' ', '')
            ret[key] = val
        return ret

    # Sort the measured qubits into decending order
    # Since bitstrings have qubit-0 as least significant bit
    if meas_qubits is True:
        meas_qubits = range(num_qubits)  # All measured
    qs = sorted(meas_qubits, reverse=True)

    # Generate bitstring keys for measured qubits
    meas_keys = count_keys(len(qs))

    # Get regex match strings for suming outcomes of other qubits
    rgx = []
    for key in meas_keys:
        def helper(x, y):
            if y in qs:
                return key[qs.index(y)] + x
            return '\\d' + x
        rgx.append(reduce(helper, range(num_qubits), ''))

    # Build the return list
    meas_counts = []
    for m in rgx:
        c = 0
        for key, val in counts.items():
            if match(m, key.replace(' ', '')):
                c += val
        meas_counts.append(c)

    # Return as counts dict on measured qubits only
    if pad_zeros is True:
        return dict(zip(meas_keys, meas_counts))
    ret = {}
    for key, val in zip(meas_keys, meas_counts):
        if val != 0:
            ret[key] = val
    return ret


def count_keys(num_qubits):
    """
    Return ordered count keys.
    """
    return [bin(j)[2:].zfill(num_qubits)
            for j in range(2 ** num_qubits)]


def combine_counts(counts1, counts2):
    """
    Combine two counts dictionaries.
    """
    ret = counts1
    for key, val in counts2.items():
        if key in ret:
            ret[key] += val
        else:
            ret[key] = val
    return ret


def expectation_counts(counts):
    """
    Converts count dict to an expectation counts dict.

    The returned dictionary is also a counts dictionary but the keys
    correspond to the which subsystems the operators are acting on
    and the counts are the un-normalized expectation values. The counts
    can be converted to expectation values by dividing by the value of the
    all '0's entry. The '0's key is the expectation value of the identity
    operator, and its value is equal to the number of shots .

    Args:
        counts (dict): a counts dictionary.

    Returns:
        A new counts dictionary where the counts are un-normalized
        expectation values for the subsystem measurement operators.


    Example:
        Consider a input counts dictionary for `s` shots of measurement of
        the two-qubit operator XZ (X on qubit-1, Z on qubit-0). The
        dictionary returned will have keys corresponding to:
            '00': s * <II>,
            '01': s * <IZ>,
            '10': s * <XI>,
            '11': s * <XZ>
    """

    # Get total shots for data set
    shots = np.sum(list(counts.values()))

    # Compute measured operators subsets in a data set
    nq = len(list(counts.keys())[0])

    # Get operator subsets
    subsets = []
    for r in range(nq):
        subsets += list(combinations(range(nq), r + 1))

    # Compute expectation values
    exp_data = {'00': shots}
    for s in subsets:
        exp_counts = 0
        exp_op = nq * ['0']

        # Get expectation operator
        for qubit in s:
            exp_op[qubit] = '1'

        # Get expectation value
        for key, val in marginal_counts(counts, s, pad_zeros=True).items():
            exp_counts += (-1) ** (key.count('1')) * val
        exp_data[''.join(exp_op)] = exp_counts

    return exp_data
