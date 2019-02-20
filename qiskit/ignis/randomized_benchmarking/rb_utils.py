# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
RB Helper functions
"""

def count_gates(qobj, basis, qubits):

    """
    Take a compiled qobj and output the number of gates in each circuit

    Args:
        qobj: compiled qobj
        basis: gates basis for the qobj
        qubits: qubits to count over

    Returns:
        n x m list of number of gates
            n: number of circuits
            m: number of gates in basis
    """

    #TO DO
    pass

def gates_per_clifford(qobj_list, clifford_length, basis, qubits):

    """
    Take a list of compiled qobjs (for each seed) and use these
    to calculate the number of gates per clifford

    Args:
        qobj_list: compiled qobjs for each seed
        clifford_length: number of cliffords in each circuit
        basis: gates basis for the qobj
        qubits: qubits to count over

    Returns:
        list of number of gates per clifford (same order as basis)
    """

    #TO DO

    pass


def coherence_limit(nQ=2, T1_list=[100.,100.], T2_list=[100.,100.], gatelen=0.1):

    """
    The error per gate (1-average_gate_fidelity) given by the T1,T2 limit

    Args:
        nQ: number of qubits (1 and 2 supported)
        T1_list: list of T1's (Q1,...,Qn)
        T2_list: list of T2's (as measured, not Tphi)
        gatelen: length of the gate

    Returns:
        coherence limited error per gate
    """


    if nQ==1:

        ###
        pass

    elif nQ==2:

        ###
        pass

    else:
        raise ValueError('Not a valid number of qubits')

    return 0.0
