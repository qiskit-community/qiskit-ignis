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

# TODO(mtreinish): Remove these disables when implementation is finished
# pylint: disable=unused-argument,unnecessary-pass

"""
RB Helper functions
"""


from typing import List, Optional, Union, Dict
from warnings import warn

import numpy as np
from qiskit import QuantumCircuit
from qiskit.qobj import QasmQobj


def count_gates(qobj, basis, qubits):
    """
    Take a compiled qobj and output the number of gates in each circuit

    Args:
        qobj: compiled qobj
        basis: gates basis for the qobj
        qubits: qubits to count over

    Returns:
        n x l x m list of number of gates
            n: number of circuits
            l: number of qubits
            m: number of gates in basis

    Additional Information:
        nQ gates are counted in each qubit's set of gates
    """
    warn('The function `count_gates` will be deprecated.', DeprecationWarning)

    nexp = len(qobj.experiments)
    ngates = np.zeros([nexp, len(qubits), len(basis)], dtype=int)

    basis_ind = {basis[i]: i for i in range(len(basis))}

    for i in range(nexp):
        for instr in qobj.experiments[i].instructions:
            if instr.name in basis:
                for qind, qubit in enumerate(qubits):
                    if qubit in instr.qubits:
                        ngates[i][qind][basis_ind[instr.name]] += 1

    return ngates


def gates_per_clifford(transpiled_circuits_list: List[List[QuantumCircuit]],
                       clifford_lengths: Union[np.ndarray, List[int]],
                       basis: List[str],
                       qubits: List[int],
                       qobj_list: Optional[List[QasmQobj]] = None,
                       clifford_length: Optional[np.ndarray] = None) \
        -> Dict[int, Dict[str, float]]:
    """Take a list of transpiled ``QuantumCircuit`` and use these to calculate
    the number of gates per Clifford. Each ``QuantumCircuit`` should be transpiled into
    given ``basis`` set. The result can be used to convert a value of error per Clifford
    into error per basis gate under appropriate assumption.

    Example:
        This example shows how to calculate gate per Clifford of 2Q RB sequence for
        qubit 0 and qubit 1. You can refer to the function
        :mod:`~qiskit.ignis.verification.randomized_benchmarking.randomized_benchmarking_seq`
        for the detail of RB circuit generation.::

            # create RB circuits
            rb_circs_list, xdata = randomized_benchmarking_seq(**rb_opts)

            # transpile
            transpiled_circuits_list = []
            for rb_circs in rb_circs_list:
                rb_circs_transpiled = qiskit.transpile(rb_circs, basis_gate=basis)
                transpiled_circuits_list.append(rb_circs_transpiled)

            # calculate gate per Clifford
            ngates = gates_per_clifford(transpiled_circuits_list, xdata[0], basis, [0, 1])

        The gate counts for qubit 0 (1) is obtained by ``ngates[0]`` (``ngates[1]``)
        as usual python dictionary. If all gate counts are zero,
        you may specify wrong ``basis`` or input circuit list is not transpiled into basis gates.

    Args:
        transpiled_circuits_list: List of transpiled RB circuit for each seed.
        clifford_lengths: number of Cliffords in each circuit
        basis: gates basis for the qobj
        qubits: qubits to count over
        qobj_list: Deprecated. see ``transpiled_circuits_list``
        clifford_length: Deprecated. see ``clifford_lengths``

    Returns:
        Nested dictionary of gate counts per Clifford.
    """
    if qobj_list is not None:
        transpiled_circuits_list = qobj_list
        warn('The argument `qobj_list` will be deprecated. Use `transpiled_circuit_list`.',
             DeprecationWarning)

    if clifford_length is not None:
        clifford_lengths = clifford_length
        warn('The argument `clifford_length` will be deprecated. Use `clifford_lengths`.',
             DeprecationWarning)

    ncliffs = 0
    ngates = {qubit: {base: 0 for base in basis} for qubit in qubits}

    for transpiled_circuits in transpiled_circuits_list:
        if isinstance(transpiled_circuits, list):
            for ncliff, transpiled_circuit in zip(clifford_lengths, transpiled_circuits):
                for instr, qregs, _ in transpiled_circuit.data:
                    for qreg in qregs:
                        try:
                            ngates[qreg.index][instr.name] += 1
                        except KeyError:
                            pass
                # include inverse
                ncliffs += ncliff + 1
        else:
            warn('`QasmQobj` input will be deprecated. Use `QuantumCircuit` instead. '
                 'Gate counts based on `QasmQobj` has no unittest and may return wrong counts.',
                 DeprecationWarning)
            # TODO: remove this code block after deprecation period
            for ncliff, experiment in zip(clifford_lengths, transpiled_circuits.experiments):
                for instr in experiment.instructions:
                    for q_ind in instr.qubits:
                        try:
                            ngates[q_ind][instr.name] += 1
                        except KeyError:
                            pass
                # include inverse
                ncliffs += ncliff + 1

    for qubit in qubits:
        for base in basis:
            ngates[qubit][base] /= ncliffs

    warn('The function now returns nested dictionary instead of numpy array.')

    return ngates


def coherence_limit(nQ=2, T1_list=None, T2_list=None,
                    gatelen=0.1):

    """
    The error per gate (1-average_gate_fidelity) given by the T1,T2 limit

    Args:
        nQ: number of qubits (1 and 2 supported)
        T1_list: list of T1's (Q1,...,Qn)
        T2_list: list of T2's (as measured, not Tphi).
            If not given assume T2=2*T1
        gatelen: length of the gate

    Returns:
        coherence limited error per gate
    """

    T1 = np.array(T1_list)

    if T2_list is None:
        T2 = 2*T1
    else:
        T2 = np.array(T2_list)

    if len(T1) != nQ or len(T2) != nQ:
        raise ValueError("T1 and/or T2 not the right length")

    coherence_limit_err = 0

    if nQ == 1:

        coherence_limit_err = 0.5*(1.-2./3.*np.exp(-gatelen/T2[0]) -
                                   1./3.*np.exp(-gatelen/T1[0]))

    elif nQ == 2:

        T1factor = 0
        T2factor = 0

        for i in range(2):
            T1factor += 1./15.*np.exp(-gatelen/T1[i])
            T2factor += 2./15.*(np.exp(-gatelen/T2[i]) +
                                np.exp(-gatelen*(1./T2[i]+1./T1[1-i])))

        T1factor += 1./15.*np.exp(-gatelen*np.sum(1/T1))
        T2factor += 4./15.*np.exp(-gatelen*np.sum(1/T2))

        coherence_limit_err = 0.75*(1.-T1factor-T2factor)

    else:
        raise ValueError('Not a valid number of qubits')

    return coherence_limit_err


def twoQ_clifford_error(ngates, gate_qubit, gate_err):
    """
    The two qubit Clifford gate error given measured errors in the primitive
    gates used to construct the Clifford (see arxiv:1712.06550). Assumes the
    error in the underlying gates is depolarizing.

    Args:
        ngates: list of the number of gates per 2Q Clifford
        gate_qubit: list of the qubit corresponding to the gate (0, 1 or -1).
            -1 corresponds to the 2Q gate.
        gate_err: list of the gate errors

    Returns:
        Error per 2Q Clifford
    """

    alpha1Q = [1.0, 1.0]
    alpha2Q = 1.0

    for gate_ind, ngate in enumerate(ngates):
        if gate_qubit[gate_ind] == -1:
            alpha2Q *= (1-4/3*gate_err[gate_ind])**ngate
        else:
            alpha1Q[gate_qubit[gate_ind]] *= \
                (1-2*gate_err[gate_ind])**ngate

    alpha2Q_cliff = 1/5*(np.sum(alpha1Q)+3*np.prod(alpha1Q))*alpha2Q

    return (1-alpha2Q_cliff)*3/4
