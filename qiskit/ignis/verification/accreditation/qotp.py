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
# pylint: disable= no-member


"""
Quantum one-time pad
"""


import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters.circuit_to_dag import circuit_to_dag
from qiskit.converters.dag_to_circuit import dag_to_circuit
from qiskit.compiler import transpile


def layer_parser(circ, two_qubit_gate='cx', coupling_map=None):
    """
    Tranforms general circuits into a nice form for a qotp.

    Args:
        circ (QuantumCircuit): A generic quantum circuit
        two_qubit_gate (string): a flag as to which 2 qubit
            gate to compile with, can be cx or cz
        coupling_map: some particular device topology as list
            of list (e.g. [[0,1],[1,2],[2,0]])
    Returns:
        singlequbit_layers (lsit):  a list of circuits describing
            the single qubit gates
        cz_layers (list): a list of circuits describing the cz layers
        meas_layer (QuantumCircuit): a circuit describing the final measurement
    """

    # transpile to single qubits and cx
    # TODO: replace cx with cz when that is available
    circ_internal = transpile(circ,
                              optimization_level=2,
                              basis_gates=['u1', 'u2', 'u3', 'cx'],
                              coupling_map=coupling_map)
    # quantum and classial registers
    qregs = circ_internal.qregs[0]
    cregs = circ_internal.cregs[0]
    # conatiners for the eventual output passed to the accred code
    singlequbitlayers = [QuantumCircuit(qregs, cregs),
                         QuantumCircuit(qregs, cregs)]
    twoqubitlayers = [QuantumCircuit(qregs, cregs)]
    measlayer = QuantumCircuit(qregs, cregs)
    # some flags for simplicity
    current2qs = []
    # loop through circuit (best to use the dag object)
    dag_internal = circuit_to_dag(circ_internal)
    for dag_layer in dag_internal.layers():
        circuit_layer = dag_to_circuit(dag_layer['graph'])
        for circelem, qsub, csub in circuit_layer:
            n = circelem.name
            if n == "barrier":
                # if a barrier separates any two qubit gates
                # start a new layer
                if current2qs != []:
                    singlequbitlayers.append(QuantumCircuit(qregs, cregs))
                    twoqubitlayers.append(QuantumCircuit(qregs, cregs))
                    current2qs = []
                singlequbitlayers[-2].append(circelem, qsub, csub)
            elif n in ('u1', 'u2', 'u3'):
                # single qubit gate
                q = qsub[0]
                if q in current2qs:
                    singlequbitlayers[-1].append(circelem, qsub, csub)
                else:
                    singlequbitlayers[-2].append(circelem, qsub, csub)
            elif n == "cx":
                # cx indices
                q0 = qsub[0]
                q1 = qsub[1]
                # check if new cnot satisfies overlap criteria
                if q0 in current2qs or q1 in current2qs:
                    singlequbitlayers.append(QuantumCircuit(qregs, cregs))
                    twoqubitlayers.append(QuantumCircuit(qregs, cregs))
                    current2qs = []
                if two_qubit_gate == 'cx':
                    # append cx
                    twoqubitlayers[-1].cx(q0, q1)
                elif two_qubit_gate == 'cz':
                    # append and correct to cz with h gates
                    twoqubitlayers[-1].cz(q0, q1)
                    singlequbitlayers[-1].h(qsub[1])
                    singlequbitlayers[-2].h(qsub[1])
                else:
                    raise Exception("Two qubit gate {0}".format(two_qubit_gate)
                                    + " is not implemented in qotp")
                # add to current
                current2qs.append(q0)
                current2qs.append(q1)
            elif n == "measure":
                measlayer.append(circelem, qsub, csub)
            else:
                raise Exception("Circuit element {0}".format(n)
                                + " is not implemented in qotp")
    if current2qs == []:
        del singlequbitlayers[-1]
        del twoqubitlayers[-1]
    for ind, circlayer in enumerate(singlequbitlayers):
        singlequbitlayers[ind] = transpile(circlayer,
                                           basis_gates=['u1', 'u2', 'u3'])
    parsedlayers = {'singlequbitlayers': singlequbitlayers,
                    'twoqubitlayers': twoqubitlayers,
                    'measlayer': measlayer,
                    'twoqubitgate': two_qubit_gate,
                    'qregs': qregs,
                    'cregs': cregs}
    return parsedlayers


def QOTP_fromlayers(layers, rng):
    """
    An intermediate step of a qotp in which we've converted the circuit
    to layers and only return a single pad or compilation

    Args:
        layers (dict): parsed layers from the layer parser
        rng (RNG): a random number generator
    Returns:
        qotp_circ (QuantumCircuit): output onetime pad circ
        qotp_postp (list): correction as liist of bits
    """

    # make some circuits
    qregs = layers['qregs']
    cregs = layers['cregs']
    twoqubitgate = layers['twoqubitgate']
    qotp_circ = QuantumCircuit(qregs, cregs)
    tempCirc = QuantumCircuit(qregs, cregs)

    # initial z gates after prep
    paulizs = rng.randint(2, size=len(qregs))
    for qind, q in enumerate(qregs):
        if paulizs[qind]:
            tempCirc.z(q)
    # step through layers
    for lnum, gates2q in enumerate(layers['twoqubitlayers']):
        # add single qubit gates to temp circuit
        tempCirc = tempCirc+layers['singlequbitlayers'][lnum]
        # generate and add single qubit paulis
        paulizs = rng.randint(2, size=len(qregs))
        paulixs = rng.randint(2, size=len(qregs))
        for qind, q in enumerate(qregs):
            if paulizs[qind]:
                tempCirc.z(q)
            if paulixs[qind]:
                tempCirc.x(q)
        # add to circuit and reset temp
        tempCirc = transpile(tempCirc,
                             basis_gates=['u1', 'u2', 'u3'])
        qotp_circ = qotp_circ+tempCirc
        tempCirc = QuantumCircuit(qregs, cregs)
        # add two qubit layers and get indices for 2qgates
        qotp_circ.barrier()
        qotp_circ = qotp_circ+gates2q
        qotp_circ.barrier()
        twoqindices = []
        for _, qsub, _ in gates2q:
            twoqindices.append([qsub[0].index, qsub[1].index])
        # update Paulis
        for inds in twoqindices:
            if twoqubitgate == 'cx':
                # iz -> zz and xi -> xx
                paulizs[inds[0]] = (paulizs[inds[0]]+paulizs[inds[1]]) % 2
                paulixs[inds[1]] = (paulixs[inds[1]]+paulixs[inds[0]]) % 2
            elif twoqubitgate == 'cz':
                # ix -> zx and xi -> xz
                paulizs[inds[0]] = (paulizs[inds[0]]+paulixs[inds[1]]) % 2
                paulizs[inds[1]] = (paulizs[inds[1]]+paulixs[inds[0]]) % 2
            else:
                raise Exception("Two qubit gate {0}".format(twoqubitgate)
                                + "is not implemented in qotp")
        for qind, q in enumerate(qregs):
            if paulixs[qind]:
                tempCirc.x(q)
            if paulizs[qind]:
                tempCirc.z(q)
    # add final single qubit layer
    tempCirc = tempCirc+layers['singlequbitlayers'][-1]
    # add final Paulis to create the one time pad
    paulizs = rng.randint(2, size=len(qregs))
    paulixs = rng.randint(2, size=len(qregs))
    for qind, q in enumerate(qregs):
        if paulizs[qind]:
            tempCirc.z(q)
        if paulixs[qind]:
            tempCirc.x(q)
    # add to circuit
    tempCirc = transpile(tempCirc,
                         basis_gates=['u1', 'u2', 'u3'])
    qotp_circ = qotp_circ+tempCirc
    # post operations
    qotp_postp = np.flip(paulixs)
    # measurements
    qotp_circ = qotp_circ+layers['measlayer']
    return qotp_circ, qotp_postp


def QOTP(circ, num, two_qubit_gate='cx', coupling_map=None, seed=None):
    """
    Performs a QOTP (or random compilation) on a generic circuit.

    This is essentially the same protocol as used in
    randomized compiling, but follows the methods in
    Samuele Ferracin, Theodoros Kapourniotis and Animesh Datta
    New Journal of Physics, Volume 21, November 2019
    https://iopscience.iop.org/article/10.1088/1367-2630/ab4fd6

    Args:
        circ (QuantumCircuit): A generic quantum circuit
        num (int): the number of one-time pads to return
        two_qubit_gate (string): a flag as to which 2 qubit
            gate to compile with, can be cx or cz
        coupling_map (list): a particular device topology as a
            list of list (e.g. [[0,1],[1,2],[2,0]])
        seed (int): seed to the random number generator
    Returns:
        qotp_circs (list): a list of circuits with qotp applied
        qotp_postps (list): a list of arrays specifying the one time pads
    """
    rng = np.random.RandomState(seed)
    # break into layers
    layers = layer_parser(circ,
                          two_qubit_gate=two_qubit_gate,
                          coupling_map=coupling_map)
    # output lists
    qotp_circs = []
    qotp_postps = []
    # generate circuits and postops
    for _ in range(num):
        circ, postp = QOTP_fromlayers(layers, rng)
        qotp_circs.append(circ)
        qotp_postps.append(postp)
    return qotp_circs, qotp_postps


def QOTPCorrectCounts(qotp_counts, qotp_postp):
    """
    Corrects a dictionary of results, shifting the qotp

    Args:
        qotp_counts (dict): a dict of exp counts
        qotp_postp (list): a binary list denoting the one time pad
    Returns:
        counts_out (dict): the corrected counts dict
    """

    counts_out = {}
    for key, val in qotp_counts.items():
        keyshift = [1 if k == "1" else 0 for k in key]
        keyshift = [(k+s) % 2 for k, s in zip(keyshift, qotp_postp)]
        keyshift = ''.join([str(k) for k in keyshift])
        counts_out[keyshift] = val
    return counts_out
