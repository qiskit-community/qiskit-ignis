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
Generates accreditation circuits
Implementation follows the methods from
Samuele Ferracin, Theodoros Kapourniotis and Animesh Datta
New Journal of Physics, Volume 21, November 2019
https://iopscience.iop.org/article/10.1088/1367-2630/ab4fd6
"""
import copy
import numpy as np
from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from .qotp import layer_parser, QOTP_fromlayers


class AccreditationCircuits:
    """
    This class generates accreditation circuits from a target.

    Implementation follows the methods from
    Samuele Ferracin, Theodoros Kapourniotis and Animesh Datta
    New Journal of Physics, Volume 21, November 2019
    https://iopscience.iop.org/article/10.1088/1367-2630/ab4fd6
    """
    def __init__(self, target_circ, two_qubit_gate='cx', coupling_map=None, seed=None):
        """
        Initialize the circuit generation class
        parse into layers

        Args:
            target_circ (QuantumCircuit): a qiskit circuit to accredit
            two_qubit_gate (string): a flag as to which 2 qubit
                gate to compile with, can be cx or cz
            coupling_map (list): some particular device topology
                as list of list (e.g. [[0,1],[1,2],[2,0]])
            seed (int): seed to the random number generator
        """
        self._rng = np.random.RandomState(seed)
        self.target_circuit(target_circ,
                            two_qubit_gate=two_qubit_gate,
                            coupling_map=coupling_map)

    def target_circuit(self, target_circ, two_qubit_gate='cx', coupling_map=None):
        """
        Load target circuit in to class, and parse into layers

        Args:
            target_circ (QuantumCircuit): a qiskit circuit to accredit
            two_qubit_gate (string): a flag as to which 2 qubit
                gate to compile with, can be cx or cz
            coupling_map (list): some particular device topology
                as list of list (e.g. [[0,1],[1,2],[2,0]])
        """
        self.target = copy.deepcopy(target_circ)
        # parse circuit into layers
        self.layers = layer_parser(self.target,
                                   two_qubit_gate=two_qubit_gate,
                                   coupling_map=coupling_map)

    def generate_circuits(self, num_trap):
        """
        Generate quantum circuits for accreditation

        Args:
            num_trap (int): number of trap circuits

        Returns:
            tuple: A tuple of the form
                (``circuit_list``, `postp_list``, ``v_zero``) where:
                circuit_list (list): accreditation circuits
                postp_list (list): strings used for classical post-processing
                v_zero (int): position of target circuit
        """
        # Position of the target
        v_zero = self._rng.randint(0, num_trap+1)
        # output lists
        circuit_list = []
        postp_list = []
        # main loop through traps
        testlayers = copy.deepcopy(self.layers)
        for k in range(num_trap+1):
            if k == v_zero:  # Generating the target circuit
                testlayers = copy.deepcopy(self.layers)
            else:  # Generating a trap circuit
                testlayers['singlequbitlayers'] = self._routine_two()
            # apply onte time pad and add to outputlist
            circ, postp = QOTP_fromlayers(testlayers, self._rng)
            circuit_list.append(circ)
            postp_list.append(postp)
        return circuit_list, postp_list, v_zero

    def _routine_two(self):
        """
        Routine 2.
        It returns random 1-qubit gate for trap circuits

        Returns:
            list: gate_trap list of all 1-qubit gates in trap circuit

        Raises:
            QiskitError: If an unsupported 2 qubit gate is present
        """
        # generate a temporary set of single qubit gate initialized to I
        qregs = self.layers['qregs']
        cregs = self.layers['cregs']
        nlayers = len(self.layers['twoqubitlayers'])+1
        gate_trap = [QuantumCircuit(qregs,
                                    cregs) for j in range(nlayers)]

        # decide if we are in x or z basis and apply first row of H's
        basis = self._rng.randint(2)
        if basis:
            for q in self.layers['qregs']:
                gate_trap[0].h(q)

        # step through cz layers
        for layer, gates2q in enumerate(self.layers['twoqubitlayers']):
            regs2q = []  # a list of all registers used by czs
            for _, qsub, _ in gates2q:
                regs2q.extend(qsub)
                g2q = self.layers['twoqubitgate']
                if g2q == 'cx':
                    # apply either H x SH or S x I (and inverses)
                    if self._rng.randint(2):
                        gate_trap[layer].h(qsub[0])
                        gate_trap[layer+1].h(qsub[0])
                        gate_trap[layer].s(qsub[1])
                        gate_trap[layer].h(qsub[1])
                        gate_trap[layer+1].h(qsub[1])
                        gate_trap[layer+1].sdg(qsub[1])
                    else:
                        gate_trap[layer].s(qsub[0])
                        gate_trap[layer+1].sdg(qsub[0])
                elif g2q == 'cz':
                    # apply either H x S or S x H (and inverses)
                    if self._rng.randint(2):
                        gate_trap[layer].h(qsub[0])
                        gate_trap[layer+1].h(qsub[0])
                        gate_trap[layer].s(qsub[1])
                        gate_trap[layer+1].sdg(qsub[1])
                    else:
                        gate_trap[layer].s(qsub[0])
                        gate_trap[layer+1].sdg(qsub[0])
                        gate_trap[layer].h(qsub[1])
                        gate_trap[layer+1].h(qsub[1])
                else:
                    raise QiskitError("Two qubit gate {0}"
                                      "is not implemented"
                                      "in accreditation circuits".format(g2q))
            for q in self.layers['qregs']:
                # if we didn't do anything to this index yet
                # apply a h or an s
                if q not in regs2q:
                    if self._rng.randint(2):
                        gate_trap[layer].h(q)
                        gate_trap[layer+1].h(q)
                    else:
                        gate_trap[layer].s(q)
                        gate_trap[layer+1].sdg(q)
        # if in x basis undo H's
        if basis:
            for q in self.layers['qregs']:
                gate_trap[-1].h(q)
        return gate_trap
