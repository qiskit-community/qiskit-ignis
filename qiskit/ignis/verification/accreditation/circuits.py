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
Generates accreditation circuits
"""
import copy
from numpy import random
from qiskit import QuantumCircuit
from .qotp import layer_parser, QOTP_fromlayers


class accreditationCircuits:
    """This class generates accreditation circuits from a target."""
    def __init__(self, targetcirc, twoqubitgate='cz', coupling_map=None):
        self.targetCircuit(targetcirc,
                           twoqubitgate=twoqubitgate,
                           coupling_map=coupling_map)

    def targetCircuit(self, targetcirc, twoqubitgate='cz', coupling_map=None):
        """
        Load target circuit in to class,
        parse into layers
            Args:
                targetcirc (circuit): a qiskit circuit to accredit
                coupling_map (list): some particular device topology
                as list of list (e.g. [[0,1],[1,2],[2,0]])
        """
        self.target = copy.deepcopy(targetcirc)
        # parse circuit into layers
        self.layers = layer_parser(self.target,
                                   twoqubitgate=twoqubitgate,
                                   coupling_map=coupling_map)

    def generateCircuits(self, num_trap):
        """
        Generate quantum circuits for accreditation
            Args:
                num_trap (int): number of trap circuits
            Returns:
                circuit_list (list): accreditation circuits
                postp_list (list): strings used for classical post-processing
                v_zero (int): position of target circuit
        """
        # Position of the target
        v_zero = random.randint(0, num_trap+1)
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
            circ, postp = QOTP_fromlayers(testlayers)
            circuit_list.append(circ)
            postp_list.append(postp)
        return circuit_list, postp_list, v_zero

    def _routine_two(self):
        """
        Routine 2.
        It returns random 1-qubit gate for trap circuits
            Args:
                czs (list): a list of circuits encoding the cz gate layers

            Returns:
                gate_trap (list): list of all 1-qubit gates in trap circuit
        """
        # generate a temporary set of single qubit gate initialized to I
        qregs = self.layers['qregs']
        cregs = self.layers['cregs']
        nlayers = len(self.layers['twoqubitlayers'])+1
        gate_trap = [QuantumCircuit(qregs,
                                    cregs) for j in range(nlayers)]

        # decide if we are in x or z basis and apply first row of H's
        basis = random.randint(2)
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
                    if random.randint(2):
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
                    if random.randint(2):
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
                    raise Exception("Two qubit gate {0}".format(g2q)
                                    + "is not implemented"
                                    + " in accreditation circuits")
            for q in self.layers['qregs']:
                # if we didn't do anything to this index yet
                # apply a h or an s
                if q not in regs2q:
                    if random.randint(2):
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
