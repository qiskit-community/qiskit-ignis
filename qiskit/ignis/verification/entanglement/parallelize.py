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

# pylint: disable=invalid-name

"""
This module provides methods to parallellize CNOT gates
in the preparation of the GHZ State, which results in
the GHZ State having a much higher fidelity
then a normal "linear" CNOT gate preparation of the
GHZ State. Additionally, there are methods within parallelize.py
that configure different characterization tests for the
GHZ State, including Multiple Quantum Coherence (MQC),
Parity Oscillations (PO), and Quantum Tomography.

It may be more suitable to put this module in Terra rather
than Ignis.
"""

from qiskit.circuit import ClassicalRegister, QuantumRegister, Parameter
from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile


class BConfig:
    """
    This class is used to create a GHZ circuit
    with parallellized CNOT gates to increase fidelity
    """

    def __init__(self, backend, indicator=True):
        self.nodes = {}
        self.backend = backend
        self.cmap = backend.configuration().coupling_map
        self.initialize_nodes()
        self.indicator = indicator

    def initialize_nodes(self):
        """
        Initializes the nodes to the dictionary based on coupling map
        """
        self.nodes = {}
        for i in range(len(self.cmap_calib())):
            self.nodes[self.cmap_calib()[i][0]] = []
        for i in range(len(self.cmap_calib())):
            self.nodes[self.cmap_calib()[i][0]].append(self.cmap_calib()[i][1])

    def cmap_calib(self):
        """
        Only intended for public devices (doubling and reversing
        each item in coupling map), but useful to run anyway
        """
        cmap_new = list(self.cmap)
        for a in self.cmap:
            if [a[1], a[0]] not in self.cmap:
                cmap_new.append([a[1], a[0]])
        cmap_new.sort(key=lambda x: x[0])
        return cmap_new

    def get_best_node(self):
        """
        First node with the most connections; Does not yet sort
        based on error, but that is probably not too useful
        """

        best_node = 0
        for i in self.nodes:
            if len(self.nodes[i]) > len(self.nodes[best_node]):
                best_node = i
            else:
                pass
        return best_node

    def indicator_off(self):
        """
        We turn off gate-based sorting of the tier_dict
        """

        self.indicator = False

    def indicator_on(self):
        """
        We turn on gate-based sorting of the tier_dict
        """

        self.indicator = True

    def get_cx_error(self):
        """
        Gets dict of relevant CNOT gate errors
        """

        a = self.backend.properties().to_dict()['gates']
        cxerrordict = {}
        for i in a:
            if len(i['qubits']) == 1:
                continue
            if len(i['qubits']) == 2:
                b = tuple(i['qubits'])
                if b in cxerrordict.keys():
                    pass
                cxerrordict[b] = i['parameters'][0]['value']
                if (b[1], b[0]) not in cxerrordict.keys():
                    cxerrordict[(b[1], b[0])] = i['parameters'][0]['value']
                else:
                    pass

        return cxerrordict

    def get_cx_length(self):
        """
        Gets dict of relevant CNOT gate lengths
        """

        a = self.backend.properties().to_dict()['gates']
        cxlengthdict = {}
        for i in a:
            if len(i['qubits']) == 1:
                continue
            if len(i['qubits']) == 2:
                b = tuple(i['qubits'])
                if b in cxlengthdict.keys():
                    pass
                cxlengthdict[b] = i['parameters'][1]['value']
                if (b[1], b[0]) not in cxlengthdict.keys():
                    cxlengthdict[(b[1], b[0])] = i['parameters'][1]['value']
                else:
                    pass

        return cxlengthdict

    def child_sorter(self, children, parent):
        """
        Sorts children nodes based on error/length
        """

        return sorted(children, key=lambda child: self.get_cx_error()[(child, parent)])

    def get_tier_dict(self):
        """
        Take the nodes of the BConfig to create a Tier Dictionary,
        where keys are the steps in the process,
        and the values are the connections following pattern of:
        [controlled qubit, NOT qubit]. Thus the
        backend's GHZ state is parallelized.

        Returns:
            dict: Tier dictionary - [step in process, control-target connection]
                Facilitates parallelized GHZ circuits
        """

        tier = {}
        tier_DM = {}
        length = len(self.nodes.keys())
        trashlist = []
        tier[0] = (self.get_best_node())
        tier_DM[self.get_best_node()] = 0

        trashlist.append(self.get_best_node())
        parent = self.get_best_node()

        tier = {x: [[]] for x in range(length)}

        parentlist = []
        parentlist.append(parent)
        ii = 0
        while True:
            totalchildren = []
            for parent in parentlist:
                children = self.nodes[parent]
                if self.indicator:
                    children = self.child_sorter(children, parent)
                totalchildren += children
                children = [a for a in children if a not in trashlist]
                j = tier_DM[parent]
                for i, _ in enumerate(children):
                    tier[j] += [[parent, children[i]]]
                    tier_DM[children[i]] = j+1
                    j += 1
                    trashlist.append(children[i])
                parentlist = totalchildren

            if len(trashlist) == length:
                break
            ii += 1
            if ii > 50:
                break

        newtier = {}
        for a in tier:
            if [] in tier[a]:
                tier[a].remove([])
        for a in tier:
            if tier[a] != []:
                newtier[a] = tier[a]
        tier = newtier

        return tier

    def get_ghz_layout(self, n, transpiled=True, barriered=True):
        """
        Feeds the Tier Dict of the backend to create a basic
        qiskit GHZ circuit with no measurement;
        Args:
           n (int): number of qubits
           transpiled (bool): toggle on/off transpilation - useful for tomography
           barriered (bool): yes/no whether to barrier each step of CNOT gates

        Returns:
           QuantumCircuit: A GHZ Circuit
       """

        tierdict = self.get_tier_dict()
        q = QuantumRegister(n, 'q')
        circ = QuantumCircuit(q)
        circ.h(q[0])
        trashlist = []
        initial_layout = {}
        for a in tierdict:
            for aa in tierdict[a]:
                if aa[0] not in trashlist:
                    trashlist.append(aa[0])
                    trashindex = trashlist.index(aa[0])
                    initial_layout[q[trashindex]] = aa[0]
                else:
                    pass
                if aa[1] not in trashlist:
                    trashlist.append(aa[1])
                    trashindex = trashlist.index(aa[1])
                    initial_layout[q[trashindex]] = aa[1]
                else:
                    pass
                circ.cx(q[trashlist.index(aa[0])],
                        q[trashlist.index(aa[1])])

                if len(trashlist) == n:
                    break
            if barriered:
                circ.barrier()
            if len(trashlist) == n:
                break

        if transpiled:
            circ = transpile(circ, backend=self.backend,
                             initial_layout=initial_layout)

        return circ, initial_layout

    def get_measurement_circ(self, n, qregname, cregname, full_measurement=True):
        """
        Creates a measurement circuit that can toggle
        between measuring the control qubit
        or measuring all qubits. The default is
        measurement of all qubits.
        """

        q = QuantumRegister(n, qregname)
        if full_measurement:
            cla = ClassicalRegister(n, cregname)
            meas = QuantumCircuit(q, cla)
            meas.barrier()
            meas.measure(q, cla)
            return meas

        cla = ClassicalRegister(1, cregname)
        meas = QuantumCircuit(q, cla)
        meas.barrier()
        meas.measure(q[0], cla)
        return meas

    def get_ghz_mqc(self, n, delta, full_measurement=True):
        """
        Get MQC circuit
        """

        circ, initial_layout = self.get_ghz_layout(n)
        q = QuantumRegister(n, 'q')
        rotate = QuantumCircuit(q)

        rotate.barrier()
        rotate.u1(delta, q)
        rotate.barrier()
        rotate.x(q)
        rotate.barrier()
        rotate = transpile(rotate,
                           backend=self.backend,
                           initial_layout=initial_layout)

        meas = self.get_measurement_circ(n, 'q', 'c', full_measurement)
        meas = transpile(meas,
                         backend=self.backend,
                         initial_layout=initial_layout)

        new_circ = circ + rotate + circ.inverse() + meas

        return new_circ, initial_layout

    def get_ghz_mqc_para(self, n, full_measurement=True):
        """
        Get a parametrized MQC circuit.
        Remember that get_counts() method accepts
        an index now, not a circuit

        Args:
            n (int): number of qubits
            full_measurement (bool): Whether to append full measurement, or
                only on the first qubit
        Returns:
            QuantumCircuit: A GHZ Circuit
        """

        circ, initial_layout = self.get_ghz_layout(n)
        q = QuantumRegister(n, 'q')
        rotate = QuantumCircuit(q)
        delta = Parameter('t')
        rotate.barrier()
        rotate.u1(delta, q)
        rotate.barrier()
        rotate.x(q)
        rotate.barrier()
        rotate = transpile(rotate,
                           backend=self.backend,
                           initial_layout=initial_layout)
        meas = self.get_measurement_circ(n, 'q', 'c', full_measurement)
        meas = transpile(meas,
                         backend=self.backend,
                         initial_layout=initial_layout)
        new_circ = circ + rotate + circ.inverse() + meas
        return new_circ, delta, initial_layout

    def get_ghz_po(self, n, delta):
        """
        Get Parity Oscillation circuit
        """

        circ, initial_layout = self.get_ghz_layout(n)
        q = QuantumRegister(n, 'q')
        rotate = QuantumCircuit(q)

        rotate.barrier()
        rotate.u2(delta, -delta, q)
        rotate.barrier()
        rotate = transpile(rotate,
                           backend=self.backend,
                           initial_layout=initial_layout)

        meas = self.get_measurement_circ(n, 'q', 'c', True)
        meas = transpile(meas,
                         backend=self.backend,
                         initial_layout=initial_layout)

        new_circ = circ + rotate + meas

        return new_circ, initial_layout

    def get_ghz_po_para(self, n):
        """
        Get a parametrized PO circuit. Remember that get_counts()
        method accepts an index now, not a circuit.
        The two phase parameters are a quirk of the Parameter module

        Args:
            n (int): number of qubits

        Returns:
            tuple: A tuple of type (``QuantumCircuit``, ``list``, ``dict``) A
                parity oscillation circuit, its Delta/minus-delta parameters,
                and the initial ghz layout
        """

        circ, initial_layout = self.get_ghz_layout(n)
        q = QuantumRegister(n, 'q')
        rotate = QuantumCircuit(q)

        delta = Parameter('t')
        deltaneg = Parameter('-t')

        rotate.barrier()
        rotate.u2(delta, deltaneg, q)
        rotate.barrier()
        rotate = transpile(rotate,
                           backend=self.backend,
                           initial_layout=initial_layout)
        meas = self.get_measurement_circ(n, 'q', 'c', True)
        meas = transpile(meas,
                         backend=self.backend,
                         initial_layout=initial_layout)
        new_circ = circ + rotate + meas
        return new_circ, [delta, deltaneg], initial_layout

    def get_ghz_simple(self, n, full_measurement):
        """
        Get simple GHZ circuit with measurement

        Args:
            n (int): number of qubits
            full_measurement (bool): Whether to append full measurement, or
                only on the first qubit
        Returns:
           QuantumCircuit: A GHZ Circuit
        """

        circ, initial_layout = self.get_ghz_layout(n)
        q = QuantumRegister(n, 'q')

        meas = self.get_measurement_circ(n, 'q', 'c', full_measurement)
        meas = transpile(meas,
                         backend=self.backend,
                         initial_layout=initial_layout)
        new_circ = circ + meas

        return new_circ, q, initial_layout
