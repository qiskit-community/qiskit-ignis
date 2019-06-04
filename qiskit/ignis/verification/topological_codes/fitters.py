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

'''
Decoders for quantum error correction codes.

Specifically, this class contains decoders that can be expressed as solving
a graph-theoretic problem.
'''

import copy
import warnings
import networkx as nx

from qiskit import QuantumCircuit, Aer, execute


class GraphDecoder():
    '''
    A range of decoders for topological quantum error correcting codes.

    Attributes
    * `code`: Object describing a compatible form of code, as supplied when
    the decoder is initialized.
    * `S`: Graph describing connections between non-trivial syndrome elements.
    '''

    def __init__(self, code, S=None):
        '''
        Initializes the decoder for the supplied `code` by running
        `_make_syndrome_graph()`. Since this process can take some time,
        it is also possible to load in a premade `S`. However,
        if this was created for a differently defined `code`,it won't work
        properly.
        '''
        self.code = code

        if S:
            self.S = S
        else:
            self.S = self._make_syndrome_graph()

    def _separate_string(self, string):

        separated_string = []
        for syndrome_type_string in string.split('  '):
            separated_string.append(syndrome_type_string.split(' '))
        return separated_string

    def _make_syndrome_graph(self):
        '''
        This method injects all possible Pauli errors into the circuit for
        `code`. This is done by examining the qubits used in each gate of the
        circuit for a stored logical 0. A graph is then created with a node
        for each non-trivial syndrome element, and an edge between all such
        elements that can be created by the same error.
        '''

        S = nx.Graph()

        qc = self.code.circuit['0']

        blank_qc = QuantumCircuit()
        for qreg in qc.qregs:
            blank_qc.add_register(qreg)
        for creg in qc.cregs:
            blank_qc.add_register(creg)

        error_circuit = {}
        circuit_name = {}
        depth = len(qc)
        for j in range(depth):
            qubits = qc.data[j][1]
            for qubit in qubits:
                for error in ['x', 'y', 'z']:
                    temp_qc = copy.deepcopy(blank_qc)
                    temp_qc.name = str((j, qubit, error))
                    temp_qc.data = qc.data[0:j]
                    eval('temp_qc.' + error + '(qubit)')
                    temp_qc.data += qc.data[j:depth + 1]
                    circuit_name[(j, qubit, error)] = temp_qc.name
                    error_circuit[temp_qc.name] = temp_qc

        job = execute(list(error_circuit.values()),
                      Aer.get_backend('qasm_simulator'))

        for j in range(depth):
            qubits = qc.data[j][1]
            for qubit in qubits:
                for error in ['x', 'y', 'z']:

                    raw_results = {}
                    raw_results['0'] = job.result().get_counts(
                        str((j, qubit, error)))
                    results = self.code.process_results(raw_results)['0']

                    for string in results:
                        separated_string = self._separate_string(string)

                        nodes = []
                        for syn_type, _ in enumerate(separated_string):
                            for syn_round in range(
                                    len(separated_string[syn_type])):
                                elements = \
                                    separated_string[syn_type][syn_round]
                                for elem_num, element in enumerate(elements):
                                    if element == '1':
                                        nodes.append((syn_type,
                                                      syn_round,
                                                      elem_num))

                        for node in nodes:
                            S.add_node(node)
                        for source in nodes:
                            for target in nodes:
                                if source != target:
                                    S.add_edge(source, target, distance=1)

        return S

    def make_error_graph(self, string, subgraphs=None):
        '''
        Makes the graph for the syndrome in `string`.
        '''

        if subgraphs is None:
            subgraphs = []
            for syndrome_type in string.split('  '):
                subgraphs.append(['0'])

        set_subgraphs = [
            subgraph for subs4type in subgraphs for subgraph in subs4type]

        E = {subgraph: nx.Graph() for subgraph in set_subgraphs}

        separated_string = self._separate_string(string)

        for syndrome_type, _ in enumerate(separated_string):
            for syndrome_round in range(len(separated_string[syndrome_type])):
                elements = separated_string[syndrome_type][syndrome_round]
                for elem_num, element in enumerate(elements):
                    if element == '1' or syndrome_type == 0:
                        for subgraph in subgraphs[syndrome_type]:
                            E[subgraph].add_node(
                                (syndrome_type,
                                 syndrome_round,
                                 elem_num))

        # for each pair of nodes in error create an edge and weight with the
        # distance
        for subgraph in set_subgraphs:
            for source in E[subgraph]:
                for target in E[subgraph]:
                    if target != (source):
                        distance = int(nx.shortest_path_length(
                            self.S, source, target))
                        E[subgraph].add_edge(source, target, weight=-distance)

        return E

    def matching(self, string):
        '''
        Given a string, performs correction using minimum weight matching and
        returns a string with corrected logical values.
        '''

        # this matching algorithm is designed for a single graph
        E = self.make_error_graph(string)['0']

        # set up graph that is like E, but each syndrome node is connected to a
        # separate copy of the nearest logical node
        E_matching = nx.Graph()
        syndrome_nodes = []
        logical_nodes = []
        logical_neighbours = []
        for node in E:
            if node[0] == 0:
                logical_nodes.append(node)
            else:
                syndrome_nodes.append(node)
        for source in syndrome_nodes:
            for target in syndrome_nodes:
                if target != (source):
                    E_matching.add_edge(
                        source, target, weight=E[source][target]['weight'])

            potential_logical = {}
            for target in logical_nodes:
                potential_logical[target] = E[source][target]['weight']
            nearest_logical = max(potential_logical, key=potential_logical.get)
            E_matching.add_edge(
                source,
                nearest_logical + source,
                weight=potential_logical[nearest_logical])
            logical_neighbours.append(nearest_logical + source)
        for source in logical_neighbours:
            for target in logical_neighbours:
                if target != (source):
                    E_matching.add_edge(source, target, weight=0)

        # do the matching on this
        matches = nx.max_weight_matching(E_matching, maxcardinality=True)

        # use it to construct and return a correcetd logical string
        logicals = self._separate_string(string)[0]
        for (source, target) in matches:
            if source[0] == 0 and target[0] != 0:
                logicals[source[1]] = str((int(logicals[source[1]]) + 1) % 2)
            if target[0] == 0 and source[0] != 0:
                logicals[target[1]] = str((int(logicals[target[1]]) + 1) % 2)

        logical_string = ''
        for logical in logicals:
            logical_string += logical + ' '
        logical_string = logical_string[:-1]

        return logical_string

    def get_logical_prob(self, results, algorithm='matching'):
        '''
        Given a dictionary of results, as produced by a code object, the
        logical error probability is calculated for the decoding method
        specified by `algorithm`.
        '''
        logical_prob = {}
        for log in results:

            shots = 0
            incorrect_shots = 0

            corrected_results = {}
            if algorithm == 'matching':
                for string in results[log]:
                    corr_str = self.matching(string)
                    if corr_str in corrected_results:
                        corrected_results[corr_str] += results[log][string]
                    else:
                        corrected_results[corr_str] = results[log][string]
            else:
                warnings.warn(
                    "The requested algorithm " +
                    str(algorithm) +
                    " is not known.",
                    Warning)

            for string in corrected_results:
                shots += corrected_results[string]
                if string[0] != str(log):
                    incorrect_shots += corrected_results[string]

            logical_prob[log] = incorrect_shots / shots

        return logical_prob
