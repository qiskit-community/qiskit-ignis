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
Decoders for quantum error correction codes, with a focus on those that can be
expressed as solving a graph-theoretic problem.
"""

import copy
import warnings
import networkx as nx
import numpy as np

from qiskit import QuantumCircuit, execute

try:
    from qiskit import Aer
    HAS_AER = True
except ImportError:
    from qiskit import BasicAer
    HAS_AER = False


class GraphDecoder():
    """
    Class to construct the graph corresponding to the possible syndromes
    of a quantum error correction code, and then run suitable decoders.
    """

    def __init__(self, code, S=None):
        """
        Args:
            code (RepitionCode): The QEC Code object for which this decoder
                will be used.
            S (networkx.Graph): Graph describing connectivity between syndrome
                elements. Will be generated automatically if not supplied.

        Additional information:
            The decoder for the supplied ``code`` is initialized by running
            ``_make_syndrome_graph()``. Since this process can take some
            time, it is also possible to load in a premade ``S``. However,
            if this was created for a differently defined ``code``, it won't
            work properly.
        """

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

    def _string2nodes(self, string):

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
        return nodes

    def _make_syndrome_graph(self):
        """
        This method injects all possible Pauli errors into the circuit for
        ``code``.

        This is done by examining the qubits used in each gate of the
        circuit for a stored logical 0. A graph is then created with a node
        for each non-trivial syndrome element, and an edge between all such
        elements that can be created by the same error.
        """

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
                    getattr(temp_qc, error)(qubit)
                    temp_qc.data += qc.data[j:depth + 1]
                    circuit_name[(j, qubit, error)] = temp_qc.name
                    error_circuit[temp_qc.name] = temp_qc

        if HAS_AER:
            simulator = Aer.get_backend('qasm_simulator')
        else:
            simulator = BasicAer.get_backend('qasm_simulator')

        job = execute(list(error_circuit.values()), simulator)

        for j in range(depth):
            qubits = qc.data[j][1]
            for qubit in qubits:
                for error in ['x', 'y', 'z']:

                    raw_results = {}
                    raw_results['0'] = job.result().get_counts(
                        str((j, qubit, error)))
                    results = self.code.process_results(raw_results)['0']

                    for string in results:

                        nodes = self._string2nodes(string)

                        assert len(nodes) in [0, 2], "Error of type " + \
                            error + " on qubit " + str(qubit) + \
                            " at depth " + str(j) + " creates " + \
                            str(len(nodes)) + \
                            " nodes in syndrome graph, instead of 2."

                        for node in nodes:
                            S.add_node(node)
                        for source in nodes:
                            for target in nodes:
                                if source != target:
                                    S.add_edge(source, target, distance=1)

        return S

    def weight_syndrome_graph(self, results):
        """Generate weighted syndrome graph from result counts.

        Args:
            results (dict): A results dictionary, as produced by the
            `process_results` method of the code.

        Additional information:
            Uses `results` to estimate the probability of the errors that
            create the pairs of nodes in S. The edge weights are then
            replaced with the corresponding -log(p/(1-p).
        """

        results = results['0']

        count = {element: {edge: 0 for edge in self.S.edges}
                 for element in ['00', '01', '10', '11']}

        for string in results:

            nodes = self._string2nodes(string)

            for edge in self.S.edges:
                element = ''
                for j in range(2):
                    if edge[j] in nodes:
                        element += '1'
                    else:
                        element += '0'
                count[element][edge] += results[string]

        for edge in self.S.edges:
            edge_data = self.S.get_edge_data(edge[0], edge[1])
            ratios = []
            for elements in [('00', '11'), ('11', '00'),
                             ('01', '10'), ('10', '01')]:
                if count[elements[1]][edge] > 0:
                    ratio = count[elements[0]][edge]/count[elements[1]][edge]
                    ratios.append(ratio)
            edge_data['distance'] = -np.log(min(ratios))

    def make_error_graph(self, string, subgraphs=None):
        """
        Args:
            string (str): A string describing the output from the code.
            subgraphs (list): Used when multiple, semi-independent graphs need
            need to created.

        Returns:
            E: The subgraph(s) of S which corresponds to the non-trivial
            syndrome elements in the given string.
        """

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
                            self.S, source, target, weight='distance'))
                        E[subgraph].add_edge(source, target, weight=-distance)

        return E

    def matching(self, string):
        """
        Args:
            string (str): A string describing the output from the code.

        Returns:
            str: A string with corrected logical values,
                computed using minimum weight perfect matching.

        Additional information:
            This function can be run directly, or used indirectly to
            calculate a logical error probability with `get_logical_prob`
        """

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

        # use it to construct and return a corrected logical string
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
        """
        Args:
            results (dict): A results dictionary, as produced by the
            `process_results` method of the code.
            algorithm (str): Choice of which decoder to use.

        Returns:
            dict: Dictionary of logical error probabilities for
            each of the encoded logical states whose results were given in
            the input.
        """

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


def postselection_decoding(results):
    """
    Calculates the logical error probability using postselection decoding.

    This postselects all results with trivial syndrome.

    Args:
        results (dict): A results dictionary, as produced by the
            `process_results` method of a code.

    Returns:
        dict: Dictionary of logical error probabilities for
           each of the encoded logical states whose results were given in
           the input.
    """
    logical_prob = {}
    postselected_results = {}
    for log in results:
        postselected_results[log] = {}
        for string in results[log]:

            syndrome_list = string.split('  ')
            syndrome_list.pop(0)
            syndrome_string = '  '.join(syndrome_list)

            error_free = True
            for char in syndrome_string:
                error_free = error_free and (char in ['0', ' '])

            if error_free:
                postselected_results[log][string] = results[log][string]

    for log in results:
        shots = 0
        incorrect_shots = 0
        for string in postselected_results[log]:
            shots += postselected_results[log][string]
            if string[0] != log:
                incorrect_shots += postselected_results[log][string]

        logical_prob[log] = incorrect_shots / shots

    return logical_prob


def lookuptable_decoding(training_results, real_results):
    """
    Calculates the logical error probability using postselection decoding.
    This postselects all results with trivial syndrome.

    Args:
        training_results (dict): A results dictionary, as produced by the
            ``process_results`` method of a code.
        real_results (dict): A results dictionary, as produced by the
            ``process_results`` method of a code.

    Returns:
        dict: Dictionary of logical error probabilities for
            each of the encoded logical states whose results were given in
            the input.


    Additional information:
        Given a two dictionaries of results, as produced by a code object,
        thelogical error probability is calculated for lookup table
        decoding. This is done using `training_results` as a guide to which
        syndrome is most probable for each logical value, and the
        probability is calculated for the results in `real_results`.
    """

    logical_prob = {}
    for log in real_results:
        shots = 0
        incorrect_shots = 0
        for string in real_results[log]:

            p = {}
            for testlog in ['0', '1']:
                if string in training_results[testlog]:
                    p[testlog] = training_results[testlog][string]
                else:
                    p[testlog] = 0

            shots += real_results[log][string]
            if p['1' * (log == '0') + '0' * (log == '1')] > p[log]:
                incorrect_shots += real_results[log][string]

        logical_prob[log] = incorrect_shots / shots

    return logical_prob
