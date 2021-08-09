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

"""Run codes and decoders."""

import unittest

from qiskit.ignis.verification.topological_codes import RepetitionCode
from qiskit.ignis.verification.topological_codes import GraphDecoder
from qiskit.ignis.verification.topological_codes import lookuptable_decoding
from qiskit.ignis.verification.topological_codes import postselection_decoding

from qiskit import execute, Aer, QuantumCircuit
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error


def get_syndrome(code, noise_model, shots=1014):
    """Runs a code to get required results."""
    circuits = [code.circuit[log] for log in ["0", "1"]]

    job = execute(
        circuits,
        Aer.get_backend("qasm_simulator"),
        noise_model=noise_model,
        shots=shots,
    )
    raw_results = {}
    for log in ["0", "1"]:
        raw_results[log] = job.result().get_counts(log)

    return code.process_results(raw_results)


def get_noise(p_meas, p_gate):
    """Define a noise model."""
    error_meas = pauli_error([("X", p_meas), ("I", 1 - p_meas)])
    error_gate1 = depolarizing_error(p_gate, 1)
    error_gate2 = error_gate1.tensor(error_gate1)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas, "measure")
    noise_model.add_all_qubit_quantum_error(error_gate1, ["u1", "u2", "u3"])
    noise_model.add_all_qubit_quantum_error(error_gate2, ["cx"])

    return noise_model


class TestCodes(unittest.TestCase):
    """Test the topological codes module."""

    def single_error_test(self, code):
        """
        Insert all possible single qubit errors into the given code,
        and check that each creates a pair of syndrome nodes.
        """
        decoder = GraphDecoder(code)

        for logical in ["0", "1"]:
            qc = code.circuit[logical]
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
                    for error in ["x", "y", "z"]:
                        temp_qc = blank_qc.copy()
                        temp_qc.name = str((j, qubit, error))
                        temp_qc.data = qc.data[0:j]
                        getattr(temp_qc, error)(qubit)
                        temp_qc.data += qc.data[j: depth + 1]
                        circuit_name[(j, qubit, error)] = temp_qc.name
                        error_circuit[temp_qc.name] = temp_qc

            simulator = Aer.get_backend("qasm_simulator")
            job = execute(list(error_circuit.values()), simulator)

            for j in range(depth):
                qubits = qc.data[j][1]
                for qubit in qubits:
                    for error in ["x", "y", "z"]:
                        raw_results = {}
                        raw_results[logical] = job.result().get_counts(
                            str((j, qubit, error))
                        )
                        results = code.process_results(raw_results)[logical]
                        for string in results:
                            nodes = decoder._string2nodes(string, logical=logical)
                            self.assertIn(
                                len(nodes),
                                [0, 2],
                                "Error of type "
                                + error
                                + " on qubit "
                                + str(qubit)
                                + " at depth "
                                + str(j)
                                + " creates "
                                + str(len(nodes))
                                + " nodes in syndrome graph, instead of 2.",
                            )

    def test_string2nodes(self):
        """Test string2nodes with different logical values."""
        code = RepetitionCode(3, 2)
        dec = GraphDecoder(code)
        s0 = "0 0  01 00 01"
        s1 = "1 1  01 00 01"
        self.assertTrue(
            dec._string2nodes(s0, logical="0") == dec._string2nodes(s1, logical="1"),
            "Error: Incorrect nodes from results string",
        )

    def test_graph_construction(self):
        """Check that single errors create a pair of nodes for all types of code."""
        for d in [2, 3]:
            for T in [1, 2]:
                for xbasis in [False, True]:
                    code = RepetitionCode(d, T, xbasis=xbasis)
                    self.single_error_test(code)

    def test_weight(self):
        """Error weighting code test."""
        error = (
            "Error: Calculated error probability not correct for "
            + "test result '0 0  11 00' in d=3, T=1 repetition code."
        )
        code = RepetitionCode(3, 1)
        dec = GraphDecoder(code)
        test_results = {"0": {"0 0  00 00": 1024, "0 0  11 00": 512}}
        p = dec.get_error_probs(test_results)
        self.assertTrue(round(p[(1, 0, 0), (1, 0, 1)], 2) == 0.33, error)

    def test_rep_probs(self):
        """Repetition code test."""
        matching_probs = {}
        lookup_probs = {}
        post_probs = {}

        max_dist = 5

        noise_model = get_noise(0.02, 0.02)

        for d in range(3, max_dist + 1, 2):

            code = RepetitionCode(d, 2)

            results = get_syndrome(code, noise_model=noise_model, shots=8192)

            dec = GraphDecoder(code)

            logical_prob_match = dec.get_logical_prob(results)
            logical_prob_lookup = lookuptable_decoding(results, results)
            logical_prob_post = postselection_decoding(results)

            for log in ["0", "1"]:
                matching_probs[(d, log)] = logical_prob_match[log]
                lookup_probs[(d, log)] = logical_prob_lookup[log]
                post_probs[(d, log)] = logical_prob_post[log]

        for d in range(3, max_dist - 1, 2):
            for log in ["0", "1"]:
                m_down = matching_probs[(d, log)] > matching_probs[(d + 2, log)]
                l_down = lookup_probs[(d, log)] > lookup_probs[(d + 2, log)]
                p_down = post_probs[(d, log)] > post_probs[(d + 2, log)]

                m_error = (
                    "Error: Matching decoder does not improve "
                    + "logical error rate between repetition codes"
                    + " of distance "
                    + str(d)
                    + " and "
                    + str(d + 2)
                    + ".\n"
                    + "For d="
                    + str(d)
                    + ": "
                    + str(matching_probs[(d, log)])
                    + ".\n"
                    + "For d="
                    + str(d + 2)
                    + ": "
                    + str(matching_probs[(d + 2, log)])
                    + "."
                )
                l_error = (
                    "Error: Lookup decoder does not improve "
                    + "logical error rate between repetition codes"
                    + " of distance "
                    + str(d)
                    + " and "
                    + str(d + 2)
                    + ".\n"
                    + "For d="
                    + str(d)
                    + ": "
                    + str(lookup_probs[(d, log)])
                    + ".\n"
                    + "For d="
                    + str(d + 2)
                    + ": "
                    + str(lookup_probs[(d + 2, log)])
                    + "."
                )
                p_error = (
                    "Error: Postselection decoder does not improve "
                    + "logical error rate between repetition codes"
                    + " of distance "
                    + str(d)
                    + " and "
                    + str(d + 2)
                    + ".\n"
                    + "For d="
                    + str(d)
                    + ": "
                    + str(post_probs[(d, log)])
                    + ".\n"
                    + "For d="
                    + str(d + 2)
                    + ": "
                    + str(post_probs[(d + 2, log)])
                    + "."
                )

                self.assertTrue(m_down or matching_probs[(d, log)] == 0.0, m_error)
                self.assertTrue(l_down or lookup_probs[(d, log)] == 0.0, l_error)
                self.assertTrue(p_down or post_probs[(d, log)] == 0.0, p_error)

    def test_graph(self):
        """Test if analytically derived SyndromeGraph is correct."""
        error = (
            "Error: The analytical SyndromeGraph does not coincide "
            + "with the brute force SyndromeGraph in d=7, T=2 RepetitionCode."
        )
        code = RepetitionCode(7, 2)
        graph_new = GraphDecoder(code, brute=False).S
        graph_old = GraphDecoder(code, brute=True).S
        test_passed = True
        for node in graph_new.nodes():
            test_passed &= node in graph_old.nodes()
        for node in graph_old.nodes():
            test_passed &= node in graph_new.nodes()
        test_passed &= rx.is_isomorphic(graph_new, graph_old, lambda x, y: x == y)
        self.assertTrue(test_passed, error)


if __name__ == "__main__":
    unittest.main()
