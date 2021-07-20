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

from qiskit import execute, Aer
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error


def get_syndrome(code, noise_model, shots=1014):
    """Runs a code to get required results."""
    circuits = [code.circuit[log] for log in ['0', '1']]

    job = execute(
        circuits,
        Aer.get_backend('qasm_simulator'),
        noise_model=noise_model,
        shots=shots)
    raw_results = {}
    for log in ['0', '1']:
        raw_results[log] = job.result().get_counts(log)

    return code.process_results(raw_results)


def get_noise(p_meas, p_gate):
    """Define a noise model."""
    error_meas = pauli_error([('X', p_meas), ('I', 1 - p_meas)])
    error_gate1 = depolarizing_error(p_gate, 1)
    error_gate2 = error_gate1.tensor(error_gate1)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(
        error_meas, "measure")
    noise_model.add_all_qubit_quantum_error(
        error_gate1, ["u1", "u2", "u3"])
    noise_model.add_all_qubit_quantum_error(
        error_gate2, ["cx"])

    return noise_model


class TestCodes(unittest.TestCase):
    """The test class. """

    def test_rep(self, weighted=False):
        """Repetition code test."""
        matching_probs = {}
        weighted_matching_probs = {}
        lookup_probs = {}
        post_probs = {}

        max_dist = 5

        noise_model = get_noise(0.02, 0.02)

        for d in range(3, max_dist + 1, 2):

            code = RepetitionCode(d, 2)

            results = get_syndrome(code, noise_model=noise_model, shots=8192)

            dec = GraphDecoder(code)
            if weighted:
                dec.weight_syndrome_graph(results=results[d])

            logical_prob_match = dec.get_logical_prob(
                results)
            logical_prob_lookup = lookuptable_decoding(
                results, results)
            logical_prob_post = postselection_decoding(
                results)

            for log in ['0', '1']:
                matching_probs[(d, log)] = logical_prob_match[log]
                weighted_matching_probs[(d, log)] = logical_prob_match[log]
                lookup_probs[(d, log)] = logical_prob_lookup[log]
                post_probs[(d, log)] = logical_prob_post[log]

        for d in range(3, max_dist-1, 2):
            for log in ['0', '1']:
                m_down = matching_probs[(d, log)] \
                    > matching_probs[(d + 2, log)]
                w_down = matching_probs[(d, log)] \
                    > weighted_matching_probs[(d + 2, log)]
                l_down = lookup_probs[(d, log)] \
                    > lookup_probs[(d + 2, log)]
                p_down = post_probs[(d, log)] \
                    > post_probs[(d + 2, log)]

                m_error = "Error: Matching decoder does not improve "\
                    + "logical error rate between repetition codes"\
                    + " of distance " + str(d) + " and " + str(d + 2) + ".\n"\
                    + "For d="+str(d)+": " + str(matching_probs[(d, log)])\
                    + ".\n"\
                    + "For d="+str(d+2)+": " + str(matching_probs[(d+2, log)])\
                    + "."
                w_error = "Error: Matching decoder does not improve "\
                    + "logical error rate between repetition codes"\
                    + " of distance " + str(d) + " and " + str(d + 2) + ".\n"\
                    + "For d="+str(d)+" (unweighted): "\
                    + str(matching_probs[(d, log)]) + ".\n"\
                    + "For d="+str(d+2)+" (weighted): "\
                    + str(weighted_matching_probs[(d+2, log)])\
                    + "."
                l_error = "Error: Lookup decoder does not improve "\
                    + "logical error rate between repetition codes"\
                    + " of distance " + str(d) + " and " + str(d + 2) + ".\n"\
                    + "For d="+str(d)+": " + str(lookup_probs[(d, log)])\
                    + ".\n"\
                    + "For d="+str(d+2)+": " + str(lookup_probs[(d+2, log)])\
                    + "."
                p_error = "Error: Postselection decoder does not improve "\
                    + "logical error rate between repetition codes"\
                    + " of distance " + str(d) + " and " + str(d + 2) + ".\n"\
                    + "For d="+str(d)+": " + str(post_probs[(d, log)])\
                    + ".\n"\
                    + "For d="+str(d+2)+": " + str(post_probs[(d+2, log)])\
                    + "."

                self.assertTrue(
                    m_down or matching_probs[(d, log)] == 0.0, m_error)
                self.assertTrue(
                    w_down or matching_probs[(d + 2, log)] == 0.0, w_error)
                self.assertTrue(
                    l_down or lookup_probs[(d, log)] == 0.0, l_error)
                self.assertTrue(
                    p_down or post_probs[(d, log)] == 0.0, p_error)

    def test_weight(self):
        """Error weighting code test."""
        error = "Error: Calculated error probability not correct for "\
            + "test result '0 0  11 00' in d=3, T=1 repetition code."
        code = RepetitionCode(3, 1)
        dec = GraphDecoder(code)
        test_results = {'0': {'0 0  00 00': 1024, '0 0  11 00': 512}}
        p = dec.get_error_probs(test_results)
        self.assertTrue(
            round(p[(1, 0, 0), (1, 0, 1)], 2) == 0.33, error)


if __name__ == '__main__':
    unittest.main()
