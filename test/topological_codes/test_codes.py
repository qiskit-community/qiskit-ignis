import unittest

from qiskit.ignis.verification.topological_codes import RepetitionCode
from qiskit.ignis.verification.topological_codes import GraphDecoder
from qiskit.ignis.verification.topological_codes.decoder_utils import *

from qiskit import execute, Aer
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error


def get_syndrome(code, noise_model, shots=1014):

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


noise_model = get_noise(0.04, 0.04)


class TestCodes(unittest.TestCase):
    """ The test class """

    def test_rep(self):
        matching_probs = {}
        lookup_probs = {}
        post_probs = {}

        max_dist = 4

        for d in range(3, max_dist + 1):

            code = RepetitionCode(d, 2)

            results = get_syndrome(code, noise_model=noise_model, shots=8192)

            dec = GraphDecoder(code)

            logical_prob_match = dec.get_logical_prob(
                results)
            logical_prob_lookup = lookuptable_decoding(
                results, results)
            logical_prob_post = postselection_decoding(
                results)

            for log in ['0', '1']:
                matching_probs[(d, log)] = logical_prob_match[log]
                lookup_probs[(d, log)] = logical_prob_lookup[log]
                post_probs[(d, log)] = logical_prob_post[log]

        for d in range(3, max_dist):
            for log in ['0', '1']:
                m_down = matching_probs[(d, log)] \
                    > matching_probs[(d + 1, log)]
                l_down = lookup_probs[(d, log)] \
                    > lookup_probs[(d + 1, log)]
                p_down = post_probs[(d, log)] \
                    > post_probs[(d + 1, log)]

                m_error = "Error: Matching decoder does not improve "\
                    + "logical error rate between repetition codes"\
                    + " of distance " + str(d) + " and " + str(d + 1) + "."
                l_error = "Error: Matching decoder does not improve "\
                    + "logical error rate between repetition codes"\
                    + " of distance " + str(d) + " and " + str(d + 1) + "."
                p_error = "Error: Matching decoder does not improve "\
                    + "logical error rate between repetition codes"\
                    + " of distance " + str(d) + " and " + str(d + 1) + "."

                self.assertTrue(
                    m_down or matching_probs[(d, log)] == 0.0, m_error)
                self.assertTrue(
                    l_down or lookup_probs[(d, log)] == 0.0, l_error)
                self.assertTrue(
                    p_down or post_probs[(d, log)] == 0.0, p_error)


if __name__ == '__main__':
    unittest.main()
