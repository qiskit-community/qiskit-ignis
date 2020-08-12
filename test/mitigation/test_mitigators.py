# -*- coding: utf-8 -*-

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
Test mitigators
"""

from itertools import combinations, product, chain, permutations
from typing import List, Tuple
import unittest

from ddt import ddt, unpack, data

from qiskit import execute, QuantumCircuit
from qiskit.result import Result
from qiskit.providers.aer import QasmSimulator, noise
from qiskit.ignis.mitigation.measurement import (
    MeasMitigatorGenerator,
    MeasMitigatorFitter,
    counts_expectation_value
)


class NoisySimulationTest(unittest.TestCase):
    """Base class that contains methods and attributes
    for doing tests of readout error noise with flexible
    readout errors.
    """

    sim = QasmSimulator()

    # Example max qubit number
    num_qubits = 4

    # Create readout errors
    readout_errors = []
    for i in range(num_qubits):
        p_error1 = (i + 1) * 0.002
        p_error0 = 2 * p_error1
        ro_error = noise.ReadoutError([[1 - p_error0, p_error0], [p_error1, 1 - p_error1]])
        readout_errors.append(ro_error)
    # TODO: Needs 2q errors?

    # Readout Error only
    noise_model = noise.NoiseModel()
    for i in range(num_qubits):
        noise_model.add_readout_error(readout_errors[i], [i])
    seed_simulator = 100

    shots = 10000

    tolerance = 0.05

    def execute_circs(self, qc_list: List[QuantumCircuit], noise_model = None) -> Result:
        """Run circuits with the readout noise defined in this class
        """
        return execute(
            qc_list,
            backend=self.sim,
            seed_simulator=self.seed_simulator,
            shots=self.shots,
            noise_model=None if noise_model is None else self.noise_model,
            backend_options={'method': 'density_matrix'}
        ).result()


class TestExpVals(NoisySimulationTest):
    """Test the expectation values of all the MeasMitigator* classes
    and compare against the exact results.
    """

    qc = QuantumCircuit(4, 4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.h(range(4))
    qc.measure(range(4), range(4))

    def setUp(self):
        result_targ = self.execute_circs(self.qc)
        result_nois = self.execute_circs(self.qc, noise_model=self.noise_model)

        counts_target = result_targ.get_counts(0)
        self.counts_noise = result_nois.get_counts(0)

        self.exp_targ = counts_expectation_value(counts_target)

        self.method = None

    def test_ctmp_exp(self):
        """Test CTMP"""
        self.method = 'CTMP'

    def test_full_exp(self):
        """Test complete"""
        self.method = 'complete'

    def test_tensored_exp(self):
        """Test tensored"""
        self.method = 'tensored'

    def tearDown(self):
        circs, meta, _ = MeasMitigatorGenerator(self.num_qubits, method=self.method).run()
        result_cal = self.execute_circs(circs, noise_model=self.noise_model)
        mitigator = MeasMitigatorFitter(result_cal, meta).fit(method=self.method)

        expval = mitigator.expectation_value(self.counts_noise)

        self.assertLess(abs(self.exp_targ - expval), self.tolerance)


# https://docs.python.org/3/library/itertools.html#recipes
def powerset(iterable):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    output_set = list(iterable)
    return chain.from_iterable(combinations(output_set, r) for r in range(len(output_set)+1))


@ddt
class TestPartialExpVals(NoisySimulationTest):
    """Test taking expectation values on only subsets of qubits and permutations
    of the original set of qubits.
    """

    @data(*product(
        [
            item for sublist in
            [*map(lambda x: list(permutations(x)), powerset(range(4)))] for item in sublist],
        ['complete', 'tensored']
        ))
    @unpack
    def test_partial_expval(self, qubits: Tuple[int], method: str):
        """Test taking the expectation value only on the set of `qubits`
        with `method` for mitigation.
        """

        if len(qubits) == 0:
            return None

        qc = QuantumCircuit(4, len(qubits))
        qc.h(qubits[0])
        for i, j in combinations(qubits, r=2):
            qc.cx(i, j)
        qc.measure(qubits, range(len(qubits)))

        result_targ = self.execute_circs(qc)
        result_nois = self.execute_circs(qc, noise_model=self.noise_model)

        counts_targ = result_targ.get_counts(0)
        counts_nois = result_nois.get_counts(0)

        exp_targ = counts_expectation_value(counts_targ)

        circs, meta, _ = MeasMitigatorGenerator(self.num_qubits, method=method).run()
        result_cal = self.execute_circs(circs, noise_model=self.noise_model)
        mitigator = MeasMitigatorFitter(result_cal, meta).fit(method=method)

        expval = mitigator.expectation_value(counts_nois, qubits=qubits)

        self.assertLess(abs(exp_targ - expval), self.tolerance)
        return None


if __name__ == '__main__':
    unittest.main()
