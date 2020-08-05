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

# pylint: disable=undefined-loop-variable,invalid-name

"""
Run through Quantum volume
"""

import unittest

import qiskit
import qiskit.ignis.verification.quantum_volume as qv
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import depolarizing_error

SEED = 42


def qv_circuit_execution(qubit_lists: list, ntrials: int, shots: int):
    """
    create quantum volume circuits, simulate the ideal state and run a noisy simulation
    Args:
        qubit_lists (list): list of lists of qubits to apply qv circuits to
        ntrials (int): number of iterations (number of circuits)
        shots (int): number of shots per simulation

    Returns:
        tuple: a tuple of 2 lists:
            list of Results of the ideal statevector simulations
            list of Results of the noisy circuits simulations

    """
    # create the qv circuit
    qv_circs, qv_circs_nomeas = qv.qv_circuits(qubit_lists, ntrials)
    # get the ideal state
    statevector_backend = qiskit.Aer.get_backend('statevector_simulator')
    ideal_results = []
    for trial in range(ntrials):
        ideal_results.append(qiskit.execute(qv_circs_nomeas[trial],
                                            backend=statevector_backend).result())

    # define noise_model
    noise_model = NoiseModel()
    p1q = 0.002
    p2q = 0.02
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p1q, 1), 'u2')
    noise_model.add_all_qubit_quantum_error(depolarizing_error(2 * p1q, 1), 'u3')
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p2q, 2), 'cx')

    # get the noisy results
    backend = qiskit.Aer.get_backend('qasm_simulator')
    basis_gates = ['u1', 'u2', 'u3', 'cx']  # use U,CX for now
    exp_results = []
    for trial in range(ntrials):
        exp_results.append(
            qiskit.execute(qv_circs[trial], basis_gates=basis_gates, backend=backend,
                           noise_model=noise_model, shots=shots,
                           seed_simulator=SEED,
                           backend_options={'max_parallel_experiments': 0}).result())

    return ideal_results, exp_results


class TestQV(unittest.TestCase):
    """ The test class """

    def test_qv_circuits(self):
        """ Test circuit generation """

        # Qubit list
        qubit_lists = [[0, 1, 2], [0, 1, 2, 4], [0, 1, 2, 4, 7]]
        ntrials = 5

        qv_circs, _ = qv.qv_circuits(qubit_lists, ntrials)

        self.assertEqual(len(qv_circs), ntrials,
                         "Error: Not enough trials")

        self.assertEqual(len(qv_circs[0]), len(qubit_lists),
                         "Error: Not enough circuits for the "
                         "number of specified qubit lists")

    def test_qv_circuits_with_seed(self):
        """Ensure seed is propogated to QuantumVolme objects."""
        qubit_lists = [list(range(5))]
        qv_circs, qv_circs_no_meas = qv.qv_circuits(qubit_lists, seed=3)
        meas_name = qv_circs[0][0].data[0][0].name
        no_meas_name = qv_circs_no_meas[0][0].data[0][0].name

        self.assertEqual(int(meas_name.split(',')[-1].rstrip(']')), 811)
        self.assertEqual(int(no_meas_name.split(',')[-1].rstrip(']')), 811)

    def test_measurements_in_circuits(self):
        """Ensure measurements are set or not on output circuits."""
        qubit_lists = [list(range(4))]
        qv_circs, qv_circs_no_meas = qv.qv_circuits(qubit_lists)
        qv_circs_measure_qubits = [
            x[1][0].index for x in qv_circs[0][0].data if x[0].name == 'measure']
        self.assertNotIn('measure',
                         [x[0].name for x in qv_circs_no_meas[0][0].data])
        self.assertEqual([0, 1, 2, 3], qv_circs_measure_qubits)

    def test_measurements_in_circuits_qubit_list_gap(self):
        """Test that there are no measurement instructions in output nomeas circuits."""
        qubit_lists = [[1, 3, 5, 7]]
        qv_circs, qv_circs_no_meas = qv.qv_circuits(qubit_lists)
        qv_circs_measure_qubits = [
            x[1][0].index for x in qv_circs[0][0].data if x[0].name == 'measure']
        self.assertNotIn('measure',
                         [x[0].name for x in qv_circs_no_meas[0][0].data])
        self.assertEqual([1, 3, 5, 7], qv_circs_measure_qubits)

    def test_qv_fitter(self):
        """ Test the fitter"""
        qubit_lists = [[0, 1, 3], [0, 1, 3, 5], [0, 1, 3, 5, 7],
                       [0, 1, 3, 5, 7, 10]]
        ntrials = 5

        ideal_results, exp_results = qv_circuit_execution(qubit_lists,
                                                          ntrials,
                                                          shots=1024)

        qv_fitter = qv.QVFitter(qubit_lists=qubit_lists)
        qv_fitter.add_statevectors(ideal_results)
        qv_fitter.add_data(exp_results)

        qv_success_list = qv_fitter.qv_success()
        self.assertFalse(qv_success_list[0][0])


if __name__ == '__main__':
    unittest.main()
