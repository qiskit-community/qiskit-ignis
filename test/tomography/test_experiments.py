import unittest
import numpy as np
from qiskit import Aer
from qiskit import QuantumCircuit
from qiskit.ignis.verification.tomography.experiment import ProcessTomographyExperiment
from qiskit.ignis.verification.tomography.experiment import StateTomographyExperiment
from qiskit.ignis.verification.tomography.experiment import GatesetTomographyExperiment
from qiskit.quantum_info import state_fidelity, partial_trace, Statevector, Choi
from qiskit.circuit.library import (HGate, SGate)
BACKEND = Aer.get_backend('qasm_simulator')


class TestStateTomography(unittest.TestCase):
    def test_basic_non_entangled(self):
        circ = QuantumCircuit(2)
        circ.x(0)
        circ.h(1)

        exp = StateTomographyExperiment(circ, [0])
        rho = exp.run(BACKEND)
        psi = partial_trace(Statevector.from_instruction(circ), [1])

        F_bell = state_fidelity(psi, rho, validate=False)
        self.assertAlmostEqual(F_bell, 1, places=1)

    def test_bell_basic(self):
        bell = QuantumCircuit(2)
        bell.h(0)
        bell.cx(0, 1)

        exp = StateTomographyExperiment(bell, [0])
        rho = exp.run(BACKEND)
        psi = partial_trace(Statevector.from_instruction(bell), [1])

        F_bell = state_fidelity(psi, rho, validate=False)
        self.assertAlmostEqual(F_bell, 1, places=1)

    def test_bell_meas(self):
        bell = QuantumCircuit(2, 2)
        bell.h(0)
        bell.cx(0, 1)

        psi = partial_trace(Statevector.from_instruction(bell), [1])

        bell.measure(1, 1)

        exp = StateTomographyExperiment(bell, [0])
        rho = exp.run(BACKEND)

        F_bell = state_fidelity(psi, rho, validate=False)
        self.assertAlmostEqual(F_bell, 1, places=1)

    def test_bell_full(self):
        bell = QuantumCircuit(2, 2)
        bell.h(0)
        bell.cx(0, 1)

        exp = StateTomographyExperiment(bell)
        rho = exp.run(BACKEND)
        psi = Statevector.from_instruction(bell)
        F_bell = state_fidelity(psi, rho, validate=False)
        self.assertAlmostEqual(F_bell, 1, places=1)

    def test_subspace_analysis(self):
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(0, 2)
        circuit.z(1)
        circuit.x(2)

        exp = StateTomographyExperiment(circuit)  # full experiment - all 3 qubits
        exp.execute(BACKEND)

        rho_01 = exp.set_target_qubits([0, 1]).run_analysis()
        rho_02 = exp.set_target_qubits([0, 2]).run_analysis()
        rho_12 = exp.set_target_qubits([1, 2]).run_analysis()

        psi_01 = partial_trace(Statevector.from_instruction(circuit), [2])
        psi_02 = partial_trace(Statevector.from_instruction(circuit), [1])
        psi_12 = partial_trace(Statevector.from_instruction(circuit), [0])

        F_bell = state_fidelity(psi_01, rho_01, validate=False)
        self.assertAlmostEqual(F_bell, 1, places=1)

        F_bell = state_fidelity(psi_02, rho_02, validate=False)
        self.assertAlmostEqual(F_bell, 1, places=1)

        F_bell = state_fidelity(psi_12, rho_12, validate=False)
        self.assertAlmostEqual(F_bell, 1, places=1)


class TestProcessTomography(unittest.TestCase):
    def test_bell_full(self):
        bell = QuantumCircuit(2)
        bell.h(0)
        bell.cx(0, 1)

        exp = ProcessTomographyExperiment(bell)
        rho = exp.run(BACKEND)
        psi = Choi(bell).data
        F_bell = state_fidelity(psi / 4, rho / 4, validate=False)
        self.assertAlmostEqual(F_bell, 1, places=1)

class TestGatesetTomography(unittest.TestCase):
    def compare_gates(self, expected_gates, result_gates, labels, delta=0.2):
        for label in labels:
            expected_gate = expected_gates[label]
            result_gate = result_gates[label].data
            msg = "Failure on gate {}: Expected gate = \n{}\n" \
                  "vs Actual gate = \n{}".format(label,
                                                 expected_gate,
                                                 result_gate)
            distance = self.hs_distance(expected_gate, result_gate)
            self.assertAlmostEqual(distance, 0, delta=delta, msg=msg)

    @staticmethod
    def hs_distance(A, B):
        return sum([np.abs(x) ** 2 for x in np.nditer(A - B)])

    @staticmethod
    def convert_from_ptm(vector):
        Id = np.sqrt(0.5) * np.array([[1, 0], [0, 1]])
        X = np.sqrt(0.5) * np.array([[0, 1], [1, 0]])
        Y = np.sqrt(0.5) * np.array([[0, -1j], [1j, 0]])
        Z = np.sqrt(0.5) * np.array([[1, 0], [0, -1]])
        v = vector.reshape(4)
        return v[0] * Id + v[1] * X + v[2] * Y + v[3] * Z

    def run_test_on_gate_and_noise(self,
                                    gate,
                                    noise_model=None,
                                    noise_ptm=None):

        exp = GatesetTomographyExperiment(gate)
        gateset_basis = exp.basis()

        labels = gateset_basis.gate_labels
        gates = gateset_basis.gate_matrices
        gates['rho'] = np.array([[np.sqrt(0.5)], [0], [0], [np.sqrt(0.5)]])
        gates['E'] = np.array([[np.sqrt(0.5), 0, 0, np.sqrt(0.5)]])
        # apply noise if given
        for label in labels:
            if label != "Id" and noise_ptm is not None:
                gates[label] = noise_ptm @ gates[label]
        Fs = [gateset_basis.spam_matrix(label)
              for label in gateset_basis.spam_labels]

        result_gates = exp.run(BACKEND)
        expected_gates = gates
        expected_gates['E'] = self.convert_from_ptm(expected_gates['E'])
        expected_gates['rho'] = self.convert_from_ptm(expected_gates['rho'])
        self.compare_gates(expected_gates, result_gates, labels + ['E', 'rho'])

    def test_noiseless(self):
        self.run_test_on_gate_and_noise(HGate())

if __name__ == '__main__':
    unittest.main()
