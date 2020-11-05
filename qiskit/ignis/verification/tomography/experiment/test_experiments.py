import unittest
from qiskit import Aer
from qiskit import QuantumCircuit
from experiment import StateTomographyExperiment, ProcessTomographyExperiment
from qiskit.quantum_info import state_fidelity, partial_trace, Statevector, Choi

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
        bell = QuantumCircuit(2,2)
        bell.h(0)
        bell.cx(0, 1)

        psi = partial_trace(Statevector.from_instruction(bell), [1])

        bell.measure(1,1)

        exp = StateTomographyExperiment(bell, [0])
        rho = exp.run(BACKEND)

        F_bell = state_fidelity(psi, rho, validate=False)
        self.assertAlmostEqual(F_bell, 1, places=1)

    def test_bell_full(self):
        bell = QuantumCircuit(2,2)
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
        circuit.cx(0,2)
        circuit.z(1)
        circuit.x(2)

        exp = StateTomographyExperiment(circuit) # full experiment - all 3 qubits
        exp.execute(BACKEND)
        rho_01 = exp.set_target_qubits([0,1]).run_analysis()
        rho_02 = exp.set_target_qubits([0,2]).run_analysis()
        rho_12 = exp.set_target_qubits([1,2]).run_analysis()

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

if __name__ == '__main__':
    unittest.main()