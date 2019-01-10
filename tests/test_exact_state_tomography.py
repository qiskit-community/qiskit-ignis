import unittest
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer
from qiskit.quantum_info import state_fidelity
import qiskit_ignis.tomography as tomo
import qiskit
import numpy

class TestExactStateTomography(unittest.TestCase):
    def run_circuit_and_tomography(self, circuit, qubits):
        job = qiskit.execute(circuit, Aer.get_backend('statevector_simulator'))
        psi = job.result().get_statevector(circuit)
        qst = tomo.state_tomography_circuits(circuit, qubits)
        job = qiskit.execute(qst, Aer.get_backend('qasm_simulator'), shots=5000)
        tomo_counts_bell = tomo.tomography_data(job.result(), qst)
        rho = tomo.fitters.exact_state_tomography(tomo_counts_bell)
        return (rho, psi)

    def test_bell_2_qubits(self):
        q2 = QuantumRegister(2)
        bell = QuantumCircuit(q2)
        bell.h(q2[0])
        bell.cx(q2[0], q2[1])

        rho, psi = self.run_circuit_and_tomography(bell, q2)
        F_bell = state_fidelity(psi, rho)
        self.assertAlmostEqual(F_bell, 1)

    def test_bell_3_qubits(self):
        q3 = QuantumRegister(3)
        bell = QuantumCircuit(q3)
        bell.h(q3[0])
        bell.cx(q3[0], q3[1])
        bell.cx(q3[1], q3[2])

        rho, psi = self.run_circuit_and_tomography(bell, q3)
        F_bell = state_fidelity(psi, rho)
        self.assertAlmostEqual(F_bell, 1)

if __name__ == '__main__':
    unittest.main()
