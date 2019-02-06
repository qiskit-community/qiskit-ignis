import unittest
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer
from qiskit.quantum_info import state_fidelity
import qiskit_ignis.tomography as tomo
import qiskit
import numpy

class TestStateTomography(unittest.TestCase):
    def run_circuit_and_tomography(self, circuit, qubits):
        job = qiskit.execute(circuit, Aer.get_backend('statevector_simulator'))
        psi = job.result().get_statevector(circuit)
        qst = tomo.state_tomography_circuits(circuit, qubits)
        job = qiskit.execute(qst, Aer.get_backend('qasm_simulator'), shots=5000)
        tomo_counts = tomo.tomography_data(job.result(), qst)
        probs, basis_matrix, weights = tomo.fitter_data(tomo_counts)
        rho_cvx = tomo.state_cvx_fit(probs, basis_matrix)
        rho_mle = tomo.state_mle_fit(probs, basis_matrix)
        return (rho_cvx, rho_mle, psi)

    def test_bell_2_qubits(self):
        q2 = QuantumRegister(2)
        bell = QuantumCircuit(q2)
        bell.h(q2[0])
        bell.cx(q2[0], q2[1])

        rho_cvx, rho_mle, psi = self.run_circuit_and_tomography(bell, q2)
        F_bell_cvx = state_fidelity(psi, rho_cvx)
        self.assertAlmostEqual(F_bell_cvx, 1, places = 1)
        F_bell_mle = state_fidelity(psi, rho_mle)
        self.assertAlmostEqual(F_bell_mle, 1, places=1)

    def test_bell_3_qubits(self):
        q3 = QuantumRegister(3)
        bell = QuantumCircuit(q3)
        bell.h(q3[0])
        bell.cx(q3[0], q3[1])
        bell.cx(q3[1], q3[2])

        rho_cvx, rho_mle, psi = self.run_circuit_and_tomography(bell, q3)
        F_bell_cvx = state_fidelity(psi, rho_cvx)
        self.assertAlmostEqual(F_bell_cvx, 1, places=1)
        F_bell_mle = state_fidelity(psi, rho_mle)
        self.assertAlmostEqual(F_bell_mle, 1, places=1)

    def test_complex_1_qubit_circuit(self):
        q = QuantumRegister(1)
        circ = QuantumCircuit(q)
        circ.u3(1, 1, 1, q[0])

        rho_cvx, rho_mle, psi = self.run_circuit_and_tomography(circ, q)
        F_bell_cvx = state_fidelity(psi, rho_cvx)
        self.assertAlmostEqual(F_bell_cvx, 1, places=1)
        F_bell_mle = state_fidelity(psi, rho_mle)
        self.assertAlmostEqual(F_bell_mle, 1, places=1)

    def test_complex_3_qubit_circuit(self):
        def rand_angles():
            return tuple(2 * numpy.pi * numpy.random.random(3) - numpy.pi)

        q = QuantumRegister(3)
        circ = QuantumCircuit(q)
        for j in range(3):
            circ.u3(*rand_angles(), q[j])

        rho_cvx, rho_mle, psi = self.run_circuit_and_tomography(circ, q)
        F_bell_cvx = state_fidelity(psi, rho_cvx)
        self.assertAlmostEqual(F_bell_cvx, 1, places=1)
        F_bell_mle = state_fidelity(psi, rho_mle)
        self.assertAlmostEqual(F_bell_mle, 1, places=1)

if __name__ == '__main__':
    unittest.main()
