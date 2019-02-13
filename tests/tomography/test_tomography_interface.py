import unittest
import qiskit_ignis.tomography as tomo
from qiskit import Aer, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import state_fidelity
from qiskit.tools.qi.qi import outer
import qiskit

class TestTomographyInterface(unittest.TestCase):
    def run_circuit_and_state_tomography(self, circuit, qubits):
        job = qiskit.execute(circuit, Aer.get_backend('statevector_simulator'))
        psi = job.result().get_statevector(circuit)
        rho = tomo.perform_state_tomography(circuit, qubits)
        return (rho, psi)

    def run_circuit_and_process_tomography(self, circuit, qubits):
        job = qiskit.execute(circuit, Aer.get_backend('unitary_simulator'))
        ideal_unitary = job.result().get_unitary(circuit)
        choi_ideal = outer(ideal_unitary.ravel(order='F'))
        choi = tomo.perform_process_tomography(circuit, qubits)
        return (choi, choi_ideal)

    def test_basic_state_tomography(self):
        q3 = QuantumRegister(3)
        bell = QuantumCircuit(q3)
        bell.h(q3[0])
        bell.cx(q3[0], q3[1])
        bell.cx(q3[1], q3[2])

        rho, psi = self.run_circuit_and_state_tomography(bell, q3)
        F_bell = state_fidelity(psi, rho)
        self.assertAlmostEqual(F_bell, 1, places=1)

    def test_basic_process_tomography(self):
        q = QuantumRegister(2)
        circ = QuantumCircuit(q)
        circ.h(q[0])
        circ.cx(q[0], q[1])

        choi, choi_ideal = self.run_circuit_and_process_tomography(circ, q)
        fidelity = state_fidelity(choi / 4, choi_ideal / 4)
        self.assertAlmostEqual(fidelity, 1, places=1)

if __name__ == '__main__':
    unittest.main()