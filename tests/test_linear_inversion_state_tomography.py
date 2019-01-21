import unittest
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer
from qiskit.quantum_info import state_fidelity
import qiskit_ignis.tomography as tomo
import qiskit
import numpy
import itertools

class TestLinearInversionStateTomography(unittest.TestCase):

    #this section is dedicated to generating tomography data as precise as possible
    #(as opposed to the usual tomography data which highly lacks in precision due to small number of trials)
    PX0 = 0.5 * numpy.array([[1, 1], [1, 1]])
    PX1 = 0.5 * numpy.array([[1, -1], [-1, 1]])

    PY0 = 0.5 * numpy.array([[1, -1j], [1j, 1]])
    PY1 = 0.5 * numpy.array([[1, 1j], [-1j, 1]])

    PZ0 = numpy.array([[1, 0], [0, 0]])
    PZ1 = numpy.array([[0, 0], [0, 1]])

    projectors = {'X': (PX0, PX1), 'Y': (PY0, PY1), 'Z': (PZ0, PZ1)}

    def prob_for(self, psi, op):
        return ((numpy.conj(psi).T @ op) @ psi).real

    def measurement_probs(self, ops, psi):
        n = len(ops)
        results = {}
        for v in itertools.product([0, 1], repeat=n):
            projector = numpy.array([[1]])
            for b, op in zip(v, ops):
                projector = numpy.kron(projector, self.projectors[op][b])
            v_string = "".join([str(x) for x in v])
            results[v_string] = self.prob_for(psi, projector)
        return results

    def generate_probs(self, op_names, psi):
        qubits_num = int(numpy.math.log2(len(psi)))
        probs = {}
        for ops in itertools.product(op_names, repeat=qubits_num):
            probs[ops] = self.measurement_probs(ops, psi)
        return probs

    def run_circuit_and_tomography(self, circuit, qubits):
        job = qiskit.execute(circuit, Aer.get_backend('statevector_simulator'))
        psi = job.result().get_statevector(circuit)
        qst = tomo.state_tomography_circuits(circuit, qubits)
        job = qiskit.execute(qst, Aer.get_backend('qasm_simulator'), shots=5000)
        tomo_counts = tomo.tomography_data(job.result(), qst)
        rho = tomo.fitters.linear_inversion_state_tomography(tomo_counts)
        tomo_probs = self.generate_probs(['X', 'Y', 'Z'], psi)
        rho_probs = tomo.fitters.linear_inversion_state_tomography(tomo_probs)
        return (rho, rho_probs, psi)

    def test_bell_2_qubits(self):
        q2 = QuantumRegister(2)
        bell = QuantumCircuit(q2)
        bell.h(q2[0])
        bell.cx(q2[0], q2[1])

        rho, rho_probs, psi = self.run_circuit_and_tomography(bell, q2)
        F_bell = state_fidelity(psi, rho)
        self.assertAlmostEqual(F_bell, 1)

    def test_bell_3_qubits(self):
        q3 = QuantumRegister(3)
        bell = QuantumCircuit(q3)
        bell.h(q3[0])
        bell.cx(q3[0], q3[1])
        bell.cx(q3[1], q3[2])

        rho, rho_probs, psi = self.run_circuit_and_tomography(bell, q3)
        F_bell = state_fidelity(psi, rho)
        self.assertAlmostEqual(F_bell, 1)

    def test_complex_1_qubit_circuit(self):
        q = QuantumRegister(1)
        circ = QuantumCircuit(q)
        circ.u3(1, 1, 1, q[0])

        rho, rho_probs, psi = self.run_circuit_and_tomography(circ, q)
        F_rho = state_fidelity(psi, rho)
        F_rho_probs = state_fidelity(psi, rho_probs)
        self.assertNotAlmostEqual(F_rho, 1) #we explicitly expect the limited-trials state tomography to fail
        self.assertAlmostEqual(F_rho_probs, 1) #with "infinite" precision it should still succeed

    def test_complex_3_qubit_circuit(self):
        def rand_angles():
            return tuple(2 * numpy.pi * numpy.random.random(3) - numpy.pi)

        q = QuantumRegister(3)
        circ = QuantumCircuit(q)
        for j in range(3):
            circ.u3(*rand_angles(), q[j])

        rho, rho_probs, psi = self.run_circuit_and_tomography(circ, q)
        F_rho = state_fidelity(psi, rho)
        F_rho_probs = state_fidelity(psi, rho_probs)
        self.assertNotAlmostEqual(F_rho, 1)  # we explicitly expect the limited-trials state tomography to fail
        self.assertAlmostEqual(F_rho_probs, 1)  # with "infinite" precision it should still succeed

if __name__ == '__main__':
    unittest.main()
