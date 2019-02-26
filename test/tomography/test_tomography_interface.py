import unittest
import qiskit.ignis.verification.tomography as tomo
from qiskit import Aer, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import state_fidelity
from qiskit.tools.qi.qi import outer
import qiskit
import numpy

class TestTomographyInterface(unittest.TestCase):
    def assertListAlmostEqual(self, lhs, rhs, places = None):
        self.assertEqual(len(lhs), len(rhs), msg = "List lengths differ: {} != {}".format(len(lhs), len(rhs)))
        for i in range(len(lhs)):
            if isinstance(lhs[i], numpy.ndarray) and isinstance(rhs[i], numpy.ndarray):
                self.assertMatricesAlmostEqual(lhs[i], rhs[i], places = places)
            else:
                self.assertAlmostEqual(lhs[i], rhs[i], places = places)

    def test_basic_state_tomography(self):
        q3 = QuantumRegister(3)
        bell = QuantumCircuit(q3)
        bell.h(q3[0])
        bell.cx(q3[0], q3[1])
        bell.cx(q3[1], q3[2])

        job = qiskit.execute(bell, Aer.get_backend('statevector_simulator'))
        psi = job.result().get_statevector(bell)
        rho = tomo.perform_state_tomography(bell, q3, ideal=False, fidelity=False)

        F_bell = state_fidelity(psi, rho)
        self.assertAlmostEqual(F_bell, 1, places=1)

    def test_state_tomography_ideal_data(self):
        q3 = QuantumRegister(3)
        bell = QuantumCircuit(q3)
        bell.h(q3[0])
        bell.cx(q3[0], q3[1])
        bell.cx(q3[1], q3[2])

        job = qiskit.execute(bell, Aer.get_backend('statevector_simulator'))
        psi = job.result().get_statevector(bell)

        tomography_results = tomo.perform_state_tomography(bell, q3)
        rho = tomography_results['rho']
        ideal_psi = tomography_results['ideal_psi']
        fidelity = tomography_results['fidelity']

        F_bell = state_fidelity(psi, rho)
        self.assertEqual(F_bell, fidelity)
        self.assertListAlmostEqual(psi, ideal_psi)


    def test_basic_process_tomography(self):
        q = QuantumRegister(2)
        circ = QuantumCircuit(q)
        circ.h(q[0])
        circ.cx(q[0], q[1])

        job = qiskit.execute(circ, Aer.get_backend('unitary_simulator'))
        ideal_unitary = job.result().get_unitary(circ)
        choi_ideal = outer(ideal_unitary.ravel(order='F'))
        choi = tomo.perform_process_tomography(circ, q, ideal=False, fidelity=False)

        fidelity = state_fidelity(choi / 4, choi_ideal / 4)
        self.assertAlmostEqual(fidelity, 1, places=1)

if __name__ == '__main__':
    unittest.main()