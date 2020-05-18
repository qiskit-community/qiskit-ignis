import qiskit
from qiskit.quantum_info.operators import Operator
import numpy as np

qr = qiskit.QuantumRegister(3, 'qr')
cr = qiskit.ClassicalRegister(3, 'cr')
circ = qiskit.QuantumCircuit(qr, cr)

circ.ccx(qr[0], qr[1], qr[2])
print ("circuit with toffoli gate:")
print (circ)

print ("unitary matrix (up to a global phase):")
backend = qiskit.Aer.get_backend('unitary_simulator')
basis_gates = ['u1','u2','u3','cx'] # use U,CX for now
job = qiskit.execute(circ, backend=backend, basis_gates=basis_gates)
print(np.around(job.result().get_unitary(),3))

qr = qiskit.QuantumRegister(3, 'qr')
cr = qiskit.ClassicalRegister(3, 'cr')
circ = qiskit.QuantumCircuit(qr, cr)

cs_op = Operator([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1j]])

# csdg = cs*cs*cs
csdg_op = Operator([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, -1j]])

circ.h(qr[2])
circ.unitary(cs_op, [qr[0], qr[2]], label='cs')
circ.cx(qr[0], qr[1])
circ.unitary(csdg_op, [qr[1], qr[2]], label='csdg')
circ.cx(qr[0], qr[1])
circ.unitary(cs_op, [qr[1], qr[2]], label='cs')
circ.h(qr[2])

print ("circuit with cs gates:")
print (circ)
circ.draw(output='mpl')

print ("unitary matrix (up to a global phase):")
backend = qiskit.Aer.get_backend('unitary_simulator')
basis_gates = ['u1','u2','u3','cx'] # use U,CX for now
job = qiskit.execute(circ, backend=backend, basis_gates=basis_gates)
print(np.around(job.result().get_unitary(),3))


qr = qiskit.QuantumRegister(1, 'qr')
cr = qiskit.ClassicalRegister(1, 'cr')
circ = qiskit.QuantumCircuit(qr, cr)

circ.rz(np.pi/2, qr[0])
circ.rx(np.pi/2, qr[0])
circ.rz(np.pi/2, qr[0])
print ("circuit with toffoli gate:")
print (circ)

print ("unitary matrix (up to a global phase):")
backend = qiskit.Aer.get_backend('unitary_simulator')
basis_gates = ['u1','u2','u3','cx'] # use U,CX for now
job = qiskit.execute(circ, backend=backend, basis_gates=basis_gates)
print(np.around(job.result().get_unitary(),3))
