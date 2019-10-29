# Needed for functions
import numpy as np

# Import QISKit classes
from qiskit import QuantumCircuit
from .tomographybasis import TomographyBasis


class GateSetBasis:
    def __init__(self, name, gates, spam):
        self.name = name
        self.gate_labels = gates[0]
        self.gate_func = gates[1]
        self.spam_labels = spam[0]
        self.spam_spec = spam[1]

    def add_to_circuit(self, circ, qubit, op):
        if op not in self.spam_spec:
            raise RuntimeError("{} is not a SPAM circuit".format(op))
        op_gates = self.spam_spec[op]
        for gate in op_gates:
            self.gate_func(circ, qubit, gate)


    def measurement_circuit(self, op, qubit, clbit):
        circ = QuantumCircuit(qubit.register, clbit.register)
        self.add_to_circuit(circ, qubit, op)
        circ.measure(qubit, clbit)
        return circ

    def measurement_matrix(self):
        pass

    def preparation_circuit(self, op, qubit):
        circ = QuantumCircuit(qubit.register)
        self.add_to_circuit(circ, qubit, op)
        return circ

    def preparation_matrix(self):
        pass

    def get_tomography_basis(self):
        return TomographyBasis(self.name,
                                 measurement=(self.spam_labels,
                                              self.measurement_circuit,
                                              self.measurement_matrix),
                                 preparation=(self.spam_labels,
                                              self.preparation_circuit,
                                              self.preparation_matrix))


def standard_gates_func(circ, qubit, op):
    if op == 'Id':
        pass
    if op == 'X_Rot_90':
        circ.u3(np.pi / 2, -np.pi / 2, np.pi / 2, qubit)
    if op == 'Y_Rot_90':
        circ.u3(np.pi / 2, np.pi, np.pi, qubit)

standard_gateset_basis = GateSetBasis('Standard GST',
                                      (('Id', 'X_Rot_90', 'Y_Rot_90'), standard_gates_func),
                                      (('F0', 'F1', 'F2', 'F3'),
                                       {'F0': ('Id',), 'F1': ('X_Rot_90',), 'F2': ('Y_Rot_90',), 'F3': ('X_Rot_90','X_Rot_90')},
                                       )
                        )
GatesetTomographyBasis = standard_gateset_basis.get_tomography_basis()