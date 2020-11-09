from qiskit import QuantumCircuit
from qiskit.ignis.experiments.base import Generator
from typing import List, Dict, Union


class TomographyGenerator(Generator):
    def __init__(self,
                 name: str,
                 circuit: QuantumCircuit,
                 meas_qubits: Union[int, List[int]] = None,
                 prep_qubits: Union[int, List[int]] = None,
                 meas_clbits: Union[int, List[int]] = None
                 ):
        super().__init__(name, circuit.num_qubits)
        self._meas_qubits = meas_qubits if meas_qubits else list(range(circuit.num_qubits))
        self._prep_qubits = prep_qubits if prep_qubits is not None else self._meas_qubits
        self._meas_clbits = meas_clbits if meas_clbits is not None else self._meas_qubits
        self._circuits = None # should be initialized by the derived class

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of experiment circuits."""
        return self._circuits

    def _extra_metadata(self) -> List[Dict[str, any]]:
        """Generate a list of experiment metadata dicts."""
        return [{
            'circuit_name': circ.name,
            'meas_qubits': self._meas_qubits,
            'prep_qubits': self._prep_qubits,
            'meas_clbits': self._meas_clbits
        }
            for circ in self._circuits]
