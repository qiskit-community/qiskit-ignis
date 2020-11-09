from qiskit import QuantumCircuit
from qiskit.ignis.verification.tomography.basis import state_tomography_circuits
from .experiment import TomographyExperiment
from .generator import TomographyGenerator
from .analysis import TomographyAnalysis
from ast import literal_eval
from typing import List, Union, Tuple, Optional

class StateTomographyExperiment(TomographyExperiment):
    # pylint: disable=arguments-differ
    def __init__(self,
                 circuit: QuantumCircuit,
                 qubits: Union[int, List[int]] = None,
                 meas_basis: str = 'Pauli',
                 meas_labels: Union[str, Tuple[str], List[Tuple[str]]] = 'Pauli',
                 method: str = 'auto',
                 job: Optional = None):

        analysis = TomographyAnalysis(method=method,
                                      meas_basis=meas_basis,
                                      )
        generator = StateTomographyGenerator(circuit, qubits,
                                             meas_basis=meas_basis,
                                             meas_labels=meas_labels
                                             )

        super().__init__(generator=generator, analysis=analysis, job=job)


class StateTomographyGenerator(TomographyGenerator):
    def __init__(self,
                 circuit: QuantumCircuit,
                 qubits: Union[int, List[int]] = None,
                 meas_basis: str = 'Pauli',
                 meas_labels: Union[str, Tuple[str], List[Tuple[str]]] = 'Pauli'
                 ):
        super().__init__("state tomography", circuit, qubits)
        self._meas_basis = meas_basis
        self._circuits = state_tomography_circuits(circuit,
                                                   self._meas_qubits,
                                                   meas_basis=self._meas_basis,
                                                   meas_labels=meas_labels
                                                   )

    def _extra_metadata(self):
        metadata_list = super()._extra_metadata()
        for metadata in metadata_list:
            metadata['prep_label'] = None
            metadata['prep_basis'] = None
            metadata['meas_label'] = literal_eval(metadata['circuit_name'])
            metadata['meas_basis'] = self._meas_basis
        return metadata_list
