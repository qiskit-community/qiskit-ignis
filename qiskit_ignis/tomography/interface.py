import qiskit
from qiskit import Aer
from .basis.circuits import state_tomography_circuits, process_tomography_circuits
from .data import tomography_data
from .fitters import fitter_data, state_cvx_fit, process_cvx_fit

def perform_state_tomography(circuit, measured_qubits, shots = 5000):
    qasm_simulator = Aer.get_backend('qasm_simulator')
    tomography_circuits = state_tomography_circuits(circuit, measured_qubits)
    job = qiskit.execute(tomography_circuits, qasm_simulator, shots=shots)
    tomography_data_results = tomography_data(job.result(), tomography_circuits)
    data, basis, weights = fitter_data(tomography_data_results)
    rho = state_cvx_fit(data, basis)
    return rho

def perform_process_tomography(circuit, measured_qubits, shots = 4000):
    qasm_simulator = Aer.get_backend('qasm_simulator')
    tomography_circuits = process_tomography_circuits(circuit, measured_qubits)
    job = qiskit.execute(tomography_circuits, qasm_simulator, shots=shots)
    tomography_data_results = tomography_data(job.result(), tomography_circuits)
    data, basis, weights = fitter_data(tomography_data_results)
    choi = process_cvx_fit(data, basis)
    return choi