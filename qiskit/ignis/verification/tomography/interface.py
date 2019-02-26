import qiskit
from qiskit import Aer
from qiskit.quantum_info import state_fidelity
from qiskit.tools.qi.qi import outer
from .basis.circuits import state_tomography_circuits, process_tomography_circuits
from .data import tomography_data
from qiskit_ignis.tomography.fitters import fitter_data, state_cvx_fit, process_cvx_fit, state_mle_fit, process_mle_fit


def perform_state_tomography(circuit, measured_qubits,
                             backend = None,
                             fitter = 'cvx',
                             ideal = True,
                             fidelity = True,
                             shots = 5000,
                             ):
    if backend is None:
        backend = Aer.get_backend('qasm_simulator')
    tomography_circuits = state_tomography_circuits(circuit, measured_qubits)
    job = qiskit.execute(tomography_circuits, backend, shots=shots)
    tomography_data_results = tomography_data(job.result(), tomography_circuits)
    data, basis, weights = fitter_data(tomography_data_results)
    rho = None
    if fitter == 'cvx':
        try:
            rho = state_cvx_fit(data, basis)
        except Exception as e:
            print("CVX run failed: {}, attempting MLE fit".format(e))
    if fitter == 'mle' or rho is None:
        rho = state_mle_fit(data, basis)

    if ideal == False and fidelity == False:
        return rho

    result = {'rho': rho}
    if ideal == True:
        job = qiskit.execute(circuit, Aer.get_backend('statevector_simulator'))
        ideal_psi = job.result().get_statevector(circuit)
        result['ideal_psi'] = ideal_psi
        if fidelity == True:
            fidelity_value = state_fidelity(ideal_psi, rho)
            result['fidelity'] = fidelity_value

    return result

def perform_process_tomography(circuit, measured_qubits,
                             backend = None,
                             fitter = 'cvx',
                             ideal = True,
                             fidelity = True,
                             shots = 4000,):
    if backend is None:
        backend = Aer.get_backend('qasm_simulator')
    tomography_circuits = process_tomography_circuits(circuit, measured_qubits)
    job = qiskit.execute(tomography_circuits, backend, shots=shots)
    tomography_data_results = tomography_data(job.result(), tomography_circuits)
    data, basis, weights = fitter_data(tomography_data_results)
    choi = None
    if fitter == 'cvx':
        try:
            choi = process_cvx_fit(data, basis)
        except Exception as e:
            print("CVX run failed: {}, attempting MLE fit".format(e))
    if fitter == 'mle' or choi is None:
        choi = process_mle_fit(data, basis)

    if ideal == False and fidelity == False:
        return choi

    result = {'choi': choi}
    if ideal == True:
        job = qiskit.execute(circuit, Aer.get_backend('unitary_simulator'))
        ideal_unitary = job.result().get_unitary(circuit)
        ideal_choi = outer(ideal_unitary.ravel(order='F'))
        result['ideal_choi'] = ideal_choi
        if fidelity == True:
            n = len(measured_qubits)
            fidelity_value = state_fidelity(ideal_choi / 2**n, choi / 2**n)
            result['fidelity'] = fidelity_value
    return result

