def perform_state_tomography(circuit, measured_qubits, shots = 5000):
    import qiskit
    from qiskit_aer import Aer
    from .basis.circuits import state_tomography_circuits
    from .data import tomography_data
    from .fitters import fitter_data, cvx_fit

    qasm_simulator = Aer.get_backend('qasm_simulator')
    tomography_circuits = state_tomography_circuits(circuit, measured_qubits)
    job = qiskit.execute(tomography_circuits, qasm_simulator, shots=shots)
    tomography_data_results = tomography_data(job.result(), tomography_circuits)
    data, basis = fitter_data(tomography_data_results)
    rho = cvx_fit(data, basis, trace=1)
    return rho