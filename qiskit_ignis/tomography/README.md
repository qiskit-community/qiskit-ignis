# Quantum Tomography

**Author:** Christopher J. Wood (cjwood@us.ibm.com)
**Date Started:** May 14, 2018.

### Contents

* Example tomography notebooks
* Quantum tomography python module
  * Quantum state and process tomography circuit generation
    * Measurement in Pauli basis *X, Y, Z*
    * *Default* preparation in 4-state Pauli basis: *Zp, Zm, Xp, Yp*
    * *Optional* preparation in symmetric informationally complete basis
  * Quantum state and process tomography reconstruction via maximum-likelihood least-squares fitting (MLE)
  * Quantum state and process tomography reconstruction via semidefinite program convex optimization (SDP)
  * Helper functions for processing count data into fitter data including
    * Calculation of weights assuming Gaussian statistics

### Using

If the root directory of the repository is added to the python path, import with

```python
import tomography as tomo
```

#### State Tomography

Given a state preparation quantum circuit `circ`, and a qubit register `qr` state tomography of qubits `qr` may be performed using

```python
# Genereate and execute tomography circuits
tomo_circuits = tomo.state_tomography_circuits(circ, qr)
result = qiskit.execute(qst_bell, 'local_qasm_simulator').result()

# Extract tomography count and basis data
tomo_data = tomo.tomography_data(result, prep)

# Generate fitter data and reconstruct density matrix
fitter_data, fitter_basis = tomo.fitter_data(tomo_data)
rho_fit = tomo.cvx_fit(fitter_data, fitter_basis, trace=1)
```

See the [State Tomography](./examples/state-tomography.ipynb) example notebook for more details.

Similarly if we wanted to perform process tomography of the circuit itself we can use the following:

```python
# Genereate and execute tomography circuits
tomo_circuits = tomo.process_tomography_circuits(circ, qr)
result = qiskit.execute(qst_bell, 'local_qasm_simulator').result()

# Extract tomography count and basis data
tomo_data = tomo.tomography_data(result, prep)

# Generate fitter data and reconstruct density matrix
fitter_data, fitter_basis = tomo.fitter_data(tomo_data)
choi_fit = tomo.cvx_fit(fitter_data, fitter_basis, trace=2)
```

See the [Process Tomography](./examples/process-tomography.ipynb) example notebook for more details.

### Requirements

This module requires *QISKit > 0.5.0*. Install using `pip install qiskit`.

Import from repository root directory with

```python
import tomography as tomo
```

#### Additional Requirements for SDP reconstruction

The SDP tomographic reconstruction requires the installation of two additional python packages: [CVXPY](http://www.cvxpy.org/en/latest/index.html) (version 0.4.11) and [CVXOPT](http://cvxopt.org/index.html). These should be installed into the same environment as QISkit using Anaconda or PIP.

**Installing with Anaconda**
```bash
conda install -c conda-forge cvxopt
conda install -c cvxgrp cvxpy libgcc
```

**Installing with PIP**
```bash
pip install cvxopt
pip install cvxpy
```

For more details see the relevant documentation for [CVXPY](http://www.cvxpy.org/en/latest/install/#) and [CVXOPT](http://cvxopt.org/install/index.html).