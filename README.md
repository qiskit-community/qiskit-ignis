# Qiskit Ignis

[![License](https://img.shields.io/github/license/Qiskit/qiskit-ignis.svg?style=popout-square)](https://opensource.org/licenses/Apache-2.0)[![Build Status](https://img.shields.io/travis/com/Qiskit/qiskit-ignis/master.svg?style=popout-square)](https://travis-ci.com/Qiskit/qiskit-ignis)[![](https://img.shields.io/github/release/Qiskit/qiskit-ignis.svg?style=popout-square)](https://github.com/Qiskit/qiskit-ignis/releases)[![](https://img.shields.io/pypi/dm/qiskit-ignis.svg?style=popout-square)](https://pypi.org/project/qiskit-ignis/)

**Qiskit** is an open-source framework for working with noisy intermediate-scale quantum computers (NISQ) at the level of pulses, circuits, and algorithms.

Qiskit is made up of elements that each work together to enable quantum computing. This element is **Ignis**, which provides tools for quantum hardware verification, noise characterization, and error correction.

## Installation

We encourage installing Qiskit via the PIP tool (a python package manager), which installs all Qiskit elements, including this one.

```bash
pip install qiskit
```

PIP will handle all dependencies automatically for us and you will always install the latest (and well-tested) version.

To install from source, follow the instructions in the [contribution guidelines](.github/CONTRIBUTING.rst).

## Creating your first quantum experiment with Qiskit Ignis
Now that you have Qiskit Ignis installed, you can start creating experiments, to reveal information about the device quality. Here is a basic example:

```
$ python
```

```python
# Import Qiskit classes
import qiskit
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.providers.aer import noise # import AER noise model

# Measurement error mitigation functions
from qiskit.ignis.error_mitigation.measurement import (complete_meas_cal,
                                                       CompleteMeasFitter, 
                                                       MeasurementFilter)

# Generate a noise model for the qubits
noise_model = noise.NoiseModel()
for qi in range(5):
    read_err = noise.errors.readout_error.ReadoutError([[0.75, 0.25],[0.1, 0.9]])
    noise_model.add_readout_error(read_err, [qi])

# Generate the measurement calibration circuits
# for running measurement error mitigation
qr = QuantumRegister(5)
meas_cals, state_labels = complete_meas_cal(qubit_list=[2,3,4], qr=qr)

# Execute the calibration circuits
backend = qiskit.Aer.get_backend('qasm_simulator')
job = qiskit.execute(meas_cals, backend=backend, shots=1000, noise_model=noise_model)
cal_results = job.result()

# Make a calibration matrix
meas_fitter = CompleteMeasFitter(cal_results, state_labels)

# Make a 3Q GHZ state
cr = ClassicalRegister(3)
ghz = QuantumCircuit(qr, cr)
ghz.h(qr[2])
ghz.cx(qr[2], qr[3])
ghz.cx(qr[3], qr[4])
ghz.measure(qr[2],cr[0])
ghz.measure(qr[3],cr[1])
ghz.measure(qr[4],cr[2])

# Execute the GHZ circuit (with the same noise model)
job = qiskit.execute(ghz, backend=backend, shots=1000, noise_model=noise_model)
results = job.result()

# Results without mitigation
raw_counts = results.get_counts()
print("Results without mitigation:", raw_counts)

# Create a measurement filter from the calibration matrix
meas_filter = meas_fitter.filter
# Apply the filter to the raw counts to mitigate 
# the measurement errors
mitigated_counts = meas_filter.apply(raw_counts)
print("Results with mitigation:", mitigated_counts)
```

```
Results without mitigation: {'000': 181, '001': 83, '010': 59, '011': 65, '100': 101, '101': 48, '110': 72, '111': 391}

Results with mitigation: {'000': 420.866934, '001': 2.1002, '011': 1.30314, '100': 53.0165, '110': 13.1834, '111': 509.5296}
```

## Contribution guidelines

## Contribution Guidelines

If you'd like to contribute to Qiskit Ignis, please take a look at our
[contribution guidelines](.github/CONTRIBUTING.md). This project adheres to Qiskit's [code of conduct](.github/CODE_OF_CONDUCT.md). By participating, you are expect to uphold to this code.

We use [GitHub issues](https://github.com/Qiskit/qiskit-ignis/issues) for tracking requests and bugs. Please use our [slack](https://qiskit.slack.com) for discussion and simple questions. To join our Slack community use the [link](https://join.slack.com/t/qiskit/shared_invite/enQtNDc2NjUzMjE4Mzc0LTMwZmE0YTM4ZThiNGJmODkzN2Y2NTNlMDIwYWNjYzA2ZmM1YTRlZGQ3OGM0NjcwMjZkZGE0MTA4MGQ1ZTVmYzk). For questions that are more suited for a forum we use the Qiskit tag in the [Stack Exchange](https://quantumcomputing.stackexchange.com/questions/tagged/qiskit).

## Next Steps

Now you're set up and ready to check out some of the other examples from our
[Qiskit Tutorials](https://github.com/Qiskit/qiskit-tutorials/tree/master/qiskit/ignis) repository.

## Authors and Citation

Qiskit Ignis is the work of [many people](https://github.com/Qiskit/qiskit-ignis/graphs/contributors) who contribute
to the project at different levels. If you use Qiskit, please cite as per the included [BibTeX file](https://github.com/Qiskit/qiskit/blob/master/Qiskit.bib).

## License

[Apache License 2.0](LICENSE.txt)
