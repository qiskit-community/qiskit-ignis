# Qiskit Ignis (_DEPRECATED_)

[![License](https://img.shields.io/github/license/Qiskit/qiskit-ignis.svg?style=popout-square)](https://opensource.org/licenses/Apache-2.0)[![Build Status](https://img.shields.io/travis/com/Qiskit/qiskit-ignis/master.svg?style=popout-square)](https://travis-ci.com/Qiskit/qiskit-ignis)[![](https://img.shields.io/github/release/Qiskit/qiskit-ignis.svg?style=popout-square)](https://github.com/Qiskit/qiskit-ignis/releases)[![](https://img.shields.io/pypi/dm/qiskit-ignis.svg?style=popout-square)](https://pypi.org/project/qiskit-ignis/)

**_NOTE_** _As of the version 0.7.0 Qiskit Ignis is deprecated and has been
superseded by the
[Qiskit Experiments](https://github.com/Qiskit/qiskit-experiments) project.
Active development on the project has stopped and only compatibility fixes
and other critical bugfixes will be accepted until the project is officially
retired and archived._

**Qiskit** is an open-source framework for working with noisy quantum computers at the level of pulses, circuits, and algorithms.

Qiskit is made up of elements that each work together to enable quantum computing. This element is **Ignis**, which provides tools for quantum hardware verification, noise characterization, and error correction.

## Migration Guide

As of version 0.7.0, Qiskit Ignis has been deprecated and some of its functionality 
was migrated into the `qiskit-experiments` package and into `qiskit-terra`.

* Ignis characterization module

  * This module was partly migrated to [`qiskit-experiments`](https://github.com/Qiskit/qiskit-experiments) and split into two different modules:
  `qiskit_experiments.library.calibration`
  `qiskit_experiments.library.characterization`
  * `AmpCal` is now replaced by `FineAmplitude`.
  * `ZZFitter` was not migrated yet.
  
* Ignis discriminator module

  * This module is in the process of migration to [`qiskit-experiments`](https://github.com/Qiskit/qiskit-experiments)

* Ignis mitigation module

  * The readout mitigator is in [`qiskit-terra`](https://github.com/Qiskit/qiskit-terra): [`qiskit.utils.mitigation`](https://qiskit.org/documentation/apidoc/utils_mitigation.html).
  * Experiments for generating the readout mitigators is in  [`qiskit-experiments`](https://github.com/Qiskit/qiskit-experiments): [local readout error characterization experiment](https://qiskit.org/documentation/experiments/stubs/qiskit_experiments.library.characterization.LocalReadoutError.html) and [correlated readout error characterization experiment](https://qiskit.org/documentation/experiments/stubs/qiskit_experiments.library.characterization.CorrelatedReadoutError.html) 
  * For use of mitigators with `qiskit.algorithms` and the [`QuantumInstance` class](https://qiskit.org/documentation/stubs/qiskit.utils.QuantumInstance.html?highlight=quantuminstance#qiskit.utils.QuantumInstance)
    this has been integrated into `qiskit-terra` directly with the `QuantumInstance`.
  
* Ignis verification module

  * Randomized benchmarking, Quantum Volume and State and Process Tomography were migrated to [`qiskit-experiments`](https://github.com/Qiskit/qiskit-experiments).
  * Migration of Gate-set tomography to [`qiskit-experiments`](https://github.com/Qiskit/qiskit-experiments) is in progress.
  * `topological_codes` will continue development under [NCCR-SPIN](https://github.com/NCCR-SPIN/topological_codes/blob/master/README.md), while the functionality is reintegrated into Qiskit. Some additional functionality can also be found in the offshoot project [qtcodes](https://github.com/yaleqc/qtcodes).
  * Currently the Accredition and Entanglement modules have not been migrated.  
The following table gives a more detailed breakdown that relates the function, as it existed in Ignis, 
to where it now lives after this move.

| Old | New | Library |
| :---: | :---: | :---: |
| qiskit.ignis.characterization.calibrations | qiskit_experiments.library.calibration | qiskit-experiments |
| qiskit.ignis.characterization.coherence | qiskit_experiments.library.characterization | qiskit-experiments |
| qiskit.ignis.mitigation | qiskit_terra.mitigation | qiskit-terra |
| qiskit.ignis.verification.quantum_volume | qiskit_experiments.library.quantum_volume | qiskit-experiments |
| qiskit.ignis.verification.randomized_benchmarking | qiskit_experiments.library.randomized_benchmarking | qiskit-experiments |
| qiskit.ignis.verification.tomography | qiskit_experiments.library.tomography | qiskit-experiments |

## Installation

We encourage installing Qiskit via the pip tool (a python package manager). The following command installs the core Qiskit components, including Ignis.

```bash
pip install qiskit
```

Pip will handle all dependencies automatically for us and you will always install the latest (and well-tested) version.

To install from source, follow the instructions in the [contribution guidelines](./CONTRIBUTING.md).

### Extra Requirements

Some functionality has extra optional requirements. If you're going to use any
visualization functions for fitters you'll need to install matplotlib. You
can do this with `pip install matplotlib` or when you install ignis with
`pip install qiskit-ignis[visualization]`. If you're going to use a cvx fitter
for running tomogography you'll need to install cvxpy. You can do this with
`pip install cvxpy` or when you install ignis with
`pip install qiskit-ignis[cvx]`. When performing expectation value measurement
error mitigation using the CTMP method performance can be improved using
just-in-time compiling if Numbda is installed. You can do this with
`pip install numba` or when you install ignis with
`pip install qiskit-ignis[jit]`. For using the discriminator classes in
`qiskit.ignis.measurement` scikit-learn needs to be installed. You can do this with
`pip install scikit-learn` or when you install ignis with
`pip install qiskit-ignis[iq]`. If you want to install all extra requirements
when you install ignis you can run `pip install qiskit-ignis[visualization,cvx,jit,iq]`.

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
from qiskit.ignis.mitigation.measurement import (complete_meas_cal,
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
print("Results with mitigation:", {l:int(mitigated_counts[l]) for l in mitigated_counts})
```

```
Results without mitigation: {'000': 181, '001': 83, '010': 59, '011': 65, '100': 101, '101': 48, '110': 72, '111': 391}

Results with mitigation: {'000': 421, '001': 2, '011': 1, '100': 53, '110': 13, '111': 510}
```

## Contribution Guidelines

If you'd like to contribute to Qiskit Ignis, please take a look at our
[contribution guidelines](./CONTRIBUTING.md). This project adheres to Qiskit's [code of conduct](./CODE_OF_CONDUCT.md). By participating, you are expect to uphold to this code.

We use [GitHub issues](https://github.com/Qiskit/qiskit-ignis/issues) for tracking requests and bugs. Please use our [slack](https://qiskit.slack.com) for discussion and simple questions. To join our Slack community use the [link](https://join.slack.com/t/qiskit/shared_invite/enQtNDc2NjUzMjE4Mzc0LTMwZmE0YTM4ZThiNGJmODkzN2Y2NTNlMDIwYWNjYzA2ZmM1YTRlZGQ3OGM0NjcwMjZkZGE0MTA4MGQ1ZTVmYzk). For questions that are more suited for a forum we use the Qiskit tag in the [Stack Exchange](https://quantumcomputing.stackexchange.com/questions/tagged/qiskit).

## Next Steps

Now you're set up and ready to check out some of the other examples from our
[Qiskit Tutorials](https://github.com/Qiskit/qiskit-iqx-tutorials/tree/master/qiskit/advanced/ignis) repository.

## Authors and Citation

Qiskit Ignis is the work of [many people](https://github.com/Qiskit/qiskit-ignis/graphs/contributors) who contribute
to the project at different levels. If you use Qiskit, please cite as per the included [BibTeX file](https://github.com/Qiskit/qiskit/blob/master/Qiskit.bib).

## License

[Apache License 2.0](LICENSE.txt)
