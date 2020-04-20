# Changelog


All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

  **Types of changes:**

  - **Added**: for new features.
  - **Changed**: for changes in existing functionality.
  - **Deprecated**: for soon-to-be removed features.
  - **Removed**: for now removed features.
  - **Fixed**: for any bug fixes.
  - **Security**: in case of vulnerabilities.


## [UNRELEASED]

### Added
- API documentation (\#345, \#346, \#347, \#348, \#353)
- CNOT-Dihedral randomized benchmarking (\#296)
- Accreditation module for output accrediation of noisy devices (\#252, \#325, \#329)
- Pulse calibrations for single qubits (\#292, \#302, \#303, \#304)
- Pulse Discriminator (\#238, \#278, \#297, \#316)
- Entanglement verification circuits (\#328)
- Gateset tomography for single-qubit gate sets (\#330)
- Adds randomized benchmarking utility functions `calculate_1q_epg`, `calculate_2q_epg` functions to calculate 1 and 2-qubit error per gate from error per Clifford (\#335)
- Adds randomized benchmarking utility functions `calculate_1q_epc`, `calculate_2q_epc` for calculating 1 and 2-qubit error per Clifford from error per gate (\#368)

### Changed
- Support integer labels for qubits in tomography (\# 359)
- Support integer labels for measurement error mitigation (\# 359)

### Deprecated

- Deprecates `twoQ_clifford_error` function. Use `calculate_2q_epc` instead.
- Python 3.5 support in qiskit-ignis is deprecated. Support will be
  removed on the upstream python community's end of life date for the version,
  which is 09/13/2020.


## [0.2.0](https://github.com/Qiskit/qiskit/compare/0.1.1...0.2.0)- 2019-08-22

### Added

- Logging Module (\#153)
- Purity RB (\#218)
- Interleaved RB (\#174)
- Repetition Code for Verification (\#210)

### Changed

- Apply measurement mitigation in parallel when applied to multiple results (\#240)
- Add multiple results to measurement mitigation (\#240)
- Fixed bug in RB fit error
- Updates for Terra Qubit class (\#200)
- Added the ability to add arbitrary seeds to RB (not just in order) (\#208)
- Fix bug in the characterization fitter when selecting a qubit index to fit
- Improved guess values for RB fitters and enabled the user to input their own guess values

### Removed

## [0.1.1] - 2019-05-02

### Added

- Tensored Measurement Mitigation
- Align cliffs option to RB
- Quantum Volume
- Subset measurement mitigation

## [0.1.0] - 2019-03-04

### Added

- Initial public release.


[UNRELEASED]: TBD
[0.1.0]: March 4, 2019
