# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""
Quantum tomography circuit generation.
"""

import logging
import itertools as it

from qiskit import QuantumRegister
from qiskit import ClassicalRegister
from qiskit import QuantumCircuit
from qiskit import QiskitError
from qiskit.circuit.measure import Measure
from qiskit.circuit.reset import Reset

from .tomographybasis import TomographyBasis
from .paulibasis import PauliBasis
from .sicbasis import SICBasis

# Create logger
logger = logging.getLogger(__name__)


# TODO: Update docstrings

###########################################################################
# State tomography circuits for measurement in Pauli basis
###########################################################################

def state_tomography_circuits(circuit, measured_qubits,
                              meas_labels='Pauli', meas_basis='Pauli',):
    """
    Return a list of quantum state tomography circuits.

    This performs measurement in the Pauli-basis resulting in 3 ** n
    circuits for an n-qubit state tomography experiment.

    Args:
        circuit (QuantumCircuit): the state preparation circuit to be
                                  tomographed.
        measured_qubits (QuantumRegister): the qubits to be measured.
            This can also be a list of whole QuantumRegisters or
            individual QuantumRegister qubit tuples.
        meas_labels (str, tuple, list(tuple)): The measurement operator
            labels. See additional information for details (Default: 'Pauli').
        meas_basis (str, TomographyBasis): The measurement basis.
            See additional information for details (Default: 'Pauli').

    Returns:
        A list of QuantumCircuit objects containing the original circuit
        with state tomography measurements appended at the end.

    Additional Information:
        The returned circuits are named by the measurement basis.

        To perform tomography measurement in a custom basis, or to generate
        a subset of state tomography circuits for a partial tomography
        experiment use the general function `tomography_circuits`.
    """
    return _tomography_circuits(circuit, measured_qubits, None,
                                meas_labels=meas_labels, meas_basis=meas_basis,
                                prep_labels=None, prep_basis=None)


###########################################################################
# Process tomography circuits for preparation and measurement in Pauli basis
###########################################################################

def process_tomography_circuits(circuit, measured_qubits,
                                prepared_qubits=None,
                                meas_labels='Pauli', meas_basis='Pauli',
                                prep_labels='Pauli', prep_basis='Pauli'):
    """
    Return a list of quantum process tomography circuits.

    This performs preparation in the minimial Pauli-basis eigenstates
    Zp, Zm, Xp, Ym (|0>, |1>, |+>, |+i>) on each qubit, and measurement in
    the Pauli-basis X, Y, Z resulting in (4 ** n) * (3 ** n) circuits for
    an n-qubit process tomography experiment.

    Args:
        circuit (QuantumCircuit): the QuantumCircuit circuit to be
                                  tomographed.
        measured_qubits (QuantumRegister): the qubits to be measured.
            This can also be a list of whole QuantumRegisters or
            individual QuantumRegister qubit tuples.
        prepared_qubits (QuantumRegister or None): the qubits to have state
            preparation applied, if different from measured_qubits. If None
            measured_qubits will be used for prepared qubits (Default: None).
        meas_labels (str, tuple, list(tuple)): The measurement operator
            labels. See additional information for details (Default: 'Pauli').
        meas_basis (str, TomographyBasis): The measurement basis.
            See additional information for details (Default: 'Pauli').
        prep_labels (str, tuple, list(tuple)): The preparation operator
            labels. See additional information for details (Default: 'Pauli').
        prep_basis (str, TomographyBasis): The preparation basis.
            See additional information for details (Default: 'Pauli').

    Returns:
        A list of QuantumCircuit objects containing the original circuit
        with state preparation circuits prepended, and measurement circuits
        appended.

    Additional Information:
        The returned circuits are named by the preparation and measurement
        basis. These circuit names can be recovered using the
        `process_tomography_circuit_names` function to retrieve count data
        from a QISKit Result object at a later time.

        To perform tomography measurement in a custom basis, or to generate
        a subset of process tomography circuits for a partial tomography
        experiment use the general function `tomography_circuits`.
    """
    return _tomography_circuits(circuit, measured_qubits, prepared_qubits,
                                meas_labels=meas_labels, meas_basis=meas_basis,
                                prep_labels=prep_labels, prep_basis=prep_basis)


###########################################################################
# General state and process tomography circuit functions
###########################################################################

def _tomography_circuits(circuit, measured_qubits, prepared_qubits=None,
                         meas_labels='Pauli', meas_basis='Pauli',
                         prep_labels=None, prep_basis=None):
    """
    Return a list of quantum tomography circuits.

    This is the general circuit preparation function called by
    `state_tomography_circuits` and `process_tomography_circuits` and
    allows partial tomography circuits to be generated, or tomography
    circuits with custom preparation and measurement operators.

    Args:
        circuit (QuantumCircuit): the QuantumCircuit circuit to be
                                  tomographed.
        measured_qubits (QuantumRegister): the qubits to be measured.
            This can also be a list of whole QuantumRegisters or
            individual QuantumRegister qubit tuples.
        prepared_qubits (QuantumRegister or None): the qubits to have state
            preparation applied, if different from measured_qubits. If None
            measured_qubits will be used for prepared qubits (Default: None).
        meas_labels (None, str, tuple, list(tuple)): The measurement operator
            labels. If None no measurements will be appended. See additional
            information for details (Default: 'Pauli').
        prep_labels (None, str, tuple, list(tuple)): The preparation operator
            labels. If None no preparations will be appended. See additional
            information for details (Default: None).
        meas_circuit_fn (None, str, function): The measurement circuit
            function. See additional information for details (Default: None).
        prep_circuit_fn (None, str, function): The preparation circuit
            function. See additional information for details (Default: None).

    Returns:
        A list of QuantumCircuit objects containing the original circuit
        with state preparation circuits prepended, and measurement circuits
        appended.

    Additional Information

        Specifying Labels
        -----------------
        `meas_labels` and `prep_labels` may be specified as either:
            - None: no measurements, or preparation, will be added.
            - str: use a built-in basis.
                For meas_labels the built-in basis is 'Pauli'.
                For prep_labels the built-in bases are 'Pauli' and 'SIC'.
            - tuple(str): specify single qubit operator labels and
                          generate all n-qubit combinations.
            - list(tuple(str)): specify a custom list of n-qubit label tuples.

        If a str argument is used then it is not necessary to specify the
        corresponding meas_circuit_fn or prep_circuit_fn as the defaults will
        be used for the corresponding basis. However, when using a tuple or
        list value these functions must be manually specified using either a
        str for the build in bases, or a function (See below for
        documentation.)

        Specifying a tuple can be used to only measure certain operators.
        For example if we specify meas_labels=('Z', ) the resulting circuits
        will only contain measurements in the Z-basis. Specifying
        meas_labels=('X','Z') will only contain 2 ** n measurements in X and Z
        basis etc.

        Specifying a tuple is necessary when using a custom `meas_cicuit_fn` or
        `prep_circuit_fn` as these will be the str passed to the function to
        return the corresponding QuantumCircuit objects.

        Specifying a list of tuples will override an automatic generation. This
        can be for partial tomography. For example for a 2-qubit state
        tomography experiment we might only specify correlated measurements eg:
            meas_labels=[('X','X'), ('Y','Y'), ('Z','Z')]

        Custom Measurement Circuit Funtion
        ----------------------------------
        Custom measurement circuit functions can be used by passing the
        function using the `meas_circuit_fn` keyword. These functions should
        have the signature:

        meas_circuit_fn(op, qubit, clbit)
            Args:
                op (str): the operator label
                qubit (Qubit): measured qubit
                clbit (Clbit): measurement clbit
            Returns:
                A QuantumCircuit object for the measurement.

        See the built-in function `pauli_measurement_circuit` for an example.
        The built-in Pauli measurement function `pauli_measurement_circuit`
        may be invoked using the meas_circuit_fn='Pauli'.

        Custom Preparation Circuit Funtion
        ----------------------------------
        Custom preparation circuit functions can be used by passing the
        function using the `prep_circuit_fn` keyword. These functions should
        have the signature:

        prep_circuit_fn(op, qubit)
            Args:
                op (str): the operator label
                qubit (Qubit): measured qubit
            Returns:
                A QuantumCircuit object for the preparation gates.

        See the build-in function `pauli_preparation_circuit` for an example.
        See the built-in function `pauli_measurement_circuit` for an example.
        The built-in Pauli preparation function `pauli_preparation_circuit`
        may be invoked using the prep_circuit_fn='Pauli'.
        The built-in SIC-POVM preparation function
        `sicpovm_preparation_circuit` may be invoked using the
        prep_circuit_fn='SIC'.
    """

    # Check for different prepared qubits
    if prepared_qubits is None:
        prepared_qubits = measured_qubits
    # Check input circuit for measurements and measured qubits
    if isinstance(measured_qubits, list):
        # Unroll list of registers
        meas_qubits = _format_registers(*measured_qubits)
    else:
        meas_qubits = _format_registers(measured_qubits)
    if isinstance(prepared_qubits, list):
        # Unroll list of registers
        prep_qubits = _format_registers(*prepared_qubits)
    else:
        prep_qubits = _format_registers(prepared_qubits)
    if len(prep_qubits) != len(meas_qubits):
        raise QiskitError(
            "prepared_qubits and measured_qubits are different length.")
    num_qubits = len(meas_qubits)
    meas_qubit_registers = set(q.register for q in meas_qubits)
    # Check qubits being measured are defined in circuit
    for reg in meas_qubit_registers:
        if reg not in circuit.qregs:
            logger.warning('WARNING: circuit does not contain '
                           'measured QuantumRegister: %s', reg.name)

    prep_qubit_registers = set(q.register for q in prep_qubits)
    # Check qubits being measured are defined in circuit
    for reg in prep_qubit_registers:
        if reg not in circuit.qregs:
            logger.warning('WARNING: circuit does not contain '
                           'prepared QuantumRegister: %s', reg.name)

    # Get combined registers
    qubit_registers = prep_qubit_registers.union(meas_qubit_registers)

    # Check if there are already measurements in the circuit
    for op in circuit:
        if isinstance(op, Measure):
            logger.warning('WARNING: circuit already contains measurements')
        if isinstance(op, Reset):
            logger.warning('WARNING: circuit contains resets')

    # Load built-in circuit functions
    if callable(meas_basis):
        measurement = meas_basis
    else:
        measurement = default_basis(meas_basis)
        if isinstance(measurement, TomographyBasis):
            if measurement.measurement is not True:
                raise QiskitError("Invalid measurement basis")
            measurement = measurement.measurement_circuit
    if callable(prep_basis):
        preparation = prep_basis
    else:
        preparation = default_basis(prep_basis)
        if isinstance(preparation, TomographyBasis):
            if preparation.preparation is not True:
                raise QiskitError("Invalid preparation basis")
            preparation = preparation.preparation_circuit

    # Check we have circuit functions defined
    if measurement is None and meas_labels is not None:
        raise ValueError("Measurement basis is not specified.")
    if preparation is None and prep_labels is not None:
        raise ValueError("Preparation basis is not specified.")

    # Load built-in basis labels
    if isinstance(meas_labels, str):
        meas_labels = _default_measurement_labels(meas_labels)
    if isinstance(prep_labels, str):
        prep_labels = _default_preparation_labels(prep_labels)

    # Generate n-qubit labels
    meas_labels = _generate_labels(meas_labels, num_qubits)
    prep_labels = _generate_labels(prep_labels, num_qubits)

    # Note if the input circuit already has classical registers defined
    # the returned circuits add a new classical register for the tomography
    # measurments which will be inserted as the first classical register in
    # the list of returned circuits.
    registers = qubit_registers.copy()
    if measurement is not None:
        clbits = ClassicalRegister(num_qubits)
        registers.add(clbits)

    # Generate the circuits
    qst_circs = []
    for pl in prep_labels:
        prep = QuantumCircuit(*registers)
        # Generate preparation circuit
        if pl is not None:
            for j in range(num_qubits):
                prep += preparation(pl[j], prep_qubits[j])
            prep.barrier(*qubit_registers)
        # Add circuit being tomographed
        prep += circuit
        # Generate Measurement circuit
        for ml in meas_labels:
            meas = QuantumCircuit(*registers)
            if ml is not None:
                meas.barrier(*qubit_registers)
                for j in range(num_qubits):
                    meas += measurement(ml[j], meas_qubits[j], clbits[j])
            circ = prep + meas
            if pl is None:
                # state tomography circuit
                circ.name = str(ml)
            else:
                # process tomography circuit
                circ.name = str((pl, ml))
            qst_circs.append(circ)
    return qst_circs


###########################################################################
# Built-in circuit functions
###########################################################################

def default_basis(basis):
    """
    Built in Tomography Bases
    """
    if basis is None:
        return None
    if isinstance(basis, str):
        if basis == 'Pauli':
            return PauliBasis
        if basis == 'SIC':
            return SICBasis
    if isinstance(basis, TomographyBasis):
        return basis
    raise ValueError('Unrecognised basis: {}'.format(basis))


def _default_measurement_labels(basis):
    """
    Built in measurement basis labels.
    """
    if default_basis(basis) == PauliBasis:
        return ('X', 'Y', 'Z')
    raise ValueError('Unrecognised basis string "{}"'.format(basis))


def _default_preparation_labels(basis):
    """
    Built in preparation basis labels.
    """
    tomo_basis = default_basis(basis)
    if tomo_basis == PauliBasis:
        return ('Zp', 'Zm', 'Xp', 'Yp')
    if tomo_basis == SICBasis:
        return ('S0', 'S1', 'S2', 'S3')
    raise ValueError('Unrecognised basis string "{}"'.format(basis))


###########################################################################
# Helper functions
###########################################################################

def tomography_circuit_tuples(measured_qubits, meas_labels='Pauli',
                              prep_labels=None):
    """
    Return list of tomography circuit label tuples.
    """

    if isinstance(meas_labels, (str, TomographyBasis)):
        meas_labels = _default_measurement_labels(meas_labels)
    if isinstance(prep_labels, (str, TomographyBasis)):
        prep_labels = _default_preparation_labels(prep_labels)

    mls = _generate_labels(meas_labels, measured_qubits)
    pls = _generate_labels(prep_labels, measured_qubits)
    return [(ml, pl) for pl, ml in it.product(mls, pls)]


def _generate_labels(labels, measured_qubits):
    """
    Return list of n-qubit measurement circuit labels.
    """
    if labels is None:
        return [None]
    # Generate n-qubit tuples for single qubit tuples
    if isinstance(labels, tuple):
        labels = _operator_tuples(labels, measured_qubits)
    if isinstance(labels, list):
        return labels
    raise ValueError(
        'Invalid labels specification: must be None, list, string, or tuple')


def _format_registers(*registers):
    """
    Return a list of qubit QuantumRegister tuples.
    """
    if not registers:
        raise Exception('No registers are being measured!')
    qubits = []
    for tuple_element in registers:
        if isinstance(tuple_element, QuantumRegister):
            for j in range(tuple_element.size):
                qubits.append(tuple_element[j])
        else:
            qubits.append(tuple_element)
    # Check registers are unique
    if len(qubits) != len(set(qubits)):
        raise Exception('Qubits to be measured are not unique!')
    return qubits


def _operator_tuples(labels, qubits):
    """
    Return a list of all length-n tuples.
    """
    if isinstance(qubits, int):
        num_qubits = qubits
    elif isinstance(qubits, list):
        num_qubits = len(_format_registers(*qubits))
    else:
        num_qubits = len(_format_registers(qubits))
    return list(it.product(labels, repeat=num_qubits))
