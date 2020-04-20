# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
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
from typing import List, Union, Tuple, Optional
import itertools as it
import re

from qiskit import QuantumRegister
from qiskit.circuit import Qubit
from qiskit import ClassicalRegister
from qiskit import QuantumCircuit
from qiskit import QiskitError
from qiskit.circuit.measure import Measure
from qiskit.circuit.reset import Reset

from .tomographybasis import TomographyBasis
from .paulibasis import PauliBasis
from .gatesetbasis import default_gateset_basis, GateSetBasis
from .sicbasis import SICBasis

# Create logger
logger = logging.getLogger(__name__)

###########################################################################
# State tomography circuits for measurement in Pauli basis
###########################################################################


def state_tomography_circuits(
        circuit: QuantumCircuit,
        measured_qubits: QuantumRegister,
        meas_labels: Union[str, Tuple[str], List[Tuple[str]]] = 'Pauli',
        meas_basis: Union[str, TomographyBasis] = 'Pauli'
) -> List[QuantumCircuit]:
    """
    Return a list of quantum state tomography circuits.

    This performs measurement in the Pauli-basis resulting in :math:`3^n`
    circuits for an n-qubit state tomography experiment.

    Args:
        circuit: the state preparation circuit to be tomographed.
        measured_qubits: the qubits to be measured.
            This can also be a list of whole QuantumRegisters or
            individual QuantumRegister qubit tuples.
        meas_labels: (default: 'Pauli') The measurement operator labels.
        meas_basis: (default: 'Pauli') The measurement basis.

    Returns:
        A list containing copies of the original circuit
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

def process_tomography_circuits(
        circuit: QuantumCircuit,
        measured_qubits: QuantumRegister,
        prepared_qubits: Optional[QuantumRegister] = None,
        meas_labels: Union[str, Tuple[str], List[Tuple[str]]] = 'Pauli',
        meas_basis: Union[str, TomographyBasis] = 'Pauli',
        prep_labels: Union[str, Tuple[str], List[Tuple[str]]] = 'Pauli',
        prep_basis: Union[str, TomographyBasis] = 'Pauli'
) -> List[QuantumCircuit]:
    r"""Return a list of quantum process tomography circuits.

    This performs preparation in the minimial Pauli-basis eigenstates
        * ``"Z_p"``: :math:`|0\rangle`
        * ``"Z_m"``: :math:`|1\rangle`
        * ``"X_p"``: :math:`|+\rangle`
        * ``"Y_m"``: :math:`|+i\rangle`

    on each qubit, and measurement in the Pauli-basis X, Y, Z resulting
    in :math:`4^n 3^n` circuits for an n-qubit process
    tomography experiment.

    Args:
        circuit: the QuantumCircuit circuit to be
            tomographed.
        measured_qubits: the qubits to be measured.
            This can also be a list of whole QuantumRegisters or
            individual QuantumRegister qubit tuples.
        prepared_qubits: the qubits to have state
            preparation applied, if different from measured_qubits. If None
            measured_qubits will be used for prepared qubits
        meas_labels: (default: 'Pauli') The measurement operator labels.
        meas_basis: (default: 'Pauli') The measurement basis.
        prep_labels: (default: 'Pauli') The preparation operator labels.
        prep_basis: (default: 'Pauli') The preparation basis.

    Returns:
        A list of QuantumCircuit objects containing the original circuit
        with state preparation circuits prepended, and measurement circuits
        appended.

    The returned circuits are named by the preparation and measurement
    basis.
    """
    return _tomography_circuits(circuit, measured_qubits, prepared_qubits,
                                meas_labels=meas_labels, meas_basis=meas_basis,
                                prep_labels=prep_labels, prep_basis=prep_basis)


###########################################################################
# Gate set tomography circuits for preparation and measurement
###########################################################################

def gateset_tomography_circuits(measured_qubits: Optional[List[int]] = None,
                                gateset_basis: Union[str,
                                                     GateSetBasis] = 'default'
                                ) -> List[QuantumCircuit]:
    r"""Return a list of quantum gate set tomography (GST) circuits.

    The circuits are fully constructed from the data given in gateset_basis.
    Note that currently this is only implemented for the single-qubits.

    Args:
        measured_qubits: The qubits to perform GST. If None GST will be
                         performed on qubit-0.
        gateset_basis: The gateset and SPAM data.

    Returns:
        A list of QuantumCircuit objects containing the original circuit
        with state preparation circuits prepended, and measurement circuits
        appended.

    Raises:
        QiskitError: If called for more than 1 measured qubit.

    Additional Information:
        Gate set tomography is performed on a gate set (G0, G1,...,Gm)
        with the additional information of SPAM circuits (F0,F1,...,Fn)
        that are constructed from the gates in the gate set.

        In gate set tomography, we assume a single initial state rho
        and a single POVM measurement operator E. The SPAM circuits
        now provide us with a complete set of initial state F_j|rho>
        and measurements <E|F_i.

        We perform three types of experiments:

        1) :math:`\langle E  | F_i G_k F_j |\rho \rangle` for 1 <= i,j <= n
            and 1 <= k <= m:
            This experiment enables us to obtain data on the gate G_k
        2) :math:`\langle E  | F_i F_j |\rho \rangle`  for 1 <= i,j <= n:
            This experiment enables us to obtain the Gram matrix required
            to "invert" the results of experiments of type 1 in order to
            reconstruct (a matrix similar to) the gate G_k
        3) :math:`\langle E  | F_j |\rho \rangle` for 1 <= j <= n:
            This experiment enables us to reconstruct <E| and rho

        The result of this method is the set of all the circuits needed for
        these experiments, suitably labeled with a tuple of the corresponding
        gate/SPAM labels
    """
    if measured_qubits is None:
        measured_qubits = [0]

    if len(measured_qubits) > 1:
        raise QiskitError("Only 1-qubit gate set tomography "
                          "is currently supported")
    num_qubits = 1 + max(measured_qubits)

    all_circuits = []
    if gateset_basis == 'default':
        gateset_basis = default_gateset_basis()
    meas_basis = gateset_basis.get_tomography_basis()
    prep_basis = gateset_basis.get_tomography_basis()
    meas_labels = meas_basis.measurement_labels
    prep_labels = prep_basis.preparation_labels
#    qubit = QuantumRegister(num_qubits)
    # Experiments of the form <E|F_i G_k F_j|rho>
    for gate in gateset_basis.gate_labels:
        circuit = QuantumCircuit(num_qubits)
        # we assume only 1 qubit for now
        qubit = circuit.qubits[measured_qubits[0]]
        gateset_basis.add_gate_to_circuit(circuit, qubit, gate)
        gst_circuits = _tomography_circuits(circuit, qubit, qubit,
                                            meas_labels=meas_labels,
                                            meas_basis=meas_basis,
                                            prep_labels=prep_labels,
                                            prep_basis=prep_basis)
        for tomography_circuit in gst_circuits:
            # Getting the names of Fi and Fj using regex
            res = re.search("'(.*)'.*'(.*)'", tomography_circuit.name)
            tomography_circuit.name = str((res.group(1), gate, res.group(2)))
        all_circuits = all_circuits + gst_circuits

    # Experiments of the form <E|F_i F_j|rho>
    # Can be skipped if one of the gates is ideal identity
    circuit = QuantumCircuit(num_qubits)
    qubit = circuit.qubits[measured_qubits[0]]
    gst_circuits = _tomography_circuits(circuit, qubit, qubit,
                                        meas_labels=meas_labels,
                                        meas_basis=meas_basis,
                                        prep_labels=prep_labels,
                                        prep_basis=prep_basis)
    for tomography_circuit in gst_circuits:
        # Getting the names of Fi and Fj using regex
        res = re.search("'(.*)'.*'(.*)'", tomography_circuit.name)
        tomography_circuit.name = str((res.group(1), res.group(2)))
    all_circuits = all_circuits + gst_circuits

    # Experiments of the form <E|F_j|rho>
    circuit = QuantumCircuit(num_qubits)
    qubit = circuit.qubits[measured_qubits[0]]
    gst_circuits = _tomography_circuits(circuit, qubit, qubit,
                                        meas_labels=meas_labels,
                                        meas_basis=meas_basis,
                                        prep_labels=None,
                                        prep_basis=None)
    for tomography_circuit in gst_circuits:
        # Getting the name of Fj using regex
        res = re.search("'(.*)'", tomography_circuit.name)
        tomography_circuit.name = str((res.group(1),))
    all_circuits = all_circuits + gst_circuits

    return all_circuits

###########################################################################
# General state and process tomography circuit functions
###########################################################################


def _tomography_circuits(
        circuit: QuantumCircuit,
        measured_qubits: QuantumRegister,
        prepared_qubits: Optional[QuantumRegister] = None,
        meas_labels: Union[str, Tuple[str], List[Tuple[str]]] = 'Pauli',
        meas_basis: Union[str, TomographyBasis] = 'Pauli',
        prep_labels: Union[str, Tuple[str], List[Tuple[str]]] = 'Pauli',
        prep_basis: Union[str, TomographyBasis] = 'Pauli'
) -> List[QuantumCircuit]:
    """Return a list of quantum tomography circuits.
    This is the general circuit preparation function called by
    `state_tomography_circuits` and `process_tomography_circuits` and
    allows partial tomography circuits to be generated, or tomography
    circuits with custom preparation and measurement operators.

    Args:
        circuit: the QuantumCircuit circuit to be tomographed.
        measured_qubits: the qubits to be measured.
            This can also be a list of whole QuantumRegisters or
            individual QuantumRegister qubit tuples.
        prepared_qubits: the qubits to have state
            preparation applied, if different from measured_qubits. If None
            measured_qubits will be used for prepared qubits
        meas_labels: (default: 'Pauli') The measurement operator
            labels. If None no measurements will be appended. See additional
            information for details
        meas_basis: (default: 'Pauli') The measurement basis.
        prep_labels: (default: 'Pauli') The preparation operator
            labels. If None no preparations will be appended. See additional
            information for details
        prep_basis: (default: 'Pauli') The preparation basis.
    Raises:
        QiskitError: If the measurement/preparation basis is invalid.
        ValueError: If the measurement/preparation basis is not specified
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
        For example if we specify `meas_labels=('Z', )` the resulting circuits
        will only contain measurements in the Z-basis. Specifying
        `meas_labels=('X','Z')` will only contain :math:`2^n` measurements
        in X and Z basis etc.

        Specifying a tuple is necessary when using a custom `meas_circuit_fn`
        or `prep_circuit_fn` as these will be the str passed to the function to
        return the corresponding QuantumCircuit objects.

        Specifying a list of tuples will override an automatic generation. This
        can be for partial tomography. For example for a 2-qubit state
        tomography experiment we might only specify correlated measurements eg:
            meas_labels=[('X','X'), ('Y','Y'), ('Z','Z')]

        Custom Measurement Circuit Function
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

        Custom Preparation Circuit Function
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
    if isinstance(measured_qubits, (list, tuple)):
        # Unroll list of registers
        if isinstance((measured_qubits[0]), int):
            measured_qubits = [circuit.qubits[i] for i in measured_qubits]
        meas_qubits = _format_registers(*measured_qubits)
    else:
        meas_qubits = _format_registers(measured_qubits)
    if isinstance(prepared_qubits, (list, tuple)):
        # Unroll list of registers
        if isinstance(prepared_qubits[0], int):
            prepared_qubits = [circuit.qubits[i] for i in prepared_qubits]
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
    # measurements which will be inserted as the first classical register in
    # the list of returned circuits.
    registers = qubit_registers.copy()
    if measurement is not None:
        clbits = ClassicalRegister(num_qubits)
        registers.add(clbits)

    # Generate the circuits
    qst_circs = []
    for prep_label in prep_labels:
        prep = QuantumCircuit(*registers)
        # Generate preparation circuit
        if prep_label is not None:
            for j in range(num_qubits):
                prep += preparation(prep_label[j], prep_qubits[j])
            prep.barrier(*qubit_registers)
        # Add circuit being tomographed
        prep += circuit
        # Generate Measurement circuit
        for meas_label in meas_labels:
            meas = QuantumCircuit(*registers)
            if meas_label is not None:
                meas.barrier(*qubit_registers)
                for j in range(num_qubits):
                    meas += measurement(meas_label[j],
                                        meas_qubits[j],
                                        clbits[j])
            circ = prep + meas
            if prep_label is None:
                # state tomography circuit
                circ.name = str(meas_label)
            else:
                # process tomography circuit
                circ.name = str((prep_label, meas_label))
            qst_circs.append(circ)
    return qst_circs


###########################################################################
# Built-in circuit functions
###########################################################################

def default_basis(basis: Optional[Union[str, TomographyBasis]]) -> TomographyBasis:
    """Built in Tomography Bases.

    Args:
        basis: the tomography basis.
    Raises:
        ValueError: In case the given basis is not recognized
    Returns:
        The requested tomography basis.
    Additional Information:
        If the input basis is ``None`` or a :class:`TomographyBasis` it will
        be returned unchanged.
        If it is a `"Pauli"` or `"SIC"` it will return the built in tomography
        basis object for that basis.
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
    raise ValueError('Unrecognized basis: {}'.format(basis))


def _default_measurement_labels(basis: Union[str, TomographyBasis]
                                ) -> Tuple[str]:
    """Built in measurement basis labels.

    Args:
        basis: A tomography basis or a name of standard one.
    Raises:
        ValueError: In case the basis is not recognized
    Returns:
        In case the base is Pauli: The labels ('X','Y','Z').
    """
    if default_basis(basis) == PauliBasis:
        return ('X', 'Y', 'Z')
    raise ValueError('Unrecognized basis string "{}"'.format(basis))


def _default_preparation_labels(basis: Union[str, TomographyBasis]
                                ) -> Tuple[str]:
    """Built in preparation basis labels.

    Args:
        basis: A tomography basis or a name of standard one.
    Raises:
        ValueError: In case the basis is not recognized
    Returns:
        - In case the base is Pauli: The labels ('Zp', 'Zm', 'Xp', 'Yp').
        - In case the base is SIC: The labels ('S0', 'S1', 'S2', 'S3').

    """
    tomo_basis = default_basis(basis)
    if tomo_basis == PauliBasis:
        return ('Zp', 'Zm', 'Xp', 'Yp')
    if tomo_basis == SICBasis:
        return ('S0', 'S1', 'S2', 'S3')
    raise ValueError('Unrecognized basis string "{}"'.format(basis))


###########################################################################
# Helper functions
###########################################################################

def tomography_circuit_tuples(
        measured_qubits: Union[int,
                               QuantumRegister,
                               List[QuantumRegister]],
        meas_labels: Union[str, TomographyBasis, Tuple] = 'Pauli',
        prep_labels: Optional[Union[str, TomographyBasis, Tuple]] = None
        ) -> List:
    r"""Return list of tomography circuit label tuples.
    Args:
        measured_qubits: Either qubits the tomography will be applied to
        or their length
        meas_labels: (default: 'Pauli') The measurement basis labels or the basis itself.
        prep_labels: The preparation basis labels or the basis itself.
    Returns:
        A list of all pairs :math:`[m_l, p_k]` where
        :math:`m_1,\ldots m_t` are all the n-qubit measurement labels
        and :math:`p_1,\ldots, p_s` are all the n-qubit preparation labels

        Note that if prep_labels are empty, it will be considered
        as containing only the empty string "", so a list with
        all measurement lables will still be generated.
    """

    if isinstance(meas_labels, (str, TomographyBasis)):
        meas_labels = _default_measurement_labels(meas_labels)
    if isinstance(prep_labels, (str, TomographyBasis)):
        prep_labels = _default_preparation_labels(prep_labels)

    mls = _generate_labels(meas_labels, measured_qubits)
    pls = _generate_labels(prep_labels, measured_qubits)
    return [(ml, pl) for pl, ml in it.product(mls, pls)]


def _generate_labels(labels: Optional[Union[Tuple[str],
                                            List[Tuple[str]],
                                            str]],
                     measured_qubits: Union[int,
                                            QuantumRegister,
                                            List[QuantumRegister]]
                     ) -> List[Tuple[str]]:
    """Return list of n-qubit measurement circuit labels.
    Args:
        labels: A tuple of the basis labels, or a string of the basis
            labels separated by spaces, or a list of pre-made tuples.
        measured_qubits: Either qubits the tomography will be applied to
            or their length
    Raises:
        ValueError: if the label specification is wrong.
    Returns:
        A list of length n-tuples containing all possible
            combinations of the given labels.
    """
    if labels is None:
        return [None]
    # Generate n-qubit tuples for single qubit tuples
    if isinstance(labels, tuple):
        labels = _operator_tuples(labels, measured_qubits)
    if isinstance(labels, str):
        labels = _operator_tuples(labels.split(" "), measured_qubits)
    if isinstance(labels, list):
        return labels
    raise ValueError(
        'Invalid labels specification: must be None, list, string, or tuple')


def _format_registers(*registers: Union[Qubit, QuantumRegister]
                      ) -> List[Qubit]:
    """Return a list of qubit QuantumRegister tuples.

    Args:
        registers: Any nonzero number of qubits or
        quantum registers, all unique
    Raises:
         QiskitError: If no qubits/registers were passed
            or non-unique qubits passed
    Returns:
        A flat list of all qubits passed.
    """
    if not registers:
        raise QiskitError('No registers are being measured!')
    qubits = []
    for tuple_element in registers:
        if isinstance(tuple_element, QuantumRegister):
            for j in range(tuple_element.size):
                qubits.append(tuple_element[j])
        else:
            qubits.append(tuple_element)
    # Check registers are unique
    if len(qubits) != len(set(qubits)):
        raise QiskitError('Qubits to be measured are not unique!')
    return qubits


def _operator_tuples(labels: Tuple[str],
                     qubits: Union[int,
                                   QuantumRegister,
                                   List[QuantumRegister]]
                     ) -> List[Tuple[str]]:
    """Return a list of all length-n tuples of labels.

    Args:
        labels: The basis labels to build tuples from
        qubits: Either qubits the tomography will be applied to
            or their length
    Returns:
        All n-length sequences of elements of **labels**.
        For example, if **labels** = ('X', 'Y', 'Z')
        and the number of qubits n=2, then the result will be the list
        [('X', 'X'), ('X', 'Y'), ('X', 'Z'),
         ('Y', 'X'), ('Y', 'Y'), ('Y', 'Z'),
         ('Z', 'X'), ('Z', 'Y'), ('Z', 'Z')]
    """
    if isinstance(qubits, int):
        num_qubits = qubits
    elif isinstance(qubits, list):
        num_qubits = len(_format_registers(*qubits))
    else:
        num_qubits = len(_format_registers(qubits))
    return list(it.product(labels, repeat=num_qubits))
