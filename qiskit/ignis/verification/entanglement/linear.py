'''
The module linear.py provides the linear
preparation analogous of parallelize.py.
'''

from qiskit import *
from qiskit.circuit import Parameter


def get_measurement_circ(n, extent='full'):
    '''
    Creates a measurement circuit that can toggle between
    measuring the first control qubit or measuring all qubits.
    The default is measurement of all qubits.
    Args:
       n: number of qubits
       extent ('full', 'one'):
        Whether to append full measurement, or only on the first qubit.
    Returns:
       The measurement suffix for a circuit
    '''
    q = QuantumRegister(n, 'q')
    if extent == 'one':
        cla = ClassicalRegister(1, 'c')
        meas = QuantumCircuit(q, cla)
        meas.barrier()
        meas.measure(q[0], cla)
        return meas
    if extent == 'full':
        cla = ClassicalRegister(n, 'c')
        meas = QuantumCircuit(q, cla)
        meas.barrier()
        meas.measure(q, cla)
        return meas
    if extent != 'one' or extent != 'full':
        raise Exception("Extent arguments must be 'full' or 'one'")
    return None


def get_ghz_simple(n, measure=True, extent='full'):
    '''
    Creates a linear GHZ state
    with the option of measurement
    Args:
       n: number of qubits
       measure (Boolean): Whether to add measurement gates
       extent ('full', 'one'):
        Whether to append full measurement, or only on the first qubit.
        Relevant only for measure=True
    Returns:
       A linear GHZ Circuit
    '''
    q = QuantumRegister(n, 'q')
    circ = QuantumCircuit(q)
    circ.h(q[0])
    for i in range(1, n):
        circ.cx(q[i - 1], q[i])
    if measure:
        meas = get_measurement_circ(n, extent)
        circ = circ + meas
    else:
        pass
    return circ


def get_ghz_mqc(n, delta, measure='full'):
    '''
    This function creates an MQC circuit with n qubits,
    where the middle phase rotation around the z axis is by delta
    '''
    q = QuantumRegister(n, 'q')
    circ = get_ghz_simple(n, measure=False)
    circinv = circ.inverse()
    circ.barrier()
    circ.u1(delta, q)
    circ.x(q)
    circ.barrier()
    circ += circinv
    meas = get_measurement_circ(n, measure)
    circ = circ + meas
    circ.draw()
    return circ


def get_ghz_mqc_para(n, measure='full'):
    '''
    This function creates an MQC circuit with n qubits,
    where the middle phase rotation around the z axis is by delta
    Args:
       n: number of qubits
       measure ('full', 'one'):
        Whether to append full measurement, or only on the first qubit.
    Returns:
       An mqc circuit and its Delta parameter
    '''
    q = QuantumRegister(n, 'q')
    circ = get_ghz_simple(n, measure=False)
    delta = Parameter('t')
    circinv = circ.inverse()
    circ.barrier()
    circ.u1(delta, q)
    circ.x(q)
    circ.barrier()
    circ += circinv
    meas = get_measurement_circ(n, measure)
    circ = circ + meas
    circ.draw()
    return circ, delta


def get_ghz_po(n, delta, measure='full'):
    '''
    This function creates an Parity Oscillation circuit
    with n qubits, where the middle superposition rotation around
    the x and y axes is by delta
    '''
    if measure != 'full':
        raise Exception("Only 'full' argument can be accepted",
                        " for measure in Parity Oscillation circuit")
    q = QuantumRegister(n, 'q')
    circ = get_ghz_simple(n, measure=False)

    circ.barrier()
    circ.u2(delta, -delta, q)
    circ.barrier()
    meas = get_measurement_circ(n, measure)
    circ = circ + meas
    circ.draw()
    return circ


def get_ghz_po_para(n, measure='full'):
    '''
    This function creates a Parity Oscillation circuit with n qubits,
    where the middle superposition rotation around
     the x and y axes is by delta
     Args:
       n: number of qubits
       measure: Must be 'full' - so as to append full measurement.
    Returns:
       A parity oscillation circuit and its Delta/minus-delta parameters
   '''
    if measure != 'full':
        raise Exception("Only 'full' argument can be accepted",
                        " for measure in Parity Oscillation circuit")
    q = QuantumRegister(n, 'q')
    delta = Parameter('t')
    deltaneg = Parameter('-t')
    circ = get_ghz_simple(n, measure=False)

    circ.barrier()
    circ.u2(delta, deltaneg, q)
    meas = get_measurement_circ(n, measure)
    circ = circ + meas
    return circ, [delta, deltaneg]
