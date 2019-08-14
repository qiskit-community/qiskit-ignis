from qiskit import *
from qiskit.circuit import Parameter
import numpy as np




def get_ghz_measurement(n,extent):
    '''
    Creates a measurement circuit that can toggle between measuring the control qubit
    or measuring all qubits. The default is measurement of all qubits.
    '''
    q = QuantumRegister(n,'q')
    if extent == 'one':
        cla = ClassicalRegister(1,'c')  
        meas = QuantumCircuit(q,cla)
        meas.barrier()
        meas.measure(q[0],cla)
        return meas
    elif extent == 'full':
        cla = ClassicalRegister(n,'c')
        meas = QuantumCircuit(q,cla)
        meas.barrier()
        meas.measure(q,cla)
        return meas
    elif extent != 'one' or extent != 'full':
        raise Exception("Extent arguments must be 'full' or 'one'")

        
def get_ghz_simple(n,obs=True, extent = 'full'):
    '''
    Creates a "dummy" linear GHZ state with the option of measurement
    '''
    q = QuantumRegister(n,'q')
    circ = QuantumCircuit(q)
    circ.h(q[0])
    for i in range(1,n):
        circ.cx(q[i-1],q[i])       
    if obs:
        meas = get_ghz_measurement(n, extent)
        circ = circ + meas
    else:
        pass
    return circ        
        
        
        
        
def get_ghz_mqc(n,delta, extent = 'full'):
    '''
    This function creates an MQC circuit with n qubits, where the middle phase rotation around the z axis is by delta 
    '''
    q = QuantumRegister(n,'q')
    circ = get_ghz_simple(n,obs=False)
    circinv = circ.inverse()
    circ.barrier()
    circ.u1(delta,q)
    circ.x(q)
    circ.barrier()
    circ += circinv                
    meas = get_ghz_measurement(n, extent)
    circ = circ + meas
    circ.draw()
    return circ

def get_ghz_mqc_para(n,extent = 'full'):
    '''
    This function creates an MQC circuit with n qubits, where the middle phase rotation around the z axis is by delta 
    '''
    q = QuantumRegister(n,'q')
    circ = get_ghz_simple(n,obs=False)
    delta = Parameter('t')
    circinv = circ.inverse()
    circ.barrier()
    circ.u1(delta,q)
    circ.x(q)
    circ.barrier()
    circ += circinv                
    meas = get_ghz_measurement(n, extent)
    circ = circ + meas
    circ.draw()
    return circ, delta

def get_ghz_po(n,delta, extent = 'full'): #n is number of qubits, delta is phase to rotate about x-y axis
    '''
    This function creates an Parity Oscillation circuit with n qubits, where the middle superposition rotation around 
    the x and y axes is by delta
    '''
    if extent != 'full':
        raise Exception("Only 'full' argument can be accepted for extent in Parity Oscillation circuit")
    q = QuantumRegister(n,'q')
    circ = get_ghz_simple(n,obs=False)
    circinv = circ.inverse()
    circ.barrier()
    circ.u2(delta,-delta,q)
    circ.barrier()
    meas = get_ghz_measurement(n, extent)
    circ = circ + meas
    circ.draw()
    return circ

def get_ghz_po_para(n, extent = 'full'): #n is number of qubits, delta is phase to rotate about x-y axis
    '''
    This function creates an Parity Oscillation circuit with n qubits, where the middle superposition rotation around 
    the x and y axes is by delta
    '''
    if extent != 'full':
        raise Exception("Only 'full' argument can be accepted for extent in Parity Oscillation circuit")
    q = QuantumRegister(n,'q')
    delta = Parameter('t')
    deltaneg = Parameter('-t')
    circ = get_ghz_simple(n,obs=False)
    circinv = circ.inverse()
    circ.barrier()
    circ.u2(delta,deltaneg,q)
    circ.barrier()
    meas = get_ghz_measurement(n, extent)
    circ = circ + meas
    circ.draw()
    return circ, [delta, deltaneg]