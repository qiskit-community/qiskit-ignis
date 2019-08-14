from qiskit import *
from qiskit.circuit import Parameter
import numpy as np
import matplotlib.pyplot as plt


def ordered_list_generator(counts_dictionary,qn):
    '''
    For parity oscillations; just arranges dictionary of counts in bitwise binary order to compute dot products more easily
    '''
    orderedlist=[]
    limit = 2**qn
    keycode = str(qn+2)
    for a in range(limit):
        keyq = str(format(a, '#0'+keycode+'b'))[2:]
        if keyq in counts_dictionary.keys():
            orderedlist.append(counts_dictionary[str(format(a, '#0'+keycode+'b'))[2:]])
        else:
            orderedlist.append(0.0)
    return np.asarray(orderedlist)

def composite_pauli_z(qn):
    '''
    Generates n tensored pauli z matrix upon input of qubit number
    '''
    composite_sigma_z = sigma_z = np.array([[1, 0],[0, -1]])
    for a in range(1,qn):
        composite_sigma_z = np.kron(composite_sigma_z,sigma_z)
        
    return composite_sigma_z

def composite_pauli_z_expvalue(counts_dictionary,qn):
    '''
    Generates expectation value of n tensored pauli matrix upon input of qubit number and composite pauli matrix
    '''    
    return np.dot(ordered_list_generator(counts_dictionary,qn),np.diag(composite_pauli_z(qn)))

class Plotter:
    '''
    Various plots of the |000...0> state in MQC and PO experiments
    '''
    def __init__(self, label):
        self.label = label

    def title_maker(self):
        if self.label == 'mqc':
            title = 'Raw counts of ground state'
            title_ext = 'Error mitigated vs Raw counts of ground state (MQC)'
        elif self.label == 'po':
            title = 'Raw counts of <pauli_z operator>'
            title_ext = 'Error mitigated vs Raw counts of <pauli_z operator>'
            
        return title,title_ext
    def sin_plotter(self,x, y, y_m = None):
        title,title_ext = self.title_maker()
        if y_m is None:
            plt.plot(x,y)
            plt.title(title)
            plt.xlabel('\phi')
            plt.ylabel('Normalized Contrast')
        else:
            plt.plot(x,y,  label='raw')
            plt.plot(x,y_m, label='error mitigated')
            plt.legend()
            plt.title(title_ext)
            plt.xlabel('\phi')
            plt.ylabel('Normalized Contrast')
    def get_fourier_info(self, qn, x, y, y_m, p_dict):
        norm = len(x)
        n = qn
        if y_m is None:
            raise Exception("Two tuples must be provided")
        if p_dict is None:
            raise Exception("P0 and P1 must be provided")
        else:
            P0, P1, P0_m, P1_m = p_dict['P0'], p_dict['P1'], p_dict['P0_m'], p_dict['P1_m']
            fft = (1/norm)*np.fft.fft(y)
            fft_m = (1/norm)*np.fft.fft(y_m)
            fftx = norm*np.fft.fftfreq(x.shape[-1])
            plt.plot(fftx,np.absolute(fft),'.',label='raw')
            plt.plot(fftx,np.absolute(fft_m),'.',label='error mitigated')
            plt.title('Phase Fourier trans. Error mitigated vs Raw counts of |00...0> state (PO Exp.)')
            plt.xlabel(r'v')
            plt.ylabel(r'I(v)')
            plt.legend()
            if self.label == 'mqc':
                I0 = fft[0]
                In = fft[n]
                I0_m = fft_m[0]
                In_m = fft_m[n]
                (LB, UB) = (2*np.sqrt(np.absolute(In)) ,
                            np.sqrt(np.absolute(In)) + np.sqrt(np.absolute(I0)/2))
                (LB_m, UB_m) = (2*np.sqrt(np.absolute(In_m)) ,
                                np.sqrt(np.absolute(In_m)) + np.sqrt(np.absolute(I0_m)/2))
                print("Upper/Lower raw fidelity bounds = ", round(UB,3)," +/- ",
                      round(.01*UB,3)," || ", round(LB,3), " +/- ",round(.01*LB,3))
                print("Upper/Lower error mitigated fidelity bounds = ", round(UB_m,3)," +/- ",
                      round(.01*UB_m,3)," || ", round(LB_m,3), " +/- ",round(.01*LB_m,3))
                F, F_m = .5*(P0_m + P1_m) + np.sqrt(np.absolute(fft[qn])), .5*(P0_m + P1_m) + np.sqrt(np.absolute(fft_m[qn]))
                print("Raw fidelity = ",round(F,3)," +/- ",round(.01*F,3))
                print("Mitigated fidelity = ",round(F_m,3)," +/- ",round(.01*F_m,3))
                
                return {'I0':I0, 'In':In, 'I0_m':I0_m, 'In_m':In_m, 'LB':LB, 'UB':UB,'LB_m':LB_m, 'UB_m':UB_m, 'F':F, 'F_m':F_m}

            elif self.label == 'po':
                In = fft[n]
                In_m = fft_m[n]
                C, C_m = 2*np.absolute(In), 2*np.absolute(In_m)
                F, F_m = .5*(P0+ P1+C), .5*(P0_m + P1_m+C_m)
                print("Raw fidelity = " , round(F,3)," +/- ",round(.01*F,3))
                print("Mitigated fidelity = ", round(F_m,3), " +/- ",round(.01*F_m,3))
                
                return {'In':In, 'In_m':In_m , 'F': F, 'F_m':F_m}

def rho_to_fidelity(rho):
    return np.abs((rho[0,0]+rho[-1,-1]+np.abs(rho[0,-1])+np.abs(rho[-1,0]))/2)
        