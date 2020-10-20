# for Mac python 3.5 compatibility
# pylint: disable=import-outside-toplevel,invalid-name

"""
This module provides several miscellaneous tools for
analysis of the GHZ State (most notably, Fourier Analysis)
"""

from typing import Dict, Tuple
import numpy as np

try:
    from matplotlib import pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def ordered_list_generator(counts_dictionary: Dict, qn: int) -> np.array:
    """
    For parity oscillations; just arranges dictionary of counts
    in bitwise binary order to compute dot products more easily

    Args:
        counts_dictionary: The counts
        qn: Number of qubits

    Returns:
        The counts rearrangement
    """
    orderedlist = []
    limit = 2**qn

    keycode = str(qn + 2)
    for a in range(limit):
        keyq = str(format(a, '#0' + keycode + 'b'))[2:]
        orderedlist.append(counts_dictionary.get(keyq, 0.0))
    return np.asarray(orderedlist)


def composite_pauli_z(qn: int) -> np.array:
    """
    Generates n tensored pauli z matrix upon input of qubit number

    Args:
        qn: Number of qubits

    Returns:
        The tensored pauli z matrices
    """
    composite_sigma_z = sigma_z = np.array([[1, 0], [0, -1]])
    for _ in range(1, qn):
        composite_sigma_z = np.kron(composite_sigma_z, sigma_z)

    return composite_sigma_z


def composite_pauli_z_expvalue(counts_dictionary: Dict, qn: int) -> float:
    """
    Generates expectation value of n tensored pauli matrix
    upon input of qubit number and composite pauli matrix

    Args:
        counts_dictionary: The counts
        qn: Number of qubits

    Returns:
        The expectation value
    """
    return np.dot(ordered_list_generator(counts_dictionary, qn),
                  np.diag(composite_pauli_z(qn)))


class Plotter:
    """
    Various plots of the ground state in MQC and PO experiments
    """
    def __init__(self, label):
        self.label = label

    def title_maker(self) -> Tuple[str, str]:
        """
        Make title depending on type of exp.

        Returns:
            The title strings
        """
        if self.label == 'mqc':
            title = 'Raw counts of ground state'
            title_ext = 'Error mitigated vs Raw counts of ground state (MQC)'
        elif self.label == 'po':
            title = 'Raw counts of <pauli_z operator>'
            title_ext = 'Error mitigated vs Raw counts of <pauli_z operator>'

        return title, title_ext

    def sin_plotter(self,
                    x: np.array,
                    y: np.array,
                    y_m: np.array = None
                    ):
        """
        Make sin plot of counts in both mqc and po exps.
        Args:
            x: Phase series
            y: Counts series
            y_m: Mitigated counts series

        Raises:
            ImportError: If matplotlib is not present
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib needs to be installed and properly "
                              "configured needed to plot. You can run "
                              "'pip install matplotlib'")
        title, title_ext = self.title_maker()
        if y_m is None:
            plt.plot(x, y)
            plt.title(title)
            plt.xlabel(r'\phi')
            plt.ylabel('Normalized Contrast')
        else:
            plt.plot(x, y, label='raw')
            plt.plot(x, y_m, label='error mitigated')
            plt.legend()
            plt.title(title_ext)
            plt.xlabel(r'\phi')
            plt.ylabel('Normalized Contrast')

    def get_fourier_info(self,
                         qn: int,
                         x: np.array,
                         y: np.array,
                         y_m: np.array,
                         p_dict: Dict
                         ) -> Dict:
        """
        Get fourier trans. data/plot of both mqc and po exps.
        Args:
            qn: Number of qubits
            x: Phase series
            y: Counts series
            y_m: Mitigated counts series
            p_dict: probabilities dictionary
        Returns:
            The fourier transform data
        Raises:
            ImportError: If matplotlib is not present
            Exception: If y_m or p_dict are not provided
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib needs to be installed and properly "
                              "configured needed to plot. You can run "
                              "'pip install matplotlib'")
        norm = len(x)
        n = qn
        if y_m is None:
            raise Exception("Two tuples must be provided")

        if p_dict is None:
            raise Exception("P0 and P1 must be provided")

        P0, P1 = p_dict['P0'], p_dict['P1']
        P0_m, P1_m = p_dict['P0_m'], p_dict['P1_m']
        fft = (1/norm)*np.fft.fft(y)
        fft_m = (1/norm)*np.fft.fft(y_m)
        fftx = norm*np.fft.fftfreq(x.shape[-1])
        plt.plot(fftx, np.absolute(fft), '.', label='raw')
        plt.plot(fftx, np.absolute(fft_m), '.', label='error mitigated')
        t_s = 'Error mitigated vs Raw counts of |00...0> state (PO Exp.)'
        plt.title(t_s)
        plt.xlabel(r'v')
        plt.ylabel(r'I(v)')
        plt.legend()
        if self.label == 'mqc':
            I0 = fft[0]
            In = fft[n]
            I0_m = fft_m[0]
            In_m = fft_m[n]
            (LB, UB) = (2*np.sqrt(np.absolute(In)),
                        np.sqrt(np.absolute(In))
                        + np.sqrt(np.absolute(I0)/2))
            (LB_m, UB_m) = (2*np.sqrt(np.absolute(In_m)),
                            np.sqrt(np.absolute(In_m))
                            + np.sqrt(np.absolute(I0_m)/2))
            print("Upper/Lower raw fidelity bounds = ",
                  round(UB, 3), " +/- ", round(.01*UB, 3),
                  " || ", round(LB, 3), " +/-", round(.01*LB, 3))
            print("Upper/Lower error mitigated fidelity bounds = ",
                  round(UB_m, 3), " +/- ", round(.01*UB_m, 3), " || ",
                  round(LB_m, 3), " +/- ", round(.01*LB_m, 3))
            (F, F_m) = (.5*(P0_m + P1_m) + np.sqrt(np.absolute(fft[qn])),
                        .5*(P0_m + P1_m) + np.sqrt(np.absolute(fft_m[qn])))
            print("Raw fidelity = ", round(F, 3), " +/- ", round(.01*F, 3))
            print("Mitigated fidelity = ", round(F_m, 3),
                  " +/- ", round(.01*F_m, 3))

            return {'I0': I0, 'In': In, 'I0_m': I0_m,
                    'In_m': In_m, 'LB': LB, 'UB': UB,
                    'LB_m': LB_m, 'UB_m': UB_m, 'F': F,
                    'F_m': F_m}

        if self.label == 'po':
            In = fft[n]
            In_m = fft_m[n]
            C, C_m = 2*np.absolute(In), 2*np.absolute(In_m)
            F, F_m = .5*(P0 + P1 + C), .5*(P0_m + P1_m + C_m)
            print("Raw fidelity = ", round(F, 3), " +/- ", round(.01*F, 3))
            print("Mitigated fidelity = ", round(F_m, 3),
                  " +/- ", round(.01*F_m, 3))

            return {'In': In, 'In_m': In_m, 'F': F, 'F_m': F_m}

        return None


def rho_to_fidelity(rho: float) -> float:
    """
    Get fidelity given rho
    Args:
            rho: The density matrix
    Returns:
        The fidelity value for rho
    """
    return np.abs((rho[0, 0] + rho[-1, -1] + np.abs(rho[0, -1])
                   + np.abs(rho[-1, 0]))/2)
