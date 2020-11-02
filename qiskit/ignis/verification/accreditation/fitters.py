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

# pylint: disable=invalid-name


"""
Class for accreditation protocol

Implementation follows the methods from
Samuele Ferracin, Theodoros Kapourniotis and Animesh Datta
New Journal of Physics, Volume 21, November 2019
https://iopscience.iop.org/article/10.1088/1367-2630/ab4fd6
"""


import numpy as np
from qiskit import QiskitError
from .qotp import QOTPCorrectString


class AccreditationFitter:
    """
    Class for fitters for accreditation

    Implementation follows the methods from
    Samuele Ferracin, Theodoros Kapourniotis and Animesh Datta
    New Journal of Physics, Volume 21, November 2019
    https://iopscience.iop.org/article/10.1088/1367-2630/ab4fd6
    """
    def __init__(self):
        self.bound = 1
        self.confidence = 1
        self.N_acc = 0
        self.n_rejects_per_run = []
        self.num_runs = 0
        self.flag = 'accepted'
        self.outputs = []
        self.num_runs = 0
        self.num_traps = 0
        self.g_num = 1

    def AppendResults(self, results, postp_list, v_zero):
        """
        Single run of accreditation protocol, data input as
        qiskit result object

        Args:
            results (Result): results of the quantum job
            postp_list (list): list of strings used to post-process outputs
            v_zero (int): position of target
        """
        strings = []
        for ind, _ in enumerate(postp_list):
            # Classical postprocessing
            # check single shot and extract string
            counts = results.get_counts(ind)
            shots = 0
            countstring = None
            for countstring, val in counts.items():
                shots += val
            if shots != 1 or countstring is None:
                QiskitError("ERROR: not single shot data")
            strings.append(countstring)
        self.single_protocol_run(strings, postp_list, v_zero)

    def AppendStrings(self, strings, postp_list, v_zero):
        """
        Single run of accreditation protocol, data input as
        a list of output strings

        Args:
            strings (list): stringlist of outputs
            postp_list (list): list of strings used to post-process outputs
            v_zero (int): position of target
        """
        self.single_protocol_run(strings, postp_list, v_zero)

    def single_protocol_run(self, strings, postp_list, v_zero):
        """
        Single protocol run of accreditation protocol

        Args:
            strings (list): outputs of the quantum job
            postp_list (list): list of strings used to post-process outputs
            v_zero (int): position of target
        """
        self.num_runs = self.num_runs + 1
        self.flag = 'accepted'

        # Check that correct number of traps is input
        if self.num_runs == 1:
            self.num_traps = len(postp_list)-1
        else:
            if len(postp_list)-1 != self.num_traps:
                QiskitError("ERROR: Run protocol with the"
                            "same number of traps")
        if self.num_traps < 3:
            QiskitError("ERROR: run the protocol with at least 3 traps")
        k = 0
        self.n_rejects_per_run.append(0)
        for s, p in zip(strings, postp_list):
            if k != v_zero:
                # Check if trap returns correct output
                meas = QOTPCorrectString(s, p)
                if meas != '0' * len(meas):
                    self.flag = 'rejected'
                    self.n_rejects_per_run[-1] += 1
            else:
                output_target = QOTPCorrectString(s, p)
            k += 1
        if self.flag == 'accepted':
            self.N_acc += 1
            self.outputs.append(output_target)

    def bound_variation_distance(self, theta):
        """
        This function computes the bound on variation distance based and
        the confidence

        Args:
            theta (float): number between 0 and 1
        """
        if self.N_acc == 0:
            QiskitError("ERROR: Variation distance requires"
                        "at least one accepted run")
        if self.N_acc/self.num_runs > theta:
            self.bound = self.g_num*1.7/(self.num_traps+1)
            self.bound = self.bound/(self.N_acc/self.num_runs-theta)
            self.bound = self.bound+1-self.g_num
            self.confidence = 1-2*np.exp(-2*theta*self.num_runs*self.num_runs)
        if self.bound > 1:
            self.bound = 1
