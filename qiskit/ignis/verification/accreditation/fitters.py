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
    Class for accreditation protocol
    Based on Ferracin et al, Accrediting outputs of noisy intermediate-scale
    quantum devices, arXiv:1811.09709
"""


import numpy as np
from qiskit import QiskitError
from .qotp import QOTPCorrectCounts


class accreditationFitter:
    '''
        Class for fitters for accreditation
    '''

    bound = 1
    confidence = 1
    N_acc = 0
    num_runs = 0
    flag = 'accepted'
    outputs = []
    num_runs = 0
    num_traps = 0
    g_num = 1

    def single_protocol_run(self, outputs_list, postp_list, v_zero):
        """
            Single protocol run of accreditation protocol on simul backend
            Args:
                circuit_list (list): list of all circuits, target and traps
                postp_list (list): list of strings used to post-process outputs
                v_zero (int): position of target
                noise_model
                basis_gates (list)
        """
        self.num_runs = self.num_runs + 1
        self.flag = 'accepted'

        # Check that correct number of traps is input
        if self.num_runs == 1:
            self.num_traps = len(outputs_list)-1
        else:
            if len(outputs_list)-1 != self.num_traps:
                QiskitError("ERROR: Run protocol with the " +
                            "same number of traps")

        if self.num_traps < 3:
            QiskitError("ERROR: run the protocol with at least 3 traps")
        allcounts = []
        for output, postp in zip(outputs_list, postp_list):

            # Classical postprocessing
            # check single shot and extract string
            counts = output.get_counts()
            counts = QOTPCorrectCounts(counts, postp)
            shots = 0
            countstring = None
            for countstring, val in counts.items():
                shots += val
            if shots != 1 or countstring is None:
                QiskitError("ERROR: not single shot data")
            allcounts.append(countstring)
        for k, count in enumerate(allcounts):
            if k != v_zero:
                # Check if trap returns correct output
                if count != '0' * len(count):
                    self.flag = 'rejected'
            else:
                output_target = count
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
                        + "at least one accepted run")
        if self.N_acc/self.num_runs > theta:
            self.bound = self.g_num*1.7/(self.num_traps+1)
            self.bound = self.bound/(self.N_acc/self.num_runs-theta)
            self.bound = self.bound+1-self.g_num
            self.confidence = 1-2*np.exp(-2*theta*self.num_runs*self.num_runs)
        if self.bound > 1:
            self.bound = 1
