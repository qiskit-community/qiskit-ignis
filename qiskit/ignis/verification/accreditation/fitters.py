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

import sys
import numpy as np
from qiskit import QiskitError


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

        for k in range(len(outputs_list)):

            # Classical postprocessing
            output = [1 if s == "1" else 0 for s in outputs_list[k][0]]
            postp = postp_list[k][0]

            for i, _ in enumerate(output):
                output[i] = (output[i] + postp[i]) % 2

            if k != v_zero:
                # Check if trap returns correct output
                if output != [0] * len(output):
                    self.flag = 'rejected'
            else:
                output_target = output

        if self.flag == 'accepted':
            if self.N_acc != 0:
                self.N_acc = self.N_acc+1
                self.outputs = np.vstack((self.outputs, output_target))
            else:
                self.N_acc = self.N_acc+1
                self.outputs = output_target

    def bound_variation_distance(self, theta):
        """
            This function computes the bound on variation distance based and
            the confidence
            Args:
                theta (float): number between 0 and 1
        """
        if self.N_acc == 0:
            sys.exit()
        if self.N_acc/self.num_runs > theta:
            self.bound = self.g_num*1.7/(self.num_traps+1)
            self.bound = self.bound/(self.N_acc/self.num_runs-theta)
            self.bound = self.bound+1-self.g_num
            self.confidence = 1-2*np.exp(-2*theta*self.num_runs*self.num_runs)
        if self.bound > 1:
            self.bound = 1
