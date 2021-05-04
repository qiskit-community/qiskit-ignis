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
as well as Ferracin, Merkel, McKay and Datta
https://arxiv.org/abs/2103.06603
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
    as well as Ferracin, Merkel, McKay and Datta
    https://arxiv.org/abs/2103.06603
    """

    def __init__(self):
        self._counts_all = {}
        self._counts_accepted = {}
        self._Ntraps = None
        self._Nrejects = []
        self._Nruns = 0
        self._Nacc = 0
        self._g = 1.0

    def Reset(self):
        """
        Reset the accreditation class object

        Args:
        """
        self._counts_all = {}
        self._counts_accepted = {}
        self._Ntraps = None
        self._Nrejects = []
        self._Nruns = 0
        self._Nacc = 0
        self._g = 1.0

    def AppendResults(self, results, postp_list, v_zero):
        """
        Single run of accreditation protocol, data input as
        qiskit result object assumed to be single shot

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
        self._AppendData(strings, postp_list, v_zero)

    def AppendStrings(self, strings, postp_list, v_zero):
        """
        Single run of accreditation protocol, data input as
        a list of output strings

        Args:
            strings (list): stringlist of outputs
            postp_list (list): list of strings used to post-process outputs
            v_zero (int): position of target
        """
        self._AppendData(strings, postp_list, v_zero)

    def FullAccreditation(self, confidence):
        """
        This function computes the bound on variation distance based and
        the confidence interval desired.  This protocol is from the earlier
        paper and fully treats non-Markovian errors

        Args:
            confidence (float): number between 0 and 1

        Returns:
            dict: dict of postselected target counts
            float: 1-norm bound from noiseless samples
            float: confidence
        """
        if self._Nacc == 0:
            QiskitError("ERROR: Variation distance requires"
                        "at least one accepted run")
        if confidence > 1 or confidence < 0:
            QiskitError("ERROR: Confidence must be"
                        "between 0 and 1")
        theta = np.sqrt(np.log(2/(1-confidence))/(2*self._Nruns))
        if self._Nacc/self._Nruns > theta:
            bound = self._g*1.7/(self._Ntraps+1)
            bound = bound/(self._Nacc/self._Nruns-theta)
            bound = bound+1-self._g
        else:
            bound = 1
        self.bound = min(self.bound, 1)
        return self._counts_accepted, bound, confidence

    def MeanAccreditation(self, confidence):
        """
        This function computes the bound on variation distance based and
        the confidence interval desired.  This protocol is from the second
        paper and assumes Markovianity

        Args:
            confidence (float): number between 0 and 1

        Returns:
            dict: dict of corrected target counts
            float: 1-norm bound from noiseless samples
            float: confidence
        """
        theta = np.sqrt(np.log(2/(1-confidence))/(2*self._Nruns))
        bound = 2*np.sum(self._Nrejects)/self._Nruns/self._Ntraps + theta
        self.bound = min(self.bound, 1)
        return self._counts_all, bound, confidence

    def _AppendData(self, strings, postp_list, v_zero):
        """
        Single protocol run of accreditation protocol on simul backend

        Args:
            strings (list): bit string results
            postp_list (list): list of strings used to post-process outputs
            v_zero (int): position of target
        """
        # Check that correct number of traps is input
        if self._Ntraps is None:
            self._Ntraps = len(postp_list)-1
        else:
            if len(postp_list)-1 != self._Ntraps:
                QiskitError("ERROR: Run protocol with the"
                            "same number of traps")
        if self._Ntraps < 3:
            QiskitError("ERROR: run the protocol with at least 3 traps")
        self._Nruns += 1
        self._Nrejects.append(0)
        flag = True
        for ind, (s, p) in enumerate(zip(strings, postp_list)):
            if ind != v_zero:
                # Check if trap returns correct output
                meas = QOTPCorrectString(s, p)
                if meas != '0' * len(meas):
                    flag = False
                    self._Nrejects[-1] += 1
            else:
                target_count = QOTPCorrectString(s, p)
                if target_count in self._counts_all.keys():
                    self._counts_all[target_count] += 1
                else:
                    self._counts_all[target_count] = 1
        if flag:
            self._Nacc += 1
            if target_count in self._counts_accepted.keys():
                self._counts_accepted[target_count] += 1
            else:
                self._counts_accepted[target_count] = 1
