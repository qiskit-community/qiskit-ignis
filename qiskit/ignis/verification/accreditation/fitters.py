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
"""


import numpy as np
from qiskit import QiskitError
from qiskit.utils.deprecation import deprecate_function
from .qotp import QOTPCorrectString, QOTPCorrectCounts


class AccreditationFitter:
    """
    Class for fitters for accreditation

    Implementation follows the methods from [1] FullAccreditation
    and [2] MeanAccreditation.

    Data can be input either as qiskit result objects, or as
    lists of bitstrings (the latter is useful for batch jobs).

    References:
        1. S. Ferracin, T. Kapourniotis, A. Datta.
           *Accrediting outputs of noisy intermediate-scale quantum computing devices*,
           New Journal of Physics, Volume 21, 113038. (2019).
           `NJP 113038 <https://iopscience.iop.org/article/10.1088/1367-2630/ab4fd6>`_
        2. S. Ferracin, S. Merkel, D. McKay, A. Datta.
           *Experimental accreditation of outputs of noisy quantum computers*,
           arxiv:2103.06603 (2021).
           `arXiv:quant-ph/2103.06603 <https://arxiv.org/abs/2103.06603>`_
    """

    def __init__(self):
        self._counts_all = {}
        self._counts_accepted = {}
        self._Ntraps = None
        self._Nrejects = []
        self._Nruns = 0
        self._Nacc = 0
        self._g = 1.0

        # all to be deprecated
        self.flag = None
        self.outputs = None
        self.num_runs = None
        self.N_acc = None
        self.bound = None
        self.confidence = None

    def Reset(self):
        """
        Reset the accreditation class object
        """
        self._counts_all = {}
        self._counts_accepted = {}
        self._Ntraps = None
        self._Nrejects = []
        self._Nruns = 0
        self._Nacc = 0
        self._g = 1.0

        # all to be deprecated
        self.flag = None
        self.outputs = None
        self.num_runs = None
        self.N_acc = None
        self.bound = None
        self.confidence = None

    def AppendResults(self, results, postp_list, v_zero):
        """
        Single run of accreditation protocol, data input as
        qiskit result object assumed to be single shot

        Args:
            results (Result): results of the quantum job
            postp_list (list): list of strings used to post-process outputs
            v_zero (int): position of target

        Raises:
            QiskitError: If the data is not single shot
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
                raise QiskitError("ERROR: not single shot data")
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
        the confidence interval desired.  This protocol is from [1]
        and fully treats non-Markovian errors

        Args:
            confidence (float): number between 0 and 1

        Returns:
            dict: dict of postselected target counts
            float: 1-norm bound from noiseless samples
            float: confidence

        Raises:
            QiskitError: If no runs are accepted
                         or confidence is outside of 0,1
        """
        if self._Nacc == 0:
            raise QiskitError("ERROR: Variation distance requires"
                              "at least one accepted run")
        if confidence > 1 or confidence < 0:
            raise QiskitError("ERROR: Confidence must be"
                              "between 0 and 1")
        theta = np.sqrt(np.log(2/(1-confidence))/(2*self._Nruns))
        if self._Nacc/self._Nruns > theta:
            bound = self._g*1.7/(self._Ntraps+1)
            bound = bound/(self._Nacc/self._Nruns-theta)
            bound = bound+1-self._g
        else:
            bound = 1
        bound = min(bound, 1)
        return self._counts_accepted, bound, confidence

    def MeanAccreditation(self, confidence):
        """
        This function computes the bound on variation distance based and
        the confidence interval desired.  This protocol is from [2]
        and assumes Markovianity but gives an improved bound

        Args:
            confidence (float): number between 0 and 1

        Returns:
            dict: dict of corrected target counts
            float: 1-norm bound from noiseless samples
            float: confidence
        """
        theta = np.sqrt(np.log(2/(1-confidence))/(2*self._Nruns))
        bound = 2*np.sum(self._Nrejects)/self._Nruns/self._Ntraps + theta
        bound = min(bound, 1)
        return self._counts_all, bound, confidence

    def _AppendData(self, strings, postp_list, v_zero):
        """
        Single protocol run of accreditation protocol

        Args:
            strings (list): bit string results
            postp_list (list): list of strings used to post-process outputs
            v_zero (int): position of target

        Raises:
            QiskitError: If the number of circuits is inconsistent or
                         if there are not at least 3 traps
        """
        # Check that correct number of traps is input
        if self._Ntraps is None:
            self._Ntraps = len(postp_list)-1
        else:
            if len(postp_list)-1 != self._Ntraps:
                raise QiskitError("ERROR: Run protocol with the"
                                  "same number of traps")
        if self._Ntraps < 3:
            raise QiskitError("ERROR: run the protocol with at least 3 traps")
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

    @deprecate_function('single_protocol_run is being deprecated. '
                        'Use AppendResult or AppendString')
    def single_protocol_run(self, results, postp_list, v_zero):
        """
        DEPRECATED-Single protocol run of accreditation protocol
        Args:
            results (Result): results of the quantum job
            postp_list (list): list of strings used to post-process outputs
            v_zero (int): position of target

        Raises:
            QiskitError: If the number of circuits is inconsistent or
                         if there are not at least 3 traps or
                         if the data is not single shot
        """
        self._Nruns = self._Nruns + 1
        self.flag = 'accepted'

        # Check that correct number of traps is input
        if self._Nruns == 1:
            self._Ntraps = len(postp_list)-1
        else:
            if len(postp_list)-1 != self._Ntraps:
                raise QiskitError("ERROR: Run protocol with the"
                                  "same number of traps")

        if self._Ntraps < 3:
            raise QiskitError("ERROR: run the protocol with at least 3 traps")
        allcounts = []
        for ind, postp in enumerate(postp_list):

            # Classical postprocessing
            # check single shot and extract string
            counts = results.get_counts(ind)
            counts = QOTPCorrectCounts(counts, postp)
            shots = 0
            countstring = None
            for countstring, val in counts.items():
                shots += val
            if shots != 1 or countstring is None:
                raise QiskitError("ERROR: not single shot data")
            allcounts.append(countstring)
        for k, count in enumerate(allcounts):
            if k != v_zero:
                # Check if trap returns correct output
                if count != '0' * len(count):
                    self.flag = 'rejected'
            else:
                output_target = count
        if self.flag == 'accepted':
            self._Nacc += 1
            if output_target in self._counts_accepted.keys():
                self._counts_accepted[output_target] += 1
            else:
                self._counts_accepted[output_target] = 1
        self.outputs = self._counts_accepted
        self.num_runs = self._Nruns
        self.N_acc = self._Nacc

    @deprecate_function('bound_variation_distance is being deprecated. '
                        'Use FullAccreditation or MeanAccreditation')
    def bound_variation_distance(self, theta):
        """
        DEPRECATED-This function computes the bound on variation distance based and
        the confidence
        Args:
            theta (float): number between 0 and 1

        Raises:
            QiskitError: If there is not an accepted run
        """
        if self._Nacc == 0:
            raise QiskitError("ERROR: Variation distance requires"
                              "at least one accepted run")
        if self._Nacc/self._Nruns > theta:
            self.bound = self._g*1.7/(self._Ntraps+1)
            self.bound = self.bound/(self._Nacc/self._Nruns-theta)
            self.bound = self.bound+1-self._g
            self.confidence = 1-2*np.exp(-2*theta*self._Nruns*self._Nruns)
        self.bound = min(self.bound, 1)
