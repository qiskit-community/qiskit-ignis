# -*- coding: utf-8 -*-

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

# pylint: disable=invalid-name

"""Generates circuits for quantum error correction."""

from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit


class RepetitionCode:
    """
    Implementation of a distance d repetition code, implemented over
    T syndrome measurement rounds.
    """

    def __init__(self, d, T=0, xbasis=False, resets=False, delay=0):
        """
        Creates the circuits corresponding to a logical 0 and 1 encoded
        using a repetition code.

        Args:
            d (int): Number of code qubits (and hence repetitions) used.
            T (int): Number of rounds of ancilla-assisted syndrome measurement.
            xbasis (bool): Whether to use the X basis to use for encoding (Z basis used by default).
            resets (bool): Whether to include a reset gate after mid-circuit measurements.
            delay (float): Time (in dt) to delay after mid-circuit measurements (and delay).


        Additional information:
            No measurements are added to the circuit if `T=0`. Otherwise
            `T` rounds are added, followed by measurement of the code
            qubits (corresponding to a logical measurement and final
            syndrome measurement round).
        """

        self.d = d
        self.T = 0

        self.code_qubit = QuantumRegister(d, "code_qubit")
        self.link_qubit = QuantumRegister((d - 1), "link_qubit")
        self.qubit_registers = {"code_qubit", "link_qubit"}

        self.link_bits = []
        self.code_bit = ClassicalRegister(d, "code_bit")

        self.circuit = {}
        for log in ["0", "1"]:
            self.circuit[log] = QuantumCircuit(
                self.link_qubit, self.code_qubit, name=log
            )

        self._xbasis = xbasis
        self._resets = resets

        self._preparation()

        for _ in range(T - 1):
            self.syndrome_measurement(delay=delay)

        if T != 0:
            self.syndrome_measurement(final=True)
            self.readout()

    def get_circuit_list(self):
        """
        Returns:
            circuit_list: self.circuit as a list, with
            circuit_list[0] = circuit['0']
            circuit_list[1] = circuit['1']
        """
        circuit_list = [self.circuit[log] for log in ["0", "1"]]
        return circuit_list

    def x(self, logs=("0", "1"), barrier=False):
        """
        Applies a logical x to the circuits for the given logical values.

        Args:
            logs (list or tuple): List or tuple of logical values expressed as
                strings.
            barrier (bool): Boolean denoting whether to include a barrier at
                the end.
        """
        for log in logs:
            if self._xbasis:
                self.circuit[log].z(self.code_qubit)
            else:
                self.circuit[log].x(self.code_qubit)
            if barrier:
                self.circuit[log].barrier()

    def _preparation(self, barrier=False):
        """
        Prepares logical bit states by applying an x to the circuit that will
        encode a 1.
        """

        for log in ["0", "1"]:
            if self._xbasis:
                self.circuit[log].h(self.code_qubit)
            if barrier:
                self.circuit[log].barrier()

        self.x(["1"])

    def syndrome_measurement(self, final=False, barrier=False, delay=0):
        """
        Application of a syndrome measurement round.

        Args:
            final (bool): Whether this is the final syndrome measurement round.
            barrier (bool): Boolean denoting whether to include a barrier at the end.
            delay (float): Time (in dt) to delay after mid-circuit measurements (and delay).
        """
        self.link_bits.append(
            ClassicalRegister((self.d - 1), "round_" + str(self.T) + "_link_bit")
        )

        for log in ["0", "1"]:

            self.circuit[log].add_register(self.link_bits[-1])

            if self._xbasis:
                self.circuit[log].h(self.link_qubit)

            for j in range(self.d - 1):
                if self._xbasis:
                    self.circuit[log].cx(self.link_qubit[j], self.code_qubit[j])
                else:
                    self.circuit[log].cx(self.code_qubit[j], self.link_qubit[j])

            for j in range(self.d - 1):
                if self._xbasis:
                    self.circuit[log].cx(self.link_qubit[j], self.code_qubit[j + 1])
                else:
                    self.circuit[log].cx(self.code_qubit[j + 1], self.link_qubit[j])

            if self._xbasis:
                self.circuit[log].h(self.link_qubit)

            for j in range(self.d - 1):
                self.circuit[log].measure(
                    self.link_qubit[j], self.link_bits[self.T][j])
                if self._resets and not final:
                    self.circuit[log].reset(self.link_qubit[j])
                if delay > 0 and not final:
                    self.circuit[log].delay(delay, self.link_qubit[j])

            if barrier:
                self.circuit[log].barrier()

        self.T += 1

    def readout(self):
        """
        Readout of all code qubits, which corresponds to a logical measurement
        as well as allowing for a measurement of the syndrome to be inferred.
        """

        for log in ["0", "1"]:
            if self._xbasis:
                self.circuit[log].h(self.code_qubit)
            self.circuit[log].add_register(self.code_bit)
            self.circuit[log].measure(self.code_qubit, self.code_bit)

    def process_results(self, raw_results):
        """
        Args:
            raw_results (dict): A dictionary whose keys are logical values,
                and whose values are standard counts dictionaries, (as
                obtained from the `get_counts` method of a ``qiskit.Result``
                object).

        Returns:
            results: Dictionary with the same structure as the input, but with
                the bit strings used as keys in the counts dictionaries
                converted to the form required by the decoder.

        Additional information:
            The circuits must be executed outside of this class, so that
            their is full freedom to compile, choose a backend, use a
            noise model, etc. The results from these executions should then
            be used to create the input for this method.
        """
        results = {}
        for log in raw_results:
            results[log] = {}
            for string in raw_results[log]:

                # logical readout taken from
                measured_log = string[0] + " " + string[self.d - 1]

                # final syndrome deduced from final code qubit readout
                full_syndrome = ""
                for j in range(self.d - 1):
                    full_syndrome += "0" * (string[j] == string[j + 1]) + "1" * (
                        string[j] != string[j + 1]
                    )
                # results from all other syndrome measurements then added
                full_syndrome = full_syndrome + string[self.d:]

                # changes between one syndrome and the next then calculated
                syndrome_list = full_syndrome.split(" ")
                syndrome_changes = ""
                for t in range(self.T + 1):
                    for j in range(self.d - 1):
                        if self._resets:
                            if t == 0:
                                change = (syndrome_list[-1][j] != "0")
                            else:
                                change = (syndrome_list[-t - 1][j]
                                          != syndrome_list[-t][j])
                        else:
                            if t <= 1:
                                if t != self.T:
                                    change = (syndrome_list[-t - 1][j] != "0")
                                else:
                                    change = (syndrome_list[-t - 1][j]
                                              != syndrome_list[-t][j])
                            elif t == self.T:
                                last3 = ""
                                for dt in range(3):
                                    last3 += syndrome_list[-t - 1 + dt][j]
                                change = last3.count("1") % 2 == 1
                            else:
                                change = (syndrome_list[-t - 1][j]
                                          != syndrome_list[-t + 1][j])
                        syndrome_changes += "0" * (not change) + "1" * change
                    syndrome_changes += " "

                # the space separated string of syndrome changes then gets a
                # double space separated logical value on the end
                new_string = measured_log + "  " + syndrome_changes[:-1]

                results[log][new_string] = raw_results[log][string]

        return results

    def _get_all_processed_results(self):
        """
        Returns:
            results: list of all processed results stemming from single qubit bitflip errors,
                which is used to create the decoder graph.
        """

        syn = RepetitionCodeSyndromeGenerator(self)

        T = syn.T  # number of rounds of stabilizer measurements
        d = syn.d  # number of data qubits

        results = []
        for r in range(T):
            for i in range(d - 1):
                syn.bitflip_ancilla(i, r)
                results.append(syn.get_processed_results())
                syn.bitflip_ancilla(i, r)  # undo the error
            for i in range(d):
                syn.bitflip_data(i, r, True)
                results.append(syn.get_processed_results())
                syn.bitflip_data(i, r, True)  # undo the error
                syn.bitflip_data(i, r, False)
                results.append(syn.get_processed_results())
                syn.bitflip_data(i, r, False)  # undo the error
        for i in range(d):
            syn.bitflip_readout(i)
            results.append(syn.get_processed_results())
            syn.bitflip_readout(i)  # undo the error

        return results


class RepetitionCodeSyndromeGenerator:
    """
    Allows to construct an error pattern in the circuit of a RepetitionCode object.
    Allows the measurement results to be retrieved without the need to run the simulation.
    """

    def __init__(self, code):
        """
        Args:
            code (RepetitionCode): Code object under consideration.
        """

        self.d = code.d  # number of qubits
        self.T = code.T  # number of rounds (of stabilizer measurements)

        # List of measurement results
        self.m_anc = {}
        self.m_fin = [0] * self.d
        for r in range(self.T):
            self.m_anc[r] = [0] * (self.d - 1)

    def bitflip_readout(self, i):
        """
        Introduces a bitflip error on data qubit i right before the (final) readout.
        Args:
            i (int): Qubit label.
        """
        self.m_fin[i] = (self.m_fin[i] + 1) % 2

    def bitflip_ancilla(self, i, r):
        """
        Introduces a bitflip error to ancilla i in round r.
        Args:
            i (int): Qubit label.
            r (int): Label of round of syndrome extraction.
        """
        self.m_anc[r][i] = (self.m_anc[r][i] + 1) % 2

    def bitflip_data(self, i, r0, middle=False):
        """
        Introduces a bitflip error to data qubit i in round r0.
        Args:
            i (int): Qubit label.
            r0 (int): Label of round of syndrome extraction.
            middle (bool): If False, the error is introduced before the first sequence of CNOTs.
                If True, the error is introduced in between the two CNOT sequences.
        """
        self.m_fin[i] = (self.m_fin[i] + 1) % 2

        # Check for "boundary" code qubits
        if i > 0:  # q[i] is not at the upper(left) boundary
            for r in range(r0, self.T):
                self.m_anc[r][i - 1] = (
                    self.m_anc[r][i - 1] + 1
                ) % 2  # error propagates across 2nd CNOT sequence

        if i < self.d - 1:  # q[i] is not at the lower(right) boundary
            for r in range(r0 + 1, self.T):
                self.m_anc[r][i] = (
                    self.m_anc[r][i] + 1
                ) % 2  # error propagates across 1st CNOT sequence

            self.m_anc[r0][i] = (
                self.m_anc[r0][i] + middle + 1
            ) % 2  # no error induced if it occurs in the middle

    def get_m_ancilla(self, i, r):
        """
        Args:
            i (int): Qubit label.
            r (int): Label of round of syndrome extraction.

        Returns:
            measurement_val: Measurement result of ancilla i in round r for current set of errors.
        """
        measurement_val = self.m_anc[r][i]
        return measurement_val

    def get_m_data(self, i, encoded=0):
        """
        Args:
            i (int): Qubit label.
            encoded (int): Initial logical value of the data qubits.
        Returns:
            measurement_val: Final measurement result of data qubit i for current set of errors.

        """
        measurement_val = (self.m_fin[i] + encoded) % 2
        return measurement_val

    def get_raw_results(self, encoded=0):
        """
        Args:
        Returns:
            raw_result: String of unprocessed results for current set of errors.
        """
        raw_result = ""
        for i in range(self.d - 1, -1, -1):  # qiskit's qubit ordering
            raw_result += str(self.get_m_data(i, encoded))
        for r in range(self.T - 1, -1, -1):  # qiskit's qubit register ordering
            raw_result += " "
            for i in range(self.d - 2, -1, -1):
                raw_result += str(self.get_m_ancilla(i, r))
        return raw_result

    def get_processed_results(self, encoded=0):
        """
        Args:
            encoded (int): Initial logical value of the data qubits.
        Returns:
            processed_result: String of processed results for current set of errors.
        """
        processed_result = (
            str(self.get_m_data(self.d - 1, encoded))
            + " "
            + str(self.get_m_data(0, encoded))
            + "  "
        )
        for i in range(self.d - 2, -1, -1):
            processed_result += str(self.get_m_ancilla(i, 0))
        for r in range(1, self.T):
            processed_result += " "
            for i in range(self.d - 2, -1, -1):  # qiskit's qubit ordering
                processed_result += str(
                    (self.get_m_ancilla(i, r) + self.get_m_ancilla(i, r - 1)) % 2
                )
        processed_result += " "
        for i in range(self.d - 2, -1, -1):  # qiskit's qubit ordering
            processed_result += str(
                (
                    self.get_m_ancilla(i, self.T - 1)
                    + self.get_m_data(i, encoded)
                    + self.get_m_data(i + 1, encoded)
                )
                % 2
            )
        return processed_result
    
    
    class SurfaceCode():
    
    def __init__(self,d,T,basis='z',resets=True):
        
        self.d = d
        self.T = 0
        self._basis = basis
        self._resets = resets
        
        self._num_xy = int((d**2 - 1)/2)
        
        # quantum registers
        self.code_qubit = QuantumRegister(d**2, 'code_qubit')
        self.zplaq_qubit = QuantumRegister(self._num_xy, 'zplaq_qubit')
        self.xplaq_qubit = QuantumRegister(self._num_xy, 'xplaq_qubit')
        self.qubit_registers = [self.code_qubit,self.zplaq_qubit,self.xplaq_qubit]
        
        # classical registers
        self.xplaq_bits = []
        self.zplaq_bits = []
        self.code_bit = ClassicalRegister(d**2, 'code_bit')
        
        # plaquettes
        self.zplaqs, self.xplaqs = self._get_plaquettes()

        # create the circuits
        self.circuit = {}
        for log in ['0', '1']:
            self.circuit[log] = QuantumCircuit(
                self.code_qubit,self.zplaq_qubit,self.xplaq_qubit,
                name=log)
            
        # apply initial logical paulis for encoded states
        self._preparation()

        # add the gates required for syndrome measurements
        for _ in range(T-1):
            self.syndrome_measurement()
        if T != 0:
            self.syndrome_measurement(final=True)
            self.readout()
            
    def _get_plaquettes(self):

        d = self.d
        
        zplaqs = []
        xplaqs = []
        for y in range(-1,d):
            for x in range(-1,d):

                bulk = x in range(d-1) and y in range(d-1)
                ztab = (x==-1 and y%2==0) or (x==d-1 and y%2==1)
                xtab = (y==-1 and x%2==1) or (y==d-1 and x%2==0)

                if (x in range(d-1) or y in range(d-1)) and (bulk or ztab or xtab):
                    plaq = []
                    for dy in range(2):
                        for dx in range(2):   
                            if x+dx in range(d) and y+dy in range(d):
                                plaq.append(x+dx+d*(y+dy))
                            else:
                                plaq.append(None)

                    if (x+y)%2==0:
                        xplaqs.append([plaq[0], plaq[1], plaq[2], plaq[3]])
                    else:
                        zplaqs.append([plaq[0], plaq[2], plaq[1], plaq[3]])
                    
        return zplaqs,xplaqs

    def _preparation(self):
        """
        Prepares logical bit states by applying an x to the circuit that will
        encode a 1.
        """
        if self._basis=='z':
            self.x(['1'])
        else:
            for log in self.circuit:
                self.circuit[log].h(self.code_qubit)
            self.z(['1'])
        
    def get_circuit_list(self):
        """
        Returns:
            circuit_list: self.circuit as a list, with
            circuit_list[0] = circuit['0']
            circuit_list[1] = circuit['1']
        """
        circuit_list = [self.circuit[log] for j,log in enumerate(['0', '1'])]
        return circuit_list

    def x(self, logs=('0', '1'), barrier=False):
        """
        Applies a logical x to the circuits for the given logical values.
        Args:
            logs (list or tuple): List or tuple of logical values expressed as
                strings.
            barrier (bool): Boolean denoting whether to include a barrier at
                the end.
        """
        for log in logs:
            for j in range(self.d):
                self.circuit[log].x(self.code_qubit[j*d])
            if barrier:
                self.circuit[log].barrier()
       
    def z(self, logs=('0', '1'), barrier=False):
        """
        Applies a logical z to the circuits for the given logical values.
        Args:
            logs (list or tuple): List or tuple of logical values expressed as
                strings.
            barrier (bool): Boolean denoting whether to include a barrier at
                the end.
        """
        for log in logs:
            for j in range(self.d):
                self.circuit[log].z(self.code_qubit[j])
            if barrier:
                self.circuit[log].barrier()

    def syndrome_measurement(self, final=False, barrier=False):
        """
        Application of a syndrome measurement round.
        Args:
            reset (bool): If set to true add a boolean at the end of each round
            barrier (bool): Boolean denoting whether to include a barrier at the end.
            
        """
        
        num_bits = int((self.d**2 - 1)/2)
        
        zplaqs,xplaqs = self.zplaqs, self.xplaqs
        
        # classical registers for this round
        self.zplaq_bits.append(ClassicalRegister(
            self._num_xy, 'round_' + str(self.T) + '_zplaq_bit'))
        self.xplaq_bits.append(ClassicalRegister(
            self._num_xy, 'round_' + str(self.T) + '_xplaq_bit'))

        for log in ['0', '1']:

            self.circuit[log].add_register(self.zplaq_bits[-1])
            self.circuit[log].add_register(self.xplaq_bits[-1])
            
            self.circuit[log].h(self.xplaq_qubit)
            
            for j in range(4):
                for p,plaq in enumerate(zplaqs):
                    c = plaq[j]
                    if c!=None:
                        self.circuit[log].cx(self.code_qubit[c], self.zplaq_qubit[p])
                for p,plaq in enumerate(xplaqs):
                    c = plaq[j]
                    if c!=None:
                        self.circuit[log].cx(self.xplaq_qubit[p], self.code_qubit[c])
            
            self.circuit[log].h(self.xplaq_qubit)
            
            for j in range(self._num_xy):
                self.circuit[log].measure(
                    self.xplaq_qubit[j], self.xplaq_bits[self.T][j])
                self.circuit[log].measure(
                    self.zplaq_qubit[j], self.zplaq_bits[self.T][j])
                if self._resets and not final:
                    self.circuit[log].reset(self.xplaq_qubit[j])
                    self.circuit[log].reset(self.zplaq_qubit[j])

            if barrier:
                self.circuit[log].barrier()

        self.T += 1
        
    def readout(self):
        """
        Readout of all code qubits, which corresponds to a logical measurement
        as well as allowing for a measurement of the syndrome to be inferred.
        """
        
        for log in ['0', '1']:
            if self._basis=='x':
                self.circuit[log].h(self.code_qubit)
            self.circuit[log].add_register(self.code_bit)
            self.circuit[log].measure(self.code_qubit, self.code_bit)
            
    def process_results(self, raw_results):
        
        zplaqs,xplaqs = self.zplaqs, self.xplaqs
        
        results = {}
        for log in raw_results:
            results[log] = {}
            for string in raw_results[log]:
                
                final_readout = string.split(' ')[0][::-1]
                
                # get logical readout
                # (though it's called Z, it actually depends on the basis)
                Z = [0,0]
                for j in range(self.d):
                    if self._basis=='z':
                        # evaluated using top row
                        Z[0] = (Z[0] + int(final_readout[j]))%2
                        # evaluated using bottom row
                        Z[1] = (Z[1] + int(final_readout[self.d**2-1-j]))%2
                    else:
                        # evaluated using left side
                        Z[0] = (Z[0] + int(final_readout[j*self.d]))%2
                        # evaluated using right side
                        Z[1] = (Z[1] + int(final_readout[(j+1)*self.d-1]))%2  
                    
                measured_Z = str(Z[0]) + ' ' + str(Z[1]) 
                
                # final syndrome for plaquettes deduced from final code qubit readout
                if self._basis=='z':
                    plaqs = zplaqs
                else:
                    plaqs = xplaqs
                full_syndrome = ''
                for plaq in plaqs:
                    parity = 0
                    for q in plaq:
                        if q!=None:
                            parity += int(final_readout[q])
                    full_syndrome = str(parity%2) + full_syndrome
                
                # results from all other plaquette syndrome measurements then added
                if self._basis=='z':
                    full_syndrome = full_syndrome + ' ' + ' '.join(string.split(' ')[2::2])
                else:
                    full_syndrome = full_syndrome + ' ' + ' '.join(string.split(' ')[1::2])

                # changes between one syndrome and the next then calculated
                syndrome_list = full_syndrome.split(' ')

                height = len(syndrome_list)
                width = len(syndrome_list[0])
                syndrome_changes = ''
                for t in range(height):
                    for j in range(width):
                        if self._resets:
                            if t == 0:
                                change = (syndrome_list[-1][j] != '0')
                            else:
                                change = (syndrome_list[-t][j]
                                          != syndrome_list[-t - 1][j])
                            syndrome_changes += '0' * (not change) + '1' * change
                        else:
                            if t <= 1:
                                if t != self.T:
                                    change = (syndrome_list[-t - 1][j] != '0')
                                else:
                                    change = (syndrome_list[-t - 1][j]
                                              != syndrome_list[-t][j])
                            elif t == self.T:
                                last3 = ''
                                for dt in range(3):
                                    last3 += syndrome_list[-t - 1 + dt][j]
                                change = last3.count('1') % 2 == 1
                            else:
                                change = (syndrome_list[-t - 1][j]
                                          != syndrome_list[-t + 1][j])
                            syndrome_changes += '0' * (not change) + '1' * change
                        syndrome_changes += ' '
                    
                # the space separated string of syndrome changes then gets a
                # double space separated logical value on the end
                new_string = measured_Z + '  ' + syndrome_changes[:-1]
                
                if new_string in results[log]:
                    results[log][new_string] += raw_results[log][string]
                else:
                    results[log][new_string] = raw_results[log][string]
                
        return results

    
