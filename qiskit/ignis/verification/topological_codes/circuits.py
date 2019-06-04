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

'''
Generates circuits for quantum error correction
'''

from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit


class RepetitionCode():
    '''
    Implementation of a distance d repetition code, implemented over
    T syndrome measurement rounds.

    Attributes
    * `d`: Code distance.
    * `T`: Number of syndrome measurement rounds.
    * `code_qubit` and `link_qubit`: Register used for code qubits and
    ancillae, respectively.
    * `qubit_registers`: Strings that can be used to reference the qubit
    registers.
    * `link_bits`: List of classical registers, with the jth being used
    for the jth measurement round.
    * `code_bit`: Classical register used for readout of code qubits.
    * `self.circuit`: A dictionary of two quantum circuits: one for the
    encoded logical bit value 0 and the other for 1, with corresponding
    keys `'0'` and `'1'`.

    See below for an explanation of each method.
    '''

    def __init__(self, d, T=0):
        '''
        Initializes all attributes according to the given code distance
        `d` and number of measurement round `T`.

        No measurements are added to the circuit if `T=0`. Otherwise `T` rounds
        are added, followed by measurement of the code qubits (corresponding to
        a logical measurement and final syndrome measurement round).
        '''

        self.d = d
        self.T = 0

        self.code_qubit = QuantumRegister(d, 'code_qubit')
        self.link_qubit = QuantumRegister((d - 1), 'link_qubit')
        self.qubit_registers = {'code_qubit', 'link_qubit'}

        self.link_bits = []
        self.code_bit = ClassicalRegister(d, 'code_bit')

        self.circuit = {}
        for log in ['0', '1']:
            self.circuit[log] = QuantumCircuit(
                self.link_qubit, self.code_qubit, name=log)

        self._preparation()

        for _ in range(T):
            self.syndrome_measurement()

        if T != 0:
            self.readout()

    def x(self, logs=('0', '1')):
        '''
        Applies a logical x to the circuits in the list or tuple `log`.
        '''
        for log in logs:
            for j in range(self.d):
                self.circuit[log].x(self.code_qubit[j])
            self.circuit[log].barrier()

    def _preparation(self):
        '''
        Prepares logical bit states by applying an x to the circuit that will
        encode a 1.
        '''
        self.x(['1'])

    def syndrome_measurement(self):
        '''
        Application of a syndrome measurement round.
        '''

        self.link_bits.append(ClassicalRegister(
            (self.d - 1), 'round_' + str(self.T) + '_link_bit'))

        for log in ['0', '1']:

            self.circuit[log].add_register(self.link_bits[-1])

            for j in range(self.d - 1):
                self.circuit[log].cx(self.code_qubit[j], self.link_qubit[j])

            for j in range(self.d - 1):
                self.circuit[log].cx(
                    self.code_qubit[j + 1], self.link_qubit[j])

            for j in range(self.d - 1):
                self.circuit[log].measure(
                    self.link_qubit[j], self.link_bits[self.T][j])
                self.circuit[log].reset(self.link_qubit[j])

            self.circuit[log].barrier()

        self.T += 1

    def readout(self):
        '''
        Readout of all code qubits, which corresponds to a logical measurement
        as well as allowing for a measurement of the syndrome to be inferred.
        '''
        for log in ['0', '1']:
            self.circuit[log].add_register(self.code_bit)
            self.circuit[log].measure(self.code_qubit, self.code_bit)

    def process_results(self, raw_results):
        '''
        The circuits must be executed outside of this class, so that the user
        has full freedom to compile, choose their backend, use a noise model,
        etc.

        Once the circuits have been executed, this method transforms the output
        bit strings to the form required by to set up a decoder.
        '''
        results = {}
        for log in raw_results:
            results[log] = {}
            for string in raw_results[log]:

                # logical readout taken from
                measured_log = string[0] + ' ' + string[self.d - 1]

                # final syndrome deduced from final code qubit readout
                full_syndrome = ''
                for j in range(self.d - 1):
                    full_syndrome += '0' * (string[j] == string[j + 1]) \
                        + '1' * (string[j] != string[j + 1])
                # results from all other syndrome measurements then added
                full_syndrome = full_syndrome + string[self.d:]

                # changes between one syndrome and the next then calculated
                syndrome_list = full_syndrome.split(' ')
                syndrome_changes = ''
                for t in range(self.T + 1):
                    for j in range(self.d - 1):
                        if t == 0:
                            change = (syndrome_list[-1][j] != '0')
                        else:
                            change = (syndrome_list[-t][j]
                                      != syndrome_list[-t - 1][j])
                        syndrome_changes += '0' * (not change) + '1' * change
                    syndrome_changes += ' '

                # the space separated string of syndrome changes then gets a
                # double space separated logical value on the end
                new_string = measured_log + '  ' + syndrome_changes[:-1]

                results[log][new_string] = raw_results[log][string]

        return results
