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

    def __init__(self, d, T=0):
        """
        Creates the circuits corresponding to a logical 0 and 1 encoded
        using a repetition code.

        Args:
            d (int): Number of code qubits (and hence repetitions) used.
            T (int): Number of rounds of ancilla-assisted syndrome measurement.


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

        self._preparation()

        for _ in range(T - 1):
            self.syndrome_measurement()

        if T != 0:
            self.syndrome_measurement(reset=False)
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
            for j in range(self.d):
                self.circuit[log].x(self.code_qubit[j])
            if barrier:
                self.circuit[log].barrier()

    def _preparation(self):
        """
        Prepares logical bit states by applying an x to the circuit that will
        encode a 1.
        """
        self.x(["1"])

    def syndrome_measurement(self, reset=True, barrier=False):
        """
        Application of a syndrome measurement round.

        Args:
            reset (bool): If set to true add a boolean at the end of each round
            barrier (bool): Boolean denoting whether to include a barrier at the end.
        """
        self.link_bits.append(
            ClassicalRegister((self.d - 1), "round_" + str(self.T) + "_link_bit")
        )

        for log in ["0", "1"]:

            self.circuit[log].add_register(self.link_bits[-1])

            for j in range(self.d - 1):
                self.circuit[log].cx(self.code_qubit[j], self.link_qubit[j])

            for j in range(self.d - 1):
                self.circuit[log].cx(self.code_qubit[j + 1], self.link_qubit[j])

            for j in range(self.d - 1):
                self.circuit[log].measure(self.link_qubit[j], self.link_bits[self.T][j])
                if reset:
                    self.circuit[log].reset(self.link_qubit[j])

            if barrier:
                self.circuit[log].barrier()

        self.T += 1

    def readout(self):
        """
        Readout of all code qubits, which corresponds to a logical measurement
        as well as allowing for a measurement of the syndrome to be inferred.
        """
        for log in ["0", "1"]:
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
                full_syndrome = full_syndrome + string[self.d :]

                # changes between one syndrome and the next then calculated
                syndrome_list = full_syndrome.split(" ")
                syndrome_changes = ""
                for t in range(self.T + 1):
                    for j in range(self.d - 1):
                        if t == 0:
                            change = syndrome_list[-1][j] != "0"
                        else:
                            change = syndrome_list[-t][j] != syndrome_list[-t - 1][j]
                        syndrome_changes += "0" * (not change) + "1" * change
                    syndrome_changes += " "

                # the space separated string of syndrome changes then gets a
                # double space separated logical value on the end
                new_string = measured_log + "  " + syndrome_changes[:-1]

                results[log][new_string] = raw_results[log][string]

        return results

    def _get_all_processed_results(self):  # NEWNESS
        """- returns a list of all processed results stemming from single qubit errors
        - needed so create the decoder graph
        """

        syn = RepetitionCodeSyndromeGenerator(self)  # new class

        T = syn.T  # number of rounds of stabilizer measurements
        d = syn.d  # number of data qubits

        out = []
        for r in range(T):
            for i in range(d - 1):
                syn.bitflip_ancilla(i, r)
                out.append(syn.get_processed_results())
                syn.bitflip_ancilla(i, r)  # undo the error
            for i in range(d):
                syn.bitflip_data(i, r, True)
                out.append(syn.get_processed_results())
                syn.bitflip_data(i, r, True)  # undo the error
                syn.bitflip_data(i, r, False)
                out.append(syn.get_processed_results())
                syn.bitflip_data(i, r, False)  # undo the error
        for i in range(d):
            syn.bitflip_readout(i)
            out.append(syn.get_processed_results())
            syn.bitflip_readout(i)  # undo the error

        return out


class RepetitionCodeSyndromeGenerator:
    """
    description
    """

    def __init__(self, code: RepetitionCode):
        """
        Keeps track how individually introduced bitflip errors affect the
        measurement outcomes of a repetition code.

        Args:
            code: RepetitionCode for which a SyndromeGraph shall be created
        """

        self.d = code.d  # number of qubits
        self.T = code.T  # number of rounds (of stabilizer measurements)

        # List of measurement results
        self.m_anc = {}  # referred to as (b_r[0], ..., b_r[n-2]) in Ref [1]
        self.m_fin = [0] * self.d  # referred to as (c[0], ..., c[n-2]) in Ref [1]
        for r in range(self.T):
            self.m_anc[r] = [0] * (self.d - 1)

    def bitflip_readout(self, i: int) -> None:
        """
        Introduces a bitflip error on data qubit i right before the (final) readout

        Args:
            i: qubit index
        """
        self.m_fin[i] = (self.m_fin[i] + 1) % 2

    def bitflip_ancilla(self, i: int, r: int) -> None:
        """
        Introduces a bitflip error to ancilla i in round r.
        Args:
            i: qubit index
            r: round index
        """
        self.m_anc[r][i] = (self.m_anc[r][i] + 1) % 2

    def bitflip_data(self, i: int, r0: int, middle: bool = False) -> None:
        """
        Introduces a bitflip error to data qubit i in round r0.
        Args:
            i: qubit index
            r0: round index
            middle: if False, the error is introduced before the first sequence of CNOTs
                    if True, the error is introduced in between the two CNOT sequences
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

    def get_m_ancilla(self, i: int, r: int) -> int:
        """
        Get ancilla measurement result.
        Args:
            i: ancilla qubit index
            r: stabilizer measurement round
        Returns:
            Measurement result of ancilla i in round r for the current set of errors.
        """
        return self.m_anc[r][i]

    def get_m_data(self, i: int, encoded: int = 0) -> int:
        """
        Get data measurement result.
        Args:
            i: qubit index
            encoded: initial logical value of the data qubits
        Returns:
             Final measurement result of data qubit i for the current set of errors.
        """
        return (self.m_fin[i] + encoded) % 2

    def get_raw_results(self, encoded: int = 0) -> str:
        """
        Get raw results of syndrome measurements and final readout.
        Args:
            encoded: logical |0> or |1>
        Returns:
            raw_results as in _make_syndrome_graph
        """
        out = ""
        for i in range(self.d - 1, -1, -1):  # qiskit's qubit ordering
            out += str(self.get_m_data(i, encoded))
        for r in range(self.T - 1, -1, -1):  # qiskit's qubit register ordering
            out += " "
            for i in range(self.d - 2, -1, -1):
                out += str(self.get_m_ancilla(i, r))
        return out

    def get_processed_results(self, encoded: int = 0) -> str:
        """
        Get processed results of syndrome measurements and final readout.
        Args:
            encoded: logical |0> or |1>
        Returns:
            processed_results as in _make_syndrome_graph
        """
        out = (
            str(self.get_m_data(self.d - 1, encoded))
            + " "
            + str(self.get_m_data(0, encoded))
            + "  "
        )
        for i in range(self.d - 2, -1, -1):
            out += str(self.get_m_ancilla(i, 0))
        for r in range(1, self.T):
            out += " "
            for i in range(self.d - 2, -1, -1):  # qiskit's qubit ordering
                out += str(
                    (self.get_m_ancilla(i, r) + self.get_m_ancilla(i, r - 1)) % 2
                )
        out += " "
        for i in range(self.d - 2, -1, -1):  # qiskit's qubit ordering
            out += str(
                (
                    self.get_m_ancilla(i, self.T - 1)
                    + self.get_m_data(i, encoded)
                    + self.get_m_data(i + 1, encoded)
                )
                % 2
            )
        return out
