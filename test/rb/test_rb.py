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

# pylint: disable=undefined-loop-variable,invalid-name,missing-type-doc

"""
Run through RB for different qubit numbers to check that it's working
and that it returns the identity
"""

import itertools
import random
import unittest

import numpy as np
from ddt import ddt, data, unpack

import qiskit
import qiskit.ignis.verification.randomized_benchmarking as rb
from qiskit import QiskitError
from qiskit.circuit.library import (XGate, YGate, ZGate, HGate, TGate,
                                    SGate, SdgGate, CXGate, CZGate,
                                    SwapGate)
from qiskit.circuit.library import U1Gate, U2Gate, U3Gate


@ddt
class TestRB(unittest.TestCase):
    """The test class."""

    @staticmethod
    def choose_pattern(pattern_type, nq):
        """
        Choose a valid field for rb_opts['rb_pattern']

        Args:
            pattern_type (int): a number between 0 and 2.
                0 - a list of all qubits, for nq=5 it is
                    [1, 2, 3, 4, 5].
                1 - a list of lists of single qubits, for nq=5
                    it is [[1], [2], [3], [4], [5]].
                2 - randomly choose a pattern which is a list of
                    two lists where the first one has 2 elements,
                    for example for nq=5 it can be
                    [[4, 1], [2, 5, 3]].
            nq (int): number of qubits

        Returns:
            tuple: of the form (``res``, ``is_purity``)
                where the tuple is  the pattern or ``None``.
                Returns ``None`` if the pattern type is not relevant to the
                number of qubits, i.e,, one of two cases:
                pattern_type = 1 and nq = 1, which implies [[1]]
                pattern_type = 2 and nq <= 2:

                - for nq=1 this is impossible
                - for nq=2 this implies
                  [[1], [2]], which is already
                  tested when pattern_type = 1.

            is_purity = True if the pattern fits for purity rb
            (namely, all the patterns have the same dimension:
            only 1-qubit or only 2-qubits).
        """

        is_purity = True
        if pattern_type == 0:
            res = [list(range(nq))]
            if nq > 2:
                is_purity = False
        elif pattern_type == 1:
            if nq == 1:
                return None, None
            res = [[x] for x in range(nq)]
        else:
            if nq <= 2:
                return None, None
            shuffled_bits = list(range(nq))
            random.shuffle(shuffled_bits)
            # split_loc = random.randint(1, nq-1)
            split_loc = 2  # deterministic test
            res = [shuffled_bits[:split_loc], shuffled_bits[split_loc:]]
            if 2*split_loc != nq:
                is_purity = False

        return res, is_purity

    @staticmethod
    def choose_multiplier(mult_opt, len_pattern):
        """

        Args:
            mult_opt (int): the multiplier option to use:
                0: fixed length
                1: vector of lengths
            len_pattern (int): number of patterns

        Returns:
            int or list: the length multiplier
        """
        if mult_opt == 0:
            res = 1
        else:
            res = [i + 1 for i in range(len_pattern)]

        return res

    @staticmethod
    def choose_interleaved_gates(rb_pattern):
        """
        Args:
            rb_pattern (list): pattern for randomized benchmarking

        Returns:
            intelreved_elmnts: A list of QuantumCircuit elements that
                will be interleaved (for interleaved randomized benchmarking)
                The length of the list would equal the length of the rb_pattern.
            intelreved_gates: A corresponding list of gates.
        """
        pattern_sizes = [len(pat) for pat in rb_pattern]
        interleaved_gates = []
        interleaved_elemnts = []
        for (_, nq) in enumerate(pattern_sizes):
            gatelist = []
            qc = qiskit.QuantumCircuit(nq)
            # The interleaved gates contain x gate on each qubit
            # and cx gate on each pair of consecutive qubits
            for qubit in range(nq):
                gatelist.append('x ' + str(qubit))
                qc.x(qubit)
            for qubit in range(nq):
                if qubit > 0:
                    gatelist.append('cx ' + '0' + ' ' + str(qubit))
                    qc.cx(0, qubit)
            interleaved_gates.append(gatelist)
            interleaved_elemnts.append(qc)
        return interleaved_elemnts, interleaved_gates

    @staticmethod
    def update_interleaved_gates(gatelist, pattern):
        """
        Args:
            gatelist: list of Clifford gates
            pattern: pattern of indexes (from rb_pattern)

        Returns:
            updated_gatelist: list of Clifford gates
            after the following updates:
            - change the indexes from [0,1,...]
            according to the pattern
        """
        updated_gatelist = []
        for op in gatelist:
            split = op.split()
            op_names = split[0]
            # updating the qubit indexes according to the pattern
            # given in rb_pattern
            op_qubits = [str(pattern[int(x)]) for x in split[1:]]
            updated_gatelist += [op_names + ' ' +
                                 (' '.join(op_qubits))]
        return updated_gatelist

    @staticmethod
    def update_purity_gates(npurity, purity_ind, rb_pattern):
        """
        Args:
            npurity: equals to 3^n
            purity_ind: purity index in [0,3^n-1]
            rb_pattern: rb pattern

        Returns:
            name_type: type of name for rb_circuit
            (e.g. XY, ZZ etc.).
            gate_list: list of purity gates
            (e.g 'rx 0', 'ry 1' etc.) according to rb_pattern
        """
        name_type = ''
        ind_d = purity_ind
        purity_qubit_num = 0
        gate_list = []
        while True:
            purity_qubit_rot = np.mod(ind_d, 3)
            ind_d = np.floor_divide(ind_d, 3)
            if purity_qubit_rot == 0:
                name_type += 'Z'
            if purity_qubit_rot == 1:
                name_type += 'X'
                for pat in rb_pattern:
                    gate_list.append('rx ' + str(pat[purity_qubit_num]))
            if purity_qubit_rot == 2:
                name_type += 'Y'
                for pat in rb_pattern:
                    gate_list.append('ry ' + str(pat[purity_qubit_num]))

            purity_qubit_num = purity_qubit_num + 1
            if ind_d == 0:
                break
        # padding the circuit name with Z's so that
        # all circuits will have names of the same length
        for _ in range(int(np.log(npurity)/np.log(3)) -
                       purity_qubit_num):
            name_type += 'Z'

        return name_type, gate_list

    @staticmethod
    def ops_to_gates(ops, op_index, stop_gate='barrier'):
        """
        Args:
            ops: of the form circ.data
            op_index: int, the operation index
            stop_gate: the gate to stop (e.g. barrier or measure)

        Returns:
            gatelist: a list of gates
            op_index: int, updated index
        """
        gatelist = []
        while ops[op_index][0].name != stop_gate:
            gate = ops[op_index][0].name
            params = ops[op_index][0].params
            if gate == 'u1':
                gate += ' ' + str(params[0])
            for x in ops[op_index][1]:
                gate += ' ' + str(x.index)
            gatelist.append(gate)
            op_index += 1
        # increment because of the barrier gate
        op_index += 1
        return gatelist, op_index

    def verify_circuit(self, circ, nq, rb_opts, vec_len, result, shots,
                       is_interleaved=False, is_cnotdihedral=False):
        """
        For a single sequence, verifies that it meets the requirements:
        - Executing it on the ground state ends up in the ground state
        - It has the correct number of elements
        - It fulfills the pattern, as specified by rb_patterns and
          length_multiplier

        Args:
            circ: the sequence to check
            nq: number of qubits
            rb_opts: the specification that generated the set of sequences
                which includes circ
            vec_len: the expected length vector of circ (one of
                rb_opts['length_vector'])
            result: the output of the simulator
                when executing all the sequences on the ground state
            shots: the number of shots in the simulator execution
            is_interleaved: True if this is an interleaved circuit
            is_cnotdihedral: True if this is a non-clifford cnot-dihedral
                circuit
        """

        if not hasattr(rb_opts['length_multiplier'], "__len__"):
            rb_opts['length_multiplier'] = [
                rb_opts['length_multiplier'] for i in range(
                    len(rb_opts['rb_pattern']))]

        ops = circ.data
        # for each cycle (the sequence should consist of vec_len cycles)
        for _ in range(vec_len):
            op_index = 0
            # for non-clifford cnot-dihedral rb
            if is_cnotdihedral:
                op_index += 2*nq
            # for each component of the pattern...
            for pat_index in range(len(rb_opts['rb_pattern'])):
                # for each element...
                for _ in range(rb_opts['length_multiplier'][pat_index]):
                    # if it is an interleaved RB circuit,
                    # then it has twice as many elements
                    for _ in range(is_interleaved+1):
                        # for each basis gate...
                        # in case of align_cliffs we may have extra barriers
                        # (after another barrier)
                        if ops[op_index][0].name != 'barrier':
                            while ops[op_index][0].name != 'barrier':
                                # Verify that the gate acts
                                # on the correct qubits.
                                # This happens if the sequence is composed
                                # of the correct sub-sequences,
                                # as specified by vec_len and rb_opts
                                self.assertTrue(
                                    all(x.index in rb_opts['rb_pattern'][
                                        pat_index]
                                        for x in ops[op_index][1]),
                                    "Error: operation acts on incorrect qubits")
                                op_index += 1
                            # increment because of the barrier gate
                            op_index += 1
        # check if the ground state returns
        self.assertEqual(result.
                         get_counts(circ)['{0:b}'.format(0).zfill(nq)], shots,
                         "Error: %d qubit RB does not return the"
                         "ground state back to the ground state" % nq)

    def compare_interleaved_circuit(self, original_circ, interleaved_circ,
                                    nq, rb_opts_interleaved, interleaved_gates,
                                    vec_len):
        """
        Verifies that interleaved RB circuits meet the requirements:
        - The non-interleaved gates are the same as the
        original gates.
        - The interleaved gates are the same as the ones
        given in: rb_opts_interleaved['interleaved_gates'].

        Args:
            original_circ: original rb circuits
            interleaved_circ: interleaved rb circuits
            nq: number of qubits
            rb_opts_interleaved: the specification that
                generated the set of sequences which includes circ
            interleaved_gates: a list of the interleaved gates
                for comparison (default = None)
            vec_len: the expected length vector of circ
                (one of rb_opts['length_vector'])
        """
        if not hasattr(rb_opts_interleaved['length_multiplier'], "__len__"):
            rb_opts_interleaved['length_multiplier'] = [
                rb_opts_interleaved['length_multiplier'] for i in range(
                    len(rb_opts_interleaved['rb_pattern']))]

        original_ops = original_circ.data
        interleaved_ops = interleaved_circ.data

        original_op_index = 0
        interleaved_op_index = 0
        # for each cycle (the sequence should consist of vec_len cycles)
        for _ in range(vec_len):
            # for each component of the pattern...
            for pat_index in range(len(rb_opts_interleaved['rb_pattern'])):
                # updating the gates in:
                # rb_opts_interleaved['interleaved_gates']
                if interleaved_gates is not None:
                    updated_gatelist = self.update_interleaved_gates(
                        interleaved_gates[pat_index],
                        rb_opts_interleaved['rb_pattern'][pat_index])
                # for each element...
                for _ in range(rb_opts_interleaved['length_multiplier']
                               [pat_index]):
                    # original RB sequence
                    original_gatelist, original_op_index = \
                        self.ops_to_gates(original_ops,
                                          original_op_index)
                    # interleaved RB sequence
                    compared_gatelist, interleaved_op_index = \
                        self.ops_to_gates(interleaved_ops,
                                          interleaved_op_index)

                    # Gates in the interleaved RB sequence
                    # should be equal to original gates
                    self.assertEqual(original_gatelist, compared_gatelist,
                                     "Error: The gates in the %d qubit"
                                     "interleaved RB are not the same as"
                                     "in the original RB circuits" % nq)
                    # Gates in the interleaved RB sequence
                    # should be equal to the given gates in
                    # rb_opts_interleaved['interleaved_gates']
                    # (after updating them)
                    interleaved_gatelist, interleaved_op_index = \
                        self.ops_to_gates(interleaved_ops,
                                          interleaved_op_index)

                    if interleaved_gates is not None:
                        self.assertEqual(sorted(interleaved_gatelist),
                                         sorted(updated_gatelist),
                                         "Error: The interleaved gates in the \
                                         %d qubit interleaved RB are not the same \
                                         as given in interleaved_gates input" % nq)

    def compare_cnotdihedral_circuit(self, cnotdihedral_Z_circ,
                                     cnotdihedral_X_circ, nq,
                                     rb_opts_cnotdihedral, vec_len):
        """
        Verifies that cnotdihedral RB circuits are the same,
        except of the first and last H gates.

        Args:
            cnotdihedral_Z_circ: original rb circuits
            (meassure |0...0> state)
            cnotdihedral_X_circ: rb circuits that
            measure |+...+> state.
            nq: number of qubits
            rb_opts_cnotdihedral: the specification that
            generated the set of sequences which includes circ
            vec_len: the expected length vector of circ
            (one of rb_opts['length_vector'])
        """

        qlist_flat, _, _ = rb.circuits.check_pattern(
            rb_opts_cnotdihedral['rb_pattern'])

        if not hasattr(rb_opts_cnotdihedral['length_multiplier'], "__len__"):
            rb_opts_cnotdihedral['length_multiplier'] = [
                rb_opts_cnotdihedral['length_multiplier'] for i in range(
                    len(rb_opts_cnotdihedral['rb_pattern']))]

        circ_Z_ops = cnotdihedral_Z_circ.data
        circ_X_ops = cnotdihedral_X_circ.data

        # for each cycle (the sequence should consist of vec_len cycles)
        for _ in range(vec_len):
            op_Z_index = 0
            op_X_index = 0
            # Measurement of the |0...0> state
            circ_Z_gatelist, op_Z_index = \
                self.ops_to_gates(circ_Z_ops, op_Z_index, stop_gate='measure')
            # Measurement of the |+...+> state
            circ_X_gatelist, op_X_index = \
                self.ops_to_gates(circ_X_ops, op_X_index, stop_gate='measure')
            h_gates = []
            for _, qb in enumerate(qlist_flat):
                h_gates.append('h %d' % qb)
                h_gates.append('barrier %d' % qb)
            circ_Z_gatelist = h_gates + circ_Z_gatelist
            h_gates = []
            for _, qb in enumerate(qlist_flat):
                h_gates.append('barrier %d' % qb)
                h_gates.append('h %d' % qb)
            circ_Z_gatelist = circ_Z_gatelist + h_gates
            # Gates in the non-Clifford cnot-dihedralRB sequence
            # should be equal (up to the H-gates)
            self.assertEqual(circ_Z_gatelist, circ_X_gatelist,
                             "Error: The non-Clifford gates in the"
                             "%d qubit non-Clifford CNOT-Dihedral RB"
                             "are not the same" % nq)

    def compare_purity_circuits(self, original_circ, purity_circ, nq,
                                purity_ind, npurity, rb_opts_purity, vec_len):
        """
        Verifies that purity RB circuits meet the requirements:
        - The gates are the same as the original gates.
        - The last gates are either Rx or Ry or nothing
        (depend on d)

        Args:
            original_circ: original rb circuits
            purity_circ: purity rb circuits
            nq: number of qubits
            purity_ind: purity index in [0,3^n-1]
            npurity: equal to 3^n
            rb_opts_purity: the specification that
                generated the set of sequences which includes circ
            vec_len: the expected length vector of circ
                (one of rb_opts['length_vector'])
        """

        original_ops = original_circ.data
        purity_ops = purity_circ.data
        op_index = 0
        pur_index = 0

        # for each cycle (the sequence should consist of vec_len cycles)
        for _ in range(vec_len):
            # for each component of the pattern...
            for pat_index in range(len(rb_opts_purity['rb_pattern'])):
                # for each element...
                for _ in range(rb_opts_purity['length_multiplier'][pat_index]):
                    # original RB sequence
                    original_gatelist, op_index = \
                        self.ops_to_gates(original_ops, op_index)
                    # purity RB sequence
                    purity_gatelist, pur_index = \
                        self.ops_to_gates(purity_ops, pur_index)
                    # Gates in the purity RB sequence
                    # should be equal to original gates
                    self.assertEqual(original_gatelist, purity_gatelist,
                                     "Error: The purity gates in the \
                                     %d qubit purity RB are not the same \
                                     as in the original RB circuits" % nq)

        # The last gate in the purity RB sequence
        # should be equal to the inverse element
        # with either Rx or Ry or nothing (depend on d)
        # original last gate
        original_gatelist, op_index = \
            self.ops_to_gates(original_ops, op_index, 'measure')
        _, purity_gates = self.update_purity_gates(
            npurity, purity_ind, rb_opts_purity['rb_pattern'])
        original_gatelist = original_gatelist + purity_gates
        # purity last gate
        purity_gatelist, pur_index = \
            self.ops_to_gates(purity_ops, pur_index, 'measure')
        self.assertEqual(original_gatelist, purity_gatelist,
                         "Error: The last purity gates in the"
                         "%d qubit purity RB are wrong" % nq)

    @data(*itertools.product([1, 2, 3, 4], range(3), range(2)))
    @unpack
    def test_rb(self, nq, pattern_type, multiplier_type):
        """Main function of the RB test."""

        # Load simulator
        backend = qiskit.Aer.get_backend('qasm_simulator')

        # See documentation of choose_pattern for the meaning of
        # the different pattern types
        # Choose options for standard (simultaneous) RB:
        rb_opts = {}
        rb_opts['nseeds'] = 3
        rb_opts['rand_seed'] = 1234
        rb_opts['length_vector'] = [1, 3, 4, 7]
        rb_opts['rb_pattern'], is_purity = self.choose_pattern(pattern_type, nq)
        # if the pattern type is not relevant for nq
        if rb_opts['rb_pattern'] is None:
            raise unittest.SkipTest('pattern type is not relevant for nq')
        rb_opts_purity = rb_opts.copy()
        rb_opts['length_multiplier'] = self.choose_multiplier(
            multiplier_type, len(rb_opts['rb_pattern']))
        # Choose options for interleaved RB:
        rb_opts_interleaved = rb_opts.copy()
        rb_opts_interleaved['interleaved_elem'], interleaved_gates = \
            self.choose_interleaved_gates(rb_opts['rb_pattern'])
        # Choose options for Non-Clifford cnot-dihedral RB:
        rb_opts_cnotdihedral = rb_opts.copy()
        rb_opts_cnotdihedral['group_gates'] = 'CNOT-Dihedral'
        rb_opts_cnotdihedral_interleaved = rb_opts_cnotdihedral.copy()
        rb_opts_cnotdihedral_interleaved['interleaved_elem'], interleaved_gates = \
            self.choose_interleaved_gates(rb_opts['rb_pattern'])
        # Choose options for purity rb
        # no length_multiplier
        rb_opts_purity['length_multiplier'] = 1
        rb_opts_purity['is_purity'] = is_purity
        if multiplier_type > 0:
            is_purity = False
        # Adding seed_offset and align_cliffs
        rb_opts['seed_offset'] = 10
        rb_opts['align_cliffs'] = True

        # Generate the sequences
        # Standard (simultaneous) RB sequences:
        rb_circs, _ = rb.randomized_benchmarking_seq(**rb_opts)
        # Interleaved RB sequences:
        rb_original_circs, _, rb_interleaved_circs = \
            rb.randomized_benchmarking_seq(**rb_opts_interleaved)
        # Non-Clifford cnot-dihedral RB sequences:
        rb_cnotdihedral_Z_circs, _, rb_cnotdihedral_X_circs = \
            rb.randomized_benchmarking_seq(**rb_opts_cnotdihedral)
        # Non-Clifford interleaved cnot-dihedral RB sequences:
        # (these circuits will not be executed to save time)
        _, _, _, _, _ = \
            rb.randomized_benchmarking_seq(**rb_opts_cnotdihedral_interleaved)
        # Purity RB sequences:
        if is_purity:
            rb_purity_circs, _, npurity = \
                rb.randomized_benchmarking_seq(**rb_opts_purity)
            # verify: npurity = 3^n
            self.assertEqual(
                npurity, 3 ** len(rb_opts['rb_pattern'][0]),
                'Error: npurity does not equal to 3^n')

        # Perform an ideal execution on the generated sequences
        basis_gates = ['id', 'u1', 'u2', 'u3', 'cx']
        shots = 100
        result = []
        result_original = []
        result_interleaved = []
        result_cnotdihedral_Z = []
        result_cnotdihedral_X = []
        if is_purity:
            result_purity = [[] for d in range(npurity)]
        for seed in range(rb_opts['nseeds']):
            result.append(
                qiskit.execute(rb_circs[seed], backend=backend,
                               basis_gates=basis_gates,
                               shots=shots).result())
            result_original.append(
                qiskit.execute(rb_original_circs[seed],
                               backend=backend,
                               basis_gates=basis_gates,
                               shots=shots).result())
            result_interleaved.append(
                qiskit.execute(rb_interleaved_circs[seed],
                               backend=backend,
                               basis_gates=basis_gates,
                               shots=shots).result())
            result_cnotdihedral_Z.append(
                qiskit.execute(rb_cnotdihedral_Z_circs[seed],
                               backend=backend,
                               basis_gates=basis_gates,
                               shots=shots).result())
            result_cnotdihedral_X.append(
                qiskit.execute(rb_cnotdihedral_X_circs[seed],
                               backend=backend,
                               basis_gates=basis_gates,
                               shots=shots).result())
            if is_purity:
                for d in range(npurity):
                    result_purity[d].append(qiskit.execute(
                        rb_purity_circs[seed][d],
                        backend=backend,
                        basis_gates=basis_gates,
                        shots=shots).result())

        # Verify the generated sequences
        for seed in range(rb_opts['nseeds']):
            length_vec = rb_opts['length_vector']
            for circ_index, vec_len in enumerate(length_vec):
                # Verify circuits names
                self.assertEqual(
                    rb_circs[seed][circ_index].name,
                    'rb_length_%d_seed_%d' % (
                        circ_index, seed +
                        rb_opts['seed_offset']),
                    'Error: incorrect circuit name')
                self.assertEqual(
                    rb_original_circs[seed][circ_index].name,
                    'rb_length_%d_seed_%d' % (
                        circ_index, seed),
                    'Error: incorrect circuit name')
                self.assertEqual(
                    rb_interleaved_circs[seed][circ_index].name,
                    'rb_interleaved_length_%d_seed_%d' % (
                        circ_index, seed),
                    'Error: incorrect interleaved circuit name')
                self.assertEqual(
                    rb_cnotdihedral_Z_circs[seed][circ_index].name,
                    'rb_cnotdihedral_Z_length_%d_seed_%d' % (
                        circ_index, seed),
                    'Error: incorrect cnotdihedral circuit name')
                self.assertEqual(
                    rb_cnotdihedral_X_circs[seed][circ_index].name,
                    'rb_cnotdihedral_X_length_%d_seed_%d' % (
                        circ_index, seed),
                    'Error: incorrect non-Clifford CNOT-Dihedral circuit name')
                if is_purity:
                    for d in range(npurity):
                        name_type, _ = self.update_purity_gates(
                            npurity, d, rb_opts_purity
                            ['rb_pattern'])
                        self.assertEqual(
                            rb_purity_circs[seed][d]
                            [circ_index].name,
                            'rb_purity_%s_length_%d_seed_%d' % (
                                name_type, circ_index, seed),
                            'Error: incorrect purity circuit name')

                self.verify_circuit(rb_circs[seed][circ_index],
                                    nq, rb_opts,
                                    vec_len, result[seed], shots)
                self.verify_circuit(rb_original_circs[seed]
                                    [circ_index],
                                    nq, rb_opts,
                                    vec_len,
                                    result_original[seed], shots)
                self.verify_circuit(rb_interleaved_circs[seed]
                                    [circ_index],
                                    nq, rb_opts_interleaved,
                                    vec_len,
                                    result_interleaved[seed],
                                    shots,
                                    is_interleaved=True)
                self.verify_circuit(rb_cnotdihedral_Z_circs[seed]
                                    [circ_index],
                                    nq, rb_opts_cnotdihedral,
                                    vec_len,
                                    result_cnotdihedral_Z[seed],
                                    shots)
                self.verify_circuit(rb_cnotdihedral_X_circs[seed]
                                    [circ_index],
                                    nq, rb_opts_cnotdihedral,
                                    vec_len,
                                    result_cnotdihedral_X[seed],
                                    shots, is_cnotdihedral=True)
                if is_purity:
                    self.verify_circuit(rb_purity_circs[seed][0]
                                        [circ_index],
                                        nq, rb_opts_purity,
                                        vec_len, result_purity
                                        [0][seed], shots)
                    # compare the purity RB circuits
                    # with the original circuit
                    for d in range(1, npurity):
                        self.compare_purity_circuits(
                            rb_purity_circs[seed][0][circ_index],
                            rb_purity_circs[seed][d][circ_index],
                            nq, d, npurity, rb_opts_purity,
                            vec_len)
                # compare the interleaved RB circuits with
                # the original RB circuits
                self.compare_interleaved_circuit(
                    rb_original_circs[seed][circ_index],
                    rb_interleaved_circs[seed][circ_index],
                    nq, rb_opts_interleaved, interleaved_gates, vec_len)
                # compare the non-Clifford cnot-dihedral RB circuits
                self.compare_cnotdihedral_circuit(
                    rb_cnotdihedral_Z_circs[seed][circ_index],
                    rb_cnotdihedral_X_circs[seed][circ_index],
                    nq, rb_opts_cnotdihedral, vec_len)

        self.assertEqual(circ_index, len(rb_circs),
                         "Error: additional circuits exist")
        self.assertEqual(circ_index, len(rb_original_circs),
                         "Error: additional interleaved circuits exist")
        self.assertEqual(circ_index, len(rb_interleaved_circs),
                         "Error: additional interleaved circuits exist")
        self.assertEqual(circ_index, len(rb_cnotdihedral_Z_circs),
                         "Error: additional CNOTDihedral circuits exist")
        self.assertEqual(circ_index, len(rb_cnotdihedral_X_circs),
                         "Error: additional CNOTDihedral circuits exist")
        if is_purity:
            self.assertEqual(circ_index, len(rb_purity_circs),
                             "Error: additional purity circuits exist")

    @data([HGate(), ['h 0']], [SGate(), ['s 0']], [SdgGate(), ['s 0', 'z 0']],
          [XGate(), ['x 0']], [YGate(), ['y 0']], [ZGate(), ['z 0']])
    def test_interleaved_randomized_benchmarking_seq_1q_clifford_gates(self, gate):
        """interleaved 1Q Clifford gates in RB"""
        rb_original_circs, _, rb_interleaved_circs = rb.randomized_benchmarking_seq(
            nseeds=1, length_vector=[5], rb_pattern=[[0]],
            interleaved_elem=[gate[0]], keep_original_interleaved_elem=False)
        # Verify the generated sequences
        rb_opts = {}
        rb_opts['nseeds'] = 1
        rb_opts['rb_pattern'] = [[0]]
        vec_len = 5
        rb_opts['length_vector'] = [vec_len]
        rb_opts['length_multiplier'] = 1
        rb_opts['interleaved_elem'] = [gate[0]]
        seed = 0
        circ_index = 0

        self.assertEqual(rb_original_circs[seed][circ_index].name,
                         'rb_length_%d_seed_%d' % (circ_index, seed),
                         'Error: incorrect circuit name')
        self.assertEqual(rb_interleaved_circs[seed][circ_index].name,
                         'rb_interleaved_length_%d_seed_%d' % (circ_index, seed),
                         'Error: incorrect circuit name')
        self.compare_interleaved_circuit(rb_original_circs[seed][circ_index],
                                         rb_interleaved_circs[seed][circ_index],
                                         1, rb_opts, [gate[1]], vec_len)

    @data(SwapGate(), CXGate(), CZGate())
    def test_interleaved_randomized_benchmarking_seq_2q_clifford_gates(self, gate):
        """interleaved 2Q Clifford gates in RB"""
        rb_original_circs, _, rb_interleaved_circs = rb.randomized_benchmarking_seq(
            nseeds=1, length_vector=[5], rb_pattern=[[0, 1]],
            interleaved_elem=[gate], keep_original_interleaved_elem=False)
        # Verify the generated sequences
        rb_opts = {}
        rb_opts['nseeds'] = 1
        rb_opts['rb_pattern'] = [[0, 1]]
        vec_len = 5
        rb_opts['length_vector'] = [vec_len]
        rb_opts['length_multiplier'] = 1
        rb_opts['interleaved_elem'] = [gate]
        seed = 0
        circ_index = 0

        self.assertEqual(rb_original_circs[seed][circ_index].name,
                         'rb_length_%d_seed_%d' % (circ_index, seed),
                         'Error: incorrect circuit name')
        self.assertEqual(rb_interleaved_circs[seed][circ_index].name,
                         'rb_interleaved_length_%d_seed_%d' % (circ_index, seed),
                         'Error: incorrect circuit name')
        self.compare_interleaved_circuit(rb_original_circs[seed][circ_index],
                                         rb_interleaved_circs[seed][circ_index],
                                         2, rb_opts, None, vec_len)

    @data(TGate(), XGate())
    def test_interleaved_randomized_benchmarking_seq_1q_cnotdihedral_gates(self, gate):
        """interleaved 1Q CNOT-Dihedral gates in RB"""
        rb_cnotdihedral_Z_circs, _, rb_cnotdihedral_X_circs, \
            rb_cnotdihedral_interleaved_Z_circs, rb_cnotdihedral_interleaved_X_circs = \
            rb.randomized_benchmarking_seq(nseeds=1, length_vector=[5], rb_pattern=[[0]],
                                           interleaved_elem=[gate],
                                           keep_original_interleaved_elem=False,
                                           group_gates='CNOT-Dihedral')
        # Verify the generated sequences
        rb_opts = {}
        rb_opts['nseeds'] = 1
        rb_opts['rb_pattern'] = [[0]]
        vec_len = 5
        rb_opts['length_vector'] = [vec_len]
        rb_opts['length_multiplier'] = 1
        rb_opts['interleaved_elem'] = [gate]
        seed = 0
        circ_index = 0
        self.assertEqual(rb_cnotdihedral_Z_circs[seed][circ_index].name,
                         'rb_cnotdihedral_Z_length_%d_seed_%d' % (circ_index, seed),
                         'Error: incorrect circuit name')
        self.assertEqual(rb_cnotdihedral_X_circs[seed][circ_index].name,
                         'rb_cnotdihedral_X_length_%d_seed_%d' % (circ_index, seed),
                         'Error: incorrect circuit name')
        self.assertEqual(rb_cnotdihedral_interleaved_Z_circs[seed][circ_index].name,
                         'rb_cnotdihedral_interleaved_Z_length_%d_seed_%d' % (circ_index, seed),
                         'Error: incorrect circuit name')
        self.assertEqual(rb_cnotdihedral_interleaved_X_circs[seed][circ_index].name,
                         'rb_cnotdihedral_interleaved_X_length_%d_seed_%d' % (circ_index, seed),
                         'Error: incorrect circuit name')
        self.compare_interleaved_circuit(rb_cnotdihedral_Z_circs[seed][circ_index],
                                         rb_cnotdihedral_interleaved_Z_circs[seed][circ_index],
                                         1, rb_opts, None, vec_len)
        self.compare_cnotdihedral_circuit(rb_cnotdihedral_Z_circs[seed][circ_index],
                                          rb_cnotdihedral_X_circs[seed][circ_index],
                                          1, rb_opts, vec_len)
        self.compare_cnotdihedral_circuit(rb_cnotdihedral_interleaved_Z_circs[seed][circ_index],
                                          rb_cnotdihedral_interleaved_X_circs[seed][circ_index],
                                          1, rb_opts, vec_len)

    def test_interleaved_randomized_benchmarking_seq_2q_cnotdihedral_gates(self):
        """interleaved 2Q CNOT-Dihedral gates in RB"""
        gate = CXGate()
        rb_cnotdihedral_Z_circs, _, rb_cnotdihedral_X_circs,  \
            rb_cnotdihedral_interleaved_Z_circs, rb_cnotdihedral_interleaved_X_circs = \
            rb.randomized_benchmarking_seq(nseeds=1, length_vector=[5], rb_pattern=[[0, 1]],
                                           interleaved_elem=[gate],
                                           keep_original_interleaved_elem=False,
                                           group_gates='CNOT-Dihedral')
        # Verify the generated sequences
        rb_opts = {}
        rb_opts['nseeds'] = 1
        rb_opts['rb_pattern'] = [[0, 1]]
        vec_len = 5
        rb_opts['length_vector'] = [vec_len]
        rb_opts['length_multiplier'] = 1
        rb_opts['interleaved_elem'] = [gate]
        seed = 0
        circ_index = 0
        self.assertEqual(rb_cnotdihedral_Z_circs[seed][circ_index].name,
                         'rb_cnotdihedral_Z_length_%d_seed_%d' % (circ_index, seed),
                         'Error: incorrect circuit name')
        self.assertEqual(rb_cnotdihedral_X_circs[seed][circ_index].name,
                         'rb_cnotdihedral_X_length_%d_seed_%d' % (circ_index, seed),
                         'Error: incorrect circuit name')
        self.assertEqual(rb_cnotdihedral_interleaved_Z_circs[seed][circ_index].name,
                         'rb_cnotdihedral_interleaved_Z_length_%d_seed_%d' % (circ_index, seed),
                         'Error: incorrect circuit name')
        self.assertEqual(rb_cnotdihedral_interleaved_X_circs[seed][circ_index].name,
                         'rb_cnotdihedral_interleaved_X_length_%d_seed_%d' % (circ_index, seed),
                         'Error: incorrect circuit name')
        self.compare_interleaved_circuit(rb_cnotdihedral_Z_circs[seed][circ_index],
                                         rb_cnotdihedral_interleaved_Z_circs[seed][circ_index],
                                         2, rb_opts, [['cx 0 1']], vec_len)
        self.compare_cnotdihedral_circuit(rb_cnotdihedral_Z_circs[seed][circ_index],
                                          rb_cnotdihedral_X_circs[seed][circ_index],
                                          2, rb_opts, vec_len)
        self.compare_cnotdihedral_circuit(rb_cnotdihedral_interleaved_Z_circs[seed][circ_index],
                                          rb_cnotdihedral_interleaved_X_circs[seed][circ_index],
                                          2, rb_opts, vec_len)

    @data(1, 2, 3)
    def test_interleaved_randomized_benchmarking_seq_random_clifford_gates(self, num_qubits):
        """interleaved random Clifford gates in RB"""
        clifford = qiskit.quantum_info.random_clifford(num_qubits)
        test_circ = clifford.to_circuit()
        test_gates = clifford.to_instruction()
        seed = 0
        circ_index = 0
        rb_opts = {}
        rb_opts['nseeds'] = 1
        rb_opts['rb_pattern'] = [list(range(num_qubits))]
        vec_len = 5
        rb_opts['length_vector'] = [vec_len]
        rb_opts['length_multiplier'] = 1

        rb_original_circs, _, rb_interleaved_circs = rb.randomized_benchmarking_seq(
            nseeds=1, length_vector=[5], rb_pattern=[list(range(num_qubits))],
            interleaved_elem=[clifford],
            keep_original_interleaved_elem=False
        )
        self.assertEqual(rb_original_circs[seed][circ_index].name,
                         'rb_length_%d_seed_%d' % (circ_index, seed),
                         'Error: incorrect circuit name')
        self.assertEqual(rb_interleaved_circs[seed][circ_index].name,
                         'rb_interleaved_length_%d_seed_%d' % (circ_index, seed),
                         'Error: incorrect circuit name')
        rb_opts['interleaved_elem'] = [clifford]
        self.compare_interleaved_circuit(rb_original_circs[seed][circ_index],
                                         rb_interleaved_circs[seed][circ_index],
                                         num_qubits, rb_opts, None, vec_len)

        rb_original_circs, _, rb_interleaved_circs = rb.randomized_benchmarking_seq(
            nseeds=1, length_vector=[5], rb_pattern=[list(range(num_qubits))],
            interleaved_elem=[test_circ],
            keep_original_interleaved_elem=False
        )
        self.assertEqual(rb_original_circs[seed][circ_index].name,
                         'rb_length_%d_seed_%d' % (circ_index, seed),
                         'Error: incorrect circuit name')
        self.assertEqual(rb_interleaved_circs[seed][circ_index].name,
                         'rb_interleaved_length_%d_seed_%d' % (circ_index, seed),
                         'Error: incorrect circuit name')
        rb_opts['interleaved_elem'] = [test_circ]
        self.compare_interleaved_circuit(rb_original_circs[seed][circ_index],
                                         rb_interleaved_circs[seed][circ_index],
                                         num_qubits, rb_opts, None, vec_len)

        rb_original_circs, _, rb_interleaved_circs = rb.randomized_benchmarking_seq(
            nseeds=1, length_vector=[5], rb_pattern=[list(range(num_qubits))],
            interleaved_elem=[test_gates],
            keep_original_interleaved_elem=False
        )
        self.assertEqual(rb_original_circs[seed][circ_index].name,
                         'rb_length_%d_seed_%d' % (circ_index, seed),
                         'Error: incorrect circuit name')
        self.assertEqual(rb_interleaved_circs[seed][circ_index].name,
                         'rb_interleaved_length_%d_seed_%d' % (circ_index, seed),
                         'Error: incorrect circuit name')
        rb_opts['interleaved_elem'] = [test_gates]
        self.compare_interleaved_circuit(rb_original_circs[seed][circ_index],
                                         rb_interleaved_circs[seed][circ_index],
                                         num_qubits, rb_opts, None, vec_len)

    @data(1, 2)
    def test_interleaved_randomized_benchmarking_seq_random_cnotdihedral_gates(self, num_qubits):
        """interleaved random CNOT-Dihedral gates in RB"""
        elem = rb.random_cnotdihedral(num_qubits)
        test_circ = elem.to_circuit()
        test_gates = elem.to_instruction()

        rb_opts = {}
        rb_opts['nseeds'] = 1
        rb_opts['rb_pattern'] = [list(range(num_qubits))]
        vec_len = 5
        rb_opts['length_vector'] = [vec_len]
        rb_opts['length_multiplier'] = 1
        rb_opts['group_gates'] = 'CNOT-Dihedral'
        seed = 0
        circ_index = 0

        rb_cnotdihedral_Z_circs, _, rb_cnotdihedral_X_circs, \
            rb_cnotdihedral_interleaved_Z_circs, rb_cnotdihedral_interleaved_X_circs = \
            rb.randomized_benchmarking_seq(nseeds=1, length_vector=[5],
                                           rb_pattern=[list(range(num_qubits))],
                                           interleaved_elem=[elem],
                                           keep_original_interleaved_elem=False,
                                           group_gates='CNOT-Dihedral')
        self.assertEqual(rb_cnotdihedral_Z_circs[seed][circ_index].name,
                         'rb_cnotdihedral_Z_length_%d_seed_%d' % (circ_index, seed),
                         'Error: incorrect circuit name')
        self.assertEqual(rb_cnotdihedral_X_circs[seed][circ_index].name,
                         'rb_cnotdihedral_X_length_%d_seed_%d' % (circ_index, seed),
                         'Error: incorrect circuit name')
        self.assertEqual(rb_cnotdihedral_interleaved_Z_circs[seed][circ_index].name,
                         'rb_cnotdihedral_interleaved_Z_length_%d_seed_%d' % (circ_index, seed),
                         'Error: incorrect circuit name')
        self.assertEqual(rb_cnotdihedral_interleaved_X_circs[seed][circ_index].name,
                         'rb_cnotdihedral_interleaved_X_length_%d_seed_%d' % (circ_index, seed),
                         'Error: incorrect circuit name')
        rb_opts['interleaved_elem'] = [elem]
        self.compare_interleaved_circuit(rb_cnotdihedral_Z_circs[seed][circ_index],
                                         rb_cnotdihedral_interleaved_Z_circs[seed][circ_index],
                                         num_qubits, rb_opts, None, vec_len)
        self.compare_cnotdihedral_circuit(rb_cnotdihedral_Z_circs[seed][circ_index],
                                          rb_cnotdihedral_X_circs[seed][circ_index],
                                          num_qubits, rb_opts, vec_len)
        self.compare_cnotdihedral_circuit(rb_cnotdihedral_interleaved_Z_circs[seed][circ_index],
                                          rb_cnotdihedral_interleaved_X_circs[seed][circ_index],
                                          num_qubits, rb_opts, vec_len)

        rb_cnotdihedral_Z_circs, _, rb_cnotdihedral_X_circs, \
            rb_cnotdihedral_interleaved_Z_circs, rb_cnotdihedral_interleaved_X_circs = \
            rb.randomized_benchmarking_seq(nseeds=1, length_vector=[5],
                                           rb_pattern=[list(range(num_qubits))],
                                           interleaved_elem=[test_circ],
                                           keep_original_interleaved_elem=False,
                                           group_gates='CNOT-Dihedral')
        self.assertEqual(rb_cnotdihedral_Z_circs[seed][circ_index].name,
                         'rb_cnotdihedral_Z_length_%d_seed_%d' % (circ_index, seed),
                         'Error: incorrect circuit name')
        self.assertEqual(rb_cnotdihedral_X_circs[seed][circ_index].name,
                         'rb_cnotdihedral_X_length_%d_seed_%d' % (circ_index, seed),
                         'Error: incorrect circuit name')
        self.assertEqual(rb_cnotdihedral_interleaved_Z_circs[seed][circ_index].name,
                         'rb_cnotdihedral_interleaved_Z_length_%d_seed_%d' % (circ_index, seed),
                         'Error: incorrect circuit name')
        self.assertEqual(rb_cnotdihedral_interleaved_X_circs[seed][circ_index].name,
                         'rb_cnotdihedral_interleaved_X_length_%d_seed_%d' % (circ_index, seed),
                         'Error: incorrect circuit name')
        rb_opts['interleaved_elem'] = [test_circ]
        self.compare_interleaved_circuit(rb_cnotdihedral_Z_circs[seed][circ_index],
                                         rb_cnotdihedral_interleaved_Z_circs[seed][circ_index],
                                         num_qubits, rb_opts, None, vec_len)
        self.compare_cnotdihedral_circuit(rb_cnotdihedral_Z_circs[seed][circ_index],
                                          rb_cnotdihedral_X_circs[seed][circ_index],
                                          num_qubits, rb_opts, vec_len)
        self.compare_cnotdihedral_circuit(rb_cnotdihedral_interleaved_Z_circs[seed][circ_index],
                                          rb_cnotdihedral_interleaved_X_circs[seed][circ_index],
                                          num_qubits, rb_opts, vec_len)

        rb_cnotdihedral_Z_circs, _, rb_cnotdihedral_X_circs, \
            rb_cnotdihedral_interleaved_Z_circs, rb_cnotdihedral_interleaved_X_circs = \
            rb.randomized_benchmarking_seq(nseeds=1, length_vector=[5],
                                           rb_pattern=[list(range(num_qubits))],
                                           interleaved_elem=[test_gates],
                                           keep_original_interleaved_elem=False,
                                           group_gates='CNOT-Dihedral')
        self.assertEqual(rb_cnotdihedral_Z_circs[seed][circ_index].name,
                         'rb_cnotdihedral_Z_length_%d_seed_%d' % (circ_index, seed),
                         'Error: incorrect circuit name')
        self.assertEqual(rb_cnotdihedral_X_circs[seed][circ_index].name,
                         'rb_cnotdihedral_X_length_%d_seed_%d' % (circ_index, seed),
                         'Error: incorrect circuit name')
        self.assertEqual(rb_cnotdihedral_interleaved_Z_circs[seed][circ_index].name,
                         'rb_cnotdihedral_interleaved_Z_length_%d_seed_%d' % (circ_index, seed),
                         'Error: incorrect circuit name')
        self.assertEqual(rb_cnotdihedral_interleaved_X_circs[seed][circ_index].name,
                         'rb_cnotdihedral_interleaved_X_length_%d_seed_%d' % (circ_index, seed),
                         'Error: incorrect circuit name')
        rb_opts['interleaved_elem'] = [test_gates]
        self.compare_interleaved_circuit(rb_cnotdihedral_Z_circs[seed][circ_index],
                                         rb_cnotdihedral_interleaved_Z_circs[seed][circ_index],
                                         num_qubits, rb_opts, None, vec_len)
        self.compare_cnotdihedral_circuit(rb_cnotdihedral_Z_circs[seed][circ_index],
                                          rb_cnotdihedral_X_circs[seed][circ_index],
                                          num_qubits, rb_opts, vec_len)
        self.compare_cnotdihedral_circuit(rb_cnotdihedral_interleaved_Z_circs[seed][circ_index],
                                          rb_cnotdihedral_interleaved_X_circs[seed][circ_index],
                                          num_qubits, rb_opts, vec_len)


class TestRBUtils(unittest.TestCase):
    """Test for RB utilities."""

    def test_coherence_limit(self):
        """Test coherence_limit."""
        t1 = 100.
        t2 = 100.
        gate2Q = 0.5
        gate1Q = 0.1
        twoq_coherence_err = rb.rb_utils.coherence_limit(2, [t1, t1],
                                                         [t2, t2], gate2Q)

        oneq_coherence_err = rb.rb_utils.coherence_limit(1, [t1],
                                                         [t2], gate1Q)

        self.assertAlmostEqual(oneq_coherence_err, 0.00049975, 6,
                               "Error: 1Q Coherence Limit")

        self.assertAlmostEqual(twoq_coherence_err, 0.00597, 5,
                               "Error: 2Q Coherence Limit")

    @staticmethod
    def create_fake_circuits(num_gates):
        """Helper function to generate list of circuits with given basis gate numbers."""
        circs = []
        for num_gate in num_gates:
            circ = qiskit.QuantumCircuit(2)
            for _ in range(num_gate[0]):
                circ.append(U1Gate(0), [0])
            for _ in range(num_gate[1]):
                circ.append(U2Gate(0, 0), [0])
            for _ in range(num_gate[2]):
                circ.append(U3Gate(0, 0, 0), [0])
            for _ in range(num_gate[3]):
                circ.cx(0, 1)
            circs.append(circ)

        return circs

    def test_gates_per_clifford(self):
        """Test gate per Clifford."""
        num_gates = [[6, 7, 5, 8], [10, 12, 8, 14]]
        clifford_lengths = np.array([4, 8])

        circs = self.create_fake_circuits(num_gates)
        gpc = rb.rb_utils.gates_per_clifford(transpiled_circuits_list=[circs],
                                             clifford_lengths=clifford_lengths,
                                             basis=['u1', 'u2', 'u3', 'cx'],
                                             qubits=[0])
        ncliffs = np.sum(clifford_lengths + 1)

        self.assertAlmostEqual(gpc[0]['u1'],
                               (num_gates[0][0] + num_gates[1][0]) / ncliffs)
        self.assertAlmostEqual(gpc[0]['u2'],
                               (num_gates[0][1] + num_gates[1][1]) / ncliffs)
        self.assertAlmostEqual(gpc[0]['u3'],
                               (num_gates[0][2] + num_gates[1][2]) / ncliffs)
        self.assertAlmostEqual(gpc[0]['cx'],
                               (num_gates[0][3] + num_gates[1][3]) / ncliffs)

    def test_gates_per_clifford_with_invalid_basis(self):
        """Test gate per Clifford when invalid gate is included in basis."""
        num_gates = [[1, 1, 1, 1]]
        clifford_lengths = np.array([1])

        circs = self.create_fake_circuits(num_gates)
        gpc = rb.rb_utils.gates_per_clifford(transpiled_circuits_list=[circs],
                                             clifford_lengths=clifford_lengths,
                                             basis=['u1', 'u2', 'u3', 'cx', 'fake_gate'],
                                             qubits=[0])

        self.assertAlmostEqual(gpc[0]['fake_gate'], 0)

    def test_calculate_1q_epg(self):
        """Test calculating EPGs of single qubit gates."""
        gpc = {0: {'cx': 0, 'u1': 0.1, 'u2': 0.3, 'u3': 0.5}}
        epc = 2.6e-4

        epg_u1 = 0
        epg_u2 = epc / (0.3 + 2 * 0.5)
        epg_u3 = 2 * epg_u2

        epgs = rb.calculate_1q_epg(gpc, epc, 0)

        # test raise error when invalid qubit is specified.
        with self.assertRaises(QiskitError):
            rb.calculate_1q_epg(gpc, epc, 1)

        # check values
        self.assertAlmostEqual(epgs['u1'], epg_u1)
        self.assertAlmostEqual(epgs['u2'], epg_u2)
        self.assertAlmostEqual(epgs['u3'], epg_u3)

    def test_calculate_1q_epg_with_wrong_basis(self):
        """Test calculating EPGs of single qubit gates with wrong basis."""
        gpc = {0: {'cx': 0, 'rx': 0.3, 'ry': 0.3, 'rz': 0.3}}
        epc = 2.6e-4

        with self.assertRaises(QiskitError):
            rb.calculate_1q_epg(gpc, epc, 0)

    def test_calculate_1q_epg_with_cx(self):
        """Test calculating EPGs of single qubit gates with nonzero two qubit gate."""
        gpc = {0: {'cx': 1.5, 'u1': 0.1, 'u2': 0.3, 'u3': 0.5}}
        epc = 2.6e-4

        with self.assertRaises(QiskitError):
            rb.calculate_1q_epg(gpc, epc, 0)

    def test_calculate_2q_epg(self):
        """Test calculating EPG of two qubit gate."""
        gpc = {0: {'cx': 1.5, 'u1': 0.1, 'u2': 0.3, 'u3': 0.5},
               1: {'cx': 1.5, 'u1': 0.1, 'u2': 0.3, 'u3': 0.5}}
        epgs_q0 = {'u1': 0, 'u2': 1e-4, 'u3': 2e-4}
        epgs_q1 = {'u1': 0, 'u2': 1e-4, 'u3': 2e-4}
        epc = 1.0e-2

        alpha_1q = (1 - 2 * 1e-4) ** 0.3 * (1 - 2 * 2e-4) ** 0.5
        alpha_c_1q = 1 / 5 * (2 * alpha_1q + 3 * alpha_1q ** 2)
        alpha_c_2q = (1 - 4 / 3 * epc) / alpha_c_1q

        epg_with_1q_epgs = 3 / 4 * (1 - alpha_c_2q) / 1.5
        epg_without_1q_epgs = epc/1.5

        # test raise error when invalid number of qubit is given.
        with self.assertRaises(QiskitError):
            rb.calculate_2q_epg(gpc, epc, [0, 1, 2], [epgs_q0, epgs_q1])

        # test raise error when invalid qubit pair is specified.
        with self.assertRaises(QiskitError):
            rb.calculate_2q_epg(gpc, epc, [0, 2], [epgs_q0, epgs_q1])

        # when 1q EPGs are not given
        self.assertAlmostEqual(
            rb.calculate_2q_epg(gpc, epc, [0, 1]),
            epg_without_1q_epgs,
        )

        # when 1q EPGs are given
        self.assertAlmostEqual(
            rb.calculate_2q_epg(gpc, epc, [0, 1], [epgs_q0, epgs_q1]),
            epg_with_1q_epgs
        )

    def test_calculate_2q_epg_with_another_gate_name(self):
        """Test calculating EPG of two qubit gate when another gate name is specified."""
        gpc = {0: {'cz': 1.5, 'u1': 0.1, 'u2': 0.3, 'u3': 0.5},
               1: {'cz': 1.5, 'u1': 0.1, 'u2': 0.3, 'u3': 0.5}}
        epc = 1.0e-2

        # test raise error when default basis name is specified
        with self.assertRaises(QiskitError):
            rb.calculate_2q_epg(gpc, epc, [0, 1])

        # pass when correct name is specified
        self.assertAlmostEqual(
            rb.calculate_2q_epg(gpc, epc, [0, 1], two_qubit_name='cz'), epc/1.5
        )

    def test_twoQ_clifford(self):
        """Test calculating EPC from EPC."""
        error_1q = 0.001
        error_2q = 0.01

        gpc = {0: {'cx': 1.5, 'u1': 0.1, 'u2': 0.3, 'u3': 0.5},
               1: {'cx': 1.5, 'u1': 0.1, 'u2': 0.3, 'u3': 0.5}}
        gate_qubit = [0, 0, 0, 1, 1, 1, -1]
        gate_err = [0, error_1q, 2*error_1q, 0, error_1q, 2*error_1q, error_2q]

        alpha_2q = (1 - 4 / 3 * error_2q) ** 1.5
        alpha_1q = (1 - 2 * error_1q) ** 0.3 * (1 - 4 * error_1q) ** 0.5

        alpha_c_2q = 1 / 5 * (2 * alpha_1q + 3 * alpha_1q * alpha_1q) * alpha_2q

        with self.assertWarns(DeprecationWarning):
            epc = rb.twoQ_clifford_error(gpc, gate_qubit, gate_err)

        self.assertAlmostEqual(epc, 3 / 4 * (1 - alpha_c_2q))

    def test_calculate_1q_epc(self):
        """Test calculating EPC from EPG of single qubit gates."""
        gpc = {0: {'cx': 1.5, 'u1': 0.1, 'u2': 0.3, 'u3': 0.5}}
        epgs = {'u1': 0, 'u2': 0.001, 'u3': 0.002}

        epc_ref = 1 - (1 - 0.001)**0.3 * (1 - 0.002)**0.5
        epc = rb.calculate_1q_epc(gpc, epgs, 0)

        self.assertAlmostEqual(epc, epc_ref)

    def test_calculate_2q_epc(self):
        """Test calculating two qubit EPC from EPGs."""
        gpc = {0: {'cx': 1.5, 'u1': 0.1, 'u2': 0.3, 'u3': 0.5},
               1: {'cx': 1.5, 'u1': 0.1, 'u2': 0.3, 'u3': 0.5}}
        epgs_q0 = {'u1': 0, 'u2': 1e-4, 'u3': 2e-4}
        epgs_q1 = {'u1': 0, 'u2': 1e-4, 'u3': 2e-4}
        epg_q01 = 1e-3

        alpha_1q = (1 - 2 * 1e-4)**0.3 * (1 - 2 * 2e-4)**0.5
        alpha_2q = (1 - 4 / 3 * 1e-3)**1.5

        alpha_c = 1 / 5 * (2 * alpha_1q + 3 * alpha_1q ** 2) * alpha_2q

        self.assertAlmostEqual(
            rb.calculate_2q_epc(gpc, epg_q01, [0, 1], [epgs_q0, epgs_q1]),
            3 / 4 * (1 - alpha_c)
        )

    def test_calculate_2q_epc_with_another_gate_name(self):
        """Test calculating two qubit EPC from EPGs when another gate name is specified."""
        gpc = {0: {'cz': 1.5, 'u1': 0.1, 'u2': 0.3, 'u3': 0.5},
               1: {'cz': 1.5, 'u1': 0.1, 'u2': 0.3, 'u3': 0.5}}
        epgs_q0 = {'u1': 0, 'u2': 1e-4, 'u3': 2e-4}
        epgs_q1 = {'u1': 0, 'u2': 1e-4, 'u3': 2e-4}
        epg_q01 = 1e-3

        alpha_1q = (1 - 2 * 1e-4) ** 0.3 * (1 - 2 * 2e-4) ** 0.5
        alpha_2q = (1 - 4 / 3 * 1e-3) ** 1.5

        alpha_c = 1 / 5 * (2 * alpha_1q + 3 * alpha_1q ** 2) * alpha_2q

        with self.assertRaises(QiskitError):
            rb.calculate_2q_epc(gpc, epg_q01, [0, 1], [epgs_q0, epgs_q1])

        self.assertAlmostEqual(
            rb.calculate_2q_epc(gpc, epg_q01, [0, 1], [epgs_q0, epgs_q1], two_qubit_name='cz'),
            3 / 4 * (1 - alpha_c)
        )


if __name__ == '__main__':
    unittest.main()
