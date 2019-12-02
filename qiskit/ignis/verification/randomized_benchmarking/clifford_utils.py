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

"""Advanced Clifford operations needed for randomized benchmarking."""

import numpy as np
from .Clifford import Clifford
from .basic_utils import BasicUtils

try:
    import cPickle as pickle
except ImportError:
    import pickle


class CliffordUtils(BasicUtils):
    """Class for util functions for the Clifford group."""

    def __init__(self, num_qubits=2, group_tables=None, elmnt=None,
                 gatelist=None, elmnt_key=None):
        """
        Args:
            num_qubits: number of qubits.
            group_table: table of Clifford objects.
            elmnt: a group element.
            elmnt_key: a unique index of a Clifford object.
            gatelist: a list of gates corresponding to a
                Clifford object.
        """

        self._num_qubits = num_qubits
        self._group_tables = group_tables
        self._elmnt = elmnt
        self._elmnt_key = elmnt_key
        self._gatelist = gatelist

    def num_qubits(self):
        """Return the number of qubits."""
        return self._num_qubits

    def group_tables(self):
        """Return the Clifford group tables."""
        return self._group_tables

    def elmnt(self):
        """Return a Clifford object."""
        return self._elmnt

    def elmnt_key(self):
        """Return a unique index of a Clifford object."""
        return self._elmnt_key

    def gatelist(self):
        """Return a list of gates corresponding to a Clifford object."""
        return self._gatelist

    # ----------------------------------------------------------------------------------------
    # Functions that convert to/from a Clifford object
    # ----------------------------------------------------------------------------------------
    def compose_gates(self, cliff, gatelist):
        """
        Add gates to a Clifford object from a list of gates.

        Args:
            cliff: A Clifford class object.
            gatelist: a list of gates.

        Returns:
            A Clifford class object.
        """

        for op in gatelist:
            split = op.split()
            q1 = int(split[1])
            if split[0] == 'v':
                cliff.v(q1)
            elif split[0] == 'w':
                cliff.w(q1)
            elif split[0] == 'x':
                cliff.x(q1)
            elif split[0] == 'y':
                cliff.y(q1)
            elif split[0] == 'z':
                cliff.z(q1)
            elif split[0] == 'cx':
                cliff.cx(q1, int(split[2]))
            elif split[0] == 'h':
                cliff.h(q1)
            elif split[0] == 's':
                cliff.s(q1)
            elif split[0] == 'sdg':
                cliff.sdg(q1)
            else:
                raise ValueError("Unknown gate type: ", op)

        self._gatelist = gatelist
        self._elmnt = cliff
        return cliff

    def clifford_from_gates(self, num_qubits, gatelist):
        """
        Generates a Clifford object from a list of gates.

        Args:
            num_qubits: the number of qubits for the Clifford.
            gatelist: a list of gates.

        Returns:
            A num-qubit Clifford class object.
        """
        cliff = Clifford(num_qubits)
        new_cliff = self.compose_gates(cliff, gatelist)
        return new_cliff

    # --------------------------------------------------------
    # Add gates to Cliffords
    # --------------------------------------------------------

    def pauli_gates(self, gatelist, q, pauli):
        """Adds a pauli gate on qubit q"""
        if pauli == 2:
            gatelist.append('x ' + str(q))
        elif pauli == 3:
            gatelist.append('y ' + str(q))
        elif pauli == 1:
            gatelist.append('z ' + str(q))

    def h_gates(self, gatelist, q, h):
        """Adds a hadamard gate or not on qubit q"""
        if h == 1:
            gatelist.append('h ' + str(q))

    def v_gates(self, gatelist, q, v):
        """Adds an axis-swap-gates on qubit q."""
        #  rotation is V=HSHS = [[0,1],[1,1]] tableau
        #  takes Z->X->Y->Z
        #  V is of order 3, and two V-gates is W-gate, so: W=VV and WV=I
        if v == 1:
            gatelist.append('v ' + str(q))
        elif v == 2:
            gatelist.append('w ' + str(q))

    def cx_gates(self, gatelist, ctrl, tgt):
        """Adds a controlled=x gates."""
        gatelist.append('cx ' + str(ctrl) + ' ' + str(tgt))

    # --------------------------------------------------------
    # Create a 1 or 2 Qubit Clifford based on a unique index
    # --------------------------------------------------------

    def clifford1_gates(self, idx: int):
        """
        Make a single qubit Clifford gate.

        Args:
            idx: the index (mod 24) of a single qubit Clifford.

        Returns:
            A single qubit Clifford gate.
        """

        gatelist = []
        # Cannonical Ordering of Cliffords 0,...,23
        cannonicalorder = idx % 24
        pauli = np.mod(cannonicalorder, 4)
        rotation = np.mod(cannonicalorder // 4, 3)
        h_or_not = np.mod(cannonicalorder // 12, 2)

        self.h_gates(gatelist, 0, h_or_not)

        self.v_gates(gatelist, 0, rotation)  # do the R-gates

        self.pauli_gates(gatelist, 0, pauli)

        return gatelist

    def clifford2_gates(self, idx: int):
        """
        Make a 2-qubit Clifford gate.

        Args:
            idx: the index (mod 11520) of a two-qubit Clifford.

        Returns:
            A 2-qubit Clifford gate.
        """

        gatelist = []
        cannon = idx % 11520

        pauli = np.mod(cannon, 16)
        symp = cannon // 16

        if symp < 36:  # 1-qubit Cliffords Class
            r0 = np.mod(symp, 3)
            r1 = np.mod(symp // 3, 3)
            h0 = np.mod(symp // 9, 2)
            h1 = np.mod(symp // 18, 2)

            self.h_gates(gatelist, 0, h0)
            self.h_gates(gatelist, 1, h1)
            self.v_gates(gatelist, 0, r0)
            self.v_gates(gatelist, 1, r1)

        elif symp < 360:  # CNOT-like Class
            symp = symp - 36
            r0 = np.mod(symp, 3)
            r1 = np.mod(symp // 3, 3)
            r2 = np.mod(symp // 9, 3)
            r3 = np.mod(symp // 27, 3)
            h0 = np.mod(symp // 81, 2)
            h1 = np.mod(symp // 162, 2)

            self.h_gates(gatelist, 0, h0)
            self.h_gates(gatelist, 1, h1)
            self.v_gates(gatelist, 0, r0)
            self.v_gates(gatelist, 1, r1)
            self.cx_gates(gatelist, 0, 1)
            self.v_gates(gatelist, 0, r2)
            self.v_gates(gatelist, 1, r3)

        elif symp < 684:  # iSWAP-like Class
            symp = symp - 360
            r0 = np.mod(symp, 3)
            r1 = np.mod(symp // 3, 3)
            r2 = np.mod(symp // 9, 3)
            r3 = np.mod(symp // 27, 3)
            h0 = np.mod(symp // 81, 2)
            h1 = np.mod(symp // 162, 2)

            self.h_gates(gatelist, 0, h0)
            self.h_gates(gatelist, 1, h1)
            self.v_gates(gatelist, 0, r0)
            self.v_gates(gatelist, 1, r1)
            self.cx_gates(gatelist, 0, 1)
            self.cx_gates(gatelist, 1, 0)
            self.v_gates(gatelist, 0, r2)
            self.v_gates(gatelist, 1, r3)

        else:  # SWAP Class
            symp = symp - 684
            r0 = np.mod(symp, 3)
            r1 = np.mod(symp // 3, 3)
            h0 = np.mod(symp // 9, 2)
            h1 = np.mod(symp // 18, 2)

            self.h_gates(gatelist, 0, h0)
            self.h_gates(gatelist, 1, h1)

            self.v_gates(gatelist, 0, r0)
            self.v_gates(gatelist, 1, r1)

            self.cx_gates(gatelist, 0, 1)
            self.cx_gates(gatelist, 1, 0)
            self.cx_gates(gatelist, 0, 1)

        self.pauli_gates(gatelist, 0, np.mod(pauli, 4))
        self.pauli_gates(gatelist, 1, pauli // 4)

        return gatelist

    # --------------------------------------------------------
    # Create a 1 or 2 Qubit Clifford tables
    # --------------------------------------------------------
    def clifford2_gates_table(self):
        """
        Generate a table of all 2-qubit Clifford gates.

        Args:
            None.

        Returns:
            A table of all 2-qubit Clifford gates.
        """
        cliffords2 = {}
        for i in range(11520):
            circ = self.clifford2_gates(i)
            key = self.clifford_from_gates(2, circ).index()
            cliffords2[key] = circ
        return cliffords2

    def clifford1_gates_table(self):
        """
        Generate a table of all 1-qubit Clifford gates.

        Args:
            None.

        Returns:
            A table of all 1-qubit Clifford gates.
        """
        cliffords1 = {}
        for i in range(24):
            circ = self.clifford1_gates(i)
            key = self.clifford_from_gates(1, circ).index()
            cliffords1[key] = circ
        return cliffords1

    def pickle_clifford_table(self, picklefile='cliffords2.pickle',
                              num_qubits=2):
        """
        Create pickled versions of the 1 and 2 qubit Clifford tables.

        Args:
            picklefile: pickle file name.
            num_qubits: number of qubits.

        Returns:
            A pickle file with the 1 and 2 qubit Clifford tables.
        """
        cliffords = {}
        if num_qubits == 1:
            cliffords = self.clifford1_gates_table()
        elif num_qubits == 2:
            cliffords = self.clifford2_gates_table()
        else:
            raise ValueError(
                "number of qubits bigger than is not supported for pickle")

        with open(picklefile, "wb") as pf:
            pickle.dump(cliffords, pf)
        pf.close()

    def load_clifford_table(self, picklefile='cliffords2.pickle'):
        """
        Load pickled files of the tables of 1 and 2 qubit Clifford tables.

        Args:
            picklefile: pickle file name.

        Returns:
            A table of 1 and 2 qubit Clifford gates.
        """
        with open(picklefile, "rb") as pf:
            pickletable = pickle.load(pf)
        pf.close()
        return pickletable

    def load_tables(self, num_qubits):
        """
        Returns the needed Clifford tables

        Args:
            num_qubits: number of qubits for the required table

        Returns:
            A table of Clifford objects
        """

        # load the clifford tables, but only if we're using that particular
        # num_qubits
        if num_qubits == 1:
            # 1Q Cliffords, load table programmatically
            clifford_tables = self.clifford1_gates_table()

        elif num_qubits == 2:
            # 2Q Cliffords
            # Try to load the table in from file. If it doesn't exist then make
            # the file
            try:
                clifford_tables = self.load_clifford_table(
                    picklefile='cliffords%d.pickle' % num_qubits)
            except (OSError, EOFError):
                # table doesn't exist, so save it
                # this will save time next run
                print('Making the n=%d Clifford Table' % num_qubits)
                self.pickle_clifford_table(
                    picklefile='cliffords%d.pickle' % num_qubits,
                    num_qubits=num_qubits)
                clifford_tables = self.load_clifford_table(
                    picklefile='cliffords%d.pickle' % num_qubits)
            except pickle.UnpicklingError:
                # handle error
                clifford_tables = self.self.clifford2_gates_table()

        else:
            raise ValueError("The number of qubits should be only 1 or 2")

        self._group_tables = clifford_tables
        return clifford_tables

    # --------------------------------------------------------
    # Main function that generates a random clifford gate
    # --------------------------------------------------------
    def random_gates(self, num_qubits):
        """
        Pick a random Clifford gate.

        Args:
            num_qubits: dimension of the Clifford.

        Returns:
            A 1 or 2 qubit Clifford gate.
        """

        if num_qubits == 1:
            cliff_gatelist = self.clifford1_gates(np.random.randint(0, 24))
        elif num_qubits == 2:
            cliff_gatelist = self.clifford2_gates(np.random.randint(0, 11520))
        else:
            raise ValueError("The number of qubits should be only 1 or 2")

        self._gatelist = cliff_gatelist
        return cliff_gatelist

    # --------------------------------------------------------
    # Main function that calculates an inverse of a clifford gate
    # --------------------------------------------------------
    def find_inverse_gates(self, num_qubits, gatelist):
        """
        Find the inverse of a Clifford gate.

        Args:
            num_qubits: the dimension of the Clifford.
            gatelist: a Clifford gate.

        Returns:
            An inverse Clifford gate.
        """

        if num_qubits in (1, 2):
            inv_gatelist = gatelist.copy()
            inv_gatelist.reverse()
            # replace v by w and w by v
            for i, _ in enumerate(inv_gatelist):
                split = inv_gatelist[i].split()
                if split[0] == 'v':
                    inv_gatelist[i] = 'w ' + split[1]
                elif split[0] == 'w':
                    inv_gatelist[i] = 'v ' + split[1]
            return inv_gatelist
        raise ValueError("The number of qubits should be only 1 or 2")

    def find_key(self, cliff, num_qubits):
        """
        Find the Clifford index.

        Args:
            cliff: a Clifford object.
            num_qubits: the dimension of the Clifford.

        Returns:
            Clifford index (an integer).
        """
        G_table = self.load_tables(num_qubits)
        assert cliff.index() in G_table, \
            "inverse not found in lookup table!\n%s" % cliff
        return cliff.index()
