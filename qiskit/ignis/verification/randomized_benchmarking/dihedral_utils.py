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
# NOTE: this is because BasicUtils doesn't define any arguments on it's methods
# so the implementations here don't match the abstract class. This needs to
# be fixed or reworked without an abstract class
# pylint: disable=arguments-differ


"""
Advanced CNOT-dihedral operations needed for randomized benchmarking.
"""

import numpy as np
from .dihedral import make_dict_0, make_dict_next
from .basic_utils import BasicUtils

try:
    import cPickle as pickle
except ImportError:
    import pickle


class DihedralUtils(BasicUtils):
    """
        Class for util functions for the CNOT-dihedral group.
    """

    def __init__(self, num_qubits=2, group_tables=None, elmnt=None,
                 gatelist=None, elmnt_key=None):
        """
        Args:
            num_qubits (int): number of qubits, dimension of the CNOT-dihedral
                object.
            group_tables (dict): table of all CNOTDihedral objects of a given
                dimension.
            elmnt (CNOTDihedral): a CNOT-dihedral group element.
            elmnt_key (str): a unique key of a CNOTDihedral object.
            gatelist (list): a list of gates corresponding to a
                CNOTDihedral object.
        """

        self._num_qubits = num_qubits
        self._group_tables = group_tables
        self._elmnt = elmnt
        self._elmnt_key = elmnt_key
        self._gatelist = gatelist

    def num_qubits(self):
        """Return the number of qubits of the CNOTDihedral object."""
        return self._num_qubits

    def group_tables(self):
        """Return the CNOT-dihedral group tables."""
        return self._group_tables

    def elmnt(self):
        """Return a CNOTDihedral object."""
        return self._elmnt

    def elmnt_key(self):
        """Return a unique key of a CNOTDihedral object."""
        return self._elmnt_key

    def gatelist(self):
        """Return a list of gates corresponding to
            a CNOTDihedral object."""
        return self._gatelist

    # --------------------------------------------------------
    # Create CNOT-dihedral group tables
    # --------------------------------------------------------
    def cnot_dihedral_tables(self, num_qubits):
        """
        Generate a table of all CNOT-dihedral group elements
        on num_qubits.

        Args:
            num_qubits (int): number of qubits for the CNOTDihedral
                object.

        Returns:
            dict: A table of all CNOT-dihedral group elements
                on num_qubits.
        """

        g_0 = make_dict_0(num_qubits)
        g_dict = [g_0]
        g_size = len(g_0)
        g_out = {**g_0}
        while g_size > 0:
            g_i = make_dict_next(num_qubits, g_dict)
            g_size = len(g_i)
            g_dict.append(g_i)
            g_out.update({**g_i})

        return g_out

    def pickle_dihedral_table(self, num_qubits=2):
        """
        Create pickled versions of the CNOT-dihedral group
        tables.

        Args:
            num_qubits (int): number of qubits of the CNOTDihedral
                object.

        Raises:
            ValueError: number of qubits bigger than 2 is
                not supported.
         """

        if num_qubits > 2:
            raise ValueError(
                "number of qubits bigger than is not supported for pickle")

        picklefile = 'cnot_dihedral_' + str(num_qubits) + '.pickle'
        table = self.cnot_dihedral_tables(num_qubits)

        with open(picklefile, "wb") as pf:
            pickle.dump(table, pf)
        pf.close()

    def load_dihedral_table(self, picklefile='cnot_dihedral_2.pickle'):
        """
          Load pickled files of the CNOT-dihedral group tables.

          Args:
              picklefile (str): pickle file name.

          Returns:
              dict: A table of all CNOT-dihedral group elements.
          """
        with open(picklefile, "rb") as pf:
            pickletable = pickle.load(pf)
        pf.close()
        return pickletable

    def load_tables(self, num_qubits):
        """
        Return the CNOT dihedral group tables.

        Args:
            num_qubits (int): number of qubits for the CNOTDihedral object.

        Returns:
            dict: A table of all the CNOTDihedral objects.

        Raises:
            ValueError: number of qubits bigger than 2 is
                not supported.
        """

        # load the cnot-dihedral tables, but only if we're using
        # that particular num_qubits
        if num_qubits == 1:
            # 1Q - load table programmatically
            dihedral_tables = self.cnot_dihedral_tables(1)
        elif num_qubits == 2:
            # 2Q
            # Try to load the table in from file. If it doesn't exist then make
            # the file
            try:
                dihedral_tables = self.load_dihedral_table(
                    picklefile='cnot_dihedral_%d.pickle' % num_qubits)
            except (OSError, EOFError):
                # table doesn't exist, so save it
                # this will save time next run
                print('Making the n=%d CNOT-dihedral Table' % num_qubits)
                self.pickle_dihedral_table(num_qubits=num_qubits)
                dihedral_tables = self.load_dihedral_table(
                    picklefile='cnot_dihedral_%d.pickle' % num_qubits)
            except pickle.UnpicklingError:
                # handle error
                dihedral_tables = self.cnot_dihedral_tables(num_qubits)
        else:
            raise ValueError("The number of qubits should be only 1 or 2")

        self._group_tables = dihedral_tables
        return dihedral_tables

    def elem_to_gates(self, circ):
        """
        Convert a CNOT-dihedral list of gates for the QuantumCircuit.

        Args:
            circ (list): list of gates of an element in the CNOT-dihedral
                group (from the group table).

        Returns:
            list: List of gates for the QuantumCircuit.

        Raises:
            ValueError: unknown gate type.
        """
        gatelist = []
        for gate in circ:
            if gate[0] == 'u1':
                angle = {
                    1: np.pi/4,
                    2: np.pi/2,
                    3: np.pi*3/4,
                    4: np.pi,
                    5: np.pi*5/4,
                    6: np.pi*3/2,
                    7: np.pi*7/4
                }[gate[1]]
                gatelist.append('u1 %f %d' % (angle, gate[2]))
            elif gate[0] == 'x':
                gatelist.append('x %d' % gate[1])
            elif gate[0] == 'cx':
                gatelist.append('cx %d %d' % (gate[1], gate[2]))
            else:
                raise ValueError("Unknown gate type", gate[0])

        return gatelist

    # ------------------------------------------------------
    # Create a CNOT-dihedral element based on a unique index
    # ------------------------------------------------------
    def cnot_dihedral_gates(self, idx, G_table, G_keys):
        """
        Make a single CNOT-dihedral element on num_qubits.

        Args:
            idx (int): a unique index of a single CNOT-dihedral element.
            G_table (dict): the CNOT-dihedral group table on num_qubits.
            G_keys (list): list of keys to the CNOT-dihedral group table.

        Returns:
            CNOTDihedral: A single CNOT-dihedral element on num_qubits.
        """

        elem_key = G_keys[idx]
        elem = G_table[elem_key]
        circ = (G_table[elem_key][1])
        gatelist = self.elem_to_gates(circ)

        self._gatelist = gatelist
        self._elmnt = elem[0]
        return elem

    # ---------------------------------------------------------
    # Main function that generates a random CNOT-dihedral gate
    # ---------------------------------------------------------
    def random_gates(self, num_qubits):
        """
        Pick a random CNOT-dihedral element on num_qubits.

        Args:
            num_qubits (int): number of qubits of the CNOTDihedral object.

        Returns:
            CNOTDihedral: A CNOTDihedral object.

        Raises:
            ValueError: number of qubits bigger than 2 is
                not supported.
        """

        if num_qubits > 2:
            raise ValueError("The number of qubits should be only 1 or 2")

        G_table = self.load_tables(num_qubits)
        G_keys = list(G_table.keys())

        elem = self.cnot_dihedral_gates(np.random.randint(
            0, len(G_table)), G_table, G_keys)
        self._elmnt = elem
        self._gatelist = elem[1]
        self._num_qubits = num_qubits

        return elem

    # -----------------------------------------------
    # Compose a new gatelist with an existing element
    # -----------------------------------------------
    def compose_gates(self, elem, next_elem):
        """
        Compose two CNOTDihedral objects.

        Args:
            elem (CNOTDihedral): a CNOTDihedral object.
            next_elem (CNOTDihedral): another CNOTDihedral object.

        Returns:
            CNOTDihedral: A CNOTDihedral object, that is a composition
            of the two CNOTDihedral objects.
        """
        self._gatelist = self.elem_to_gates(next_elem[1])
        elem = next_elem[0] * elem
        self._elmnt = elem
        return elem

    # -------------------------------------------------------------------
    # Main function that calculates an inverse of a CNOT dihedral element
    # -------------------------------------------------------------------
    def find_inverse_gates(self, num_qubits, elem):
        """
        Find the inverse of a CNOT-dihedral element.

        Args:
            num_qubits (int): the dimension of the CNOTDihedral element.
            elem (CNOTDihedral): an element in the CNOTDihedral group table.

        Returns:
            list: An inverse list of gates.

        Raises:
            ValueError: number of qubits bigger than 2 is
                not supported.
        """
        gatelist = elem[1]
        if num_qubits in (1, 2):
            inverse = []
            for gate in reversed(gatelist):
                if gate[0] == "u1":
                    inverse.append(("u1", 8 - gate[1], gate[2]))
                else:
                    inverse.append(gate)
            return self.elem_to_gates(inverse)
        raise ValueError("The number of qubits should be only 1 or 2")

    def find_key(self, elem, num_qubits):
        """
        Find the key of a CNOTDihedral object in the group table.

        Args:
            elem (CNOTDihedral): CNOTDihedral object
            num_qubits (int): the dimension of the CNOTDihedral object.

        Returns:
            str: A unique key to the CNOT-dihedral group table.
        """
        G_table = self.load_tables(num_qubits)
        elem.poly.weight_0 = 0  # set global phase
        assert elem.key in G_table, \
            "inverse not found in lookup table!\n%s" % elem
        return elem.key
