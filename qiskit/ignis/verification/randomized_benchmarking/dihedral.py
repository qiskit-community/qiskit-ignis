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

"""Methods for working with the CNOT-dihedral group.

Example:

  from dihedral import CNOTDihedral
  g = CNOTDihedral(3)  # create identity element on 3 qubits
  g.cnot(0,1)          # apply CNOT from qubit 0 to qubit 1
  g.flip(2)            # apply X on qubit 2
  g.phase(3, 1)        # apply T^3 on qubit 1
  print(g)             # pretty print g

  phase polynomial =
   0 + 3*x_0 + 3*x_1 + 2*x_0*x_1
  affine function =
   (x_0,x_0 + x_1,x_2 + 1)

 This means that |x_0 x_1 x_2> transforms to omega^{p(x)}|f(x)>,
 where omega = exp(i*pi/4) from which we can read that
 T^3 on qubit 1 AFTER CNOT_{0,1} is the same as
 T^3 on qubit 0, T^3 on qubit 1, and CS_{0,1} BEFORE CNOT_{0,1}.
"""
import itertools
import copy
from functools import reduce
from operator import mul


class SpecialPolynomial():
    """Multivariate polynomial with special form.

    Maximum degree 3, n Z_2 variables, coefficients in Z_8.
    """

    def __init__(self, n_vars):
        """Construct the zero polynomial on n_vars variables."""
        #   1 constant term
        #   n linear terms x_1, ..., x_n
        #   {n choose 2} quadratic terms x_1x_2, x_1x_3, ..., x_{n-1}x_n
        #   {n choose 3} cubic terms x_1x_2x_3, ..., x_{n-2}x_{n-1}x_n
        # and coefficients in Z_8
        assert n_vars >= 1, "n_vars too small!"
        self.n_vars = n_vars
        self.nc2 = int(n_vars * (n_vars-1) / 2)
        self.nc3 = int(n_vars * (n_vars-1) * (n_vars-2) / 6)
        self.weight_0 = 0
        self.weight_1 = n_vars * [0]
        self.weight_2 = self.nc2 * [0]
        self.weight_3 = self.nc3 * [0]

    def mul_monomial(self, indices):
        """Multiply by a monomial given by indices.

        Returns the product.
        """
        length = len(indices)
        assert length < 4, "no term!"
        assert True not in [x < 0 or x >= self.n_vars for x in indices], \
            "indices out of bounds!"
        assert False not in [indices[i] < indices[i+1]
                             for i in range(length-1)], \
            "indices non-increasing!"
        result = SpecialPolynomial(self.n_vars)
        if length == 0:
            result = copy.deepcopy(self)
        else:
            terms0 = [[]]
            terms1 = [[i] for i in range(self.n_vars)]
            terms2 = [[i, j] for i in range(self.n_vars-1)
                      for j in range(i+1, self.n_vars)]
            terms3 = [[i, j, k] for i in range(self.n_vars-2)
                      for j in range(i+1, self.n_vars-1)
                      for k in range(j+1, self.n_vars)]
            for term in terms0 + terms1 + terms2 + terms3:
                value = self.get_term(term)
                new_term = list(set(term).union(set(indices)))
                result.set_term(new_term, (result.get_term(new_term) +
                                           value) % 8)
        return result

    def __mul__(self, other):
        """Multiply two polynomials."""
        assert isinstance(other, (SpecialPolynomial, int)), \
            "other isn't poly or int!: %s" % str(other)
        result = SpecialPolynomial(self.n_vars)
        if isinstance(other, int):
            result.weight_0 = (self.weight_0 * other) % 8
            result.weight_1 = [(x * other) % 8 for x in self.weight_1]
            result.weight_2 = [(x * other) % 8 for x in self.weight_2]
            result.weight_3 = [(x * other) % 8 for x in self.weight_3]
        else:
            assert self.n_vars == other.n_vars, "different n_vars!"
            terms0 = [[]]
            terms1 = [[i] for i in range(self.n_vars)]
            terms2 = [[i, j] for i in range(self.n_vars-1)
                      for j in range(i+1, self.n_vars)]
            terms3 = [[i, j, k] for i in range(self.n_vars-2)
                      for j in range(i+1, self.n_vars-1)
                      for k in range(j+1, self.n_vars)]
            for term in terms0 + terms1 + terms2 + terms3:
                value = other.get_term(term)
                if value != 0:
                    temp = copy.deepcopy(self)
                    temp = temp.mul_monomial(term)
                    temp = temp * value
                    result = result + temp
        return result

    def __rmul__(self, other):
        """Right multiplication.

        This operation is commutative.
        """
        return self.__mul__(other)

    def __add__(self, other):
        """Add two polynomials."""
        assert isinstance(other, SpecialPolynomial), "other isn't poly!"
        assert self.n_vars == other.n_vars, "different n_vars!"
        result = SpecialPolynomial(self.n_vars)
        result.weight_0 = (self.weight_0 + other.weight_0) % 8
        result.weight_1 = [(x[0] + x[1]) % 8
                           for x in zip(self.weight_1, other.weight_1)]
        result.weight_2 = [(x[0] + x[1]) % 8
                           for x in zip(self.weight_2, other.weight_2)]
        result.weight_3 = [(x[0] + x[1]) % 8
                           for x in zip(self.weight_3, other.weight_3)]
        return result

    def evaluate(self, xval):
        """Evaluate the multinomial at xval.

        if xval is a length n z2 vector, return element of Z8.
        if xval is a length n vector of multinomials, return
        a multinomial. The multinomials must all be on n vars.
        """
        assert len(xval) == self.n_vars, "wrong number of variables!"
        check_int = list(map(lambda x: isinstance(x, int), xval))
        check_poly = list(map(lambda x: isinstance(x, SpecialPolynomial),
                              xval))
        assert False not in check_int or False not in check_poly, "wrong type!"
        is_int = (False not in check_int)
        if not is_int:
            assert False not in [i.n_vars == self.n_vars for i in xval], \
                "incompatible polynomials!"
        else:
            xval = [x % 2 for x in xval]
        # Examine each term of this polynomial
        terms0 = [[]]
        terms1 = [[i] for i in range(self.n_vars)]
        terms2 = [[i, j] for i in range(self.n_vars-1)
                  for j in range(i+1, self.n_vars)]
        terms3 = [[i, j, k] for i in range(self.n_vars-2)
                  for j in range(i+1, self.n_vars-1)
                  for k in range(j+1, self.n_vars)]
        # Set the initial result and start for each term
        if is_int:
            result = 0
            start = 1
        else:
            result = SpecialPolynomial(self.n_vars)
            start = SpecialPolynomial(self.n_vars)
            start.weight_0 = 1
        # Compute the new terms and accumulate
        for term in terms0 + terms1 + terms2 + terms3:
            value = self.get_term(term)
            if value != 0:
                newterm = reduce(mul, [xval[j] for j in term], start)
                result = result + value * newterm
        if isinstance(result, int):
            result = result % 8
        return result

    def set_pj(self, indices):
        """Set to special form polynomial on subset of variables.

        p_J(x) := sum_{a subseteq J,|a| neq 0} (-2)^{|a|-1}x^a
        """
        assert True not in [x < 0 or x >= self.n_vars for x in indices], \
            "indices out of bounds!"
        indices = sorted(indices)
        subsets_2 = itertools.combinations(indices, 2)
        subsets_3 = itertools.combinations(indices, 3)
        self.weight_0 = 0
        self.weight_1 = self.n_vars * [0]
        self.weight_2 = self.nc2 * [0]
        self.weight_3 = self.nc3 * [0]
        for j in indices:
            self.set_term([j], 1)
        for j in subsets_2:
            self.set_term(list(j), 6)
        for j in subsets_3:
            self.set_term(list(j), 4)

    def get_term(self, indices):
        """Get the value of a term given the list of variables.

        Example: indices = [] returns the constant
                 indices = [0] returns the coefficient of x_0
                 indices = [0,3] returns the coefficient of x_0x_3
                 indices = [0,1,3] returns the coefficient of x_0x_1x_3

        If len(indices) > 3 the method fails.
        If the indices are out of bounds the method fails.
        If the indices are not increasing the method fails.
        """
        length = len(indices)
        assert length < 4, "no term!"
        assert True not in [x < 0 or x >= self.n_vars for x in indices], \
            "indices out of bounds!"
        assert False not in [indices[i] < indices[i+1]
                             for i in range(length-1)], \
            "indices non-increasing!"
        if length == 0:
            return self.weight_0
        if length == 1:
            return self.weight_1[indices[0]]
        if length == 2:
            # sum(self.n_vars-j, {j, 1, indices[0]})
            offset_1 = int(indices[0] * self.n_vars -
                           ((indices[0] + 1) * indices[0])/2)
            offset_2 = int(indices[1] - indices[0] - 1)
            return self.weight_2[offset_1 + offset_2]

        # sum({self.n_vars-j choose 2}, {j, 1, indices[0]})
        offset_1 = int(indices[0] * (2 + indices[0]**2 - 3*indices[0] *
                                     (self.n_vars - 1) -
                                     6 * self.n_vars +
                                     3 * self.n_vars**2)/6)
        # sum(self.n_vars-j, {j, 2, indices[1]-indices[0]})
        offset_2 = int((indices[1] - indices[0] - 1) *
                       (2 * self.n_vars - indices[1] + indices[0] - 2)/2)
        offset_3 = int(indices[2] - indices[1] - 1)
        return self.weight_3[offset_1 + offset_2 + offset_3]

    def set_term(self, indices, value):
        """Set the value of a term given the list of variables.

        Example: indices = [] returns the constant
                 indices = [0] returns the coefficient of x_0
                 indices = [0,3] returns the coefficient of x_0x_3
                 indices = [0,1,3] returns the coefficient of x_0x_1x_3

        If len(indices) > 3 the method fails.
        If the indices are out of bounds the method fails.
        If the indices are not increasing the method fails.
        The value is reduced modulo 8.
        """
        length = len(indices)
        assert length < 4, "no term!"
        assert True not in [x < 0 or x >= self.n_vars for x in indices], \
            "indices out of bounds!"
        assert False not in [indices[i] < indices[i+1]
                             for i in range(length-1)], \
            "indices non-increasing!"
        value = value % 8
        if length == 0:
            self.weight_0 = value
        elif length == 1:
            self.weight_1[indices[0]] = value
        elif length == 2:
            # sum(self.n_vars-j, {j, 1, indices[0]})
            offset_1 = int(indices[0] * self.n_vars -
                           ((indices[0] + 1) * indices[0])/2)
            offset_2 = int(indices[1] - indices[0] - 1)
            self.weight_2[offset_1 + offset_2] = value
        else:
            # sum({self.n_vars-j choose 2}, {j, 1, indices[0]})
            offset_1 = int(indices[0] * (2 + indices[0]**2 - 3*indices[0] *
                                         (self.n_vars - 1) -
                                         6 * self.n_vars +
                                         3 * self.n_vars**2)/6)
            # sum(self.n_vars-j, {j, 2, indices[1]-indices[0]})
            offset_2 = int((indices[1] - indices[0] - 1) *
                           (2 * self.n_vars - indices[1] + indices[0] - 2)/2)
            offset_3 = int(indices[2] - indices[1] - 1)
            self.weight_3[offset_1 + offset_2 + offset_3] = value

    @property
    def key(self):
        """Return a string representation."""
        tup = (self.weight_0, tuple(self.weight_1),
               tuple(self.weight_2), tuple(self.weight_3))
        return str(tup)

    def __eq__(self, x):
        """Test equality."""
        return isinstance(x, SpecialPolynomial) and self.key == x.key

    def __str__(self):
        """Return formatted string representation."""
        out = str(self.weight_0)
        for i in range(self.n_vars):
            value = self.get_term([i])
            if value != 0:
                out += " + "
                if value != 1:
                    out += (str(value) + "*")
                out += ("x_" + str(i))
        for i in range(self.n_vars-1):
            for j in range(i+1, self.n_vars):
                value = self.get_term([i, j])
                if value != 0:
                    out += " + "
                    if value != 1:
                        out += (str(value) + "*")
                    out += ("x_" + str(i) + "*x_" + str(j))
        for i in range(self.n_vars-2):
            for j in range(i+1, self.n_vars-1):
                for k in range(j+1, self.n_vars):
                    value = self.get_term([i, j, k])
                    if value != 0:
                        out += " + "
                        if value != 1:
                            out += (str(value) + "*")
                        out += ("x_" + str(i) + "*x_" + str(j) +
                                "*x_" + str(k))
        return out


class CNOTDihedral():
    """CNOT-dihedral Object Class.
    The CNOT-dihedral group on n qubits is generated by the gates
    CNOT, T and X."""

    def __init__(self, n_qubits):
        # Construct the identity element on n_qubits qubits.
        self.n_qubits = n_qubits
        # phase polynomial
        self.poly = SpecialPolynomial(n_qubits)
        # n x n invertible matrix over Z_2
        self.linear = [[int(r == c) for c in range(n_qubits)]
                       for r in range(n_qubits)]
        # binary shift, n coefficients in Z_2
        self.shift = n_qubits * [0]

    def _z2matmul(self, left, right):
        """Compute product of two n x n z2 matrices."""
        prod = [[0 for col in range(self.n_qubits)]
                for row in range(self.n_qubits)]
        for i in range(self.n_qubits):
            for j in range(self.n_qubits):
                for k in range(self.n_qubits):
                    prod[i][j] = (prod[i][j] +
                                  left[i][k]*right[k][j]) % 2
        return prod

    def _z2matvecmul(self, mat, vec):
        """Compute mat*vec of n x n z2 matrix and vector."""
        prod = [0 for row in range(self.n_qubits)]
        for i in range(self.n_qubits):
            for j in range(self.n_qubits):
                prod[i] = (prod[i] + mat[i][j] * vec[j]) % 2
        return prod

    def __mul__(self, other):
        """Left multiplication self * other."""
        assert self.n_qubits == other.n_qubits, "not same n_qubits!"
        result = CNOTDihedral(self.n_qubits)
        result.shift = [(x[0] + x[1]) % 2
                        for x in zip(self._z2matvecmul(self.linear,
                                                       other.shift),
                                     self.shift)]
        result.linear = self._z2matmul(self.linear, other.linear)
        # Compute x' = B1*x + c1 using the p_j identity
        new_vars = []
        for i in range(self.n_qubits):
            support = [j for j, e in enumerate(other.linear[i]) if e != 0]
            poly = SpecialPolynomial(self.n_qubits)
            poly.set_pj(support)
            if other.shift[i] == 1:
                poly = -1 * poly
                poly.weight_0 = (poly.weight_0 + 1) % 8
            new_vars.append(poly)
        # p' = p1 + p2(x')
        result.poly = other.poly + self.poly.evaluate(new_vars)
        return result

    def __rmul__(self, other):
        """Right multiplication other * self."""
        assert self.n_qubits == other.n_qubits, "not same n_qubits!"
        result = CNOTDihedral(self.n_qubits)
        result.shift = [(x[0] + x[1]) % 2
                        for x in zip(self._z2matvecmul(other.linear,
                                                       self.shift),
                                     other.shift)]
        result.linear = self._z2matmul(other.linear, self.linear)
        # Compute x' = B1*x + c1 using the p_j identity
        new_vars = []
        for i in range(self.n_qubits):
            support = [j for j, e in enumerate(self.linear[i]) if e != 0]
            poly = SpecialPolynomial(self.n_qubits)
            poly.set_pj(support)
            if other.shift[i] == 1:
                poly = -1 * poly
                poly.weight_0 = (poly.weight_0 + 1) % 8
            new_vars.append(poly)
        # p' = p1 + p2(x')
        result.poly = self.poly + other.poly.evaluate(new_vars)
        return result

    @property
    def key(self):
        """Return a string representation of a CNOT-dihedral object."""
        tup = (self.poly.key, tuple(map(tuple, self.linear)),
               tuple(self.shift))
        return str(tup)

    def __eq__(self, x):
        """Test equality."""
        return isinstance(x, CNOTDihedral) and self.key == x.key

    def cnot(self, i, j):
        """Apply a CNOT gate to this element.
        Left multiply the element by CNOT_{i,j}.
        """
        assert i >= 0, "i negative!"
        assert j >= 0, "j negative!"
        assert i < self.n_qubits, "i too big!"
        assert j < self.n_qubits, "j too big!"
        assert i != j, "i == j!"
        self.linear[j] = [(self.linear[i][k] + self.linear[j][k]) % 2
                          for k in range(self.n_qubits)]
        self.shift[j] = (self.shift[i] + self.shift[j]) % 2

    def phase(self, k, i):
        """Apply an k-th power of T to this element.
        Left multiply the element by T_i^k.
        """
        assert i >= 0, "i negative!"
        assert i < self.n_qubits, "i too big!"
        # If the kth bit is flipped, conjugate this gate
        if self.shift[i] == 1:
            k = (7*k) % 8
        # Take all subsets \alpha of the support of row i
        # of weight up to 3 and add k*(-2)**(|\alpha| - 1) mod 8
        # to the corresponding term.
        support = [j for j, e in enumerate(self.linear[i]) if e != 0]
        subsets_2 = itertools.combinations(support, 2)
        subsets_3 = itertools.combinations(support, 3)
        for j in support:
            value = self.poly.get_term([j])
            self.poly.set_term([j], (value + k) % 8)
        for j in subsets_2:
            value = self.poly.get_term(list(j))
            self.poly.set_term(list(j), (value + -2 * k) % 8)
        for j in subsets_3:
            value = self.poly.get_term(list(j))
            self.poly.set_term(list(j), (value + 4 * k) % 8)

    def flip(self, i):
        """Apply X to this element.
        Left multiply the element by X_i.
        """
        assert i >= 0, "i negative!"
        assert i < self.n_qubits, "i too big!"
        self.shift[i] = (self.shift[i] + 1) % 2

    def __str__(self):
        """Return formatted string representation."""
        out = "phase polynomial = \n"
        out += str(self.poly)
        out += "\naffine function = \n"
        out += " ("
        for row in range(self.n_qubits):
            wrote = False
            for col in range(self.n_qubits):
                if self.linear[row][col] != 0:
                    if wrote:
                        out += (" + x_" + str(col))
                    else:
                        out += ("x_" + str(col))
                        wrote = True
            if self.shift[row] != 0:
                out += " + 1"
            if row != self.n_qubits - 1:
                out += ","
        out += ")\n"
        return out


def make_dict_0(n_qubits):
    """Make the zero-CNOT dictionary.

    This returns the dictionary of CNOT-dihedral elements on
    n_qubits using no CNOT gates. There are 16^n elements.
    The key is a unique string and the value is a pair:
    a CNOTDihedral object and a list of gates as a string.
    """
    assert n_qubits >= 1, "n_qubits too small!"
    obj = {}
    for i in range(16**n_qubits):
        elem = CNOTDihedral(n_qubits)
        circ = []
        num = i
        for j in range(n_qubits):
            xpower = int(num % 2)
            tpower = int(((num - num % 2) / 2) % 8)
            if tpower > 0:
                elem.phase(tpower, j)
                circ.append(("u1", tpower, j))
            if xpower == 1:
                elem.flip(j)
                circ.append(("x", j))
            num = int((num - num % 16) / 16)
        obj[elem.key] = (elem, circ)
    return obj


def make_dict_next(n_qubits, dicts_prior):
    """Make the m+1 CNOT dictionary given the prior dictionaries.

    This returns the dictionary of CNOT-dihedral elements on
    n_qubits using m+1 CNOT gates given the list of dictionaries
    of circuits using 0, 1, ..., m CNOT gates.
    There are no more than 4*(n^2 - n)*|G(m)| elements.
    The key is a unique string and the value is a pair:
    a CNOTDihedral object and a list of gates as a string.
    """
    assert n_qubits >= 1, "n_qubits too small!"
    obj = {}
    for elem, circ in dicts_prior[-1].values():
        for i in range(n_qubits):
            for j in range(n_qubits):
                if i != j:
                    for tpower in range(4):
                        new_elem = copy.deepcopy(elem)
                        new_circ = copy.deepcopy(circ)
                        new_elem.cnot(i, j)
                        new_circ.append(("cx", i, j))
                        if tpower > 0:
                            new_elem.phase(tpower, j)
                            new_circ.append(("u1", tpower, j))
                        if True not in [(new_elem.key in d)
                                        for d in dicts_prior] \
                                and new_elem.key not in obj:
                            obj[new_elem.key] = (new_elem, new_circ)
    return obj
