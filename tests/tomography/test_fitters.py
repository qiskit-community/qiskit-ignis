# -*- coding: utf-8 -*-
#
# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring

import collections
import unittest

import numpy

import qiskit.ignis.tomography.fitters as fitters


class TestFitters(unittest.TestCase):
    def assertMatricesAlmostEqual(self, lhs, rhs, places=None):
        self.assertEqual(lhs.shape, rhs.shape,
                         "Marix shapes differ: {} vs {}".format(lhs, rhs))
        n, m = lhs.shape
        for x in range(n):
            for y in range(m):
                self.assertAlmostEqual(
                    lhs[x, y], rhs[x, y], places=places,
                    msg="Matrices {} and {} differ on ({}, {})".format(
                        lhs, rhs, x, y))
    # the basis matrix for 1-qubit measurement in the Pauli basis
    A = numpy.array([
        [0.5 + 0.j, 0.5 + 0.j, 0.5 + 0.j, 0.5 + 0.j],
        [0.5 + 0.j, -0.5 + 0.j, -0.5 + 0.j, 0.5 + 0.j],
        [0.5 + 0.j, 0. - 0.5j, 0. + 0.5j, 0.5 + 0.j],
        [0.5 + 0.j, 0. + 0.5j, 0. - 0.5j, 0.5 + 0.j],
        [1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
        [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j]
    ])

    def test_trace_constraint(self):
        p = numpy.array([1/2, 1/2, 1/2, 1/2, 1/2, 1/2])

        for trace_value in [1, 0.3, 2, 0, 42]:
            rho = fitters.cvx_fit(p, self.A, trace=trace_value)
            self.assertAlmostEqual(numpy.trace(rho), trace_value, places=3)

    def test_fitter_data(self):
        data = collections.OrderedDict()
        data[('X',)] = {'0': 5000}
        data[('Y',)] = {'0': 2508, '1': 2492}
        data[('Z',)] = {'0': 2490, '1': 2510}
        p, A, _ = fitters.fitter_data(data)
        self.assertMatricesAlmostEqual(self.A, A)
        n = 5000
        expected_p = [5000 / n, 0 / n, 2508 / n, 2492 / n, 2490 / n, 2510 / n]
        self.assertListEqual(expected_p, p)


if __name__ == '__main__':
    unittest.main()
