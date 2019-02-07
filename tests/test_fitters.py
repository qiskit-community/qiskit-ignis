# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Test the fitters
"""

import unittest
import random
import numpy as np
import qiskit
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import depolarizing_error
import qiskit_ignis.randomized_benchmarking.standard_rb.rb_fitters as fitters
import qiskit_ignis.randomized_benchmarking.standard_rb.randomizedbenchmarking as rb

class TestFitters(unittest.TestCase):
    """ Test the fitters """

    def test_fitters(self):
        """ Test the fitters """

        # Load simulator
        backend = qiskit.Aer.get_backend('qasm_simulator')

        tests = [{'rb_opts':
                  {'nseeds': 2,
                   'n_qubits': 3,
                   'length_vector': [10, 15, 25, 40],
                   'rb_pattern': [[0], [1, 2]],
                   'length_multiplier': [1, 2]
                  },
                  'expected':
                  {'xdata': [[10, 15, 25, 40], [20, 30, 50, 80]],
                   'raw_data': [[0.71, 0.58, 0.44, 0.38], [0.58, 0.62, 0.46, 0.3]],
                   'mean': [0.645, 0.6, 0.45, 0.34],
                   'std': [0.065, 0.02, 0.01, 0.04],
                   'fit_params': [[0.75539728, 0.95548813, 0.20966888],
                                  [0.75539745, 0.97749087, 0.2096672]],
                   'fit_err': [[0.08205335, 0.02344961, 0.15338413],
                               [0.08205379, 0.01199477, 0.15338564]],
                   'epc': [0.022255934833535218, 0.016881850903803647],
                   'epc_err': [0.000546205525578543, 0.00020715691181849986]
                  }
                 },
                 {'rb_opts':
                  {'nseeds': 1,
                   'n_qubits': 3,
                   'length_vector': [10, 15, 28, 45],
                   'rb_pattern': [[0], [1, 2]],
                   'length_multiplier': [1, 2]
                  },
                  'expected':
                  {'xdata': [[10, 15, 28, 45], [20, 30, 56, 90]],
                   'raw_data': [[0.65, 0.59, 0.42, 0.32]],
                   'mean': [0.65, 0.59, 0.42, 0.32],
                   'std': None,
                   'fit_params': [[0.67621574, 0.96316045, 0.19225674],
                                  [0.6762157, 0.98140737, 0.19225687]],
                   'fit_err': [[0.05350837, 0.01428833, 0.1000942],
                               [0.05350833, 0.00727951, 0.10009414]],
                   'epc': [0.018419777304980955, 0.013944472137794234],
                   'epc_err': [0.00027325441942194396, 0.00010343200410712699]
                  }
                 },
                 {'rb_opts':
                  {'nseeds': 2,
                   'n_qubits': 3,
                   'length_vector': [10, 15, 25, 40],
                   'rb_pattern': [[0, 1, 2]],
                   'length_multiplier': [1]
                  },
                  'expected':
                  {'xdata': [[10, 15, 25, 40]],
                   'raw_data': [[0.6, 0.49, 0.31, 0.21], [0.57, 0.55, 0.41, 0.25]],
                   'mean': [0.585, 0.52, 0.36, 0.23],
                   'std': [0.015, 0.03, 0.05, 0.02],
                   'fit_params': [[0.92395288, 0.97838208, -0.15621131]],
                   'fit_err': [[0.31235316, 0.01469988, 0.364496]],
                   'epc': [0.018915683021742555],
                   'epc_err': [0.0002842022079552383]
                  }
                 }
                ]

        for tst_index, tst in enumerate(tests):

            random.seed(tst_index+1)
            np.random.seed(tst_index+1)

            # Generate the sequences
            try:
                rb_circs, xdata, rb_opts = rb.randomized_benchmarking_seq(tst['rb_opts'])
            except OSError:
                print('Skipping test no. ' + str(i) + ' because tables are missing')
                continue

            # Define a noise model
            noise_model = NoiseModel()
            noise_model.add_all_qubit_quantum_error(depolarizing_error(0.002, 1),
                                                    ['u1', 'u2', 'u3'])
            noise_model.add_all_qubit_quantum_error(depolarizing_error(0.002, 2), 'cx')

            # Perform an execution on the generated sequences
            basis_gates = 'u1,u2,u3,cx' # use U, CX for now
            shots = 100
            result = qiskit.execute(rb_circs, backend=backend, seed=tst_index+1,
                                    basis_gates=basis_gates, shots=shots,
                                    noise_model=noise_model,
                                    backend_options={'seed': tst_index+1}).result()

            # Test xdata
            self.assertEqual(xdata.tolist(), tst['expected']['xdata'],
                             'Incorrect xdata in test no. ' + str(tst_index))

            # See TODOs for calc_raw_data in rb_fitters.py
            raw_data = fitters.calc_raw_data(result, rb_circs, rb_opts, shots)
            self.assertEqual(raw_data, tst['expected']['raw_data'],
                             'Incorrect raw data in test no. ' + str(tst_index))

            # See TODOs for calc_statistics in rb_fitters.py
            ydata = fitters.calc_statistics(raw_data)
            self.assertTrue(all(np.isclose(a, b) for a, b in
                                zip(ydata['mean'], tst['expected']['mean'])),
                            'Incorrect mean in test no. ' + str(tst_index))
            if tst['expected']['std'] is None:
                self.assertIsNone(ydata['std'],
                                  'Incorrect std in test no. ' + str(tst_index))
            else:
                self.assertTrue(all(np.isclose(a, b) for a, b in
                                    zip(ydata['std'], tst['expected']['std'])),
                                'Incorrect std in test no. ' + str(tst_index))

            fit = fitters.calc_rb_fit(xdata, ydata, rb_opts['rb_pattern'])

            self.assertTrue(all(np.isclose(a, b) for c, d in
                                zip(fit, tst['expected']['fit_params'])
                                for a, b in zip(c['params'], d)),
                            'Incorrect fit parameters in test no. ' + str(tst_index))
            self.assertTrue(all(np.isclose(a, b) for c, d in
                                zip(fit, tst['expected']['fit_err'])
                                for a, b in zip(c['params_err'], d)),
                            'Incorrect fit error in test no. ' + str(tst_index))
            self.assertTrue(all(np.isclose(a['epc'], b) for a, b in
                                zip(fit, tst['expected']['epc'])),
                            'Incorrect EPC in test no. ' + str(tst_index))
            self.assertTrue(all(np.isclose(a['epc_err'], b) for a, b in
                                zip(fit, tst['expected']['epc_err'])),
                            'Incorrect EPC error in test no. ' + str(tst_index))


if __name__ == '__main__':
    unittest.main()
