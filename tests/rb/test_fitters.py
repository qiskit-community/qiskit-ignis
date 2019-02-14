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

        tests = [
            {'rb_opts':
                 {'nseeds': 3,
                  'length_vector': [1, 3, 4, 7, 10],
                  'n_qubits': 2,
                  'rb_pattern': [[0], [1]],
                  'length_multiplier': 1
                  },
             'expected':
                 {'xdata': [[1, 3, 4, 7, 10], [1, 3, 4, 7, 10]],
                  'raw_data': [[1.0, 0.98, 0.97, 0.97, 0.94],
                               [1.0, 0.99, 0.97, 0.95, 0.94],
                               [0.99, 0.98, 0.98, 0.94, 0.93]],
                  'mean': [0.99666667, 0.98333333, 0.97333333, 0.95333333, 0.93666667],
                  'std': [0.00471405, 0.00471405, 0.00471405, 0.01247219, 0.00471405],
                  'fit_params': [[0.21033493, 0.96121183, 0.79492823],
                                 [0.21033493, 0.96121183, 0.79492823]],
                  'fit_err': [[0.1071722 , 0.02469688, 0.1090674 ],
                              [0.1071722, 0.02469688, 0.1090674]],
                  'epc': [0.019394082609075713, 0.019394082609075713],
                  'epc_err': [0.000498301633611352, 0.000498301633611352]
                  }
             },
            {'rb_opts':
                {'nseeds': 3,
                'length_vector': [1, 10, 20, 30, 40, 50],
                'n_qubits': 1,
                'rb_pattern': [[0]],
                'length_multiplier': 1
                },
            'expected':
                {'xdata': [[1, 10, 20, 30, 40, 50]],
                'raw_data': [[0.98, 0.97, 0.93, 0.85, 0.84, 0.85],
                             [0.97, 0.95, 0.96, 0.89, 0.84, 0.8],
                             [0.97, 0.95, 0.93, 0.87, 0.89, 0.88]],
                'mean': [0.97333333, 0.95666667, 0.94, 0.87, 0.85666667, 0.84333333],
                'std': [0.00471405, 0.00942809, 0.01414214, 0.01632993, 0.02357023, 0.03299832],
                'fit_params': [[ 2.,  0.99853465, -1.02171782]],
                'fit_err': [[3.10537620e+01, 2.33472957e-02, 3.10566719e+01]],
                'epc': [0.0007326746000217987],
                'epc_err': [1.7131073541769797e-05]
                }
             }
        ]

        for tst_index, tst in enumerate(tests):

            random.seed(tst_index+1)
            np.random.seed(tst_index+1)

            # Generate the sequences
            rb_opts = tst['rb_opts'].copy()
            rb_circs, xdata = rb.randomized_benchmarking_seq(**rb_opts)

            # Define a noise model
            noise_model = NoiseModel()
            noise_model.add_all_qubit_quantum_error(depolarizing_error(0.002, 1),
                                                    ['u1', 'u2', 'u3'])
            noise_model.add_all_qubit_quantum_error(depolarizing_error(0.002, 2), 'cx')

            # Perform an execution on the generated sequences
            basis_gates = 'u1,u2,u3,cx' # use U, CX for now #Shelly: changed format to fit qiskit current version
            shots = 100
            result_list = []
            for i in range(rb_opts['nseeds']):
                result_list.append(qiskit.execute(rb_circs[i], backend=backend, seed=tst_index+1,
                                        basis_gates=basis_gates, shots=shots,
                                        noise_model=noise_model,
                                        backend_options={'seed': tst_index+1}).result())

            # Test xdata
            self.assertEqual(xdata.tolist(), tst['expected']['xdata'],
                             'Incorrect xdata in test no. ' + str(tst_index))

            # See TODOs for calc_raw_data in rb_fitters.py
            raw_data = fitters.calc_raw_data(result_list, rb_circs, rb_opts, shots)
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
