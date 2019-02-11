# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Base class for fitters of characteristic times
"""

from scipy.optimize import curve_fit
import numpy as np

class BaseCoherenceFitter:
    """
    Base class for fitters of characteristic times
    """

    def __init__(self, description,
                 backend_result, shots,
                 num_of_gates, gate_time,
                 num_of_qubits, measured_qubit,
                 fit_fun, fit_p0, fit_bounds):
        """
        Args:
           description: a string describing the fitter's purpose, e.g. 'T1'
           backend_result: result of backend execution (qiskit.Result).
           num_of_gates is a vector of integers in an increasing order.
           len(num_of_gates) circuits will be generated.
           In the first circuit the initial X gate will be followed by
           num_of_gates[0] identity gates.
           In the second circuit it will be followed by num_of_gates[1] identity gates.
           And so on.
           Each gate lasts gate_time micro-seconds.
           The circuits have num_of_qubits qubits.
           The index of the qubit whose time is measured is measured_qubit.
           fit_fun, fit_p0, fir_bounds: equivalent to parameters of scipy.curve_fit.
        """

        self.description = description

        self.num_of_qubits = num_of_qubits
        self.qubit = measured_qubit

        self.xdata = gate_time * num_of_gates
        self.ydata = self.calc_data(backend_result, shots, len(num_of_gates))

        self.fit_fun = fit_fun
        self.params, self.params_err = self.calc_fit(fit_p0, fit_bounds)


    def calc_data(self, backend_result, shots, num_of_circuits):
        """
        Rerieve probabilities of success from execution results, i.e.,
        probability to measure a state where all qubits are 0,
        except for self.qubit, which is 1.
        Return a dictionary ydata:
        - ydata['mean'] is a list, where item no. j is the probability of success
                        for a circuit of length self.num_of_gates[j].
        - ydata['std'] is a list, where ydata['std'][j] is the
                       standard deviation of the success.
        """

        expected_state_list = ['0'] * self.num_of_qubits
        expected_state_list[self.num_of_qubits - self.qubit - 1] = '1'
        expected_state_str = ''.join(expected_state_list)

        ydata = {'mean': [], 'std': []}
        for circ in range(num_of_circuits):
            counts = backend_result.get_counts(circ)
            if expected_state_str in counts:
                success_prob = counts[expected_state_str] / shots
            else:
                success_prob = 0
            ydata['mean'].append(success_prob)
            ydata['std'].append(np.sqrt(success_prob * (1-success_prob) / shots))

        return ydata


    def calc_fit(self, p0, bounds):
        """
        Fit the curve.
        Returns:
        - fit - same as the first returned value of curve_fit.
        - ferr - error for each parameter.
        """

        fit, fcov = curve_fit(self.fit_fun, self.xdata,
                              self.ydata['mean'], sigma=self.ydata['std'],
                              p0=p0, bounds=bounds)
        ferr = np.sqrt(np.diag(fcov))
        return fit, ferr


    def plot_coherence(self):
        """
        Plot coherence data.
        """

        from matplotlib import pyplot as plt

        plt.errorbar(self.xdata, self.ydata['mean'], self.ydata['std'],
                     marker='.', markersize=9, c='b', linestyle='')
        plt.plot(self.xdata, self.fit_fun(self.xdata, *self.params),
                 c='r', linestyle='--',
                 label=self.description+': '+str(round(self.time))+' micro-seconds')

        plt.xticks(fontsize=14, rotation=70)
        plt.yticks(fontsize=14)
        plt.xlabel('time [micro-seconds]', fontsize=16)
        plt.ylabel('Probability of success', fontsize=16)
        plt.title(self.description + ' for qubit ' + str(self.qubit), fontsize=18)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.show()


    @staticmethod
    def exp_fit_fun(x, a, tau, c):
        """
        Function used to fit the exponential decay
        """

        return a * np.exp(-x / tau) + c
