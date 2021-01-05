import numpy as np
from scipy.optimize import curve_fit
from typing import Dict, Optional, List, Union
from qiskit.result import Result, Counts
from qiskit.providers import BaseJob
from qiskit.ignis.verification.tomography import marginal_counts
from qiskit.ignis.utils import build_counts_dict_from_list

from qiskit.ignis.experiments.base import Analysis

try:
    from matplotlib import pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class RBAnalysisBase(Analysis):
    def __init__(self,
                 qubits: List[int],
                 lengths: List[int],
                 data: Optional[Union[BaseJob, Result, List[any], any]] = None,
                 metadata: Optional[Union[List[Dict[str, any]], Dict[str, any]]] = None,
                 name: Optional[str] = None,
                 exp_id: Optional[str] = None
                 ):
        self._qubits = qubits
        self._lengths = lengths
        super().__init__(data, metadata, name, exp_id, analysis_fn=self.fit)

    def num_qubits(self):
        return len(self._qubits)

    def _format_data(self, data: Result,
                     metadata: Dict[str, any],
                     index: int) -> Counts:
        counts = data.get_counts(metadata['circuit_name'])
        if 'meas_clbits' in metadata:
            counts = marginal_counts(counts, metadata['meas_clbits'])
        return counts

    def compute_prob(self, counts):
        prob = 0
        if len(counts) > 0:
            n = len(list(counts)[0])
            ground_result = '0' * n
            if ground_result in counts:
                prob = counts[ground_result] / sum(counts.values())
        return prob

    @staticmethod
    def _rb_fit_fun(x, a, alpha, b):
        """Function used to fit RB."""
        # pylint: disable=invalid-name
        return a * alpha ** x + b

    def collect_data(self, data, metadata, key_fn, conversion_fn=None):
        result = {}
        for (d, m) in zip(data, metadata):
            key = key_fn(m)
            if key not in result:
                result[key] = []
            result[key].append(d)

        for key in result:
            result[key] = build_counts_dict_from_list(result[key])
            if conversion_fn is not None:
                result[key] = conversion_fn(result[key])

        return result

    def organize_data(self, data, metadata):
        # changes the flat probability list to a list [seed_0_probs, seed_1_probs...]
        # where seed_i_prob is a list of the probs for seed i for every length
        seeds = sorted(list(set([m['seed'] for m in metadata])))
        length_indices = sorted(list(set([m['length_index'] for m in metadata])))
        prob_dict = self.collect_data(data, metadata,
                                      key_fn=lambda m: (m['seed'], m['length_index']),
                                      conversion_fn=self.compute_prob)
        return np.array([[prob_dict[(seed, length_index)] for length_index in length_indices] for seed in seeds])

    def calc_statistics(self, xdata):
        ydata = {}
        ydata['mean'] = np.mean(xdata, 0)
        ydata['std'] = None
        if xdata.shape[0] != 1:  # more than 1 seed
            ydata['std'] = np.std(xdata, 0)
        return ydata

    def generate_fit_guess(self, ydata):
        fit_guess = [0.95, 0.99, 1 / 2 ** self.num_qubits()]
        # Use the first two points to guess the decay param
        y0 = ydata['mean'][0]
        y1 = ydata['mean'][1]
        dcliff = (self._lengths[1] - self._lengths[0])
        dy = ((y1 - fit_guess[2]) / (y0 - fit_guess[2]))
        alpha_guess = dy ** (1 / dcliff)
        if alpha_guess < 1.0:
            fit_guess[1] = alpha_guess

        if y0 > fit_guess[2]:
            fit_guess[0] = ((y0 - fit_guess[2]) /
                            fit_guess[1] ** self._lengths[0])

        return tuple(fit_guess)

    def fit(self, data, metadata):
        xdata = self.organize_data(data, metadata)
        ydata = self.calc_statistics(xdata)
        fit_guess = self.generate_fit_guess(ydata)


        # if at least one of the std values is zero, then sigma is replaced
        # by None
        if ydata['std'] is None or 0 in ydata['std']:
            sigma = None
        else:
            sigma = ydata['std'].copy()

        params, pcov = curve_fit(self._rb_fit_fun, self._lengths,
                                 ydata['mean'],
                                 sigma=sigma,
                                 p0=fit_guess,
                                 bounds=([0, 0, 0], [1, 1, 1]))
        alpha = params[1]  # exponent
        params_err = np.sqrt(np.diag(pcov))
        alpha_err = params_err[1]

        nrb = 2 ** self.num_qubits()
        epc = (nrb - 1) / nrb * (1 - alpha)
        epc_err = (nrb - 1) / nrb * alpha_err / alpha

        return RBResult(
            params=params,
            params_err=params_err,
            epc=epc,
            epc_err=epc_err,
            qubits=self._qubits,
            xdata=xdata,
            ydata=ydata,
            lengths=self._lengths,
            fit_function=self._rb_fit_fun
        )

class RBResult():
    def __init__(self,
                 params,
                 params_err,
                 epc,
                 epc_err,
                 qubits,
                 xdata,
                 lengths,
                 ydata,
                 fit_function
                 ):
        self._params = params
        self._params_err = params_err
        self._epc = epc
        self._epc_err = epc_err
        self._qubits = qubits
        self._xdata = xdata
        self._lengths = lengths
        self._ydata = ydata
        self._fit_function = fit_function

    def num_qubits(self):
        return len(self._qubits)

    def plot(self, ax=None, add_label=True, show_plt=True):
        """Plot randomized benchmarking data of a single pattern.

        Args:
            ax (Axes): plot axis (if passed in).
            add_label (bool): Add an EPC label.
            show_plt (bool): display the plot.

        Raises:
            ImportError: if matplotlib is not installed.
        """

        if not HAS_MATPLOTLIB:
            raise ImportError('The function plot_rb_data needs matplotlib. '
                              'Run "pip install matplotlib" before.')

        if ax is None:
            plt.figure()
            ax = plt.gca()

        # Plot the result for each sequence
        for one_seed_data in self._xdata:
            ax.plot(self._lengths, one_seed_data, color='gray', linestyle='none',
                    marker='x')

        # Plot the mean with error bars
        ax.errorbar(self._lengths, self._ydata['mean'],
                    yerr=self._ydata['std'],
                    color='r', linestyle='--', linewidth=3)

        # Plot the fit
        ax.plot(self._lengths,
                self._fit_function(self._lengths, *self._params),
                color='blue', linestyle='-', linewidth=2)
        ax.tick_params(labelsize=14)

        ax.set_xlabel('Clifford Length', fontsize=16)
        ax.set_ylabel('Ground State Population', fontsize=16)
        ax.grid(True)

        if add_label:
            bbox_props = dict(boxstyle="round,pad=0.3",
                              fc="white", ec="black", lw=2)

            ax.text(0.6, 0.9,
                    "alpha: %.3f(%.1e) EPC: %.3e(%.1e)" %
                    (self._params[1],
                     self._params_err[1],
                     self._epc,
                     self._epc_err),
                    ha="center", va="center", size=14,
                    bbox=bbox_props, transform=ax.transAxes)

        if show_plt:
            plt.show()