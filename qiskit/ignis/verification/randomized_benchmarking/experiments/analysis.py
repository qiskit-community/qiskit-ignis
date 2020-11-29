import numpy as np
from scipy.optimize import curve_fit
from typing import Dict, Optional, List, Union
from qiskit.result import Result, Counts
from qiskit.providers import BaseJob
from qiskit.ignis.verification.tomography import marginal_counts

from qiskit.ignis.experiments.base import Analysis
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

    def organize_data(self, data, metadata):
        # changes the flat probability list to a list [seed_0_probs, seed_1_probs...]
        # where seed_i_prob is a list of the probs for seed i for every length

        seeds = sorted(list(set([m['seed'] for m in metadata])))
        length_indices = sorted(list(set([m['length_index'] for m in metadata])))
        prob_dict = {(m['seed'], m['length_index']): prob for (m,prob) in zip(metadata, data)}
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
        print(metadata)
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

        return {'params': params, 'params_err': params_err,
                               'epc': epc, 'epc_err': epc_err}

