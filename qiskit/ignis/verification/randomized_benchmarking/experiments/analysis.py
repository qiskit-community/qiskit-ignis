import numpy as np
from scipy.optimize import curve_fit
from typing import Dict, Optional, List, Union, Callable
from qiskit.result import Result, Counts
from qiskit.providers import BaseJob
from qiskit.ignis.verification.tomography import marginal_counts
from qiskit.ignis.utils import build_counts_dict_from_list
from qiskit.exceptions import QiskitError
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
                 analysis_fn: Callable,
                 data: Optional[Union[BaseJob, Result, List[any], any]] = None,
                 metadata: Optional[Union[List[Dict[str, any]], Dict[str, any]]] = None,
                 name: Optional[str] = None,
                 exp_id: Optional[str] = None,
                 group_type: Optional[str] = 'clifford'
                 ):
        self._qubits = qubits
        self._lengths = lengths
        self._group_type = group_type
        super().__init__(data, metadata, name, exp_id, analysis_fn=analysis_fn)

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

    def extract_data_and_metadata(self, data, metadata):
        if data is None:
            return (None, None)
        if isinstance(data, BaseJob):
            data = data.result()
        if isinstance(data, Result):
            # Extract metadata from result object if not provided
            if metadata is None:
                if not hasattr(data.header, "metadata"):
                    raise QiskitError("Experiment is missing metadata.")
                metadata = data.header.metadata
        return (data, metadata)

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


class RBAnalysis(RBAnalysisBase):
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
        super().__init__(qubits, lengths, self.fit, data, metadata, name, exp_id)

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

        return RBResult({
            'A': params[0],
            'alpha': params[1],
            'B': params[2],
            'A_err': params_err[0],
            'alpha_err': params_err[1],
            'B_err': params_err[2],
            'epc': epc,
            'epc_err': epc_err,
            'qubits': self._qubits,
            'xdata': xdata,
            'ydata': ydata,
            'lengths': self._lengths,
            'fit_function': self._rb_fit_fun,
            'group_type': self._group_type
        })

class CNOTDihedralRBAnalysis(RBAnalysisBase):
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
        self.z_fitter = RBAnalysis(self._qubits, self._lengths)
        self.x_fitter = RBAnalysis(self._qubits, self._lengths)
        super().__init__(qubits, lengths, self.fit, data, metadata, name, exp_id, group_type='cnot_dihedral')

    def add_data(self,
                 data: Union[BaseJob, Result, any],
                 metadata: Optional[Dict[str, any]] = None):
        data, metadata = self.extract_data_and_metadata(data, metadata)
        if data is not None:
            z_metadata = [m for m in metadata if m['cnot_basis'] == 'Z']
            x_metadata = [m for m in metadata if m['cnot_basis'] == 'X']
            self.z_fitter.add_data(data, z_metadata)
            self.x_fitter.add_data(data, x_metadata)

    def run(self, **params) -> any:
        z_fit_results = self.z_fitter.run()
        x_fit_results = self.x_fitter.run()
        self._result = self._analysis_fn(z_fit_results, x_fit_results)
        return self._result

    def fit(self, z_fit_results, x_fit_results):
        # calculate nrb=d=2^n:
        nrb = 2 ** len(self._qubits)

        # Calculate alpha_Z and alpha_R:
        alpha_Z = z_fit_results['alpha']
        alpha_R = x_fit_results['alpha']
        # Calculate their errors:
        alpha_Z_err = z_fit_results['alpha_err']
        alpha_R_err = x_fit_results['alpha_err']

        # Calculate alpha:
        alpha = (alpha_Z + nrb * alpha_R) / (nrb + 1)

        # Calculate alpha_err:
        alpha_Z_err_sq = (alpha_Z_err / alpha_Z / (nrb + 1)) ** 2
        alpha_R_err_sq = (nrb * alpha_R_err / alpha_R / (nrb + 1)) ** 2
        alpha_err = np.sqrt(alpha_Z_err_sq + alpha_R_err_sq)

        # Calculate epg_est:
        epg_est = (nrb - 1) * (1 - alpha) / nrb

        # Calculate epg_est_error
        epg_est_err = (nrb - 1) / nrb * alpha_err / alpha

        cnotdihedral_result = {'alpha': alpha,
                               'alpha_err': alpha_err,
                               'epg_est': epg_est,
                               'epg_est_err': epg_est_err,
                               'lengths': self._lengths,
                               'group_type': self._group_type}

        return CNOTDihedralRBResult(z_fit_results, x_fit_results, cnotdihedral_result)

class InterleavedRBAnalysis(RBAnalysisBase):
    def __init__(self,
                 qubits: List[int],
                 lengths: List[int],
                 data: Optional[Union[BaseJob, Result, List[any], any]] = None,
                 metadata: Optional[Union[List[Dict[str, any]], Dict[str, any]]] = None,
                 name: Optional[str] = None,
                 exp_id: Optional[str] = None,
                 group_type: Optional[str] = 'clifford'
                 ):
        self._qubits = qubits
        self._lengths = lengths
        if group_type == 'clifford':
            self.std_fitter = RBAnalysis(self._qubits, self._lengths)
            self.int_fitter = RBAnalysis(self._qubits, self._lengths)
        if group_type == 'cnot_dihedral':
            self.std_fitter = CNOTDihedralRBAnalysis(self._qubits, self._lengths)
            self.int_fitter = CNOTDihedralRBAnalysis(self._qubits, self._lengths)
        super().__init__(qubits, lengths, self.fit, data, metadata, name, exp_id, group_type)

    def add_data(self,
                 data: Union[BaseJob, Result, any],
                 metadata: Optional[Dict[str, any]] = None):

        data, metadata = self.extract_data_and_metadata(data, metadata)
        if data is not None:
            std_metadata = [m for m in metadata if m['circuit_type'] == 'standard']
            int_metadata = [m for m in metadata if m['circuit_type'] == 'interleaved']
            self.std_fitter.add_data(data, std_metadata)
            self.int_fitter.add_data(data, int_metadata)

    def run(self, **params) -> any:
        std_fit_results = self.std_fitter.run()
        int_fit_results = self.int_fitter.run()
        self._result = self._analysis_fn(std_fit_results, int_fit_results)
        return self._result

    def fit(self, std_fit_results, int_fit_results):
        # calculate nrb=d=2^n:
        nrb = 2 ** len(self._qubits)

        # Calculate alpha (=p) and alpha_c (=p_c):
        alpha = std_fit_results['alpha']
        alpha_c = int_fit_results['alpha']
        # Calculate their errors:
        alpha_err = std_fit_results['alpha_err']
        alpha_c_err = int_fit_results['alpha_err']

        # Calculate epc_est (=r_c^est) - Eq. (4):
        epc_est = (nrb - 1) * (1 - alpha_c / alpha) / nrb

        # Calculate the systematic error bounds - Eq. (5):
        systematic_err_1 = (nrb - 1) * (abs(alpha - alpha_c / alpha)
                                        + (1 - alpha)) / nrb
        systematic_err_2 = 2 * (nrb * nrb - 1) * (1 - alpha) / \
            (alpha * nrb * nrb) + 4 * (np.sqrt(1 - alpha)) * \
            (np.sqrt(nrb * nrb - 1)) / alpha
        systematic_err = min(systematic_err_1, systematic_err_2)
        systematic_err_L = epc_est - systematic_err
        systematic_err_R = epc_est + systematic_err

        # Calculate epc_est_error
        alpha_err_sq = (alpha_err / alpha) * (alpha_err / alpha)
        alpha_c_err_sq = (alpha_c_err / alpha_c) * (alpha_c_err / alpha_c)
        epc_est_err = ((nrb - 1) / nrb) * (alpha_c / alpha) \
            * (np.sqrt(alpha_err_sq + alpha_c_err_sq))

        interleaved_result = {'alpha': alpha,
                              'alpha_err': alpha_err,
                              'alpha_c': alpha_c,
                              'alpha_c_err': alpha_c_err,
                              'epc_est': epc_est,
                              'epc_est_err': epc_est_err,
                              'systematic_err':
                                  systematic_err,
                              'systematic_err_L':
                                  systematic_err_L,
                              'systematic_err_R':
                                  systematic_err_R,
                              'lengths': self._lengths,
                              'group_type': self._group_type}
        if self._group_type == 'clifford':
            return InterleavedRBResult(std_fit_results, int_fit_results, interleaved_result)
        if self._group_type == 'cnot_dihedral':
            return InterleavedCNOTDihedralRBResult(std_fit_results, int_fit_results, interleaved_result)



class RBResult():
    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __str__(self):
        return str(self._data)

    def group_type_name(self):
        names_dict = {
            'clifford': 'Clifford',
            'cnot_dihedral': 'CNOT-Dihedral'
        }
        return names_dict[self._data['group_type']]

    def num_qubits(self):
        return len(self._data.get('qubits', []))

    def params(self):
        return [self._data['A'], self._data['alpha'], self._data['B']]

    def lengths(self):
        return self._data['lengths']

    def plot_data_series(self, ax, error_bar = False, color='blue', label=None):
        for one_seed_data in self._data['xdata']:
            ax.plot(self.lengths(), one_seed_data, color=color, linestyle='none',
                    marker='x')
        if error_bar:
            ax.errorbar(self.lengths(), self._data['ydata']['mean'],
                        yerr=self._data['ydata']['std'],
                        color='red', linestyle='--', linewidth=3)
        ax.plot(self.lengths(),
                self._data['fit_function'](self._data['lengths'], *self.params()),
                color=color, linestyle='-', linewidth=2,
                label=label)

    def plot_all_data_series(self, ax):
        self.plot_data_series(ax, error_bar=True)

    def plot_label(self):
        return "alpha: %.3f(%.1e) EPC: %.3e(%.1e)" % (self._data['alpha'],
                                                      self._data['alpha_err'],
                                                      self._data['epc'],
                                                      self._data['epc_err'])

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

        self.plot_all_data_series(ax)

        ax.tick_params(labelsize=14)

        ax.set_xlabel('{} Length'.format(self.group_type_name()), fontsize=16)
        ax.set_ylabel('Ground State Population', fontsize=16)
        ax.grid(True)

        if add_label:
            bbox_props = dict(boxstyle="round,pad=0.3",
                              fc="white", ec="black", lw=2)

            ax.text(0.6, 0.9, self.plot_label(),
                    ha="center", va="center", size=14,
                    bbox=bbox_props, transform=ax.transAxes)

        if show_plt:
            plt.show()


class InterleavedRBResult(RBResult):
    def __init__(self, std_fit_result, int_fit_result, interleaved_result):
        self._std_fit_result = std_fit_result
        self._int_fit_result = int_fit_result
        self._data = interleaved_result

    def num_qubits(self):
        return self._std_fit_result.num_qubits()

    def params(self):
        raise QiskitError("params() is not fully determined in results of interleaved RB")

    def plot_all_data_series(self, ax):
        self._std_fit_result.plot_data_series(ax, color='blue', label='Standard RB')
        self._int_fit_result.plot_data_series(ax, color='red', label='Interleaved RB')
        ax.legend(loc='lower left')

    def plot_label(self):
        return "alpha: %.3f(%.1e) alpha_c: %.3e(%.1e) \n \
                            EPC_est: %.3e(%.1e)" % (self['alpha'],
                                                    self['alpha_err'],
                                                    self['alpha_c'],
                                                    self['alpha_c_err'],
                                                    self['epc_est'],
                                                    self['epc_est_err'])

class CNOTDihedralRBResult(RBResult):
    def __init__(self, z_fit_result, x_fit_result, cnotdihedral_result):
        self._z_fit_result = z_fit_result
        self._x_fit_result = x_fit_result
        self._data = cnotdihedral_result

    def num_qubits(self):
        return self._z_fit_result.num_qubits()

    def params(self):
        raise QiskitError("params() is not fully determined in results of cnot dihedral RB")

    def plot_all_data_series(self, ax):
        self._z_fit_result.plot_data_series(ax, color='blue', label='Measure state |0...0>')
        self._x_fit_result.plot_data_series(ax, color='red', label='Measure state |+...+>')
        ax.legend(loc='lower left')

    def plot_label(self):
        "alpha: %.3f(%.1e) EPG_est: %.3e(%.1e)" % (self['alpha'],
                                                   self['alpha_err'],
                                                   self['epg_est'],
                                                   self['epg_est_err'])

class InterleavedCNOTDihedralRBResult(RBResult):
    def __init__(self, cnot_std_fit_result, cnot_int_fit_result ,interleaved_result):
        self._cnot_std_fit_result = cnot_std_fit_result
        self._cnot_int_fit_result = cnot_int_fit_result
        self._data = interleaved_result

    def num_qubits(self):
        return self._cnot_std_fit_result.num_qubits()

    def params(self):
        raise QiskitError("params() is not fully determined in results of interleaved RB")

    def plot_all_data_series(self, ax):
        self._cnot_std_fit_result._z_fit_result.plot_data_series(ax, color='cyan', label='Standard RB in |0...0>')
        self._cnot_int_fit_result._z_fit_result.plot_data_series(ax, color='blue', label='Standard RB in |+...+>')
        self._cnot_std_fit_result._x_fit_result.plot_data_series(ax, color='yellow', label='Interleaved RB in |0...0>')
        self._cnot_int_fit_result._x_fit_result.plot_data_series(ax, color='red', label='Interleaved RB in |+...+>')
        ax.legend(loc='lower left')

    def plot_label(self):
        return "alpha: %.3f(%.1e) alpha_c: %.3e(%.1e) \n \
                                    EPC_est: %.3e(%.1e)" % (self['alpha'],
                                                            self['alpha_err'],
                                                            self['alpha_c'],
                                                            self['alpha_c_err'],
                                                            self['epc_est'],
                                                            self['epc_est_err'])
