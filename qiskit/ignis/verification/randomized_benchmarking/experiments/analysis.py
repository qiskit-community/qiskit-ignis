# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Randomized benchmarking analysis classes
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Union, Callable, Tuple
import numpy as np
from scipy.optimize import curve_fit
from qiskit.result import Result, Counts
from qiskit.providers import BaseJob
from qiskit.ignis.verification.tomography import marginal_counts
from qiskit.ignis.utils import build_counts_dict_from_list
from qiskit.exceptions import QiskitError
from qiskit.ignis.experiments.base import Analysis
from qiskit.quantum_info.analysis.average import average_data

try:
    from matplotlib import pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class RBAnalysisBase(Analysis):
    """Base analysis class for randomized benchmarking experiments"""
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
        """Initialize the result analyzer for a randomized banchmarking experiment.
            Args:
                qubits: the qubits participating in the experiment
                lengths: for each seed, the lengths of the circuits used for that seed.
                analysis_fn: the fitting function used within `run()`
                data: The result from running a backend of the RB circuits
                metadata: the metadata corresponding to the RB circuits
                name: the name of the analyzier
                exp_id: the id number for the experiment
                group_type: the group used for the RB gates ('clifford' or 'cnot_dihedral')
        """
        self._qubits = qubits
        self._lengths = lengths
        self._group_type = group_type
        super().__init__(data, metadata, name, exp_id, analysis_fn=analysis_fn)

    def num_qubits(self) -> int:
        """Returns the number of measured qubits in the RB experiment
            Returns: the number of qubits"""
        return len(self._qubits)

    def _format_data(self, data: Result,
                     metadata: Dict[str, any],
                     index: int) -> Counts:
        """Format the required data by getting the counts and marginalizing if needed
            Args:
                data: The result from running a backend of the RB circuits
                metadata: The metadata corresponding to the RB circuits
                index: The index of the metadata
            Returns:
                The formatted counts dictionary
        """
        counts = data.get_counts(metadata['circuit_name'])
        if 'meas_clbits' in metadata:
            counts = marginal_counts(counts, metadata['meas_clbits'])
        return counts

    def compute_prob(self, counts: Counts) -> float:
        """Computes the probability of getting the ground result
        Args:
            counts: The count dictionary
        Returns:
            The probability of the ground ("0") result from the counts dictionary
        """
        prob = 0
        if len(counts) > 0:
            n = len(list(counts)[0])
            ground_result = '0' * n
            if ground_result in counts:
                prob = counts[ground_result] / sum(counts.values())
        return prob

    @staticmethod
    def _rb_fit_fun(x: float, a: float, alpha: float, b: float) -> float:
        """The function used to fit RB: :math:`A\alpha^x + B`
            Args:
                x: The input to the function's variable
                a: The A parameter of the function
                alpha: The :math:`\alpha` parameter of the function
                b: The B parameter of the function
            Returns:
                The functions value on the specified parameters and input
        """
        # pylint: disable=invalid-name
        return a * alpha ** x + b

    def extract_data_and_metadata(self,
                                  data: Union[BaseJob, Result, List[any], any],
                                  metadata: Optional[List[Dict[str, any]]] = None
                                  ) -> Tuple[Result, List[Dict[str, any]]]:
        """Extracting the data object and its metadata, if present
            Args:
                data: The base job or result object containing the data
                metadata: the metadata, if already present
            Returns:
                A tuple of the extracted data and metadata
            Raises:
                QiskitError: If metadata was not already present and is not contained in `data`
        """
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

    def collect_data(self,
                     data: List[Counts],
                     metadata: List[Dict[str, any]],
                     key_fn: Callable[[Dict[str, any]], any],
                     conversion_fn: Optional[Callable[[Counts], any]] = None
                     ) -> Dict:
        """
        Args:
            data: List of formatted data (counts)
            metadata: The corresponding metadata
            key_fn: Function acting on the metadata to produce a key used to identify counts
            that should be counted together.
            conversion_fn: A function to be applied to the counts after the combined count
            object is obtained (e.g. computing ground-state probability)
        Returns:
            The list of collected data elements, after merge and conversion.
        """
        result = {}
        for (data_elem, meta_elem) in zip(data, metadata):
            key = key_fn(meta_elem)
            if key not in result:
                result[key] = []
            result[key].append(data_elem)

        for key in result:
            result[key] = build_counts_dict_from_list(result[key])
            if conversion_fn is not None:
                result[key] = conversion_fn(result[key])

        return result

    def organize_data(self,
                      data: List[Counts],
                      metadata: List[Dict[str, any]],
                      ) -> np.array:
        """Converts the data to a list of probabilities for each seed
            Args:
                data: The counts data
                metadata: The corresponding metadata
            Returns:
                a list [seed_0_probs, seed_1_probs...] where seed_i_prob is
                a list of the probabilities for seed i for every length
        """
        seeds = sorted(list({m['seed'] for m in metadata}))
        length_indices = sorted(list({m['length_index'] for m in metadata}))
        prob_dict = self.collect_data(data, metadata,
                                      key_fn=lambda m: (m['seed'], m['length_index']),
                                      conversion_fn=self.compute_prob)
        return np.array([[prob_dict[(seed, length_index)]
                          for length_index in length_indices]
                         for seed in seeds])

    def calc_statistics(self, xdata: np.array) -> Dict[str, List[float]]:
        """Computes the mean and standard deviation of the probability data
        Args:
            xdata: List of lists of probabilities (for each seed and length)
        Returns:
            A dictionary {'mean': m, 'std': s} for the mean and standard deviation
            Standard deviation is computed only for more than 1 seed
        """
        ydata = {}
        ydata['mean'] = np.mean(xdata, 0)
        ydata['std'] = None
        if xdata.shape[0] != 1:  # more than 1 seed
            ydata['std'] = np.std(xdata, 0)
        return ydata

    def generate_fit_guess(self, mean: np.array) -> Tuple[float]:
        """Generate initial guess for the fitter from the mean data
            Args:
                mean: A list of mean probabilities for each length
            Returns:
                The initial guess for the fit parameters (A, alpha, B)
        """
        # pylint: disable=invalid-name
        fit_guess = [0.95, 0.99, 1 / 2 ** self.num_qubits()]
        # Use the first two points to guess the decay param
        y0 = mean[0]
        y1 = mean[1]
        dcliff = (self._lengths[1] - self._lengths[0])
        dy = ((y1 - fit_guess[2]) / (y0 - fit_guess[2]))
        alpha_guess = dy ** (1 / dcliff)
        if alpha_guess < 1.0:
            fit_guess[1] = alpha_guess

        if y0 > fit_guess[2]:
            fit_guess[0] = ((y0 - fit_guess[2]) /
                            fit_guess[1] ** self._lengths[0])

        return tuple(fit_guess)

    def run_curve_fit(self,
                      ydata: Dict[str, np.array],
                      fit_guess: Tuple[float]
                      ) -> Tuple[Tuple[float], Tuple[float]]:
        """Runs the curve fir algorithm from the initial guess and based on the statistical data
            Args:
                ydata: The statistical data
                fit_guess: The initial guess
            Returns:
                The resulting fit data
        """
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
        params_err = np.sqrt(np.diag(pcov))
        return (params, params_err)


class RBResultBase(ABC):
    """Base class for randomized benchmarking analysis results"""
    def __init__(self, data: Dict):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __str__(self):
        return str(self._data)

    def group_type_name(self) -> str:
        """Returns "Clifford" or "CNOT_Dihedral" based on the underlying group"""
        names_dict = {
            'clifford': 'Clifford',
            'cnot_dihedral': 'CNOT-Dihedral'
        }
        return names_dict[self._data['group_type']]

    def num_qubits(self) -> int:
        """Returns the number of qubits used in the RB experiment"""
        return len(self._data.get('qubits', []))

    def lengths(self) -> List[int]:
        """Returns the list of RB circuits lengths"""
        return self._data['lengths']

    @abstractmethod
    def plot_all_data_series(self, ax):
        """Plots all data series of the RB; meant to be overridden by derived classes"""

    @abstractmethod
    def plot_label(self) -> str:
        """The label to be added as a top-right box in the plot"""

    def plot_y_axis_label(self) -> str:
        """Returns the string to be used as the plot's y label"""
        return "Ground State Population"

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
        ax.set_ylabel(self.plot_y_axis_label(), fontsize=16)
        ax.grid(True)

        if add_label:
            bbox_props = dict(boxstyle="round,pad=0.3",
                              fc="white", ec="black", lw=2)

            ax.text(0.6, 0.9, self.plot_label(),
                    ha="center", va="center", size=14,
                    bbox=bbox_props, transform=ax.transAxes)

        if show_plt:
            plt.show()


class RBResult(RBResultBase):
    """Class for standard RB analysis results"""
    def params(self) -> Tuple[float]:
        """Returns the parameters (A, alpha, B) of the fitting function"""
        return (self._data['A'], self._data['alpha'], self._data['B'])

    def plot_label(self) -> str:
        """Add the string to be used as the plot's header label"""
        return "alpha: %.3f(%.1e) EPC: %.3e(%.1e)" % (self._data['alpha'],
                                                      self._data['alpha_err'],
                                                      self._data['epc'],
                                                      self._data['epc_err'])

    def plot_data_series(self, ax, error_bar=False, color='blue', label=None):
        """Plots the RB data of a single series to a matplotlib axis"""
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
        """Plots the data series of the RB"""
        self.plot_data_series(ax, error_bar=True)


class InterleavedRBResult(RBResultBase):
    """Class for interleaved randomized benchmarking analysis results"""
    def __init__(self, std_fit_result, int_fit_result, interleaved_result):
        self._std_fit_result = std_fit_result
        self._int_fit_result = int_fit_result
        super().__init__(interleaved_result)

    def num_qubits(self) -> int:
        """Returns the number of qubits used in the RB experiment"""
        return self._std_fit_result.num_qubits()

    def plot_all_data_series(self, ax):
        """Plots the standard and interleaved data series"""
        self._std_fit_result.plot_data_series(ax, color='blue', label='Standard RB')
        self._int_fit_result.plot_data_series(ax, color='red', label='Interleaved RB')
        ax.legend(loc='lower left')

    def plot_label(self):
        """Plots interleaved fit results"""
        return "alpha: %.3f(%.1e) alpha_c: %.3e(%.1e) \n \
                            EPC_est: %.3e(%.1e)" % (self['alpha'],
                                                    self['alpha_err'],
                                                    self['alpha_c'],
                                                    self['alpha_c_err'],
                                                    self['epc_est'],
                                                    self['epc_est_err'])


class CNOTDihedralRBResult(RBResultBase):
    """Class for cnot-dihedral RB analysis results"""
    def __init__(self, z_fit_result, x_fit_result, cnotdihedral_result):
        self._z_fit_result = z_fit_result
        self._x_fit_result = x_fit_result
        super().__init__(cnotdihedral_result)

    def num_qubits(self) -> int:
        """Returns the number of qubits used in the RB experiment"""
        return self._z_fit_result.num_qubits()

    def plot_all_data_series(self, ax):
        """Plots the Z and X basis data series"""
        self._z_fit_result.plot_data_series(ax, color='blue', label='Measure state |0...0>')
        self._x_fit_result.plot_data_series(ax, color='red', label='Measure state |+...+>')
        ax.legend(loc='lower left')

    def plot_label(self):
        """Plots cnot-dihedral fit results"""
        return "alpha: %.3f(%.1e) EPG_est: %.3e(%.1e)" % (self['alpha'],
                                                          self['alpha_err'],
                                                          self['epg_est'],
                                                          self['epg_est_err'])


class InterleavedCNOTDihedralRBResult(RBResultBase):
    """Class for interleaved cnot-dihedral RB analysis results"""
    def __init__(self, cnot_std_fit_result, cnot_int_fit_result, interleaved_result):
        self._cnot_std_fit_result = cnot_std_fit_result
        self._cnot_int_fit_result = cnot_int_fit_result
        super().__init__(interleaved_result)

    def num_qubits(self) -> int:
        """Returns the number of qubits used in the RB experiment"""
        return self._cnot_std_fit_result.num_qubits()

    def plot_all_data_series(self, ax):
        """Plots the Z and X basis data series for both standard and interleaved"""
        std_fit = self._cnot_std_fit_result
        int_fit = self._cnot_int_fit_result

        std_fit._z_fit_result.plot_data_series(ax, color='cyan', label='Standard RB in |0...0>')
        int_fit._z_fit_result.plot_data_series(ax, color='blue', label='Interleaved RB in |0...0>')
        std_fit._x_fit_result.plot_data_series(ax, color='yellow', label='Standard RB in |+...+>')
        int_fit._x_fit_result.plot_data_series(ax, color='red', label='Interleaved RB in |+...+>')
        ax.legend(loc='lower left')

    def plot_label(self):
        """Plots interleaved cnot-dihedral fit results"""
        return "alpha: %.3f(%.1e) alpha_c: %.3e(%.1e) \n \
                                    EPC_est: %.3e(%.1e)" % (self['alpha'],
                                                            self['alpha_err'],
                                                            self['alpha_c'],
                                                            self['alpha_c_err'],
                                                            self['epc_est'],
                                                            self['epc_est_err'])


class PurityRBResult(RBResult):
    """Class for purity RB analysis results"""
    def plot_all_data_series(self, ax):
        """Plots the purity RB data series"""
        self.plot_data_series(ax, error_bar=True)

    def plot_label(self):
        """Plots the purity RB fit results"""
        return "alpha: %.3f(%.1e) PEPC: %.3e(%.1e)" % (self['alpha_pur'],
                                                       self['alpha_pur_err'],
                                                       self['pepc'],
                                                       self['pepc_err'])

    def plot_y_axis_label(self) -> str:
        """Plots the y label for purity rB results"""
        return "Trace of Rho Square"


class RBAnalysis(RBAnalysisBase):
    """Analysis class for standard RB experiments"""
    def __init__(self,
                 qubits: List[int],
                 lengths: List[int],
                 data: Optional[Union[BaseJob, Result, List[any], any]] = None,
                 metadata: Optional[Union[List[Dict[str, any]], Dict[str, any]]] = None,
                 name: Optional[str] = None,
                 exp_id: Optional[str] = None
                 ):
        """Initialize the result analyzer for a standard RB experiment.
            Args:
                qubits: the qubits participating in the experiment
                lengths: for each seed, the lengths of the circuits used for that seed.
                data: The result from running a backend of the RB circuits
                metadata: the metadata corresponding to the RB circuits
                name: the name of the analyzier
                exp_id: the id number for the experiment

            Additional information:
                We pass to `RBAnalysisBase` the standard RB fitting function implicitly
                pass the clifford group as `group_type`
        """
        self._qubits = qubits
        self._lengths = lengths
        super().__init__(qubits, lengths, self.fit, data, metadata, name, exp_id)

    def fit(self,
            data: List[Counts],
            metadata: List[Dict[str, any]]) -> RBResult:
        """Computes the RB fit for the given data"""
        xdata = self.organize_data(data, metadata)
        ydata = self.calc_statistics(xdata)
        fit_guess = self.generate_fit_guess(ydata['mean'])
        params, params_err = self.run_curve_fit(ydata, fit_guess)

        alpha = params[1]  # exponent
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
    """Analysis class for cnot-dihedral RB experiments"""
    def __init__(self,
                 qubits: List[int],
                 lengths: List[int],
                 data: Optional[Union[BaseJob, Result, List[any], any]] = None,
                 metadata: Optional[Union[List[Dict[str, any]], Dict[str, any]]] = None,
                 name: Optional[str] = None,
                 exp_id: Optional[str] = None
                 ):
        """Initialize the result analyzer for a cnot-dihedral RB experiment.
            Args:
                qubits: the qubits particiapting in the experiment
                lengths: for each seed, the lengths of the circuits used for that seed.
                data: The result from running a backend of the RB circuits
                metadata: the metadata corresponding to the RB circuits
                name: the name of the analyzier
                exp_id: the id number for the experiment

            Additional information:
                We pass to `RBAnalysisBase` the cnot-dihedral RB fitting function and explicitly
                pass the cnot-dihedral group as `group_type`.
                We use two standard fitters to fit the results for the Z and X basis.
        """
        self._qubits = qubits
        self._lengths = lengths
        self.z_fitter = RBAnalysis(self._qubits, self._lengths)
        self.x_fitter = RBAnalysis(self._qubits, self._lengths)
        super().__init__(qubits, lengths, self.fit, data, metadata,
                         name, exp_id, group_type='cnot_dihedral')

    def add_data(self,
                 data: Union[BaseJob, Result, any],
                 metadata: Optional[Dict[str, any]] = None):
        """Adds data into the fitters, directed by the `cnot_basis` metadata
            Args:
                data: The count data to add
                metadata: the corresponding metadata
        """
        data, metadata = self.extract_data_and_metadata(data, metadata)
        if data is not None:
            z_metadata = [m for m in metadata if m['cnot_basis'] == 'Z']
            x_metadata = [m for m in metadata if m['cnot_basis'] == 'X']
            self.z_fitter.add_data(data, z_metadata)
            self.x_fitter.add_data(data, x_metadata)

    def run(self, **params) -> CNOTDihedralRBResult:
        """Runs the analysis on the currently fed data
            Returns:
                The fitting result object
        """
        z_fit_results = self.z_fitter.run()
        x_fit_results = self.x_fitter.run()
        self._result = self._analysis_fn(z_fit_results, x_fit_results)
        return self._result

    def fit(self, z_fit_results: RBResult, x_fit_results: RBResult) -> CNOTDihedralRBResult:
        """Computes the cnot-dihedral fit from the results of the Z, X basis fits
            Args:
                z_fit_results: The results for the Z-basis fit
                x_fit_results: The results for the X-basis fit
            Returns:
                The cnot-dihedral result (which contains the Z, X results)
        """
        # pylint: disable=invalid-name
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
    """Analysis class for interleaved RB experiments"""
    def __init__(self,
                 qubits: List[int],
                 lengths: List[int],
                 data: Optional[Union[BaseJob, Result, List[any], any]] = None,
                 metadata: Optional[Union[List[Dict[str, any]], Dict[str, any]]] = None,
                 name: Optional[str] = None,
                 exp_id: Optional[str] = None,
                 group_type: Optional[str] = 'clifford'
                 ):
        """Initialize the result analyzer for a interleaved RB experiment.
            Args:
                qubits: the qubits particiapting in the experiment
                lengths: for each seed, the lengths of the circuits used for that seed.
                data: The result from running a backend of the RB circuits
                metadata: the metadata corresponding to the RB circuits
                name: the name of the analyzier
                exp_id: the id number for the experiment
                group_type: the group used for the RB gates ('clifford' or 'cnot_dihedral')

            Additional information:
                We use two standard fitters to fit the results for the standard and interleaved RB.
        """
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
        """Adds data into the fitters, directed by the `circuit_type` metadata
           Args:
               data: The count data to add
               metadata: the corresponding metadata
       """
        data, metadata = self.extract_data_and_metadata(data, metadata)
        if data is not None:
            std_metadata = [m for m in metadata if m['circuit_type'] == 'standard']
            int_metadata = [m for m in metadata if m['circuit_type'] == 'interleaved']
            self.std_fitter.add_data(data, std_metadata)
            self.int_fitter.add_data(data, int_metadata)

    def run(self, **params) -> RBResultBase:
        """Runs the analysis on the currently fed data
            Returns:
                The fitting result object
        """
        std_fit_results = self.std_fitter.run()
        int_fit_results = self.int_fitter.run()
        self._result = self._analysis_fn(std_fit_results, int_fit_results)
        return self._result

    def fit(self, std_fit_results: RBResult, int_fit_results: RBResult) -> RBResultBase:
        """Computes the interleaved fit from the results of the two input fits
            Args:
                std_fit_results: The results for the standard RB fit
                int_fit_results: The results for the interleaved RB fit
            Returns:
                The interleaved result (which contains the input results)
        """
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
        systematic_err_l = epc_est - systematic_err
        systematic_err_r = epc_est + systematic_err

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
                                  systematic_err_l,
                              'systematic_err_R':
                                  systematic_err_r,
                              'lengths': self._lengths,
                              'group_type': self._group_type}
        if self._group_type == 'clifford':
            return InterleavedRBResult(std_fit_results, int_fit_results, interleaved_result)
        if self._group_type == 'cnot_dihedral':
            return InterleavedCNOTDihedralRBResult(std_fit_results,
                                                   int_fit_results,
                                                   interleaved_result)
        return None


class PurityRBAnalysis(RBAnalysisBase):
    """Analysis class for purity RB experiments"""
    def __init__(self,
                 qubits: List[int],
                 lengths: List[int],
                 data: Optional[Union[BaseJob, Result, List[any], any]] = None,
                 metadata: Optional[Union[List[Dict[str, any]], Dict[str, any]]] = None,
                 name: Optional[str] = None,
                 exp_id: Optional[str] = None
                 ):
        """Initialize the result analyzer for a purity RB experiment.
            Args:
                qubits: the qubits participating in the experiment
                lengths: for each seed, the lengths of the circuits used for that seed.
                data: The result from running a backend of the RB circuits
                metadata: the metadata corresponding to the RB circuits
                name: the name of the analyzier
                exp_id: the id number for the experiment

            Additional information:
                We use two standard fitters to fit the results for the standard and interleaved RB.
        """
        self._qubits = qubits
        self._lengths = lengths
        self.add_zdict_ops()
        super().__init__(qubits, lengths, self.fit, data, metadata, name, exp_id)

    def add_zdict_ops(self):
        """Creating all Z-correlators
        in order to compute the expectation values."""
        self._zdict_ops = []
        statedict = {("{0:0%db}" % self.num_qubits()).format(i): 1 for i in
                     range(2 ** self.num_qubits())}

        for i in range(2 ** self.num_qubits()):
            self._zdict_ops.append(statedict.copy())
            for j in range(2 ** self.num_qubits()):
                if bin(i & j).count('1') % 2 != 0:
                    self._zdict_ops[-1][("{0:0%db}"
                                         % self.num_qubits()).format(j)] = -1

    @staticmethod
    def F234(n: int, a: int, b: int) -> int:
        """Function that maps:
        2^n x 3^n --> 4^n ,
        namely:
        (a,b) --> c where
        a in 2^n, b in 3^n, c in 4^n
        """
        # pylint: disable=invalid-name
        # 0 <--> I
        # 1 <--> X
        # 2 <--> Y
        # 3 <--> Z
        LUT = [[0, 0, 0], [3, 1, 2]]

        # compute bits
        aseq = []
        bseq = []

        aa = a
        bb = b
        for i in range(n):
            aseq.append(np.mod(aa, 2))
            bseq.append(np.mod(bb, 3))
            aa = np.floor_divide(aa, 2)
            bb = np.floor_divide(bb, 3)

        c = 0
        for i in range(n):
            c += (4 ** i) * LUT[aseq[i]][bseq[i]]

        return c

    def purity_op_key(self, op: str) -> int:
        """Key function to help sort the op array
        The order is: ZZ < XZ < YZ < ZX < XX < YX < ZY < XY < YY etc.
        """
        op_vals = {'Z': 0, 'X': 1, 'Y': 2}
        return sum([op_vals[o] * (3**k) for (k, o) in enumerate(op)])

    def organize_data(self,
                      data: List[Counts],
                      metadata: List[Dict[str, any]],
                      ) -> np.array:
        """Converts the data to a list of probabilities for each seed
            Args:
                data: The counts data
                metadata: The corresponding metadata
            Returns:
                a list [seed_0_probs, seed_1_probs...] where seed_i_prob is
                a list of the probabilities for seed i for every length
        """
        seeds = sorted(list({m['seed'] for m in metadata}))
        length_indices = sorted(list({m['length_index'] for m in metadata}))
        purity_ops = sorted(list({m['purity_meas_ops'] for m in metadata}), key=self.purity_op_key)
        shots_dict = self.collect_data(data, metadata,
                                       key_fn=lambda m: (m['seed'],
                                                         m['length_index'],
                                                         m['purity_meas_ops']))
        return np.array([[self.compute_purity_data(shots_dict, seed, length_index, purity_ops)
                          for length_index in length_indices]
                         for seed in seeds])

    def compute_purity_data(self,
                            shots_dict: Dict[Tuple, Counts],
                            seed: int,
                            length_index: int,
                            purity_ops: List[str]
                            ) -> float:
        """Computes the purity data from the shots dictionary for a given seed and length index
            Args:
                shots_dict: The shots dictionary for the experiment
                seed: The seed
                length_index: The length index
                purity_ops: A correctly-ordered list of the purity ops for the experiment
            Returns:
                The purity value corresponding to (seed, length_index)
        """
        corr_vec = [0] * (4 ** self.num_qubits())
        count_vec = [0] * (4 ** self.num_qubits())
        for i, purity_op in enumerate(purity_ops):
            # vector of the 4^n correlators and counts
            # calculating the vector of 4^n correlators
            for indcorr in range(2 ** self.num_qubits()):
                zcorr = average_data(shots_dict[(seed, length_index, purity_op)],
                                     self._zdict_ops[indcorr])
                zind = self.F234(self.num_qubits(), indcorr, i)

                corr_vec[zind] += zcorr
                count_vec[zind] += 1

        # calculating the purity
        purity = 0
        for idx, _ in enumerate(corr_vec):
            purity += (corr_vec[idx] / count_vec[idx]) ** 2
        purity = purity / (2 ** self.num_qubits())
        return purity

    def fit(self,
            data: List[Counts],
            metadata: List[Dict[str, any]]) -> PurityRBResult:
        """Computes the purity RB fit for the given data"""
        xdata = self.organize_data(data, metadata)
        ydata = self.calc_statistics(xdata)
        fit_guess = self.generate_fit_guess(ydata['mean'])
        params, params_err = self.run_curve_fit(ydata, fit_guess)

        alpha = params[1]  # exponent
        alpha_err = params_err[1]

        # Calculate alpha (=p):
        # fitting the curve: A*p^(2m)+B
        # where m is the Clifford length
        alpha_pur = np.sqrt(alpha)

        # calculate the error of alpha
        alpha_pur_err = alpha_err / (2 * np.sqrt(alpha_pur))

        # calculate purity error per clifford (pepc)
        nrb = 2 ** self.num_qubits()
        pepc = (nrb - 1) / nrb * (1 - alpha_pur)
        pepc_err = (nrb - 1) / nrb * alpha_pur_err / alpha_pur

        return PurityRBResult({
            'A': params[0],
            'alpha': params[1],
            'B': params[2],
            'A_err': params_err[0],
            'alpha_err': params_err[1],
            'B_err': params_err[2],
            'alpha_pur': alpha_pur,
            'alpha_pur_err': alpha_pur_err,
            'pepc': pepc,
            'pepc_err': pepc_err,
            'qubits': self._qubits,
            'xdata': xdata,
            'ydata': ydata,
            'lengths': self._lengths,
            'fit_function': self._rb_fit_fun,
            'group_type': self._group_type
        })
