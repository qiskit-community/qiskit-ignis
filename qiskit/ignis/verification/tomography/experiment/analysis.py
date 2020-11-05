from qiskit import QiskitError
from qiskit.ignis.experiments.base import Analysis
from qiskit.providers import BaseJob
from qiskit.result import Result, Counts
from typing import List, Dict, Union, Optional
from qiskit.ignis.verification.tomography.data import marginal_counts, combine_counts, count_keys
from utils import _fitter_data
from qiskit.ignis.verification.tomography.basis import TomographyBasis, default_basis
from qiskit.ignis.verification.tomography.fitters.lstsq_fit import lstsq_fit
from qiskit.ignis.verification.tomography.fitters.cvx_fit import cvx_fit, _HAS_CVX
import numpy as np

class TomographyAnalysis(Analysis):
    _HAS_SDP_SOLVER = None
    def __init__(self,
                 data: Optional[Union[BaseJob, Result, List[any], any]] = None,
                 metadata: Optional[Union[List[Dict[str, any]], Dict[str, any]]] = None,
                 meas_basis: Union[TomographyBasis, str] = 'Pauli',
                 prep_basis: Union[TomographyBasis, str] = 'Pauli',
                 method: str = 'auto',
                 name: Optional[str] = None,
                 exp_id: Optional[str] = None):
        super().__init__(data, metadata, name, exp_id)
        self._method = method

        meas_basis = default_basis(meas_basis)
        if isinstance(meas_basis, TomographyBasis):
            if meas_basis.measurement is not True:
                raise QiskitError("Invalid measurement basis")
        self.meas_matrix_func = meas_basis.measurement_matrix

        prep_basis = default_basis(prep_basis)
        if isinstance(prep_basis, TomographyBasis):
            if prep_basis.preparation is not True:
                raise QiskitError("Invalid preparation basis")
        self.prep_matrix_func = prep_basis.preparation_matrix

        self._analysis_fn = self.fit
        self._target_qubits = None

    def set_target_qubits(self, qubits: List[int]):
        self._target_qubits = qubits

    def _set_target_meta(self, metadata: Dict[str, any]):
        # restrict meas_qubits, meas_clbits and meas_label to the target qubits
        target_qubits = self._target_qubits
        meas_qubits = metadata['meas_qubits']
        if any([qubit not in meas_qubits for qubit in target_qubits]):
            raise RuntimeError("Target qubit set {} "
                               "not contained in measurement "
                               "qubit set {}".format(target_qubits, meas_qubits))
        qubit_indices = [meas_qubits.index(qubit) for qubit in target_qubits]
        metadata['meas_qubits'] = [metadata['meas_qubits'][i] for i in qubit_indices]
        metadata['meas_clbits'] = [metadata['meas_clbits'][i] for i in qubit_indices]
        metadata['meas_label'] = [metadata['meas_label'][i] for i in qubit_indices]

    def _format_data(self, data: Result,
                     metadata: Dict[str, any],
                     index: int) -> Counts:
        if self._target_qubits is not None:
            self._set_target_meta(metadata)
        meas_qubits = metadata['meas_qubits']
        counts = data.get_counts(index)
        counts = marginal_counts(counts, meas_qubits)
        counts = [counts.get(key, 0) for key in count_keys(len(meas_qubits))]
        return counts

    def fit(self, data, metadata,
            method: str = None,
            standard_weights: bool = True,
            beta: float = 0.5,
            psd: bool = True,
            trace: Optional[int] = None,
            trace_preserving: bool = False,
            **kwargs) -> np.array:
        if method is None:
            method = self._method

        data, basis_matrix, weights = _fitter_data(
            counts=data,
            metadata=metadata,
            measurement=self.meas_matrix_func,
            preparation=self.prep_matrix_func,
            standard_weights=standard_weights,
            beta=beta)
        # Choose automatic method
        if method == 'auto':
            self._check_for_sdp_solver()
            if self._HAS_SDP_SOLVER:
                method = 'cvx'
            else:
                method = 'lstsq'
        if method == 'lstsq':
            return lstsq_fit(data, basis_matrix,
                             weights=weights,
                             psd=psd,
                             trace=trace,
                             **kwargs)

        if method == 'cvx':
            return cvx_fit(data, basis_matrix,
                           weights=weights,
                           psd=psd,
                           trace=trace,
                           trace_preserving=trace_preserving,
                           **kwargs)

        raise QiskitError('Unrecognized fit method {}'.format(method))

    @classmethod
    def _check_for_sdp_solver(cls):
        """Check if CVXPY solver is available"""
        if cls._HAS_SDP_SOLVER is None:
            if _HAS_CVX:
                # pylint:disable=import-error
                import cvxpy
                solvers = cvxpy.installed_solvers()
                if 'CVXOPT' in solvers:
                    cls._HAS_SDP_SOLVER = True
                    return
                if 'SCS' in solvers:
                    # Try example problem to see if built with BLAS
                    # SCS solver cannot solver larger than 2x2 matrix
                    # problems without BLAS
                    try:
                        var = cvxpy.Variable((4, 4), PSD=True)
                        obj = cvxpy.Minimize(cvxpy.norm(var))
                        cvxpy.Problem(obj).solve(solver='SCS')
                        cls._HAS_SDP_SOLVER = True
                        return
                    except cvxpy.error.SolverError:
                        pass
            cls._HAS_SDP_SOLVER = False