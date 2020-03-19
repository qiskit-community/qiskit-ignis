# -*- coding: utf-8 -*-
#
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

# pylint: disable=missing-docstring

"""
Quantum gate set tomography fitter
"""

import itertools
from typing import Union, List, Dict, Tuple, Optional
import numpy as np
import scipy.optimize as opt
from qiskit.result import Result
from qiskit.quantum_info import Choi, PTM
from ..basis.gatesetbasis import default_gateset_basis, GateSetBasis
from .base_fitter import TomographyFitter


class GatesetTomographyFitter:
    def __init__(self,
                 result: Result,
                 circuits: List,
                 gateset_basis: Union[GateSetBasis, str] = 'Default'
                 ):
        """Initialize gateset tomography fitter with experimental data.

        Args:
            result: a Qiskit Result object obtained from executing
                            tomography circuits.
            circuits: a list of circuits or circuit names to extract
                            count information from the result object.
            gateset_basis: (default: 'Default') Representation of
            the gates and SPAM circuits of the gateset
        """
        self.gateset_basis = gateset_basis
        if gateset_basis == 'Default':
            self.gateset_basis = default_gateset_basis()
        data = TomographyFitter(result, circuits).data
        self.probs = {}
        for key, vals in data.items():
            self.probs[key] = vals.get('0', 0) / sum(vals.values())

    def linear_inversion(self) -> Dict[str, np.array]:
        """
        Reconstruct a gate set from measurement data using linear inversion.

        Returns:
            For each gate in the gateset: its approximation found
            using the linear inversion process.

        Additional Information:
            Given a gate set (G1,...,Gm)
            and SPAM circuits (F1,...,Fn) constructed from those gates
            the data should contain the probabilities of the following types:
            p_ijk = E*F_i*G_k*F_j*rho
            p_ij = E*F_i*F_j*rho

            One constructs the Gram matrix g = (p_ij)_ij
            which can be described as a product g=AB
            where A = sum (i> <E F_i) and B=sum (F_j rho><j)
            For each gate Gk one can also construct the matrix Mk=(pijk)_ij
            which can be described as Mk=A*Gk*B
            Inverting g we obtain g^-1 = B^-1A^-1 and so
            g^1 * Mk = B^-1 * Gk * B
            This gives us a matrix similiar to Gk's representing matrix.
            However, it will not be the same as Gk,
            since the observable results cannot distinguish
            between (G1,...,Gm) and (B^-1*G1*B,...,B^-1*Gm*B)
            a further step of *Gauge optimization* is required on the results
            of the linear inversion stage.
            One can also use the linear inversion results as a starting point
            for a MLE optimization for finding a physical gateset, since
            unless the probabilities are accurate, the resulting gateset
            need not be physical.
        """
        n = len(self.gateset_basis.spam_labels)
        m = len(self.gateset_basis.gate_labels)
        gram_matrix = np.zeros((n, n))
        gate_matrices = []
        for i in range(m):
            gate_matrices.append(np.zeros((n, n)))

        for i in range(n):  # row
            for j in range(n):  # column
                F_i = self.gateset_basis.spam_labels[i]
                F_j = self.gateset_basis.spam_labels[j]
                gram_matrix[i][j] = self.probs[(F_i, F_j)]

                for k in range(m):  # gate
                    G_k = self.gateset_basis.gate_labels[k]
                    gate_matrices[k][i][j] = self.probs[(F_i, G_k, F_j)]

        gram_inverse = np.linalg.inv(gram_matrix)

        gates = [gram_inverse @ gate_matrix for gate_matrix in gate_matrices]
        return dict(zip(self.gateset_basis.gate_labels, gates))

    def default_init_state(self, size):
        if size == 4:
            return np.array([[0.5], [0], [0], [0.5]])
        raise RuntimeError("No default init state for more than 1 qubit")

    def default_measurement_op(self, size):
        if size == 4:
            return np.array([[1, 0, 0, 1]])
        raise RuntimeError("No default measurement op for more than 1 qubit")

    def fit(self) -> Dict:
        """
        Reconstruct a gate set from measurement data using optimization.

        Returns:
           For each gate in the gateset: its approximation found using the
           optimization process.

        Additional Information:
            The gateset optimization process con/.sists of three phases:
            1) Use linear inversion to obtain an initial approximation.
            2) Use gauge optimization to ensure the linear inversion results
            are close enough to the expected optimization outcome to serve
            as a suitable starting point
            3) Use MLE optimization to obtain the final outcome
        """
        linear_inversion_results = self.linear_inversion()
        n = len(self.gateset_basis.spam_labels)
        E = self.default_measurement_op(n)
        rho = self.default_init_state(n)
        Gs = [self.gateset_basis.gate_matrices[label]
              for label in self.gateset_basis.gate_labels]
        Fs = [self.gateset_basis.spam_matrix(label)
              for label in self.gateset_basis.spam_labels]
        Gs_E = [linear_inversion_results[label]
                for label in self.gateset_basis.gate_labels]
        gauge_opt = GaugeOptimize(Gs, Gs_E, Fs, rho)
        Gs_E = gauge_opt.optimize()
        optimizer = GST_Optimize(self.gateset_basis.gate_labels,
                                 self.gateset_basis.spam_spec,
                                 self.probs)
        optimizer.set_initial_value(E, rho, Gs_E)
        optimization_results = optimizer.optimize()
        return optimization_results


class GaugeOptimize():
    def __init__(self,
                 Gs: List[np.array],
                 initial_Gs_E: List[np.array],
                 Fs: List[np.array],
                 rho: np.array
                 ):
        """Initialize gauge optimizer fitter with the ideal and expected
            outcomes.
        Args:
            Gs: The ideal expected gate matrices
            initial_Gs_E: The experimentally-obtained gate approximations.
            Fs: The SPAM circuit matrices
            rho: The system's initial value (usually the |0> qubits)

        Additional information:
            Gauge optimization aims to find a basis in which the tomography
            results are as close as possible to the ideal (noiseless) results

            Given a gateset specification (E, rho, G1,...,Gn) and any
            invertible matrix B, the gateset specification
            (E*B^-1, B*rho, B*G1*B^-1,...,B*Gn*B^-1)
            is indistinguishable from it by the tomography results.

            B is called the gauge matrix and the goal of gauge optimization
            is finding the B for which the resulting gateset description
            is optimal in some sense; we choose to minimize the norm
            difference between the gates found by experiment
            and the "expected" gates in the ideal (noiseless) case.
        """
        self.Gs = Gs
        self.Fs = Fs
        self.initial_Gs_E = initial_Gs_E
        self.d = np.shape(Gs[0])[0]
        self.n = len(Gs)
        self.rho = rho

    def x_to_Gs_E(self, x: np.array) -> List[np.array]:
        """Converts the gauge to the gateset defined by it
                Args:
                    x: An array representation of the B matrix

                Returns:
                    The gateset obtained from B

                Additional information:
                    Given a vector representation of B, this functions
                    produces the list [B*G1*B^-1,...,B*Gn*B^-1]
                    of gates correpsonding to the gauge B

            """
        B = np.array(x).reshape((self.d, self.d))
        try:
            BB = np.linalg.inv(B)
            return [BB @ G @ B for G in self.initial_Gs_E]
        except np.linalg.LinAlgError:
            return np.inf

    def obj_fn(self, x: np.array) -> float:
        """The norm-based score function for the gauge optimizer
            Args:
                x: An array representation of the B matrix

            Returns:
                The sum of norm differences between the ideal gateset
                and the one corresponding to B
        """
        Gs_E = self.x_to_Gs_E(x)
        return sum([np.linalg.norm(G - G_E)
                    for (G, G_E)
                    in zip(self.Gs, Gs_E)])

    def optimize(self) -> List[np.array]:
        """The main optimization method
            Returns:
                The optimal gateset found by the gauge optimization
        """
        initial_value = np.array([(F @ self.rho).T[0] for F in self.Fs]).T
        result = opt.minimize(self.obj_fn, initial_value)
        return self.x_to_Gs_E(result.x)


def split_list(l: List, sizes: List) -> List[List]:
    """Splits a list to several lists of given size
        Args:
            l: A list
            sizes: The sizes of the splitted lists
        Returns:
            The splitted lists
        Example:
            >> split_list([1,2,3,4,5,6,7], [1,4,2])
            [[1],[2,3,4,5],[6,7]]
    """
    if sum(sizes) != len(l):
        msg = "Length of list ({}) " \
              "differs from sum of split sizes ({})".format(len(l), sizes)
        raise RuntimeError(msg)
    result = []
    i = 0
    for s in sizes:
        result.append(l[i:i + s])
        i = i + s
    return result


def matrix_to_vec(A: np.array) -> np.array:
    """Converts a complex matrix to vector representation
        Args:
            A: A nxn complex matrix
        Returns:
            A row-stacking vector representation of A; first the real part
            of A and then the imaginary part
    """
    real = []
    imag = []
    for row in A:
        for x in row:
            real.append(np.real(x))
            imag.append(np.imag(x))
    return real + imag


def vec_to_complex_matrix(vec: np.array, n: int) -> np.array:
    """Constructs a nxn matrix from the given vector specification
        Args:
            vec: A vector representation of a complex matrix
            n: the dimension of the matrix
        Returns:
            The reconstructed matrix
    """
    vects = split_list(vec, [n ** 2, n ** 2])
    real = np.array([vects[0][n * i:n * (i + 1)] for i in range(n)])
    imag = np.array([vects[1][n * i:n * (i + 1)] for i in range(n)])
    return real + 1.j * imag


def complex_matrix_to_vec(M, n):
    return list(np.real(M).reshape(n ** 2)) + \
           list(np.imag(M).reshape(n ** 2))


def complex_matrix_var_nums(n: int):
    """The number of entries in a vector representing nxn complex matrix
    Args:
        n: the dimension of the matrix
    Returns:
        The number of entries in the vector representation
    """
    return 2 * n ** 2


def get_cholesky_like_decomposition(mat: np.array) -> np.array:
    """Given a PSD matrix A, finds a matrix T such that TT^{dagger}
    is an approximation of A
        Args:
            mat: A nxn matrix, assumed to be positive semidefinite.
        Returns:
            A matrix T such that TT^{dagger} approximates A
    """
    (eigenvals, P) = np.linalg.eigh(mat)
    # if a 0 eigenvalue is represented by infinitisimal negative float
    eigenvals[eigenvals < 0] = 0
    DD = np.diag(np.sqrt(eigenvals))
    return P @ DD


class GST_Optimize():
    def __init__(self,
                 Gs: List[str],
                 Fs: Dict[str, Tuple[str]],
                 probs: Dict[Tuple[str], float],
                 qubits: int = 1
                 ):
        """Initializes the data for the MLE optimizer
            Args:
                Gs: The names of the gates in the gateset
                Fs: The SPAM specification (SPAM name -> gate names)
                probs: The probabilities obtained experimentally
                qubits: the size of the gates in the gateset
        """
        self.probs = probs
        self.Gs = Gs
        self.Fs = Fs
        self.Fs_names = list(Fs.keys())
        self.qubits = qubits
        self.obj_fn_data = self.compute_objective_function_data()
        self.initial_value = None

    def compute_objective_function_data(self) -> List:
        """Computes auxiliary data needed for efficient computation
        of the objective function.

            Returns:
                 The objective function data list
            Additional information:
                The objective function is
                sum_{ijk}(<|E*R_Fi*G_k*R_Fj*Rho|>-m_{ijk})^2
                We expand R_Fi*G_k*R_Fj to a sequence of G-gates and store
                indices. We also obtain the m_{ijk} value from the probs list
                all that remains when computing the function is thus
                performing the matrix multiplications and remaining algebra.
        """
        m = len(self.Fs)
        n = len(self.Gs)
        obj_fn_data = []
        for (i, j) in itertools.product(range(m), repeat=2):
            for k in range(n):
                Fi = self.Fs_names[i]
                Fj = self.Fs_names[j]
                m_ijk = (self.probs[(Fi, self.Gs[k], Fj)])
                Fi_matrices = [self.Gs.index(gate) for gate in self.Fs[Fi]]
                Fj_matrices = [self.Gs.index(gate) for gate in self.Fs[Fj]]
                matrices = Fi_matrices + [k] + Fj_matrices
                obj_fn_data.append((matrices, m_ijk))
        return obj_fn_data

    def split_input_vector(self, x: np.array) -> Tuple:
        """Reconstruct the GST data from its vector representation
        Args:
            x: The vector representation of the GST data

        Returns:
            The GST data (E, rho, Gs) (see additional info)

        Additional information:
            The gate set tomography data is a tuple (E, rho, Gs) consisting of
            1) A POVM measurement operator E
            2) An initial quantum state rho
            3) A list Gs = (G1, G2, ..., Gk) of gates, represented as matrices

            This function reconstructs (E, rho, Gs) from the vector x
            Since the MLE optimization procedure has PSD constraints on
            E, rho and the Choi represetnation of the PTM of the Gs,
            we rely on the following property: M is PSD iff there exists
            T such that M = T @ T^{dagger}.
            Hence, x stores those T matrices for E, rho and the Gs
        """
        n = len(self.Gs)
        d = (2 ** self.qubits)
        ds = d ** 2  # d squared - the dimension of the density operator

        d_t = complex_matrix_var_nums(d)
        ds_t = complex_matrix_var_nums(ds)
        T_vars = split_list(x, [d_t, d_t] + [ds_t] * n)

        E_T = vec_to_complex_matrix(T_vars[0], d)
        rho_T = vec_to_complex_matrix(T_vars[1], d)
        Gs_T = [vec_to_complex_matrix(T_vars[2+i], ds) for i in range(n)]

        E = np.reshape(E_T @ np.conj(E_T.T), (1, ds))
        rho = np.reshape(rho_T @ np.conj(rho_T.T), (ds, 1))
        Gs = [PTM(Choi(G_T @ np.conj(G_T.T))).data for G_T in Gs_T]

        return (E, rho, Gs)

    def join_input_vector(self,
                          E: np.array,
                          rho: np.array,
                          Gs: List[np.array]
                          ) -> np.array:
        """Converts the GST data into a vector representation
            Args:
                E: The POVM measurement operator
                rho: The initial state
                Gs: The gates list

            Returns:
                The vector representation of (E, rho, Gs)

            Additional information:
                This function performs the inverse operation to
                split_input_vector; the notations are the same.
        """
        d = (2 ** self.qubits)
        ds = d ** 2  # d squared - the dimension of the density operator

        E_T = get_cholesky_like_decomposition(E.reshape((d, d)))
        rho_T = get_cholesky_like_decomposition(rho.reshape((d, d)))
        Gs_Choi = [Choi(PTM(G)).data for G in Gs]
        Gs_T = [get_cholesky_like_decomposition(G) for G in Gs_Choi]
        E_vec = complex_matrix_to_vec(E_T, d)
        rho_vec = complex_matrix_to_vec(rho_T, d)
        result = E_vec + rho_vec
        for G_T in Gs_T:
            result += complex_matrix_to_vec(G_T, ds)
        return result

    def obj_fn(self, x: np.array) -> float:
        """The MLE objective function
            Args:
                x: The vector representation of the GST data (E, rho, Gs)

            Return value:
                The MLE cost function (see additional information)

            Additional information:
                The MLE objective function is obtained by approximating
                the MLE estimator using the central limit theorem.

                It is computed as the sum of all terms of the form
                (m_{ijk} - p_{ijk})^2
                Where m_{ijk} are the experimental results, and
                p_{ijk} are the predicted results for the given GST data:
                p_{ijk} = E*F_i*G_k*F_j*rho.

                For additional info, see section 3.5 in arXiv:1509.02921
        """
        E, rho, G_matrices = self.split_input_vector(x)
        val = 0
        for term in self.obj_fn_data:
            term_val = E
            for G_index in term[0]:
                term_val = term_val @ G_matrices[G_index]
            term_val = term_val @ rho
            term_val = np.real(term_val[0][0])
            term_val = term_val - term[1]  # m_{ijk}
            term_val = term_val ** 2
            val = val + term_val
        return val

    def ptm_matrix_values(self, x: np.array) -> List[np.array]:
        """Returns a vectorization of the gates matrices
            Args:
                x: The vector representation of the GST data

            Returns:
                A vectorization of all the PTM matrices for the gates
                in the GST data

            Additional information:
                This function is not trivial since the returned vector
                is not a subset of x, since for each gate G, what x
                stores in practice is a matrix T, such that the
                Choi matrix of G is T@T^{dagger}. This needs to be
                converted into the PTM representation of G.
        """
        _, _, G_matrices = self.split_input_vector(x)
        result = []
        for G in G_matrices:
            result = result + matrix_to_vec(G)
        return result

    def rho_trace(self, x: np.array) -> Tuple[float]:
        """Returns the trace of the GST initial state
            Args:
                x: The vector representation of the GST data
            Returns:
                The trace of rho - the initial state of the GST. The real
                and imaginary part are returned separately.
        """
        _, rho, _ = self.split_input_vector(x)
        d = (2 ** self.qubits)  # rho is dxd and starts at variable d^2
        rho = rho.reshape((d, d))
        trace = sum([rho[i][i] for i in range(d)])
        return (np.real(trace), np.imag(trace))

    def bounds_eq_constraint(self, x: np.array) -> List[float]:
        """Equality MLE constraints on the GST data

        Args:
            x: The vector representation of the GST data

        Returns:
            The list of computed constraint values (should equal 0)

        Additional information:
            We have the following constraints on the GST data, due to
            the PTM representation we are using:
            1) G_{0,0} is 1 for every gate G
            2) The rest of the first row of each G is 0.
            3) G only has real values, so imaginary part is 0.

            For additional info, see section 3.5.2 in arXiv:1509.02921
        """
        ptm_matrix = self.ptm_matrix_values(x)
        bounds_eq = []
        n = len(self.Gs)
        d = (2 ** self.qubits)  # rho is dxd and starts at variable d^2
        ds = d ** 2

        i = 0
        for _ in range(n):  # iterate over all Gs
            bounds_eq.append(ptm_matrix[i] - 1)  # G^k_{0,0} is 1
            i += 1
            for _ in range(ds - 1):
                bounds_eq.append(ptm_matrix[i] - 0)  # G^k_{0,i} is 0
                i += 1
            for _ in range((ds - 1) * ds):  # rest of G^k
                i += 1
            for _ in range(ds ** 2):  # the complex part of G^k
                bounds_eq.append(ptm_matrix[i] - 0)  # G^k_{0,i} is 0
                i += 1
        return bounds_eq

    def bounds_ineq_constraint(self, x: np.array) -> List[float]:
        """Inequality MLE constraints on the GST data

            Args:
                x: The vector representation of the GST data

            Returns:
                The list of computed constraint values (should be >= 0)

            Additional information:
                We have the following constraints on the GST data, due to
                the PTM representation we are using:
                1) Every row of G except the first has entries in [-1,1]

                We implement this as two inequalities per entry.

                For additional info, see section 3.5.2 in arXiv:1509.02921
        """
        ptm_matrix = self.ptm_matrix_values(x)
        bounds_ineq = []
        n = len(self.Gs)
        d = (2 ** self.qubits)  # rho is dxd and starts at variable d^2
        ds = d ** 2

        i = 0
        for _ in range(n):  # iterate over all Gs
            i += 1
            for _ in range(ds - 1):
                i += 1
            for _ in range((ds - 1) * ds):  # rest of G^k
                bounds_ineq.append(ptm_matrix[i] + 1)  # G_k[i] >= -1
                bounds_ineq.append(-ptm_matrix[i] + 1)  # G_k[i] <= 1
                i += 1
            for _ in range(ds ** 2):  # the complex part of G^k
                i += 1
        return bounds_ineq

    def rho_trace_constraint(self, x: np.array) -> List[float]:
        """ The constraint Tr(rho) = 1
            Args:
                x: The vector representation of the GST data

            Return:
                The list of computed constraint values (should be equal 0)

            Additional information:
                We demand real(Tr(rho)) == 1 and imag(Tr(rho)) == 0
        """
        trace = self.rho_trace(x)
        return [trace[0] - 1, trace[1]]

    def constraints(self) -> List[Dict]:
        """Generates the constraints for the MLE optimization

        Returns:
            A list of constraints.

        Additional information:
            Each constraint is a dictionary containing
            type ('eq' for equality == 0, 'ineq' for inequality >= 0)
            and a function generating from the input x the values
            that are being constrained.
        """
        cons = []
        cons.append({'type': 'eq', 'fun': self.rho_trace_constraint})
        cons.append({'type': 'eq', 'fun': self.bounds_eq_constraint})
        cons.append({'type': 'ineq', 'fun': self.bounds_ineq_constraint})
        return cons

    def process_result(self, x: np.array) -> Dict:
        """Transforms the optimization result to a friendly format
            Args:
                x: the optimization result vector

            Returns:
                The final GST data, as dictionary.
        """
        E, rho, G_matrices = self.split_input_vector(x)
        result = {}
        result['E'] = E
        result['rho'] = rho
        for i in range(len(self.Gs)):
            result[self.Gs[i]] = G_matrices[i]
        return result

    def set_initial_value(self,
                          E: np.array,
                          rho: np.array,
                          Gs: List[np.array]
                          ):
        """Sets the initial value for the MLE optimization
            Args:
                E: The POVM measurement operator.
                rho: The inital state.
                Gs: A list of the gate matrices.

        """
        self.initial_value = self.join_input_vector(E, rho, Gs)

    def optimize(self, initial_value: Optional[np.array] = None) -> Dict:
        """Performs the MLE optimization for gate set tomography
            Args:
                initial_value: Vector representation of the initial value data
            Returns:
                The formatted results of the MLE optimization.
        """
        if initial_value is not None:
            self.initial_value = initial_value
        result = opt.minimize(self.obj_fn, self.initial_value,
                              method='SLSQP',
                              constraints=self.constraints())
        formatted_result = self.process_result(result.x)
        return formatted_result
