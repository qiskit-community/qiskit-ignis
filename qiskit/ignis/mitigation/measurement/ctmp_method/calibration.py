# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Perform CTMP calibration for error mitigation.
"""

# pylint: disable=invalid-name
# pylint: disable=logging-format-interpolation

import logging
from abc import abstractmethod
from itertools import combinations, product
from typing import List, Union, Tuple, Dict, Set

import numpy as np
from scipy import sparse
from scipy.linalg import logm

from qiskit import QuantumCircuit
from qiskit.result import Result

logger = logging.getLogger(__name__)

"""Generators are uniquely determined by two bitstrings,
and a list of qubits on which the bitstrings act. For instance,
the generator `|b^i><a^i| - |a^i><a^i|` acting on the (ordered)
set `C_i` is represented by `g = ('1', '0', [5])`
"""
Generator = Tuple[str, str, Tuple[int]]


_KET_BRA_DICT = {
    '00': np.array([[1, 0], [0, 0]]),
    '01': np.array([[0, 1], [0, 0]]),
    '10': np.array([[0, 0], [1, 0]]),
    '11': np.array([[0, 0], [0, 1]])
}


def tensor_list(parts: List[np.array]) -> np.array:  # pylint: disable=invalid-name
    """Given a list [a, b, c, ...], return
    the array a otimes b otimes c otimes ...
    """
    res = parts[0]
    for m in parts[1:]:
        res = sparse.kron(res, m)
    return res


def generator_to_sparse_matrix(gen: Generator, num_qubits: int) -> sparse.coo_matrix:
    """Compute the matrix form of a generator.

    Args:
        gen (Generator): Generator to use.
        num_qubits: Number of qubits for final operator.

    Returns:
        sparse.coo_matrix: Sparse representation of generator.
    """
    b, a, c = gen
    shape = (2 ** num_qubits,) * 2
    res = sparse.coo_matrix(shape)
    # pylint: disable=unnecessary-lambda
    ba_strings = list(map(lambda x: ''.join(x), list(zip(*[b, a]))))
    aa_strings = list(map(lambda x: ''.join(x), list(zip(*[a, a]))))
    ba_mats = [sparse.eye(2, 2).tocoo()] * num_qubits
    aa_mats = [sparse.eye(2, 2).tocoo()] * num_qubits
    for _c, _ba, _aa in zip(c, ba_strings, aa_strings):
        ba_mats[_c] = _KET_BRA_DICT[_ba]
        aa_mats[_c] = _KET_BRA_DICT[_aa]
    res += tensor_list(ba_mats[::-1])
    res -= tensor_list(aa_mats[::-1])
    return res


def match_on_set(str_1: str, str_2: str, qubits: Set[int]) -> bool:
    """Ask whether or not two bitstrings are equal on a set of bits.

    Args:
        str_1 (str): First string.
        str_2 (str): Second string.
        qubits (Set[int]): Qubits to check.

    Returns:
        bool: Whether or not the strings match on the given indices.

    Raises:
        ValueError: When the strings do not have equal length.
    """
    num_qubits = len(str_1)
    if len(str_1) != len(str_2):
        raise ValueError('Strings must have same length')
    q_inds = [num_qubits - i - 1 for i in qubits]
    for i in q_inds:
        if str_1[i] != str_2[i]:
            return False
    return True


def no_error_out_set(
        in_set: Set[int],
        counts_dict: Dict[str, int],
        input_state: str
) -> Dict[str, int]:
    """Given a counts dictionary, a desired bitstring, and an "input set",
    return the dictionary of counts where there are no errors on qubits
    not in `in_set`, as determined by `input_state`.
    """
    output_dict = {}
    num_qubits = len(input_state)
    out_set = set(range(num_qubits)) - in_set
    for output_state, counts in counts_dict.items():
        if match_on_set(output_state, input_state, out_set):
            output_dict[output_state] = counts
    return output_dict


def compute_gamma(g_matrix: sparse.coo_matrix) -> float:
    """Compute the factor `gamma` for a given generator matrix. See paper
    for details.

    Args:
        g_matrix (sparse.coo_matrix): Generator matrix to check.

    Returns:
        float: `gamma` factor.

    Raises:
        ValueError: When the given `g_matrix` does not have a non-negative value. Typically occurs
            if there is no readout error in the backend used to run the calibration circuits.
    """
    cg = -g_matrix.tocoo()
    current_max = -np.inf
    logger.info('Computing gamma...')
    for i, j, v in zip(cg.row, cg.col, cg.data):
        if i == j:
            if v > current_max:
                current_max = v
    logger.info('Computed gamma={}'.format(current_max))
    if current_max < 0:
        raise ValueError('gamma should be non-negative, found gamma={}'.format(current_max))
    return current_max


def local_a_matrix(j: int, k: int, counts_dicts: Dict[str, Dict[str, int]]) -> np.array:
    """Computes the A(j,k) matrix in the basis:
    00, 01, 10, 11
    """
    if j == k:
        raise ValueError('Encountered j=k={}'.format(j))
    a_out = np.zeros((4, 4))
    indices = ['00', '01', '10', '11']
    index_dict = {b: int(b, 2) for b in indices}
    for w, v in product(indices, repeat=2):
        v_to_w_err_cts = 0
        tot_cts = 0
        for input_str, c_dict in counts_dicts.items():
            if input_str[::-1][j] == v[0] and input_str[::-1][k] == v[1]:
                no_err_out_dict = no_error_out_set({j, k}, c_dict, input_str)
                tot_cts += np.sum(list(no_err_out_dict.values()))
                for output_str, counts in no_err_out_dict.items():
                    if output_str[::-1][j] == w[0] and output_str[::-1][k] == w[1]:
                        v_to_w_err_cts += counts
        a_out[index_dict[w], index_dict[v]] = v_to_w_err_cts / tot_cts
    return a_out


class BaseGeneratorSet:
    """Set of generators used to calibrate CTMP method. Includes information about how to
    calibrate the generator coefficients.
    """

    def __init__(self, num_qubits: int):
        """
        Args:
            num_qubits: Number of qubits on which the set of generators acts.
        """
        self.num_qubits = num_qubits
        self._generators = []

    def __len__(self) -> int:
        """
        Returns:
            int: The number of generators in the set.
        """
        return len(self._generators)

    def __getitem__(self, i: int) -> Generator:
        """
        Args:
            i: Index of a generator.

        Returns:
            Generator: The i-th generator in the set.
        """
        return self._generators[i]

    def __list__(self) -> List[Generator]:
        """Return the set of generators as a list.
        """
        return self._generators

    def add_generators(self, gen_list: List[Generator]):
        """Add a generator to the set.

        Args:
            gen_list: Generator to add.
        """
        self._generators.extend(gen_list)

    @abstractmethod
    def get_ctmp_error_rate(self, gen: Generator, g_mat_dict: Dict[Generator, np.array]) -> float:
        """Compute the error rate for a given generator, given the associated list of G(j,k)
        arrays.

        Args:
            gen (Generator): Generator to compute the coefficient of.
            g_mat_dict (Dict[Generator, np.array]): Dictionary of arrays for G(j,k) for various
                generators.

        Returns:
            float: The coefficient r_i corresponding to the generator G_i.
        """
        pass

    @abstractmethod
    def local_g_matrix(self, gen: Generator, counts_dicts: Dict[str, Dict[str, int]]) -> np.array:
        """Compute the matrix G(j,k) given all the counts dictionaries needed to calibrate.

        Args:
            gen (Generator): Generator to calibrate.
            counts_dicts (Dict[str, Dict[str, int]]): Dictionary of calibration results to use.
                Keys in the outer dict correspond to the bitstring prepared on the device. Values in
                the outer dict correspond to the associated counts dictionary.

        Returns:
            np.array: The G(j,k) matrix for the given generator.
        """
        pass

    @abstractmethod
    def supplementary_generators(self, gen_list: List[Generator]) -> List[Generator]:
        """These generators do not have rates directly associated with them, but are used to compute
        rates for other generators.
        """
        pass

    @classmethod
    def from_generator_list(cls, gen_list: List[Generator], num_qubits: int):
        """Create from generator list, used for testing.
        """
        res = cls(num_qubits=num_qubits)
        res.add_generators(gen_list=gen_list)
        return res


class StandardGeneratorSet(BaseGeneratorSet):
    """
    Set of generators on 1 and 2 qubits. Corresponds to the following readout errors:
    `0 -> 1`
    `1 -> 0`
    `01 -> 10`
    `11 -> 00`
    `00 -> 11`
    """

    @staticmethod
    def standard_single_qubit_bitstrings(num_qubits: int) -> List[Generator]:
        """Returns a list of tuples `[(C_1, b_1, a_1), (C_2, b_2, a_2), ...]` that represent
        the generators .
        """
        res = [('1', '0', (i,)) for i in range(num_qubits)]
        res += [('0', '1', (i,)) for i in range(num_qubits)]
        if len(res) != 2 * num_qubits:
            raise ValueError('Should have gotten 2n qubits, got {}'.format(len(res)))
        return res

    @staticmethod
    def standard_two_qubit_bitstrings_symmetric(num_qubits: int, pairs=None) -> List[Generator]:
        """Return the 11->00 and 00->11 generators on a given number of qubits.
        """
        if pairs is None:
            pairs = list(combinations(range(num_qubits), r=2))
        res = [('11', '00', (i, j)) for i, j in pairs if i < j]
        res += [('00', '11', (i, j)) for i, j in pairs if i < j]
        if len(res) != num_qubits * (num_qubits - 1):
            raise ValueError('Should have gotten n(n-1) qubits, got {}'.format(len(res)))
        return res

    @staticmethod
    def standard_two_qubit_bitstrings_asymmetric(num_qubits: int, pairs=None) -> List[Generator]:
        """Return the 01->10 generators on a given number of qubits.
        """
        if pairs is None:
            pairs = list(combinations(range(num_qubits), r=2))
        res = [('10', '01', (i, j)) for i, j in pairs]
        res += [('10', '01', (j, i)) for i, j in pairs]
        if len(res) != num_qubits * (num_qubits - 1):
            raise ValueError('Should have gotten n(n-1) qubits, got {}'.format(len(res)))
        return res

    @classmethod
    def from_num_qubits(cls, num_qubits: int, pairs=None):
        """Construct this generator set given a number of qubits, and optionally the pairs of
        qubits to use.

        Args:
            num_qubits (int): Number of qubits to calibrate.
            pairs (Optional[List[Tuple[int, int]]]): The pairs of qubits to compute generators for.
                If None, then use all pairs. Defaults to None.

        Returns:
            StandardGeneratorSet: The generator set.

        Raises:
            ValueError: Either the number of qubits is negative or not an int.
            ValueError: The number of generators does not match what it should be by counting.
        """
        if not isinstance(num_qubits, int):
            raise ValueError('num_qubits needs to be an int')
        if num_qubits <= 0:
            raise ValueError('Need num_qubits at least 1')
        res = cls(num_qubits=num_qubits)
        res.add_generators(res.standard_single_qubit_bitstrings(num_qubits))
        if num_qubits > 1:
            res.add_generators(res.standard_two_qubit_bitstrings_symmetric(num_qubits, pairs=pairs))
            res.add_generators(
                res.standard_two_qubit_bitstrings_asymmetric(num_qubits, pairs=pairs))
            if len(res) != 2 * num_qubits ** 2:
                raise ValueError('Should have gotten 2n^2 generators, got {}...'.format(len(res)))
        return res

    def supplementary_generators(self, gen_list: List[Generator]) -> List[Generator]:
        """Supplementary generators needed to run 1q calibrations.

        Args:
            gen_list (List[Generator]): List of generators.

        Returns:
            List[Generator]: List of additional generators needed.
        """
        pairs = {tuple(gen[2]) for gen in gen_list}
        supp_gens = []
        for tup in pairs:
            supp_gens.append(('10', '00', tup))
            supp_gens.append(('00', '10', tup))
            supp_gens.append(('11', '01', tup))
            supp_gens.append(('01', '11', tup))
        return supp_gens

    def get_ctmp_error_rate(self, gen: Generator, g_mat_dict: Dict[Generator, np.array]) -> float:
        """Compute the error rate r_i for generator G_i.

        Args:
            gen (Generator): Generator to calibrate.
            g_mat_dict (Dict[Generator, np.array]): Dictionary of local G(j,k) matrices.

        Returns:
            float: The coefficient r_i for generator G_i.

        Raises:
            ValueError: The provided generator is not already in the set of generators.
        """
        b, a, c = gen
        if gen not in self._generators:
            raise ValueError('Invalid generator encountered: {}'.format(gen))
        if len(b) == 1:
            rate = self._ctmp_err_rate_1_q(a=a, b=b, j=c[0], g_mat_dict=g_mat_dict)
        elif len(b) == 2:
            rate = self._ctmp_err_rate_2_q(gen, g_mat_dict)
        logger.info('Generator {} calibrated with error rate {}'.format(
            gen, rate
        ))
        return rate

    def _ctmp_err_rate_1_q(self, a: str, b: str, j: int,
                           g_mat_dict: Dict[Generator, np.array]) -> float:
        """Compute the 1q error rate for a given generator.
        """
        rate_list = []
        if a == '0' and b == '1':
            g1 = ('00', '10')
            g2 = ('01', '11')
        elif a == '1' and b == '0':
            g1 = ('10', '00')
            g2 = ('11', '01')
        else:
            raise ValueError('Invalid a,b encountered...')
        for k in range(self.num_qubits):
            if k == j:
                continue
            c = (j, k)
            for g_strs in [g1, g2]:
                gen = g_strs + (c,)
                r = self._ctmp_err_rate_2_q(gen, g_mat_dict)
                rate_list.append(r)
        if len(rate_list) != 2 * (self.num_qubits - 1):
            raise ValueError('Rate list has wrong number of elements')
        rate = np.mean(rate_list)
        return rate

    def _ctmp_err_rate_2_q(self, gen, g_mat_dict) -> float:
        """Compute the 2 qubit error rate for a given generator.
        """
        g_mat = g_mat_dict[gen]
        b, a, _ = gen
        r = g_mat[int(b, 2), int(a, 2)]
        return r

    def local_g_matrix(self, gen: Generator, counts_dicts: Dict[str, Dict[str, int]]) -> np.array:
        """Computes the G(j,k) matrix in the basis:
        00, 01, 10, 11
        """
        _, _, c = gen
        j, k = c
        a = local_a_matrix(j, k, counts_dicts)
        g = self.amat_to_gmat(a)
        if np.linalg.norm(np.imag(g)) > 1e-3:
            raise ValueError('Encountered complex entries in G_i={}'.format(g))
        g = np.real(g)
        for i in range(4):
            for j in range(4):
                if i != j:
                    if g[i, j] < 0:
                        logger.debug('Found negative element of size: {}'.format(g[i, j]))
                        g[i, j] = 0
        return g

    @staticmethod
    def amat_to_gmat(a_mat: np.array) -> np.array:
        """Map the A(j,k) matrix to the G(j,k) matrix.
        """
        return logm(a_mat)


class BaseCalibrationCircuitSet:
    """Set of circuits needed to perform CTMP calibration.
    """

    def __init__(self, num_qubits: int):
        """Initialize with a given number of qubits.
        """
        self.num_qubits = num_qubits
        self.cal_circ_dict = {}  # type: Dict[str, QuantumCircuit]

    @property
    def circs(self) -> List[QuantumCircuit]:
        """Return the circuits in the calibration set.
        """
        return list(self.cal_circ_dict.values())

    def __dict__(self):
        """Return the set of calibration circuits as a dictionary. The keys are the bitstring
        to prepare on the register, and the keys are the circuits that accomplish this.
        """
        return self.cal_circ_dict

    def bitstring_to_circ(self, bits: Union[str, int]) -> QuantumCircuit:
        """Helper function to turn bitstrings into circuits.

        Args:
            bits (Union[str, int]): Bitstring to prepare.

        Returns:
            QuantumCircuit: The circuit that prepares the given bitstring.

        Raises:
            ValueError: When the input for `bits` is not a string or an int.
        """
        if isinstance(bits, int):
            bitstring = np.binary_repr(bits, width=self.num_qubits)  # type: str
        elif isinstance(bits, str):
            bitstring = bits  # type: str
        else:
            raise ValueError('Input bits must be either str or int')
        circ = QuantumCircuit(self.num_qubits, name='cal-{}'.format(bitstring))
        for i, b in enumerate(bitstring[::-1]):
            if b == '1':
                circ.x(i)
        circ.measure_all()
        return circ

    def get_weight_1_str(self, index: int) -> str:
        """Return a one-hot bitstring.
        """
        out = ['0'] * self.num_qubits
        out[index] = '1'
        return ''.join(out)[::-1]

    @classmethod
    def from_dict(cls, num_qubits: int, cal_circ_dict: Dict[str, QuantumCircuit]):
        """Construct a circuit set from a dict, used for testing.
        """
        res = cls(num_qubits)
        res.cal_circ_dict = cal_circ_dict
        return res


class StandardCalibrationCircuitSet(BaseCalibrationCircuitSet):
    """
    The set of calibration circuits including all weight 1 bitstrings, plus 1...1 and 0...0.
    """

    @classmethod
    def from_num_qubits(cls, num_qubits: int):
        """Construct this calibration circuit set for a given number of qubits.

        Args:
            num_qubits (int): Number of qubits.

        Returns:
            StandardCalibrationCircuitSet: The resulting circuit set.
        """
        res = cls(num_qubits=num_qubits)
        cal_strings = ['0' * num_qubits, '1' * num_qubits]
        cal_strings.extend([res.get_weight_1_str(i) for i in range(num_qubits)])
        res.cal_circ_dict = {cal_str: res.bitstring_to_circ(cal_str) for cal_str in cal_strings}
        return res


class WeightTwoCalibrationCircuitSet(BaseCalibrationCircuitSet):
    """
    The set of calibration circuits with all weight 2 bitstrings.
    """

    @classmethod
    def from_num_qubits(cls, num_qubits: int):
        """Construct this calibration circuit set for a given number of qubits.

        Args:
            num_qubits (int): Number of qubits.

        Returns:
            WeightTwoCalibrationCircuitSet: The resulting circuit set.
        """
        res = cls(num_qubits=num_qubits)
        cal_strings = []
        cal_strings.extend(['0' * num_qubits])
        cal_strings.extend([res.get_weight_1_str(i) for i in range(num_qubits)])
        for i, j in product(range(num_qubits), repeat=2):
            if i != j:
                two_hot_str = ['0'] * num_qubits
                two_hot_str[i] = '1'
                two_hot_str[j] = '1'
                cal_strings.append(''.join(two_hot_str))
        res.cal_circ_dict = {cal_str: res.bitstring_to_circ(cal_str) for cal_str in cal_strings}
        return res


class MeasurementCalibrator:
    """
    Perform the calibration of the CTMP method given a set of circuits and set of generators.
    Also holds the results of this calibration.
    """

    def __init__(
            self,
            cal_circ_set: BaseCalibrationCircuitSet,
            gen_set: BaseGeneratorSet
    ):
        """Construct the calibrator.

        Args:
            cal_circ_set (BaseCalibrationCircuitSet): The calibration circuit set to use.
            gen_set (BaseGeneratorSet): The generator set to use.
        """
        self.cal_circ_set = cal_circ_set
        self.gen_set = gen_set
        self._num_qubits = self.gen_set.num_qubits
        self._gamma = None
        self._r_dict = {}  # type: Dict[Generator, float]
        self._tot_g_mat = None
        self._b_mat = None
        self.calibrated = False

    @classmethod
    def standard_construction(cls, num_qubits: int, method: str = 'weight_2'):
        """Construct the calibrator using one of the standard constructions.

        Args:
            num_qubits (int): Number of qubits to calibrate.
            method (str): Circuit set to use, either `'weight_1'` or `'weight_2'`.

        Returns:
            MeasurementCalibrator: The calibrator.

        Raises:
            ValueError: Invalid method given for calibration circuit set.
        """
        if method == 'weight_2':
            circ_set = WeightTwoCalibrationCircuitSet.from_num_qubits(num_qubits)
        elif method == 'weight_1':
            circ_set = StandardCalibrationCircuitSet.from_num_qubits(num_qubits)
        else:
            raise ValueError('Invalid method given, needs to be either weight_1 or weight_2.')
        res = cls(
            cal_circ_set=circ_set,
            gen_set=StandardGeneratorSet.from_num_qubits(num_qubits)
        )
        return res

    @property
    def gamma(self) -> float:
        """The coefficient `gamma` for simulating the Markov process.
        """
        if not self.calibrated:
            raise ValueError('Calibration has not been run yet')
        return self._gamma

    @property
    def r_dict(self) -> Dict[Generator, float]:
        """A dictionary with keys as `Generator`s, and the values as their associated error rates.
        These are the `r_i` and `G_i` pairs.
        """
        if not self.calibrated:
            raise ValueError('Calibration has not been run yet')
        return self._r_dict

    @property
    def G(self) -> float:
        """The generator of the matrix `A` with `A=e^G`.
        """
        if not self.calibrated:
            raise ValueError('Calibration has not been run yet')
        return self._tot_g_mat

    @property
    def B(self):
        """The stochastic matrix `B = Id + G / gamma`.
        """
        if not self.calibrated:
            raise ValueError('Calibration has not been run yet')
        return self._b_mat

    def __repr__(self):
        """A string representation of the calibrator.
        """
        res = "Operator error mitigation calibrator\n"
        res += "Num generators: {}, Num qubits: {}\n".format(len(self.gen_set), self._num_qubits)
        if self.calibrated:
            r_min = np.min(list(self.r_dict.values()))
            r_max = np.max(list(self.r_dict.values()))
            r_mean = np.mean(list(self.r_dict.values()))
            res += "gamma={}, r_mean={}\nr_min={}, r_max={}".format(self.gamma, r_mean, r_min,
                                                                    r_max)
        else:
            res += "Not yet calibrated"
        return res

    def circ_dicts(self, result: Result) -> Dict[str, Dict[str, int]]:
        """Compute the dictionary of counts dictionaries for all the calibration circuits.

        Args:
            result (Result): The Result object from running the calibration circuits.

        Returns:
            Dict[str, Dict[str, int]]: The dictionary whose keys are the prepared bitstrings and
                keys are the counts dictionaries for each bitstring.
        """
        circ_dicts = {}
        for bits, circ in self.cal_circ_set.cal_circ_dict.items():
            circ_dicts[bits] = result.get_counts(circ)
        return circ_dicts

    def calibrate(self, result: Result) -> Tuple[float, Dict[Generator, float]]:
        """Perform the CTMP calibration given a result.

        Args:
            result (Result): The Result object from running all of the calibration circuits.

        Returns:
            Tuple[float, Dict[Generator, float]]: The value of `gamma` along with the dictionary
                for the `r_i` and `G_i` values, i.e. the calibration parameters for each generator.
        """
        logger.info('Beginning calibration with {} generators on {} qubits'.format(
            len(self.gen_set), self._num_qubits
        ))
        gen_mat_dict = {}
        # Compute G(j,k) matrices
        logger.info('Computing local G matrices...')
        circ_dicts = self.circ_dicts(result)
        for gen in list(self.gen_set) + self.gen_set.supplementary_generators(list(self.gen_set)):
            if len(gen[2]) > 1:
                mat = self.gen_set.local_g_matrix(gen, circ_dicts)
                gen_mat_dict[gen] = mat
        logger.info('Computed local G matrices')
        # Compute r-parameters
        logger.info('Computing generator coefficients...')
        for gen in self.gen_set:
            r = self.gen_set.get_ctmp_error_rate(gen, gen_mat_dict)
            self._r_dict[gen] = r
            logger.info('Generator G={} has error rate r={}'.format(gen, r))
        logger.info('Computed generator coefficients')
        # Compute gamma
        self._tot_g_mat = self.total_g_matrix(self._r_dict)
        self._gamma = compute_gamma(self._tot_g_mat)
        self._b_mat = sparse.eye(2 ** self._num_qubits) + self._tot_g_mat / self._gamma
        self._b_mat = self._b_mat.tocsc()
        logger.info('Finished calibration...')
        logger.info('num_qubits = {}, gamma = {}'.format(
            self._num_qubits, self._gamma
        ))
        r_vec = np.array(list(self._r_dict.values()))
        logger.info('r_min={}, r_max={}, r_mean={}'.format(
            np.min(r_vec), np.max(r_vec), np.mean(r_vec)
        ))
        logger.info('Finished calibration')
        self.calibrated = True

        return self.gamma, self.r_dict

    def total_g_matrix(self, r_dict: Dict[Generator, float]) -> sparse.coo_matrix:
        """Compute the total `G` matrix given the coefficients `r_i` and the generators `G_i`.
        """
        res = sparse.coo_matrix((2 ** self._num_qubits, 2 ** self._num_qubits), dtype=np.float)
        logger.info('Computing sparse G matrix')
        for gen, r in r_dict.items():
            res += r * generator_to_sparse_matrix(gen, self._num_qubits)
        num_elts = res.shape[0] ** 2
        try:
            nnz = res.nnz
            if nnz == num_elts:
                sparsity = '+inf'
            else:
                sparsity = nnz / (num_elts - nnz)
            logger.info('Computed sparse G matrix with sparsity {}'.format(sparsity))
        except AttributeError:
            pass
        return res
