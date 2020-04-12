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
"""
Generate data for characterization fitters tests
"""

import os
import pickle

from qiskit.ignis.characterization.coherence.fitters import T1Fitter, \
                                                            T2Fitter, \
                                                            T2StarFitter

from qiskit.ignis.characterization.hamiltonian.fitters import ZZFitter

def generate_data(filename_prefix):
    """
    Generate pickle files
    """

    files_with_pickles = ['test_fitters_t1.pkl', 'test_fitters_t2.pkl',
                          'test_fitters_t2star.pkl',
                          'test_fitters_zz.pkl']

    for picfile in files_with_pickles:
        fo = open(os.path.join(os.path.dirname(__file__),
                               picfile), 'rb')
        file_content = pickle.load(fo)
        fo.close()

        fit_type = file_content['type']

        input_to_fitter = file_content['input_to_fitter']
        if fit_type == 't1':
            fit = T1Fitter(**input_to_fitter)
        elif fit_type == 't2':
            fit = T2Fitter(**input_to_fitter)
        elif fit_type == 't2star':
            fit = T2StarFitter(**input_to_fitter)
        elif fit_type == 'zz':
            fit = ZZFitter(**input_to_fitter)
        else:
            raise NotImplementedError('Unrecognized fitter type '
                                      + fit_type)

        file_content['expected_fit'] = fit
        fo = open(filename_prefix + '_' + fit_type + '.pkl', 'wb')
        pickle.dump(file_content, fo)
        fo.close()
