# -*- coding: utf-8 -*-

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

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
    
def postselection_decoding (results):
    
    logical_prob = {}
    postselected_results = {}
    for log in results:
        postselected_results[log] = {}
        for string in results[log]:

            syndrome_list = string.split('  ')
            syndrome_list.pop(0)
            syndrome_string = '  '.join( syndrome_list )

            error_free = True
            for char in syndrome_string:
                error_free = error_free and (char in ['0',' '])

            if error_free:
                postselected_results[log][string] = results[log][string]
                
    for log in results:
        shots = 0
        incorrect_shots = 0
        for string in postselected_results[log]:
            shots += postselected_results[log][string]
            if ( string[0] != log ):
                incorrect_shots += postselected_results[log][string]

        logical_prob[log] = incorrect_shots/shots
        
    return logical_prob


def lookuptable_decoding (training_results,real_results):
    
    logical_prob = {}
    for log in real_results:
        shots = 0
        incorrect_shots = 0
        for string in real_results[log]:
            
            p = {}
            for testlog in ['0','1']:
                try:
                    p[testlog] = training_results[testlog][string]
                except:
                    p[testlog] = 0
             
            shots += real_results[log][string]
            if p['1'*(log=='0')+'0'*(log=='1')]>p[log]:
                incorrect_shots += real_results[log][string]
            

        logical_prob[log] = incorrect_shots/shots
        
    return logical_prob
      
    
