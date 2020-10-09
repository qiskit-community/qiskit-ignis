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

"""
Functions used for the analysis of quantum volume results.

Based on Cross et al. "Validating quantum computers using
randomized model circuits", arXiv:1811.12926
"""

import math
import warnings
import numpy as np
from qiskit import QiskitError
from qiskit.ignis.verification.quantum_volume import circuits
from qiskit.ignis.verification.quantum_volume import fitters, QVFitter
import qiskit.execute
from qiskit.circuit.quantumcircuit import QuantumCircuit
import qiskit.providers.aer
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.compiler import transpile
from qiskit.providers.jobstatus import JobStatus
import time


class quantum_volume:
	def __init__(self, backend, ntrials=50, qubit_list=None, noise_model=None, basis_gates=None):
		self._noise_model=None
		if 'qasm' in backend.name():
			if noise_model == None:
				raise QiskitError("qasm-simulator backends requires a noise model")
			else:
				self._noise_model = noise_model
		
		if qubit_list == None:
			raise QiskitError("WIP: Please input a list of qubits to run the quantum volume calculation on")
		else:
			self._qubit_list = qubit_list

		self._basis_gates = basis_gates
		self._backend = backend
		self._is_simulator = backend.configuration().simulator
    
		self._ntrials = ntrials
		self._max_depth = len(qubit_list)
		self._list_of_qubit_lists = self.generate_qubit_lists()
		self._simulator_backend = qiskit.Aer.get_backend('statevector_simulator')

	#TODO: This is a simple approach. I am just generating, from [0,1,2,3,4,5,6], [[0,1],[0,1,2].[0,1,2,3],[0,1,2,3,4],[0,1,2,3,4,5],[0,1,2,3,4,5,6]]
		#Ideally, I would choose the best of the qubits in the subset of the qubit_list
	def generate_qubit_lists(self):

		"""
        Subdivides the input qubit list to test smaller quantum volumes

        Args:
            qubit_list (list): List of m qubits over which the maximum quantum volume should be calculated
		"""
		
		#Creates a qubit list for each depth to be tested, up to the maxiumum depth == len(qubit_list)
		return [self._qubit_list[:depth] for depth in range(3, self._max_depth + 1)]

	def generate_list_of_trails(self, depth):
		return(circuits.qv_circuits([self._list_of_qubit_lists[depth - 3]], self._ntrials)) #The -2 corrects the depth -> index


	def find_quantum_volume(self):
		managed_job_list = []
		sim_results_list = []
		
		for depth in range(3, self._max_depth + 1): #Don't check depth == 1
			sim_results = []
			real_results = []
			qv_circs_real, qv_circs_sim = self.generate_list_of_trails(depth) #Creates one list of circuits to run and one to simulate. 
			#Simulates results stored in a list of results
			for trial in range(self._ntrials):
				sim_results.append(qiskit.execute(qv_circs_sim[trial], backend=self._simulator_backend).result())
			sim_results_list.append(sim_results)
			# qv_circs_sim = [circ[0] for circ in qv_circs_sim]
			# sim_results = execute(qv_circs_sim, backend=self._simulator_backend).result()
			#Create the job manager for the depth run
			job_manager = IBMQJobManager()
			#Strips the outer list from qv_circs_real
			qv_circs_real = [circ[0] for circ in qv_circs_real]
			if self._is_simulator:
				managed_job = execute(qv_circs_real, backend = self._backend, noise_model = self._noise_model ,basis_gates=self._basis_gates)
			else:
				#Transpile circuits, and create a ManagedJobSet object to later call the results, then stores it in a list
				qv_circs_real = transpile(qv_circs_real, backend = self._backend)
				managed_job = job_manager.run(qv_circs_real, backend = self._backend)
			managed_job_list.append(managed_job)
		if self._is_simulator:
			QV = self.find_best_qv_sim(managed_job_list,sim_results_list)
		else:
			QV = self.find_best_qv_real(managed_job_list,sim_results_list)
		return QV
		
		#For each managed job in the list, try to call the results. Once it gets one result, it processes it, then determines if the rest of the 
		#results should continue to run or be stopped before running to save resources. If the QV calc fails, there is no point in running higher 
		#width circuits
	def find_best_qv_real(self,managed_job_list,sim_results_list):
		current_best_qv = 1
		for managed_job_index in range(len(managed_job_list)):

			sim_results = sim_results_list[managed_job_index]
			waiting = True
			#While the results are pending, keep looping and waiting. Break the loop and return the result once the job is finished

			print("Began requesting results")
			while waiting:
				print(managed_job_list[managed_job_index].statuses()[-1])
				if (managed_job_list[managed_job_index].statuses()[-1] is JobStatus.DONE):
					real_results = [managed_job_list[managed_job_index].results()._get_result(index)[0] for index in range(self._ntrials)]
					waiting = False
				else:
					time.sleep(5)

			print("Successfully retrieved Results for depth = " + str(managed_job_index + 3))
			
			current_list = self._list_of_qubit_lists[managed_job_index]
			success, qv_fitter  = self.qvolume(current_list,sim_results,real_results)
			# qv_fitter.plot_qv_data()
			# print(success)
			#Determine if the QV calculation completed. If it did, store the QV and continue running.  If not, then cancel the rest of the 
			#Jobs and return the previous best
			if success:
				current_best_qv = qv_fitter.quantum_volume()[0]
				if (managed_job_index == self._max_depth - 3):
					return current_best_qv
			else:
				for remaining_index in range(managed_job_index + 1, self._max_depth - 1):
					managed_job_list[remaining_index].cancel()
				return current_best_qv #TODO THIS SHOULD ALSO RETURN THE CIRCUITS AND PLOTS

	def find_best_qv_sim(self,managed_job_list,sim_results_list):
		current_best_qv = 1
		for managed_job_index in range(len(managed_job_list)):

			sim_results = sim_results_list[managed_job_index]
			real_results = managed_job_list[managed_job_index].result()
			current_list = self._list_of_qubit_lists[managed_job_index]
			success, qv_fitter  = self.qvolume(current_list,sim_results,real_results)
			# qv_fitter.plot_qv_data()
			# print(success)
			#Determine if the QV calculation completed. If it did, store the QV and continue running.  If not, then cancel the rest of the 
			#Jobs and return the previous best
			if success:
				current_best_qv = qv_fitter.quantum_volume()[0]
				if (managed_job_index == self._max_depth - 3):
					return current_best_qv
			else:
				return current_best_qv #TODO THIS SHOULD ALSO RETURN THE CIRCUITS AND PLOTS

	def qvolume(self,qlist,sim_results,real_results):
		QV = 1
		qv_fitter = QVFitter(qubit_lists=[qlist])
		# fitter_data = fitter_single(qlist)
		qv_fitter.add_statevectors(sim_results)
		qv_fitter.add_data(real_results)
		qv_success_list = qv_fitter.qv_success()
		qv_list = qv_fitter.ydata
		if qv_success_list[0][0]:
			QV = qv_fitter.quantum_volume()[0]
		return (qv_success_list[0][0], qv_fitter)

	def q_fitter(self):
		sim_results=[]
		real_results=[]
		qv_circs_real, qv_circs_sim = self.generate_list_of_trails(self._max_depth)
		for trial in range(self._ntrials):
			sim_results.append(qiskit.execute(qv_circs_sim[trial], backend=self._simulator_backend).result())
			if self._is_simulator:
				real_results.append(execute(qv_circs_real[trial], backend = self._backend, noise_model = self._noise_model ,basis_gates=self._basis_gates).result())
		if not self._is_simulator:
			all_circs = []
			for circ in qv_circs_real:
				all_circs += circ
			job_manager = IBMQJobManager()
			managed_job = job_manager.run(all_circs, backend = self._backend)
		while (managed_job.statuses()[-1] is not JobStatus.DONE):
			time.sleep(5)
		real_results = []
		for ind in range(self._ntrials):
			real_results.append([])
			for idx in range(self._max_depth-3):
				real_results[ind]+=(managed_job.result()._get_result(idx)[0])
		return QVFitter(real_results,sim_results,self.generate_qubit_lists())