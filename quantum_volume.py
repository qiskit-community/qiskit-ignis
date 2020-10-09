import matplotlib.pyplot as plt

#Import Qiskit classes
import qiskit
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import depolarizing_error, thermal_relaxation_error

#Import the qv function
import qiskit.ignis.verification.quantum_volume as qv
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroll3qOrMore, NoiseAdaptiveLayout

from qiskit.test.mock.backends import FakeMelbourne

def get_layout(qv_circs, n_qubits, n_trials, backend, transpile_trials=None, n_desired_layouts=1):
    """
    Multiple runs of transpiler level 3
    Counting occurrences of different layouts
    Return a list of layouts, ordered by occurrence/probability for good QV

    qv_circs(int): qv circuits
    n_qubits(int): number qubits for which to find a layout
    backend(): the backend onto which the QV measurement is done
    n_trials(int): total number of trials for QV measurement
    transpile_trials(int): number of transpiler trials to search for a layout, less or equal to n_trials
    """

    n_qubit_idx = 0
    if not transpile_trials:
        transpile_trials = n_trials

    for idx, qv in enumerate(qv_circs[0]):
        if qv.n_qubits == n_qubits:
            n_qubit_idx = idx
            break

    layouts_list = []
    layouts_counts = []
    for trial in range(transpile_trials):
        pm = PassManager()
        pm.append(Unroll3qOrMore())
        pm.append(NoiseAdaptiveLayout(backend.properties))
        pm.run(qv_circs[trial][n_qubit_idx])
        layout = pm.property_set['layout']
        if layout in layouts_list:
            idx = layouts_list.index(layout)
            layouts_counts[idx] += 1
        else:
            layouts_list.append(layout)
            layouts_counts.append(1)

    # Sort the layout list based on max occurrences
    sorted_layouts = sorted(layouts_list, key=lambda x: layouts_counts[layouts_list.index(x)], reverse=True)

    return sorted_layouts[:n_desired_layouts]

"""
Plan:

1. Find all connected subsets of backend
2. Get connectivity of a chosen subset of qubits
3. Get fidelity of a chosen subset of qubits
4. Calculate quantum volume for different subsets.
5. Calculate the values for a chosen cost-function, ideally these values order
the subset like the quantum volume.
6. Change the cost-functions to improve the alignment between the order of the
cost-function values and the quantum volume.

"""

# qubit_lists: list of list of qubit subsets to generate QV circuits
qubit_lists = [[0,1,3], [0,1,3,5], [0,1,3,5,7], [0,1,3,5,7,10]]
# ntrials: Number of random circuits to create for each subset
ntrials = 50

qv_circs, qv_circs_nomeas = qv.qv_circuits(qubit_lists, ntrials)

backend = FakeMelbourne()

get_layout(qv_circs, n_qubits=4, n_trials=ntrials, backend=backend)


#pass the first trial of the nomeas through the transpiler to illustrate the circuit
qv_circs_nomeas[0] = qiskit.compiler.transpile(qv_circs_nomeas[0], basis_gates=['u1','u2','u3','cx'])

#The Unitary is an identity (with a global phase)
backend = qiskit.Aer.get_backend('statevector_simulator')
ideal_results = []
for trial in range(ntrials):
    print('Simulating trial %d'%trial)
    ideal_results.append(qiskit.execute(qv_circs_nomeas[trial], backend=backend).result())

qv_fitter = qv.QVFitter(qubit_lists=qubit_lists)
qv_fitter.add_statevectors(ideal_results)

for qubit_list in qubit_lists:
    l = len(qubit_list)
    print ('qv_depth_'+str(l)+'_trial_0:', qv_fitter._heavy_outputs['qv_depth_'+str(l)+'_trial_0'])

for qubit_list in qubit_lists:
    l = len(qubit_list)
    print ('qv_depth_'+str(l)+'_trial_0:', qv_fitter._heavy_output_prob_ideal['qv_depth_'+str(l)+'_trial_0'])

noise_model = NoiseModel()
p1Q = 0.002
p2Q = 0.02
noise_model.add_all_qubit_quantum_error(depolarizing_error(p1Q, 1), 'u2')
noise_model.add_all_qubit_quantum_error(depolarizing_error(2*p1Q, 1), 'u3')
noise_model.add_all_qubit_quantum_error(depolarizing_error(p2Q, 2), 'cx')
#noise_model = None

backend = qiskit.Aer.get_backend('qasm_simulator')
basis_gates = ['u1','u2','u3','cx'] # use U,CX for now
shots = 1024
exp_results = []
for trial in range(ntrials):
    print('Running trial %d'%trial)
    exp_results.append(qiskit.execute(qv_circs[trial], basis_gates=basis_gates, backend=backend, noise_model=noise_model, backend_options={'max_parallel_experiments': 0}).result())

qv_fitter.add_data(exp_results)
for qubit_list in qubit_lists:
    l = len(qubit_list)
    #print (qv_fitter._heavy_output_counts)
    print ('qv_depth_'+str(l)+'_trial_0:', qv_fitter._heavy_output_counts['qv_depth_'+str(l)+'_trial_0'])

plt.figure(figsize=(10, 6))
ax = plt.gca()

# Plot the essence by calling plot_rb_data
qv_fitter.plot_qv_data(ax=ax, show_plt=False)

# Add title and label
ax.set_title('Quantum Volume for up to %d Qubits \n and %d Trials'%(len(qubit_lists[-1]), ntrials), fontsize=18)

plt.show()

qv_success_list = qv_fitter.qv_success()
qv_list = qv_fitter.ydata
QV = 1
my_trial = 0
for qidx, qubit_list in enumerate(qubit_lists):
    if qv_list[my_trial][qidx]>2/3:
        if qv_success_list[qidx][my_trial]:
            print("Width/depth %d greater than 2/3 (%f) with confidence %f (successful). Quantum volume %d"%
                  (len(qubit_list),qv_list[0][qidx],qv_success_list[qidx][1],qv_fitter.quantum_volume()[qidx]))
            QV = qv_fitter.quantum_volume()[qidx]
        else:
            print("Width/depth %d greater than 2/3 (%f) with confidence %f (unsuccessful)."%
                  (len(qubit_list),qv_list[0][qidx],qv_success_list[qidx][1]))
    else:
        print("Width/depth %d less than 2/3 (unsuccessful)."%len(qubit_list))

print ("The Quantum Volume is:", QV)
