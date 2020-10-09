from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroll3qOrMore, NoiseAdaptiveLayout

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
        pm.append(NoiseAdaptiveLayout(backend.properties()))
        pm.run(qv_circs[trial][n_qubit_idx])
        layout = list(pm.property_set['layout'].get_physical_bits().keys())
        if layout in layouts_list:
            idx = layouts_list.index(layout)
            layouts_counts[idx] += 1
        else:
            layouts_list.append(layout)
            layouts_counts.append(1)

    # Sort the layout list based on max occurrences
    sorted_layouts = sorted(layouts_list, key=lambda x: layouts_counts[layouts_list.index(x)], reverse=True)

    return sorted_layouts[:n_desired_layouts]

def mock_cct(num_qubits, backend):     
    """     
    Creates a mock circuit to figure out the best qubits    
    """     
    if (num_qubits > backend.configuration().n_qubits): 
        return None     
    
    cct = QuantumCircuit(num_qubits)     
    cct_meas = QuantumCircuit(num_qubits,num_qubits)      
    comb = combinations(range(num_qubits), 2)    

    for i in comb:         
        cct.cx(i[0],i[1])         
        cct_meas.cx(i[0],i[1])           

    cct_meas.measure(range(num_qubits), range(num_qubits)) 
    
    return cct, cct_meas