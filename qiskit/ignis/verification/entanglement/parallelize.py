

'''
In this package, there are three modules, linear.py,
parallellize.py, and analysis.py.It may be more suitable
to put these in Terra rather than Ignis. The most
important module, parallellize.py provides methods to
parallellize CNOT gates in the preparation of the GHZ State,
which results to the GHZ State having a much higher fidelity
then a normal "linear" CNOT gate preparation of the
GHZ State. Additionally, there are methods within parallelize.py
that configure different characterization tests for the
GHZ State, including Multiple Quantum Coherences (MQC),
Parity Oscillations (PO), and Quantum Tomography.
'''

from qiskit import *
from qiskit.circuit import Parameter


class BConfig:
    '''
    This class creates an object that creates a GHZ circuit
    with parallellized CNOT gates to increase fidelity
    '''

    def __init__(self, backend, indicator=True):
        self.nodes = {}
        self.backend = backend
        self.cmap = backend.configuration().coupling_map
        self.add_nodes()
        self.indicator = indicator

    def add_nodes(self):
        '''
        Adds nodes to the dictionary based on coupling map
        '''
        self.nodes = {}
        for i in range(len(self.cmap_calib())):
            self.nodes[self.cmap_calib()[i][0]] = []
        for i in range(len(self.cmap_calib())):
            self.nodes[self.cmap_calib()[i][0]].append(self.cmap_calib()[i][1])

    def cmap_calib(self):
        '''
        Only intended for public devices (doubling and reversing
        each item in coupling map), but useful to run anyway
        '''
        cmap_new = [i for i in self.cmap]
        for a in self.cmap:
            if [a[1], a[0]] not in self.cmap:
                cmap_new.append([a[1], a[0]])
        cmap_new.sort(key=lambda x: x[0])
        return cmap_new

    def get_best_node(self):
        '''
        First node with the most connections; Does not yet sort
        based on error, but that is probably not too useful
        '''

        bestNode = 0
        for i in self.nodes:
            if len(self.nodes[i]) > len(self.nodes[bestNode]):
                bestNode = i
            else:
                pass
        return bestNode

    def indicator_off(self):
        '''
        We turn off gate-based sorting of the tier_dict
        '''

        self.indicator = False

    def indicator_on(self):
        '''
        We turn off gate-based sorting of the tier_dict
        '''

        self.indicator = True

    def get_cx_error(self):
        '''
        Gets dict of relevant CNOT gate errors
        '''

        a = self.backend.properties().to_dict()['gates']
        cxerrordict = {}
        for i in a:
            if len(i['qubits']) == 1:
                continue
            if len(i['qubits']) == 2:
                b = tuple(i['qubits'])
                if b in cxerrordict.keys():
                    pass
                cxerrordict[b] = i['parameters'][0]['value']
                if (b[1], b[0]) not in cxerrordict.keys():
                    cxerrordict[(b[1], b[0])] = i['parameters'][0]['value']
                else:
                    pass

        newcxerrordict = cxerrordict

#         for a in list(cxerrordict.keys()):
#             if (a[1],a[0]) not in cxerrordict.keys():
#                 newcxerrordict[(a[1],a[0])] = cxerrordict[a]
        return newcxerrordict

    def get_cx_length(self):
        '''
        Gets dict of relevant CNOT gate lengths
        '''

        a = self.backend.properties().to_dict()['gates']
        cxlengthdict = {}
        for i in a:
            if len(i['qubits']) == 1:
                continue
            if len(i['qubits']) == 2:
                b = tuple(i['qubits'])
                if b in cxlengthdict.keys():
                    pass
                cxlengthdict[b] = i['parameters'][1]['value']
                if (b[1], b[0]) not in cxlengthdict.keys():
                    cxlengthdict[(b[1], b[0])] = i['parameters'][1]['value']
                else:
                    pass

        newcxlengthdict = cxlengthdict
#         for a in list(cxlengthdict.keys()):
#             if (a[1],a[0]) not in cxlengthdict.keys():
#                 newcxlengthdict[(a[1],a[0])] = cxlengthdict[a]

        return newcxlengthdict

    def child_sorter(self, children, parent):
        '''
        Sorts children nodes based on error/length
        '''

        newchildren = {}
        for i in children:
            newchildren[(i, parent)] = self.get_cx_error()[(i, parent)]
        ss = sorted(newchildren.items(), key=lambda x: x[1])
        newchildrenlist = [a[0][0] for a in ss]
        return newchildrenlist

    def get_tier_dict(self):
        '''
        Take the nodes of the BConfig to create a Tier Dictionary,
        where keys are the steps in the process,
        and the values are the connections following pattern of:
        [controlled qubit, NOT qubit]. Thus the
        backend's GHZ state is parallelized.
        '''

        tier = {}
        tierDM = {}
        length = len(self.nodes.keys())
        trashlist = []
        tier[0] = (self.get_best_node())
        tierDM[self.get_best_node()] = 0

        trashlist.append(self.get_best_node())
        parent = self.get_best_node()

        for x in range(length):
            tier[x] = [[]]

        parentlist = []
        parentlist.append(parent)
        ii = 0
        while True:
            totalchildren = []
            for parent in parentlist:
                children = self.nodes[parent]
                if self.indicator:
                    children = self.child_sorter(children, parent)
                totalchildren += children
                children = [a for a in children if a not in trashlist]
                j = tierDM[parent]
                for i, _ in enumerate(children):
                    tier[j] += [[parent, children[i]]]
                    tierDM[children[i]] = j+1
                    j += 1
                    trashlist.append(children[i])
                parentlist = totalchildren

            if len(trashlist) == length:
                break
            ii += 1
            if ii > 50:
                break

        newtier = {}
        for a in tier:
            if [] in tier[a]:
                tier[a].remove([])
        for a in tier:
            if tier[a] != []:
                newtier[a] = tier[a]
        tier = newtier

        return tier

    def get_ghz_layout(self, n, transpiled=True, barriered=True):
        '''
        Feeds the Tier Dict of the backend to create a basic
        qiskit GHZ circuit with no mesaurement;
        Can also toggle on/off transpilation,
        which is useful for Tomography. Also,
        the barriered argument barriers each "step" of CNOT gates
        '''

        tierdict = self.get_tier_dict()
        q = QuantumRegister(n, 'q')
        circ = QuantumCircuit(q)
        circ.h(q[0])
        trashlist = []
        initial_layout = {}
        for a in tierdict:
            for aa in tierdict[a]:
                if aa[0] not in trashlist:
                    trashlist.append(aa[0])
                    trashindex = trashlist.index(aa[0])
                    initial_layout[q[trashindex]] = aa[0]
                else:
                    pass
                if aa[1] not in trashlist:
                    trashlist.append(aa[1])
                    trashindex = trashlist.index(aa[1])
                    initial_layout[q[trashindex]] = aa[1]
                else:
                    pass
                circ.cx(q[trashlist.index(aa[0])],
                        q[trashlist.index(aa[1])])

                if len(trashlist) == n:
                    break
            if barriered:
                circ.barrier()
            if len(trashlist) == n:
                break

        if transpiled:
            circ = qiskit.compiler.transpile(circ, backend=self.backend,
                                             initial_layout=initial_layout)

        return circ, initial_layout

    def get_ghz_measurement(self, n, extent):
        '''
        Creates a measurement circuit that can toggle
        between measuring the control qubit
        or measuring all qubits. The default is
        measurement of all qubits.
        '''

        q = QuantumRegister(n, 'q')
        if extent == 'one':
            cla = ClassicalRegister(1, 'c')
            meas = QuantumCircuit(q, cla)
            meas.barrier()
            meas.measure(q[0], cla)
            return meas
        if extent == 'full':
            cla = ClassicalRegister(n, 'c')
            meas = QuantumCircuit(q, cla)
            meas.barrier()
            meas.measure(q, cla)
            return meas
        if extent != 'one' or extent != 'full':
            raise Exception("Extent arguments must be 'full' or 'one'")
        return None

    def get_ghz_mqc(self, n, delta, extent='full'):
        '''
        Get MQC circuit
        '''

        circ, initial_layout = self.get_ghz_layout(n)
        q = QuantumRegister(n, 'q')
        rotate = QuantumCircuit(q)

        rotate.barrier()
        rotate.u1(delta, q)
        rotate.barrier()
        rotate.x(q)
        rotate.barrier()
        rotate = qiskit.compiler.transpile(rotate,
                                           backend=self.backend,
                                           initial_layout=initial_layout)

#         cla = ClassicalRegister(n,'c')
#         meas = QuantumCircuit(q,cla)
#         meas.barrier()
#         meas.measure(q,cla)

        meas = self.get_ghz_measurement(n, extent)
        meas = qiskit.compiler.transpile(meas,
                                         backend=self.backend,
                                         initial_layout=initial_layout)

        new_circ = circ + rotate + circ.inverse() + meas

        return new_circ, initial_layout

    def get_ghz_mqc_para(self, n, extent='full'):
        '''
        Get a parametrized MQC circuit.
        Remember that get_counts() method accepts
        an index now, not a circuit
        '''

        circ, initial_layout = self.get_ghz_layout(n)
        q = QuantumRegister(n, 'q')
        rotate = QuantumCircuit(q)

        delta = Parameter('t')

        rotate.barrier()
        rotate.u1(delta, q)
        rotate.barrier()
        rotate.x(q)
        rotate.barrier()
        rotate = qiskit.compiler.transpile(rotate,
                                           backend=self.backend,
                                           initial_layout=initial_layout)


#         cla = ClassicalRegister(n,'c')
#         meas = QuantumCircuit(q,cla)
#         meas.barrier()
#         meas.measure(q,cla)

        meas = self.get_ghz_measurement(n, extent)
        meas = qiskit.compiler.transpile(meas,
                                         backend=self.backend,
                                         initial_layout=initial_layout)

        new_circ = circ + rotate + circ.inverse() + meas

        return new_circ, delta, initial_layout

    def get_ghz_po(self, n, delta, extent='full'):
        '''
        Get Parity Oscillation circuit
        '''

        if extent != 'full':
            raise Exception("Only 'full' argument can be accepted",
                            " for extent in Parity Oscillation circuit")
        circ, initial_layout = self.get_ghz_layout(n)
        q = QuantumRegister(n, 'q')
        rotate = QuantumCircuit(q)

        rotate.barrier()
        rotate.u2(delta, -delta, q)
        rotate.barrier()
        rotate = qiskit.compiler.transpile(rotate,
                                           backend=self.backend,
                                           initial_layout=initial_layout)

#         cla = ClassicalRegister(n,'c')
#         meas = QuantumCircuit(q,cla)
#         meas.barrier()
#         meas.measure(q,cla)

        meas = self.get_ghz_measurement(n, extent)
        meas = qiskit.compiler.transpile(meas,
                                         backend=self.backend,
                                         initial_layout=initial_layout)

        new_circ = circ + rotate + meas

        return new_circ, initial_layout

    def get_ghz_po_para(self, n, extent='full'):
        '''
        Get a parametrized PO circuit. Remember that get_counts()
        method accepts an index now, not a circuit.
        The two phase parameters are a quirk of the Parameter module
        '''

        if extent != 'full':
            raise Exception("Only 'full' argument can be accepted",
                            " for extent in Parity Oscillation circuit")
        circ, initial_layout = self.get_ghz_layout(n)
        q = QuantumRegister(n, 'q')
        rotate = QuantumCircuit(q)

        delta = Parameter('t')
        deltaneg = Parameter('-t')

        rotate.barrier()
        rotate.u2(delta, deltaneg, q)
        rotate.barrier()
        rotate = qiskit.compiler.transpile(rotate,
                                           backend=self.backend,
                                           initial_layout=initial_layout)


#         cla = ClassicalRegister(n,'c')
#         meas = QuantumCircuit(q,cla)
#         meas.barrier()
#         meas.measure(q,cla)

        meas = self.get_ghz_measurement(n, extent)
        meas = qiskit.compiler.transpile(meas,
                                         backend=self.backend,
                                         initial_layout=initial_layout)
        new_circ = circ + rotate + meas

        return new_circ, [delta, deltaneg], initial_layout

    def get_ghz_simple(self, n, extent='full'):
        '''
        Get simple GHZ circuit with measurement
        '''

        circ, initial_layout = self.get_ghz_layout(n)
        q = QuantumRegister(n, 'q')

#         cla = ClassicalRegister(n,'c')
#         meas = QuantumCircuit(q,cla)
#         meas.barrier()
#         meas.measure(q,cla)

        meas = self.get_ghz_measurement(n, extent)
        meas = qiskit.compiler.transpile(meas,
                                         backend=self.backend,
                                         initial_layout=initial_layout)
        new_circ = circ + meas

        return new_circ, q, initial_layout
