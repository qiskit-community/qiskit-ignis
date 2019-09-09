'''
In this package, there are three modules, linear.py,
parallellize.py, and analysis.py. Once Terra completes design
for circuit repository, may be more suitable to move it there. 
The most important module, parallellize.py provides methods to
parallellize CNOT gates in the preparation of the GHZ State,
which results to the GHZ State having a much higher fidelity
then a normal "linear" CNOT gate preparation of the
GHZ State. Additionally, there are methods within parallelize.py
that configure different characterization tests for the
GHZ State, including Multiple Quantum Coherences (MQC),
Parity Oscillations (PO), and Quantum Tomography.
Nevertheless, the module linear.py provides the linear
preparation analogous of parallelize.py. The module analysis.py
provides several miscellaneous tools for analysis of the
GHZ State (most notably, Fourier Analysis)
'''
