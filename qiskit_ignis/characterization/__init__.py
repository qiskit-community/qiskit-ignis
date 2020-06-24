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


r"""
=======================================================
Characterization (:mod:`qiskit.ignis.characterization`)
=======================================================

.. currentmodule:: qiskit.ignis.characterization

Calibrations
============

.. autosummary::
   :toctree: ../stubs/

   rabi_schedules
   drag_schedules
   RabiFitter
   DragFitter
   get_single_q_pulse
   update_u_gates


Coherence
=========

Design and analyze experiments for characterizing device coherence
(e.g. T\ :sub:`1`\ , T\ :sub:`2`\ ). See the following example of T\ :sub:`1` estimation.

Generation of coherence circuits: these circuits set the qubit in
the excited state, wait different time intervals, then measure
the qubit.

.. jupyter-execute::

    import numpy as np
    from qiskit.ignis.characterization.coherence import t1_circuits

    num_of_gates = np.linspace(10, 300, 5, dtype='int')
    gate_time = 0.1

    # Note that it is possible to measure several qubits in parallel
    qubits = [0, 2]

    t1_circs, t1_xdata = t1_circuits(num_of_gates, gate_time, qubits)

Backend execution: actually performing the experiment on the device
(or simulator).

.. jupyter-execute::

    import qiskit
    from qiskit.providers.aer.noise.errors.standard_errors \
                import thermal_relaxation_error
    from qiskit.providers.aer.noise import NoiseModel

    backend = qiskit.Aer.get_backend('qasm_simulator')
    shots = 400

    # Let the simulator simulate the following times for qubits 0 and 2:
    t_q0 = 25.0
    t_q2 = 15.0

    # Define T\ :sub:`1` noise:
    t1_noise_model = NoiseModel()
    t1_noise_model.add_quantum_error(
    thermal_relaxation_error(t_q0, 2*t_q0, gate_time),
                            'id', [0])
    t1_noise_model.add_quantum_error(
        thermal_relaxation_error(t_q2, 2*t_q2, gate_time),
        'id', [2])

    # Run the simulator
    t1_backend_result = qiskit.execute(t1_circs, backend, shots=shots,
                                       noise_model=t1_noise_model,
                                       optimization_level=0).result()

Analysis of results: deduction of T\ :sub:`1`\ , based on the experiments outcomes.

.. jupyter-execute::

    import matplotlib.pyplot as plt
    from qiskit.ignis.characterization.coherence import T1Fitter

    plt.figure(figsize=(15, 6))

    t1_fit = T1Fitter(t1_backend_result, t1_xdata, qubits,
                      fit_p0=[1, t_q0, 0],
                      fit_bounds=([0, 0, -1], [2, 40, 1]))
    print(t1_fit.time())
    print(t1_fit.time_err())
    print(t1_fit.params)
    print(t1_fit.params_err)

    for i in range(2):
        ax = plt.subplot(1, 2, i+1)
        t1_fit.plot(i, ax=ax)
    plt.show()

Combine with new results:

.. jupyter-execute::

    t1_backend_result_new = qiskit.execute(t1_circs, backend,
                                           shots=shots,
                                           noise_model=t1_noise_model,
                                           optimization_level=0).result()
    t1_fit.add_data(t1_backend_result_new)

    plt.figure(figsize=(15, 6))
    for i in range(2):
        ax = plt.subplot(1, 2, i+1)
        t1_fit.plot(i, ax=ax)
    plt.show()

.. autosummary::
   :toctree: ../stubs/

   t1_circuits
   t2_circuits
   t2star_circuits
   T1Fitter
   T2Fitter
   T2StarFitter


Gates
=====

.. autosummary::
   :toctree: ../stubs/

   ampcal_1Q_circuits
   anglecal_1Q_circuits
   ampcal_cx_circuits
   anglecal_cx_circuits
   AmpCalFitter
   AngleCalFitter
   AmpCalCXFitter
   AngleCalCXFitter


Hamiltonian
===========

.. autosummary::
   :toctree: ../stubs/

   zz_circuits
   ZZFitter


Base Fitters
============

.. autosummary::
   :toctree: ../stubs/

   BaseCoherenceFitter
   BaseGateFitter

"""

from .fitters import BaseCoherenceFitter, BaseGateFitter
from .calibrations import (rabi_schedules, drag_schedules,
                           RabiFitter, DragFitter,
                           get_single_q_pulse, update_u_gates)
from .coherence import (t1_circuits, t2_circuits,
                        t2star_circuits,
                        T1Fitter, T2Fitter, T2StarFitter)
from .gates import (ampcal_1Q_circuits, anglecal_1Q_circuits,
                    ampcal_cx_circuits, anglecal_cx_circuits,
                    AmpCalFitter, AngleCalFitter,
                    AmpCalCXFitter, AngleCalCXFitter)
from .hamiltonian import (zz_circuits, ZZFitter)
