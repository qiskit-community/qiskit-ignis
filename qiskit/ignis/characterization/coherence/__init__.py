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
=================================================================
Coherence module (:mod:`qiskit.ignis.characterization.coherence`)
=================================================================

.. currentmodule:: qiskit.ignis.characterization.coherence

Generation of coherence circuits
================================

.. jupyter-execute::

    import numpy as np
    from qiskit.ignis.characterization.coherence import t1_circuits, \\
                                                        t2_circuits, \\
                                                        t2star_circuits

    num_of_gates = (np.linspace(10, 300, 50)).astype(int)
    gate_time = 0.1

    # Note that it is possible to measure several qubits in parallel
    qubits = [0, 2]

    t1_circs, t1_xdata = t1_circuits(num_of_gates, gate_time, qubits)
    t2star_circs, t2star_xdata, osc_freq = t2star_circuits(num_of_gates,
                                                           gate_time,
                                                           qubits, nosc=5)
    t2echo_circs, t2echo_xdata = \\
       t2_circuits(np.floor(num_of_gates/2).astype(int),
                   gate_time, qubits)
    t2cpmg_circs, t2cpmg_xdata = \\
       t2_circuits(np.floor(num_of_gates/6).astype(int),
                   gate_time, qubits,
                   n_echos=5, phase_alt_echo=True)

Backend execution
=================

.. jupyter-execute::

    import qiskit
    from qiskit.providers.aer.noise.errors.standard_errors \\
                import thermal_relaxation_error
    from qiskit.providers.aer.noise import NoiseModel

    backend = qiskit.Aer.get_backend('qasm_simulator')
    shots = 400

    # Let the simulator simulate the following times for qubits 0 and 2:
    t_q0 = 25.0
    t_q2 = 15.0

    # Define T1 and T2 noise:
    t1_noise_model = NoiseModel()
    t1_noise_model.add_quantum_error(
    thermal_relaxation_error(t_q0, 2*t_q0, gate_time),
                            'id', [0])
    t1_noise_model.add_quantum_error(
        thermal_relaxation_error(t_q2, 2*t_q2, gate_time),
        'id', [2])

    t2_noise_model = NoiseModel()
    t2_noise_model.add_quantum_error(
    thermal_relaxation_error(np.inf, t_q0, gate_time, 0.5),
        'id', [0])
    t2_noise_model.add_quantum_error(
        thermal_relaxation_error(np.inf, t_q2, gate_time, 0.5),
        'id', [2])

    # Run the simulator
    t1_backend_result = qiskit.execute(t1_circs, backend, shots=shots,
                                       noise_model=t1_noise_model,
                                       optimization_level=0).result()
    t2star_backend_result = qiskit.execute(t2star_circs, backend, shots=shots,
                                           noise_model=t2_noise_model,
                                           optimization_level=0).result()
    t2echo_backend_result = qiskit.execute(t2echo_circs, backend, shots=shots,
                                           noise_model=t2_noise_model,
                                           optimization_level=0).result()

    # It is possible to split the circuits into multiple jobs and
    # then give the results to the fitter as a list:
    t2cpmg_backend_result1 = qiskit.execute(t2cpmg_circs[0:5], backend,
                                            shots=shots,
                                            noise_model=t2_noise_model,
                                            optimization_level=0).result()
    t2cpmg_backend_result2 = qiskit.execute(t2cpmg_circs[5:], backend,
                                            shots=shots,
                                            noise_model=t2_noise_model,
                                            optimization_level=0).result()

Analysis of results
===================

    Fitting T1

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

    Combine with new results

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

    Fitting T2-star

    .. jupyter-execute::

       from qiskit.ignis.characterization.coherence import T2StarFitter

       t2star_fit = T2StarFitter(t2star_backend_result, t2star_xdata, qubits,
                                 fit_p0=[0.5, t_q0, osc_freq, 0, 0.5],
                                 fit_bounds=([-0.5, 0, 0, -np.pi, -0.5],
                                             [1.5, 40, 2*osc_freq, np.pi, 1.5]))

       plt.figure(figsize=(15, 6))
       for i in range(2):
            ax = plt.subplot(1, 2, i+1)
            t2star_fit.plot(i, ax=ax)
       plt.show()

    Fitting T2 single echo

    .. jupyter-execute::

       from qiskit.ignis.characterization.coherence import T2Fitter

       t2echo_fit = T2Fitter(t2echo_backend_result, t2echo_xdata, qubits,
                             fit_p0=[0.5, t_q0, 0.5],
                             fit_bounds=([-0.5, 0, -0.5],
                                         [1.5, 40, 1.5]))

       print(t2echo_fit.params)

       plt.figure(figsize=(15, 6))
       for i in range(2):
           ax = plt.subplot(1, 2, i+1)
           t2echo_fit.plot(i, ax=ax)
       plt.show()

    Fitting T2 CPMG

    .. jupyter-execute::

        t2cpmg_fit = T2Fitter([t2cpmg_backend_result1, t2cpmg_backend_result2],
                               t2cpmg_xdata, qubits,
                               fit_p0=[0.5, t_q0, 0.5],
                               fit_bounds=([-0.5, 0, -0.5],
                                           [1.5, 40, 1.5]))

        plt.figure(figsize=(15, 6))
        for i in range(2):
            ax = plt.subplot(1, 2, i+1)
            t2cpmg_fit.plot(i, ax=ax)
        plt.show()
"""

from .circuits import t1_circuits, t2_circuits, t2star_circuits
from .fitters import T1Fitter, T2Fitter, T2StarFitter
