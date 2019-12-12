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
Pulse Schedule Generation for calibration experiments
"""

import copy
import qiskit
import qiskit.pulse as pulse
import qiskit.pulse.pulse_lib as pulse_lib
from qiskit.circuit import Gate
from qiskit.scheduler import schedule_circuit, ScheduleConfig


def rabi_schedules(amp_list, qubits, pulse_width, pulse_sigma=None,
                   width_sigma_ratio=4, drives=None, cmd_def=None,
                   meas_map=None):
    """
    Generates schedules for a rabi experiment using a Gaussian pulse

    Args:
        amp_list (list of floats): List of amplitudes for the Gaussian
        pulse [-1,1]
        qubits (list of integers): indices of the qubits to perform a rabi
        pulse_width: width of gaussian (in dt units)
        pulse_sigma: sigma of gaussian
        width_sigma_ratio: set sigma to a certain ratio of the width (use if
        pulse_sigma is None)
        drives: list of DriveChannel objects
        cmd_def: CmdDef object to use
        meas_map: meas_map to use

    Returns:
       A list of QuantumSchedules
       xdata: a list of amps
    """

    xdata = amp_list

    # copy the command def
    cmd_def = copy.deepcopy(cmd_def)

    if pulse_sigma is None:
        pulse_sigma = pulse_width / width_sigma_ratio

    # Construct the circuits
    qr = qiskit.QuantumRegister(max(qubits) + 1)
    cr = qiskit.ClassicalRegister(len(qubits))

    circuits = []

    for circ_index, g_amp in enumerate(amp_list):

        circ = qiskit.QuantumCircuit(qr, cr)
        circ.name = 'rabicircuit_%d_0' % circ_index

        rabi_pulse = pulse_lib.gaussian(duration=pulse_width,
                                        amp=g_amp,
                                        sigma=pulse_sigma,
                                        name='rabi_pulse_%d' % circ_index)

        rabi_gate = Gate(name='rabi_%d' % circ_index, num_qubits=1, params=[])

        for _, qubit in enumerate(qubits):

            # add commands to schedule
            schedule = pulse.Schedule(name='rabi_pulse_%f_%d' % (g_amp,
                                                                 qubit))

            schedule += rabi_pulse(drives[qubit])

            # append this schedule to the cmd_def
            cmd_def.add('rabi_%d' % circ_index, qubits=[qubit],
                        schedule=schedule)

            circ.append(rabi_gate, [qr[qubit]])

        for qind, qubit in enumerate(qubits):
            circ.measure(qr[qubit], cr[qind])

        circuits.append(circ)

        # schedule
        schedule_config = ScheduleConfig(cmd_def, meas_map)
        rabi_sched = [schedule_circuit(qcirc,
                                       schedule_config)
                      for qcirc in circuits]

    return rabi_sched, xdata


def drag_schedules(beta_list, qubits, pulse_amp, pulse_width,
                   pulse_sigma=None,
                   width_sigma_ratio=4, drives=None, cmd_def=None,
                   meas_map=None):
    """
    Generates schedules for a drag experiment doing a pulse then
    the - pulse

    Args:
        beta_list (list of floats): List of relative amplitudes
        for the derivative pulse
        qubits (list of integers): indices of the qubits to perform a rabi
        pulse_amp: amp of the gaussian (list of length qubits)
        pulse_width: width of gaussian (in dt units)
        pulse_sigma: sigma of gaussian
        width_sigma_ratio: set sigma to a certain ratio of the width (use if
        pulse_sigma is None)
        drives: list of DriveChannel objects
        cmd_def: CmdDef object to use
        meas_map: meas_map to use

    Returns:
       A list of QuantumSchedules
       xdata: a list of amps
    """

    xdata = beta_list

    # copy the command def
    cmd_def = copy.deepcopy(cmd_def)

    if pulse_sigma is None:
        pulse_sigma = pulse_width / width_sigma_ratio

    # Construct the circuits
    qr = qiskit.QuantumRegister(max(qubits) + 1)
    cr = qiskit.ClassicalRegister(len(qubits))

    circuits = []

    for circ_index, b_amp in enumerate(beta_list):

        circ = qiskit.QuantumCircuit(qr, cr)
        circ.name = 'dragcircuit_%d_0' % circ_index

        for qind, qubit in enumerate(qubits):

            # positive drag pulse
            drag_pulse = pulse_lib.drag(duration=pulse_width,
                                        amp=pulse_amp[qind],
                                        beta=b_amp,
                                        sigma=pulse_sigma,
                                        name='drag_pulse_%d_%d' % (circ_index,
                                                                   qubit))

            drag_gate = Gate(name='drag_%d_%d' % (circ_index, qubit),
                             num_qubits=1, params=[])

            # add commands to schedule
            schedule = pulse.Schedule(name='drag_pulse_%f_%d' % (b_amp,
                                                                 qubit))

            schedule += drag_pulse(drives[qubit])

            # append this schedule to the cmd_def
            cmd_def.add('drag_%d_%d' % (circ_index, qubit), qubits=[qubit],
                        schedule=schedule)

            # negative pulse
            drag_pulse2 = pulse_lib.drag(duration=pulse_width,
                                         amp=-1*pulse_amp[qind],
                                         beta=b_amp,
                                         sigma=pulse_sigma,
                                         name='drag_pulse_%d_%d' % (circ_index,
                                                                    qubit))

            drag_gate2 = Gate(name='drag2_%d_%d' % (circ_index, qubit),
                              num_qubits=1, params=[])

            # add commands to schedule
            schedule2 = pulse.Schedule(name='drag_pulse2_%f_%d' % (b_amp,
                                                                   qubit))

            schedule2 += drag_pulse2(drives[qubit])

            # append this schedule to the cmd_def
            cmd_def.add('drag2_%d_%d' % (circ_index, qubit), qubits=[qubit],
                        schedule=schedule2)

            circ.append(drag_gate, [qr[qubit]])
            # circ.u1(np.pi, [qr[qubit]])
            circ.append(drag_gate2, [qr[qubit]])

        for qind, qubit in enumerate(qubits):
            circ.measure(qr[qubit], cr[qind])

        circuits.append(circ)

        # schedule
        schedule_config = ScheduleConfig(cmd_def, meas_map)
        drag_sched = [schedule_circuit(qcirc,
                                       schedule_config)
                      for qcirc in circuits]

    return drag_sched, xdata
