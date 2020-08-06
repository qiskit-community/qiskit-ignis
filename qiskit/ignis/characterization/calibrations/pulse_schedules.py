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

import qiskit.pulse as pulse
import qiskit.pulse.library as pulse_lib
from qiskit.exceptions import QiskitError
from qiskit.pulse.macros import measure


def rabi_schedules(amp_list, qubits, pulse_width, pulse_sigma=None,
                   width_sigma_ratio=4, drives=None,
                   inst_map=None, meas_map=None):
    """
    Generates schedules for a rabi experiment using a Gaussian pulse

    Args:
        amp_list (list): A list of floats of amplitudes for the Gaussian
            pulse [-1,1]
        qubits (list): A list of integers for indices of the qubits to perform
            a rabi
        pulse_width (float): width of gaussian (in dt units)
        pulse_sigma (float): sigma of gaussian
        width_sigma_ratio (int): set sigma to a certain ratio of the width
            (use if pulse_sigma is None)
        drives (list): list of :class:`~qiskit.pulse.DriveChannel` objects
        inst_map (qiskit.pulse.InstructionScheduleMap): InstructionScheduleMap
            object to use
        meas_map (list): meas_map to use

    Returns:
       A list of QuantumSchedules
       xdata: a list of amps

    Raises:
        QiskitError: when necessary variables are not supplied.
    """
    xdata = amp_list

    # copy the instruction to schedule mapping
    inst_map = copy.deepcopy(inst_map)

    # Following variables should not be optional.
    # To keep function interface constant, errors are inserted here.
    # TODO: redesign this function in next release
    if inst_map is None:
        QiskitError('Instruction schedule map is not provided. ',
                    'Run `backend.defaults().instruction_schedule_map` to get inst_map.')
    if meas_map is None:
        QiskitError('Measurement map is not provided. ',
                    'Run `backend.configuration().meas_map` to get meas_map.')

    if pulse_sigma is None:
        pulse_sigma = pulse_width / width_sigma_ratio

    # Construct the schedules
    rabi_scheds = []
    for index, g_amp in enumerate(amp_list):
        rabi_pulse = pulse_lib.gaussian(duration=pulse_width,
                                        amp=g_amp,
                                        sigma=pulse_sigma,
                                        name='rabi_pulse_%d' % index)
        sched = pulse.Schedule(name='rabisched_%d_0' % index)
        for qubit in qubits:
            sched += pulse.Play(rabi_pulse, drives[qubit])
        sched += measure(qubits, inst_map=inst_map, meas_map=meas_map).shift(pulse_width)
        rabi_scheds.append(sched)

    return rabi_scheds, xdata


def drag_schedules(beta_list, qubits, pulse_amp, pulse_width,
                   pulse_sigma=None,
                   width_sigma_ratio=4, drives=None,
                   inst_map=None, meas_map=None):
    """
    Generates schedules for a drag experiment doing a pulse then
    the - pulse

    Args:
        beta_list (list of floats): List of relative amplitudes
        for the derivative pulse
        qubits (list of integers): indices of the qubits to perform a rabi
        pulse_amp (list): amp of the gaussian (list of length qubits)
        pulse_width (float): width of gaussian (in dt units)
        pulse_sigma (float): sigma of gaussian
        width_sigma_ratio (int): set sigma to a certain ratio of the width (use
            if pulse_sigma is None)
        drives (list): list of :class:`~qiskit.pulse.DriveChannel` objects
        inst_map (InstructionScheduleMap): InstructionScheduleMap object to use
        meas_map (list): meas_map to use

    Returns:
       A list of QuantumSchedules
       xdata: a list of amps

    Raises:
        QiskitError: when necessary variables are not supplied.
    """
    xdata = beta_list

    # copy the instruction to schedule mapping
    inst_map = copy.deepcopy(inst_map)

    # Following variables should not be optional.
    # To keep function interface constant, errors are inserted here.
    # TODO: redesign this function in next release
    if inst_map is None:
        QiskitError('Instruction schedule map is not provided. ',
                    'Run `backend.defaults().instruction_schedule_map` to get inst_map.')
    if meas_map is None:
        QiskitError('Measurement map is not provided. ',
                    'Run `backend.configuration().meas_map` to get meas_map.')

    if pulse_sigma is None:
        pulse_sigma = pulse_width / width_sigma_ratio

    # Construct the schedules
    drag_scheds = []
    for index, b_amp in enumerate(beta_list):
        sched = pulse.Schedule(name='dragsched_%d_0' % index)
        for qind, qubit in enumerate(qubits):
            drag_pulse_p = pulse_lib.drag(duration=pulse_width,
                                          amp=pulse_amp[qind],
                                          beta=b_amp,
                                          sigma=pulse_sigma,
                                          name='drag_pulse_%d_%d' % (index, qubit))
            drag_pulse_m = pulse_lib.drag(duration=pulse_width,
                                          amp=-pulse_amp[qind],
                                          beta=b_amp,
                                          sigma=pulse_sigma,
                                          name='drag_pulse_%d_%d' % (index, qubit))
            sched += pulse.Play(drag_pulse_p, drives[qubit])
            sched += pulse.Play(drag_pulse_m, drives[qubit])
        sched += measure(qubits, inst_map=inst_map, meas_map=meas_map).shift(2*pulse_width)
        drag_scheds.append(sched)

    return drag_scheds, xdata
