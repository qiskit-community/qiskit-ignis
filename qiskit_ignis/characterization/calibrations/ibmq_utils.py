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
Utility functions for calibrating IBMQ Devices
"""

from collections import defaultdict
from scipy.optimize import least_squares
import numpy as np

from qiskit.pulse import library as pulse_lib, Schedule, Play, ShiftPhase
from qiskit.pulse.schedule import ParameterizedSchedule


def _fit_drag_func(duration, amp, sigma, beta, exp_samples):
    """
    Helper function to compare a drag pulse to samples from
    the experiment

    Args:
        duration (int): pulse duration
        amp (complex): gauss amp
        sigma (float): gauss sigma
        beta (complex): drag amp
        exp_samples (ndarray): the experiment pulse, split into real and imag

    Returns:
        ndarray: difference between the drag and experimental samples

    """

    fit_pulse = pulse_lib.drag(duration=duration, amp=amp, sigma=sigma,
                               beta=beta).samples*np.exp(-1j*np.pi/2)

    return np.concatenate((fit_pulse.real, fit_pulse.imag))-exp_samples


def get_single_q_pulse(inst_map, qubits):
    """
    Get the DRAG parameters for the single qubit pulse

    Args:
        inst_map (InstMap): Instruction schedule map object for the device
        qubits (list): list of qubits to extract the parameters

    Returns:
        list: List of dictionaries with the parameters for the DRAG

    Notes:
        Deprecated once parameterized pulses are supported
    """

    fit_params = []

    for q in qubits:

        # get u2 command
        u2_q = inst_map.get('u2', qubits=q, P0=1, P1=1)

        for _, instr in u2_q.instructions:
            if isinstance(instr, Play):

                pulse_samples = instr.pulse.samples
                pulse_dur = len(pulse_samples)
                pulse_max = np.max(np.abs(pulse_samples))
                pulse_samples = np.concatenate((pulse_samples.real,
                                                pulse_samples.imag))
                break

        # fit a drag pulse
        def fit_func_2(x):
            return _fit_drag_func(pulse_dur, x[0], x[1], x[2], pulse_samples)

        opt_result = least_squares(fit_func_2, [pulse_max, pulse_dur/4,
                                                pulse_max/5])

        fit_params.append({'amp': opt_result.x[0],
                           'duration': pulse_dur,
                           'sigma': opt_result.x[1],
                           'beta': opt_result.x[2]})

    return fit_params


def update_u_gates(drag_params, pi2_pulse_schedules=None,
                   qubits=None, inst_map=None, drives=None):
    """Update the cmd_def with new single qubit gate values

    Will update U2, U3

    Args:
        drag_params (list): list of drag params
        pi2_pulse_schedules (list): list of new pi/2 gate as a pulse schedule
                             will use the drag_params if this is None.
        qubits (list): list of qubits to update
        inst_map (InstructionScheduleMap): InstructionScheduleMap providing
            circuit instruction to schedule definitions.
        drives (list): List of drive chs
    """

    # U2 is -P1.Y90p.-P0
    # U3 is -P2.X90p.-P0.X90m.-P1

    def parametrized_fc(kw_name, phi0, chan, t_offset):
        def _parametrized_fc(**kwargs):
            return ShiftPhase(phase=-kwargs[kw_name]+phi0, channel=chan).shift(t_offset)
        return _parametrized_fc

    for qubit in qubits:

        drive_ch = drives[qubit]

        if pi2_pulse_schedules is None:
            x90_pulse = pulse_lib.drag(**drag_params[qubit])
            x90_sched = Schedule()
            x90_sched += Play(x90_pulse, drive_ch).shift(0)
        else:
            x90_sched = pi2_pulse_schedules[qubit]

        pulse_dur = x90_sched.duration

        # find channel dependency for u2
        for _u2_group in _find_channel_groups('u2', qubits=qubit, inst_map=inst_map):
            if drive_ch in _u2_group:
                break
        else:
            _u2_group = (drive_ch, )

        u2_fc1s = [parametrized_fc('P1', np.pi/2, ch, 0) for ch in _u2_group]
        u2_fc2s = [parametrized_fc('P0', -np.pi/2, ch, pulse_dur) for ch in _u2_group]

        # find channel dependency for u3
        for _u3_group in _find_channel_groups('u3', qubits=qubit, inst_map=inst_map):
            if drive_ch in _u3_group:
                break
        else:
            _u3_group = (drive_ch, )

        u3_fc1s = [parametrized_fc('P2', 0, ch, 0) for ch in _u3_group]
        u3_fc2s = [parametrized_fc('P0', -np.pi, ch, pulse_dur) for ch in _u3_group]
        u3_fc3s = [parametrized_fc('P1', np.pi, ch, 2*pulse_dur) for ch in _u3_group]

        # add commands to schedule
        # u2
        sched_components = [*u2_fc1s, x90_sched, *u2_fc2s]
        schedule1 = ParameterizedSchedule(*sched_components,
                                          parameters=['P0', 'P1'], name='u2_%d' % qubit)

        # u3
        sched_components = [*u3_fc1s, x90_sched, *u3_fc2s, x90_sched.shift(pulse_dur), *u3_fc3s]
        schedule2 = ParameterizedSchedule(*sched_components,
                                          parameters=['P0', 'P1', 'P2'], name='u3_%d' % qubit)

        inst_map.add('u2', qubits=qubit, schedule=schedule1)
        inst_map.add('u3', qubits=qubit, schedule=schedule2)


def _find_channel_groups(command, qubits, inst_map):
    """
    Extract frame dependency of control channel on drive channel.

    Args:
        command (str): name of command.
        qubits (int): target qubit index.
        inst_map (InstructionScheduleMap): InstructionScheduleMap providing
            circuit instruction to schedule definitions.
    Returns:
        list: group of channels in the same frame.
    """
    params = inst_map.get_parameters(command, qubits=qubits)
    temp_sched = inst_map.get(command, qubits=qubits,
                              **dict(zip(params, np.zeros(len(params)))))

    synced_fcs = defaultdict(list)
    for t_0, inst in temp_sched.instructions:
        if isinstance(inst, ShiftPhase):
            synced_fcs[t_0, inst.phase].extend(inst.channels)

    channel_groups = set()
    for synced_fc in synced_fcs.values():
        channel_groups.add(tuple(synced_fc))

    return list(channel_groups)
