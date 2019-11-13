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
import numpy as np
from scipy.optimize import least_squares
from qiskit.pulse import SamplePulse, pulse_lib, Schedule, FrameChange
from qiskit.pulse.schedule import ParameterizedSchedule
from qiskit.pulse.commands import FrameChangeInstruction


def _fit_drag_func(duration, amp, sigma, beta, exp_samples):
    """
    Helper function to compare a drag pulse to samples from
    the experiment

    Args:
        duration: pulse duration
        amp: gauss amp
        sigma: gauss sigma
        beta: drag amp
        exp_samples: the experiment pulse, split into real and imag

    Returns:
        difference between the drag and experimental samples

    """

    fit_pulse = pulse_lib.drag(duration=duration, amp=amp, sigma=sigma,
                               beta=beta).samples*np.exp(-1j*np.pi/2)

    return np.concatenate((fit_pulse.real, fit_pulse.imag))-exp_samples


def get_single_q_pulse(cmd_def, qubits):
    """
    Get the DRAG parameters for the single qubit pulse

    Args:
        cmd_def: CmdDef object for the device
        qubits: list of qubits to extract the parameters

    Returns:
        List of dictionaries with the parameters for the DRAG

    TODO:
        Deprecated once parameterized pulses are supported
    """

    fit_params = []

    for q in qubits:

        # get u2 command
        u2_q = cmd_def.get('u2', qubits=q, P0=1, P1=1)

        for instr in u2_q.instructions:
            if isinstance(instr[1].command, SamplePulse):

                pulse_samples = instr[1].command.samples
                pulse_dur = len(pulse_samples)
                pulse_max = np.max(np.abs(pulse_samples))
                pulse_samples = np.concatenate((pulse_samples.real,
                                                pulse_samples.imag))
                break

        # fit a drag pulse
        def fit_func_2(x):
            return _fit_drag_func(pulse_dur, x[0],
                                  x[1], x[2], pulse_samples)

        opt_result = least_squares(fit_func_2, [pulse_max, pulse_dur/4,
                                                pulse_max/5])

        fit_params.append({'amp': opt_result.x[0],
                           'duration': pulse_dur,
                           'sigma': opt_result.x[1],
                           'beta': opt_result.x[2]})

    return fit_params


def update_u_gates(drag_params, pi2_pulse_schedules=None,
                   qubits=None, cmd_def=None, drives=None):
    """
    Update the cmd_def with new single qubit gate values
    Will update U2, U3
    Args:
        drag_params: list of drag params
        pi2_pulse_schedules: list of new pi/2 gate as a pulse schedule
        will use the drag_params if this is None
        qubits: list of qubits to update
        cmd_def: CmdDef object for the device
        drives: List of drive chs
    Returns:
        updated cmd_def
    """

    # U2 is -P1.Y90p.-P0
    # U3 is -P2.X90p.-P0.X90m.-P1

    def parametrized_fc(kw_name, phi0, chan, t_offset):
        def _parametrized_fc(**kwargs):
            return FrameChange(phase=-kwargs[kw_name]+phi0)(chan) << t_offset
        return _parametrized_fc

    for qubit in qubits:

        drive_ch = drives[qubit]

        if pi2_pulse_schedules is None:
            x90_pulse = pulse_lib.drag(**drag_params[qubit])
            x90_pulse = Schedule(x90_pulse(drive_ch))
        else:
            x90_pulse = pi2_pulse_schedules[qubit]

        pulse_dur = x90_pulse.duration

        # find channel dependency for u2
        for _u2_group in _find_channel_groups('u2',
                                              qubits=qubit,
                                              cmd_def=cmd_def):
            if drive_ch in _u2_group:
                break
        else:
            _u2_group = (drive_ch, )

        u2_fc1s = [parametrized_fc('P1', -np.pi/2, ch, 0)
                   for ch in _u2_group]
        u2_fc2s = [parametrized_fc('P0', np.pi/2, ch, pulse_dur)
                   for ch in _u2_group]

        # find channel dependency for u2
        for _u3_group in _find_channel_groups('u3',
                                              qubits=qubit,
                                              cmd_def=cmd_def):
            if drive_ch in _u3_group:
                break
        else:
            _u3_group = (drive_ch, )

        u3_fc1s = [parametrized_fc('P2', 0, ch, 0) for ch in _u3_group]
        u3_fc2s = [parametrized_fc('P0', np.pi, ch, pulse_dur)
                   for ch in _u3_group]
        u3_fc3s = [parametrized_fc('P1', -np.pi, ch, 2*pulse_dur)
                   for ch in _u3_group]

        # add commands to schedule
        # u2
        schedule1 = ParameterizedSchedule(*[*u2_fc1s,
                                            x90_pulse,
                                            *u2_fc2s],
                                          parameters=['P0', 'P1'],
                                          name='u2_%d' % qubit)

        # u3
        schedule2 = ParameterizedSchedule(*[*u3_fc1s,
                                            x90_pulse,
                                            *u3_fc2s,
                                            x90_pulse << pulse_dur,
                                            *u3_fc3s],
                                          parameters=['P0', 'P1', 'P2'],
                                          name='u3_%d' % qubit)

        cmd_def.add(cmd_name='u2', qubits=qubit, schedule=schedule1)
        cmd_def.add(cmd_name='u3', qubits=qubit, schedule=schedule2)


def _find_channel_groups(command, qubits, cmd_def):
    """
    Extract frame dependency of control channel on drive channel.

    Args:
        command: name of command.
        qubits: target qubit index.
    Returns:
        channel_groups: group of channels in the same frame.
    """
    params = cmd_def.get_parameters(command, qubits=qubits)
    temp_sched = cmd_def.get(command, qubits=qubits,
                             **dict(zip(params, np.zeros(len(params)))))

    synced_fcs = defaultdict(list)
    for t0, inst in temp_sched.instructions:
        if isinstance(inst, FrameChangeInstruction):
            synced_fcs[t0, inst.command.phase].extend(inst.channels)

    channel_groups = set()
    for synced_fc in synced_fcs.values():
        channel_groups.add(tuple(synced_fc))

    return list(channel_groups)
