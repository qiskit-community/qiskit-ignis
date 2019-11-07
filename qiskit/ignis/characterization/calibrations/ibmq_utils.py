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

import numpy as np
from scipy.optimize import least_squares
from qiskit.pulse import SamplePulse, pulse_lib, Schedule, FrameChange
from qiskit.pulse.schedule import ParameterizedSchedule


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
                   qubits=None, cmd_def=None, system=None):
    """
    Update the cmd_def with new single qubit gate values

    Will update U2, U3

    Args:
        drag_params: list of drag params
        pi2_pulse_schedules: list of new pi/2 gate as a pulse schedule
        will use the drag_params if this is None
        qubits: list of qubits to update
        cmd_def: CmdDef object for the device
        system: PulseSpec for drives

    Returns:
        updated cmd_def

    """

    # U2 is -P1.Y90p.-P0
    # U3 is -P2.X90p.-P0.X90m.-P1

    for qind in qubits:

        drive_ch = system.qubits[qind].drive

        def param_u2_fc1(drive_ch=drive_ch, **kwargs):
            return FrameChange(phase=-kwargs['P1']+np.pi/2)(drive_ch)

        def param_u3_fc1(drive_ch=drive_ch, **kwargs):
            return FrameChange(phase=-kwargs['P2'])(drive_ch)

        if pi2_pulse_schedules is None:

            x90_pulse = pulse_lib.drag(**drag_params[qind])

            x90_pulse = Schedule(x90_pulse(drive_ch))

        else:

            x90_pulse = pi2_pulse_schedules[qind]

        pulse_dur = x90_pulse.duration

        def param_u2_fc2(drive_ch=drive_ch, pulse_dur=pulse_dur, **kwargs):
            return FrameChange(phase=-kwargs['P0']-np.pi/2)(drive_ch) << \
                                                    pulse_dur

        def param_u3_fc2(drive_ch=drive_ch, pulse_dur=pulse_dur, **kwargs):
            return FrameChange(phase=-kwargs['P0']+np.pi)(drive_ch) << \
                                                    pulse_dur

        def param_u3_fc3(drive_ch=drive_ch, pulse_dur=pulse_dur, **kwargs):
            return FrameChange(phase=-kwargs['P1']-np.pi)(drive_ch) << \
                                                    2*pulse_dur

        # add commands to schedule
        # u2
        schedule1 = ParameterizedSchedule(*[param_u2_fc1, x90_pulse,
                                            param_u2_fc2],
                                          parameters=['P0', 'P1'],
                                          name='u2_%d' % qind)

        # u3
        schedule2 = ParameterizedSchedule(*[param_u3_fc1, x90_pulse,
                                            param_u3_fc2,
                                            x90_pulse << pulse_dur,
                                            param_u3_fc3],
                                          parameters=['P0', 'P1', 'P2'],
                                          name='u3_%d' % qind)

        cmd_def.add(cmd_name='u2', qubits=[qind], schedule=schedule1)
        cmd_def.add(cmd_name='u3', qubits=[qind], schedule=schedule2)
