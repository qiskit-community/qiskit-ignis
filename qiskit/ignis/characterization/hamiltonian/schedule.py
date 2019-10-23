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
Schedule generation for measuring hamiltonian parametes
"""

from qiskit import pulse
from qiskit.pulse import CmdDef, Schedule, PulseError
from qiskit.providers import BaseBackend
from typing import List, Tuple, Optional, Callable
from math import pi
from qiskit.ignis.characterization import CharacterizationError


def cr_schedules(c_qubit: int,
                 t_qubit: int,
                 cmd_def: CmdDef,
                 backend: BaseBackend,
                 cr_samples: List[int],
                 cr_pulse: Callable,
                 cancellation_pulse: Optional[Callable]) -> Tuple[int, List[Schedule]]:
    """
    Generate `Schedule`s for measuring CR Hamiltonian [1].
    Measuring CR pulse sequences in all Pauli Basis with control qubit states of both 0 and 1.
    CR pulse sequence is generated for each duration in `cr_samples` to measure CR Rabi oscillation.
    Partial tomography on target qubit yields information of CR Hamiltonian.
    Generated schedules should be executed with `meas_level=2` or use discriminator to get counts.

    [1] Sheldon, S., Magesan, E., Chow, J. M. & Gambetta, J. M.
    Procedure for systematically tuning up cross-talk in the cross-resonance gate.
    Phys. Rev. A 93, 060302 (2016).

    Args:
        c_qubit: index of control qubit.
        t_qubit: index of target qubit.
        cmd_def: command definition of target system.
        backend: backend object of target system.
        cr_samples: list of cr pulse durations to create Rabi experiments.
        cr_pulse: callback function to create `SamplePulse` with argument of `duration`.
        cancellation_pulse: callback function to create `SamplePulse` with argument of `duration`.
                            cancellation pulse is omitted if it is not specified.

    Additional Information:
        Parameter bind
        --------------
        OpenPulse provides typical pulse templates as `pulse_lib`.
        `pulse_lib.gaussian_square` is usually used to create CR pulse,
        and we can pass this function as `cr_pulse` and `cancellation_pulse` arguments.
        However, this function takes four arguments of `duration`, `amp`, `sigma`, `risefall`.
        Additional parameters can be bound by using `functools.partial` in python standard library.

        ```python
            import functools

            cr_def = pulse.pulse_lib.gaussian_square
            cr_def_bound = functools.partial(cr_def, amp=0.1, sigma=10, risefall=50)
        ```

        `cr_pulse_bound` now only takes `duration`, and this can be passed to `cr_pulse`.
        `cancellation_pulse` can be generated in the same way.
    """

    # pulse channels
    try:
        cx_ref = cmd_def.get('cx', qubits=(c_qubit, t_qubit))
    except PulseError:
        raise CharacterizationError('Cross resonance is not defined for qubits %d-%d.' % (c_qubit, t_qubit))

    cx_ref = cx_ref.filter(instruction_types=[pulse.commands.PulseInstruction])
    for channel in cx_ref.channels:
        if isinstance(channel, pulse.ControlChannel):
            cr_drive = channel
            break
    else:
        raise CharacterizationError('No valid control channel to drive cross resonance.')
    t_drive = pulse.PulseChannelSpec.from_backend(backend).qubits[t_qubit].drive

    # pi pulse to flip control qubit
    flip_ctrl = cmd_def.get('x', qubits=c_qubit)
    # measurement and acquisition
    measure = cmd_def.get('measure', qubits=backend.configuration().meas_map[0])

    # schedules to convert measurement axis
    xproj_sched = cmd_def.get('u2', qubits=t_qubit, P0=0, P1=pi)
    yproj_sched = cmd_def.get('u2', qubits=t_qubit, P0=0, P1=0.5*pi)

    proj_delay = max(xproj_sched.duration, yproj_sched.duration)

    meas_basis = {
        'x': xproj_sched.insert(proj_delay, measure),
        'y': yproj_sched.insert(proj_delay, measure),
        'z': measure.shift(proj_delay)
    }

    schedules = []
    for exp_index, cr_sample in enumerate(cr_samples):

        # cross resonance pulse schedule
        try:
            cr_sched = cr_pulse(duration=cr_sample)(cr_drive)
        except TypeError:
            raise CharacterizationError('Pulse parameters should be bound except for duration.')

        # cancellation pulse schedule if defined
        if cancellation_pulse:
            try:
                cancellation_sched = cancellation_pulse(duration=cr_sample)(t_drive)
            except TypeError:
                raise CharacterizationError('Pulse parameters should be bound except for duration.')
        else:
            cancellation_sched = pulse.Schedule()

        for basis, meas_sched in meas_basis.items():
            for c_state in (0, 1):
                sched = pulse.Schedule(name='i=%d_b=%s_c=%d' % (exp_index, basis, c_state))
                # flip control qubit
                if c_state:
                    sched = sched.insert(0, flip_ctrl)
                # add cross resonance pulse
                sched = sched.insert(flip_ctrl.duration, cr_sched)
                # add cancellation pulse
                sched = sched.insert(flip_ctrl.duration, cancellation_sched)
                # add measurement
                sched = sched.insert(sched.duration, meas_sched)

                schedules.append(sched)

    return t_qubit, schedules
