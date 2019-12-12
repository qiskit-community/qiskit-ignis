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
Cross Resonance pulses.
"""
from copy import deepcopy
from typing import Callable, Dict

import qiskit
import qiskit.pulse.pulse_lib as pulse_lib

var_duration = qiskit.circuit.Parameter('duration')


def cr_designer_variable_duration(sched_params: Dict[str, float],
                                  c_qubit: int,
                                  t_qubit: int,
                                  u_index: int,
                                  negative: bool = False) -> Callable:
    """
    Create parametrized cross resonance schedule with variable pulse duration.

    Args:
        sched_params: parameters to construct schedule.
        c_qubit: index of control qubit.
        t_qubit: index of target qubit.
        u_index: index of control channel.
        negative: flip cross resonance pulse if `True`.
    """
    # get valid cross resonance pulse parameters
    valid_pnames = 'cr_amp', 'ca_amp', 'sigma', 'risefall'
    valid_params = {pname: sched_params.get(pname, 0) for pname in valid_pnames}
    if negative:
        cr_amp = -valid_params.pop('cr_amp')
        ca_amp = -valid_params.pop('ca_amp')
    else:
        cr_amp = valid_params.pop('cr_amp')
        ca_amp = valid_params.pop('ca_amp')

    # parametrized cr schedule without cancellation
    def cr_var_sched(duration):
        # circuit Parameter object is currently uncastable into integer.
        # functional pulse requires duration to be integer type.
        # TODO: remove this
        if not isinstance(duration, int):
            try:
                duration = int(float(duration))
            except ValueError:
                raise ValueError('Duration is uncastable into integer.')

        valid_params['duration'] = duration
        # cross resonance pulse
        cr_params = deepcopy(valid_params)
        cr_params['amp'] = cr_amp
        cr_params['name'] = 'CR90%s_u_var' % ('m' if negative else 'p')
        # cancellation pulse
        ca_params = deepcopy(valid_params)
        ca_params['amp'] = ca_amp
        ca_params['name'] = 'CR90%s_d_var' % ('m' if negative else 'p')

        # create channels
        c_drive = qiskit.pulse.DriveChannel(c_qubit)
        t_drive = qiskit.pulse.DriveChannel(t_qubit)
        cr_drive = qiskit.pulse.ControlChannel(u_index)

        # create CR schedule
        sched = qiskit.pulse.Schedule()
        sched = sched.union((0, pulse_lib.gaussian_square(**cr_params)(cr_drive)))
        sched = sched.union((0, qiskit.pulse.Delay(duration)(c_drive)))
        if ca_amp != 0:
            sched = sched.union((0, pulse_lib.gaussian_square(**ca_params)(t_drive)))
        else:
            sched = sched.union((0, qiskit.pulse.Delay(duration)(t_drive)))
        return sched

    params = [var_duration.name]

    return qiskit.pulse.schedule.ParameterizedSchedule(cr_var_sched, parameters=params)
