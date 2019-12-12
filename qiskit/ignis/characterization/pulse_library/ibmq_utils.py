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
from qiskit import pulse
from qiskit.ignis.characterization import CharacterizationError
from qiskit.pulse import InstructionScheduleMap


def get_control_channels(c_qubit: int,
                         t_qubit: int,
                         circ_inst_map: InstructionScheduleMap) \
        -> int:
    """
    A helper function to get control channel index.

    Args:
        c_qubit: index of control qubit.
        t_qubit: index of target qubit.
        circ_inst_map: command definition of target system if customized.

    Returns:
        control channel index.
    """
    try:
        cx_ref = circ_inst_map.get('cx', qubits=(c_qubit, t_qubit))
    except pulse.PulseError:
        raise CharacterizationError('Cross resonance is not defined'
                                    'for qubits %d-%d.' % (c_qubit, t_qubit))

    cx_ref = cx_ref.filter(instruction_types=[pulse.commands.PulseInstruction])
    for channel in cx_ref.channels:
        if isinstance(channel, pulse.ControlChannel):
            return channel.index

    raise CharacterizationError('No valid control channel for CR drive.')
