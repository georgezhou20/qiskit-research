# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Dynamical decoupling."""

from __future__ import annotations

from typing import Iterable, Union

from qiskit import QuantumCircuit, pulse
from qiskit.circuit.library import XGate, YGate
from qiskit.providers.backend import Backend
from qiskit.pulse import DriveChannel
from qiskit.qasm import pi
from qiskit.transpiler import InstructionDurations, CouplingMap
from qiskit.transpiler.basepasses import BasePass
from qiskit.transpiler.instruction_durations import InstructionDurationsType
from qiskit.transpiler.passes import (
    TimeUnitConversion, 
    ALAPScheduleAnalysis,
    PadDelay,
    ConstrainedReschedule,
)

from qiskit_research.utils.combine_adjacent_delays import CombineAdjacentDelays
from qiskit_research.utils.dynamical_decoupling_multi import DynamicalDecouplingMulti
from qiskit_research.utils.dynamical_decoupling_single import DynamicalDecoupling
# from qiskit_research.utils.dynamical_decoupling_multi import DynamicalDecouplingMulti
from qiskit_research.utils.gates import XmGate, XpGate, YmGate, YpGate
# from qiskit_research.utils.combine_adjacent_delays import CombineAdjacentDelays
# from qiskit_research.utils.dynamical_decoupling_new import DynamicalDecoupling




import numpy as np

X = XGate()
Xp = XpGate()
Xm = XmGate()
Y = YGate()
Yp = YpGate()
Ym = YmGate()


DD_SEQUENCE = {
    "X2": (X, X),
    "X2pm": (Xp, Xm),
    "XY4": (X, Y, X, Y),
    "XY4pm": (Xp, Yp, Xm, Ym),
    "XY8": (X, Y, X, Y, Y, X, Y, X),
    "XY8pm": (Xp, Yp, Xm, Ym, Ym, Xm, Yp, Xp),
}


def dynamical_decoupling_passes(
    backend, dd_str: str, num_dd_passes: int, uhrig_spacing: bool, concat_layers: int, scheduler: BasePass = ALAPScheduleAnalysis
) -> Iterable[BasePass]:
    """Yields transpilation passes for dynamical decoupling."""
    durations = get_instruction_durations(backend)
    pulse_alignment = backend.configuration().timing_constraints['pulse_alignment']
    acquire_alignment = backend.configuration().timing_constraints['acquire_alignment']
    coupling_map = CouplingMap(backend.configuration().coupling_map)
    # skip_threshold = [0.3, 0.7]

    yield TimeUnitConversion(durations)
    yield scheduler(durations)
    yield ConstrainedReschedule(acquire_alignment=acquire_alignment, pulse_alignment=pulse_alignment)
    yield PadDelay()
    yield CombineAdjacentDelays(coupling_map)
    sequence2 = generate_concatenated_dd_seqence(dd_str, concat_layers)
    sequence1 = generate_concatenated_dd_seqence('XY8pm', concat_layers)
    dd_spacing = None
    
    # if uhrig_spacing:
    #     dd_spacing = []
    #     n = len(sequence) 
    #     for i in range(n):
    #         spacing = np.sin(np.pi * (i + 1) / (2 * n + 2)) ** 2
    #         dd_spacing.append(spacing - sum(dd_spacing))
    #     dd_spacing.append(1 - sum(dd_spacing))
    
    # yield DynamicalDecoupling(durations=durations, dd_sequence=DD_SEQUENCE['XY8pm'], spacing=dd_spacing, pulse_alignment=pulse_alignment,skip_threshold=skip_threshold)
    # yield DynamicalDecoupling(durations=durations, dd_sequence=DD_SEQUENCE['XY4pm'], spacing=dd_spacing, pulse_alignment=pulse_alignment,skip_threshold=skip_threshold)
    # yield DynamicalDecoupling(durations=durations, dd_sequence=sequence1, spacing=dd_spacing, pulse_alignment=pulse_alignment, skip_threshold=[0.2, 0.8])
    yield DynamicalDecoupling(durations=durations, dd_sequence=sequence2, spacing=dd_spacing, pulse_alignment=pulse_alignment)

    yield DynamicalDecouplingMulti(durations=durations, coupling_map=coupling_map, pulse_alignment=pulse_alignment)



# TODO this should take instruction schedule map instead of backend
def get_instruction_durations(backend: Backend) -> InstructionDurations:
    """
    Retrieves gate timing information for the backend from the instruction
    schedule map, and returns the type InstructionDurations for use by
    Qiskit's scheduler (i.e., ALAP) and DynamicalDecoupling passes.

    This method relies on IBM backend knowledge such as

      - all single qubit gates durations are the same
      - the 'x' gate, used for echoed cross resonance, is also the basis for
        all othe dynamical decoupling gates (currently)
    """
    inst_durs: InstructionDurationsType = []
    inst_sched_map = backend.defaults().instruction_schedule_map
    num_qubits = backend.configuration().num_qubits

    # single qubit gates
    for qubit in range(num_qubits):
        for inst_str in inst_sched_map.qubit_instructions(qubits=[qubit]):
            inst = inst_sched_map.get(inst_str, qubits=[qubit])
            inst_durs.append((inst_str, qubit, inst.duration))

            # create DD pulses from CR echo 'x' pulse
            if inst_str == "x":
                for new_gate in ["xp", "xm", "y", "yp", "ym"]:
                    inst_durs.append((new_gate, qubit, inst.duration))

    # two qubit gates
    for qc in range(num_qubits):
        for qt in range(num_qubits):
            for inst_str in inst_sched_map.qubit_instructions(qubits=[qc, qt]):
                inst = inst_sched_map.get(inst_str, qubits=[qc, qt])
                inst_durs.append((inst_str, [qc, qt], inst.duration))

    return InstructionDurations(inst_durs)


# TODO refactor this as a CalibrationBuilder transpilation pass
def add_pulse_calibrations(
    circuits: Union[QuantumCircuit, list[QuantumCircuit]],
    backend: Backend,
) -> None:
    """Add pulse calibrations for custom gates to circuits in-place."""
    inst_sched_map = backend.defaults().instruction_schedule_map
    num_qubits = backend.configuration().num_qubits

    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]

    for qubit in range(num_qubits):
        with pulse.build(f"xp gate for qubit {qubit}") as sched:
            # def of XpGate() in terms of XGate()
            x_sched = inst_sched_map.get("x", qubits=[qubit])
            pulse.call(x_sched)

            # add calibrations to circuits
            for circ in circuits:
                circ.add_calibration("xp", [qubit], sched)

        with pulse.build(f"xm gate for qubit {qubit}") as sched:
            # def of XmGate() in terms of XGate() and amplitude inversion
            x_sched = inst_sched_map.get("x", qubits=[qubit])
            x_pulse = x_sched.instructions[0][1].pulse
            # HACK is there a better way?
            x_pulse._amp = -x_pulse.amp  # pylint: disable=protected-access
            pulse.play(x_pulse, DriveChannel(qubit))

            # add calibrations to circuits
            for circ in circuits:
                circ.add_calibration("xm", [qubit], sched)

        with pulse.build(f"y gate for qubit {qubit}") as sched:
            # def of YGate() in terms of XGate() and phase_offset
            with pulse.phase_offset(pi / 2, DriveChannel(qubit)):
                x_sched = inst_sched_map.get("x", qubits=[qubit])
                pulse.call(x_sched)

            # add calibrations to circuits
            for circ in circuits:
                circ.add_calibration("y", [qubit], sched)

        with pulse.build(f"yp gate for qubit {qubit}") as sched:
            # def of YpGate() in terms of XGate() and phase_offset
            with pulse.phase_offset(pi / 2, DriveChannel(qubit)):
                x_sched = inst_sched_map.get("x", qubits=[qubit])
                pulse.call(x_sched)

            # add calibrations to circuits
            for circ in circuits:
                circ.add_calibration("yp", [qubit], sched)

        with pulse.build(f"ym gate for qubit {qubit}") as sched:
            # def of YGate() in terms of XGate() and phase_offset
            with pulse.phase_offset(-pi / 2, DriveChannel(qubit)):
                x_sched = inst_sched_map.get("x", qubits=[qubit])
                x_pulse = x_sched.instructions[0][1].pulse
                # HACK is there a better way?
                x_pulse._amp = -x_pulse.amp  # pylint:disable=protected-access
                pulse.play(x_pulse, DriveChannel(qubit))

            # add calibrations to circuits
            for circ in circuits:
                circ.add_calibration("ym", [qubit], sched)

def generate_concatenated_dd_seqence(dd_str : str, concat_layers: int):
    sequence = []
    for i in range(concat_layers):
        sequence = _generate_concatenated_dd_sequence(dd_str, sequence)
    return sequence

def _generate_concatenated_dd_sequence(dd_str: str, sequence):
    concat_sequence = []
    for gate in DD_SEQUENCE[dd_str]:
        concat_sequence.extend(sequence)
        concat_sequence.append(gate)
    return concat_sequence
