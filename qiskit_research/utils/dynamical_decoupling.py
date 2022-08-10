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

from typing import Iterable, List, Union, Optional

from qiskit import QuantumCircuit, pulse
from qiskit.circuit import Gate
from qiskit.circuit.library import XGate, YGate, ZGate, IGate
from qiskit.providers.backend import Backend
from qiskit.pulse import DriveChannel
from qiskit.qasm import pi
from qiskit.transpiler import InstructionDurations
from qiskit.transpiler.instruction_durations import InstructionDurationsType
from qiskit.transpiler.passes import PadDynamicalDecoupling
from qiskit.transpiler.passes.scheduling import ALAPScheduleAnalysis
from qiskit.transpiler.basepasses import BasePass
from qiskit.transpiler.passes.scheduling.scheduling.base_scheduler import BaseScheduler
from qiskit_research.utils.gates import XmGate, XpGate, YmGate, YpGate
from qiskit_research.utils.periodic_dynamical_decoupling import (
    PeriodicDynamicalDecoupling,
)

X = XGate()
Xp = XpGate()
Xm = XmGate()
Y = YGate()
Yp = YpGate()
Ym = YmGate()
Z = ZGate()
I = IGate()


DD_SEQUENCE = {
    "X2": (X, X),
    "X2pm": (Xp, Xm),
    "XY4": (X, Y, X, Y),
    "XY4pm": (Xp, Yp, Xm, Ym),
    "XY8": (X, Y, X, Y, Y, X, Y, X),
    "XY8pm": (Xp, Yp, Xm, Ym, Ym, Xm, Yp, Xp),
}


def dynamical_decoupling_passes(
    backend: Backend, dd_str: str, scheduler: BaseScheduler = ALAPScheduleAnalysis
) -> Iterable[BasePass]:
    """Yields transpilation passes for dynamical decoupling."""
    durations = get_instruction_durations(backend)
    pulse_alignment = backend.configuration().timing_constraints["pulse_alignment"]

    sequence = DD_SEQUENCE[dd_str]
    yield scheduler(durations)
    yield PadDynamicalDecoupling(
        durations, list(sequence), pulse_alignment=pulse_alignment
    )


def periodic_dynamical_decoupling(
    backend: Backend,
    base_dd_sequence: Optional[List[Gate]] = None,
    base_spacing: Optional[List[float]] = None,
    avg_min_delay: int = None,
    max_repeats: int = 1,
    scheduler: BaseScheduler = ALAPScheduleAnalysis,
) -> Iterable[BasePass]:
    """Yields transpilation passes for periodic dynamical decoupling."""
    durations = get_instruction_durations(backend)
    pulse_alignment = backend.configuration().timing_constraints["pulse_alignment"]

    if base_dd_sequence is None:
        base_dd_sequence = [XGate(), XGate()]

    yield scheduler(durations)
    yield PeriodicDynamicalDecoupling(
        durations,
        base_dd_sequence,
        base_spacing=base_spacing,
        avg_min_delay=avg_min_delay,
        max_repeats=max_repeats,
        pulse_alignment=pulse_alignment,
    )

def walsh_dynamical_decoupling(
    backend: Backend,
    nx: int,
    ny: int, 
    nz: int,
    scheduler: BaseScheduler = ALAPScheduleAnalysis,
) -> Iterable[BasePass]:

    durations = get_instruction_durations(backend)
    pulse_alignment = backend.configuration().timing_constraints["pulse_alignment"]
    sequence, spacing = generate_walsh_sequence(nx, ny, nz)

    yield scheduler(durations)
    yield PadDynamicalDecoupling(durations, sequence, spacing=spacing, pulse_alignment=pulse_alignment)


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
    circuits: Union[QuantumCircuit, List[QuantumCircuit]],
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

def generate_walsh_sequence(nx : int, ny : int, nz : int):
    m = len(format(nx + ny + nz, 'b'))
    nx = format(nx, f'0{m}b')
    ny = format(ny, f'0{m}b')
    nz = format(nz, f'0{m}b')
    x_switchings = [False] * (2 ** m)
    y_switchings = [False] * (2 ** m)
    z_switchings = [False] * (2 ** m)
    for i in range(m):
        if nx[i] == '1':
            for j in range(2 ** m - 1, -1, -(2 ** i)):
                x_switchings[j] = not x_switchings[j]
        elif ny[i] == '1':
            for j in range(2 ** m - 1, -1, -(2 ** i)):
                y_switchings[j] = not y_switchings[j]
        elif nz[i] == '1':
            for j in range(2 ** m - 1, -1, -(2 ** i)):
                z_switchings[j] = not z_switchings[j]

    dd_sequence = [I] * (2 ** m)
    spacing = [1 / (2 ** m)]
    for i in range(2 ** m):
        switchings = (x_switchings[i], y_switchings[i], z_switchings[i])
        if switchings == (True, True, False) or switchings == (False, False, True):
            dd_sequence[i] = Z
        elif switchings == (True, False, True) or switchings == (False, True, False):
            dd_sequence[i] = Y
        elif switchings == (False, True, True) or switchings == (True, False, False):
            dd_sequence[i] = X
        
        if dd_sequence[i] == I:
            spacing[-1] += 1 / (2 ** m)
        else:
            spacing.append(1 / (2 ** m))
    if not (dd_sequence[-1] == I):
        spacing[-1] = 0
    while dd_sequence.count(I) > 0:
        dd_sequence.remove(I)
    return dd_sequence, spacing