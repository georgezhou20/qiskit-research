# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Dynamical Decoupling insertion pass on multiple qubits."""

import itertools
import warnings

import numpy as np
import retworkx as rx

from qiskit.circuit.delay import Delay
from qiskit.circuit.reset import Reset
from qiskit.circuit.library.standard_gates import IGate, XGate, RZGate
from qiskit.dagcircuit import DAGOpNode, DAGInNode
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.quantum_info.synthesis import OneQubitEulerDecomposer
from qiskit.transpiler.passes.optimization import Optimize1qGates
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError


class DynamicalDecouplingMulti(TransformationPass):
    """Dynamical decoupling insertion pass on multi-qubit delays.

    """

    def __init__(self, durations, coupling_map,
                 skip_reset_qubits=True, pulse_alignment=1, skip_threshold=1):
        """Dynamical decoupling initializer.

        Args:
            durations (InstructionDurations): Durations of instructions to be
                used in scheduling.
            coupling_map (CouplingMap): qubit couplings which influences the pattern
                of multi-qubit DD.
            skip_reset_qubits (bool): if True, does not insert DD on idle
                periods that immediately follow initialized/reset qubits (as
                qubits in the ground state are less susceptile to decoherence).
            pulse_alignment: The hardware constraints for gate timing allocation.
                This is usually provided om ``backend.configuration().timing_constraints``.
                If provided, the delay length, i.e. ``spacing``, is implicitly adjusted to
                satisfy this constraint.
            skip_threshold (float): a number in range [0, 1]. If the DD sequence
                amounts to more than this fraction of the idle window, we skip.
                Default: 1 (i.e. always insert, even if filling up the window).
        """
        super().__init__()
        self._durations = durations
        self._coupling_map = coupling_map
        self._skip_reset_qubits = skip_reset_qubits
        self._alignment = pulse_alignment
        self._dd_sequence = [XGate(), RZGate(np.pi), XGate(), RZGate(-np.pi)]
        self._spacing_odd = [1/2, 1/2, 0, 0, 0]
        self._spacing_even = [1/4, 1/2, 0, 0, 1/4]
        self._addition_odd = [1, 1, 0, 0, 0]
        self._addition_even = [0, 1, 0, 0, 1]
        self._skip_threshold = skip_threshold

    def run(self, dag):
        """Run the DynamicalDecoupling pass on dag.

        Args:
            dag (DAGCircuit): a scheduled DAG.

        Returns:
            DAGCircuit: equivalent circuit with delays interrupted by DD,
                where possible.

        Raises:
            TranspilerError: if the circuit is not mapped on physical qubits.
        """
        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("DD runs on physical circuits only.")

        if dag.duration is None:
            raise TranspilerError("DD runs after circuit is scheduled.")

        num_pulses = len(self._dd_sequence)
        sequence_gphase = 0
        if num_pulses != 1:
            if num_pulses % 2 != 0:
                raise TranspilerError("DD sequence must contain an even number of gates (or 1).")
            noop = np.eye(2)
            for gate in self._dd_sequence:
                noop = noop.dot(gate.to_matrix())
            if not matrix_equal(noop, IGate().to_matrix(), ignore_phase=True):
                raise TranspilerError("The DD sequence does not make an identity operation.")
            sequence_gphase = np.angle(noop[0][0])

        new_dag = dag.copy_empty_like()

        qubit_index_map = {qubit: index for index, qubit in enumerate(new_dag.qubits)}
        index_sequence_duration_map = {}
        for qubit in new_dag.qubits:
            physical_qubit = qubit_index_map[qubit]
            dd_sequence_duration = 0
            for gate in self._dd_sequence:
                gate.duration = self._durations.get(gate, physical_qubit)
                dd_sequence_duration += gate.duration
            index_sequence_duration_map[physical_qubit] = dd_sequence_duration

        def _constrained_length(values):
            return self._alignment * np.floor(values / self._alignment)

        for nd in dag.topological_op_nodes():
            if not isinstance(nd.op, Delay) or nd.op.num_qubits == 1:
                new_dag.apply_operation_back(nd.op, nd.qargs, nd.cargs)
                continue

            dag_qubits = nd.qargs
            physical_qubits = [qubit_index_map[q] for q in dag_qubits]

            pred = next(dag.predecessors(nd))
            succ = next(dag.successors(nd))
            if self._skip_reset_qubits:  # discount initial delays
                if isinstance(pred, DAGInNode) or isinstance(pred.op, Reset):
                    new_dag.apply_operation_back(nd.op, nd.qargs, nd.cargs)
                    continue

            dd_sequence_durations = [index_sequence_duration_map[q] for q in physical_qubits]
            slacks = [(nd.op.duration - d) for d in dd_sequence_durations]
            slack_fractions = [slack / nd.op.duration for slack in slacks]
            if any(1 - sf >= self._skip_threshold for sf in slack_fractions):  # dd doesn't fit
                for q in dag_qubits:
                    new_dag.apply_operation_back(Delay(nd.op.duration), [q], [])
                continue

            # insert the actual DD sequence
            sub_coupling_map = self._coupling_map.reduce(physical_qubits)
            coloring = rx.graph_greedy_color(sub_coupling_map.graph.to_undirected())

            for dag_qubit, physical_qubit in zip(dag_qubits, physical_qubits):
                i = physical_qubits.index(physical_qubit)
                slack = slacks[i]
                xx = dd_sequence_durations[i] # FIXME: assumes same X duration on odd & even
                slack_prime = slack - xx
                if coloring[i] == 0:
                    taus = _constrained_length(slack_prime * np.asarray(self._spacing_odd))
                    taus += _constrained_length(.5 * xx * np.asarray(self._addition_odd))
                    unused_slack = slack - sum(taus)  # unused, due to rounding to int multiples of dt
                    middle_index = int((len(taus) - 1) / 2)  # arbitrary: redistribute to middle
                    taus[middle_index] += unused_slack  # now we add up to original delay duration
                else:
                    # N.B. If 2-coloring doesn't exist or wasn't found, allow for conflicts.
                    taus = _constrained_length(slack_prime * np.asarray(self._spacing_even))
                    taus += _constrained_length(.5 * xx * np.asarray(self._addition_even))
                    unused_slack = slack - sum(taus)  # unused, due to rounding to int multiples of dt
                    middle_index = int((len(taus) - 1) / 2)  # arbitrary: redistribute to middle
                    taus[middle_index] += unused_slack  # now we add up to original delay duration

                for tau, gate in itertools.zip_longest(taus, self._dd_sequence):
                    if tau > 0:
                        new_dag.apply_operation_back(Delay(tau), [dag_qubit])
                    if gate is not None:
                        new_dag.apply_operation_back(gate, [dag_qubit])

            new_dag.global_phase = _mod_2pi(new_dag.global_phase + sequence_gphase)

        return new_dag


def _mod_2pi(angle: float, atol: float = 0):
    """Wrap angle into interval [-π,π). If within atol of the endpoint, clamp to -π"""
    wrapped = (angle + np.pi) % (2 * np.pi) - np.pi
    if abs(wrapped - np.pi) < atol:
        wrapped = -np.pi
    return wrapped