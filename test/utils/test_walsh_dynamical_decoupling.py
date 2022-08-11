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

"""Test dynamical decoupling."""

import unittest

from qiskit.circuit import QuantumCircuit

from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.passes import PadDynamicalDecoupling
from qiskit.transpiler.passes.scheduling import ALAPScheduleAnalysis
from qiskit.transpiler import PassManager

from qiskit_research.utils.dynamical_decoupling import generate_walsh_sequence


class TestWalshDynamicalDecoupling(unittest.TestCase):
    """Test walsh_dynamical_decoupling."""

    def test_add_periodic_dynamical_decoupling(self):
        """Test Walsh sequence corresponding to nx = 001, ny = 010, nz = 100"""
        circuit = QuantumCircuit(4)
        circuit.h(0)
        for i in range(3):
            circuit.cx(i, i + 1)
        circuit.measure_all()

        durations = InstructionDurations(
            [
                ("h", 0, 50),
                ("cx", [0, 1], 700),
                ("cx", [1, 2], 200),
                ("cx", [2, 3], 300),
                ("x", None, 50),
                ("y", None, 50),
                ("z", None, 50),
                ("measure", None, 1000),
                ("reset", None, 1500),
            ]
        )
        pulse_alignment = 25
        sequence, spacing = generate_walsh_sequence(0b001, 0b010, 0b100)
        pm = PassManager([
            ALAPScheduleAnalysis(durations),
            PadDynamicalDecoupling(
                durations=durations,
                dd_sequence=sequence,
                spacing=spacing,
                pulse_alignment=pulse_alignment,
                skip_reset_qubits=False,
            )
        ])
        circ_dd = pm.run(circuit)

        self.assertTrue(
            str(circ_dd.draw()).strip() == """
              ┌───┐           ┌───────────────┐┌───┐┌───────────────┐┌───┐»
   q_0: ──────┤ H ├────────■──┤ Delay(25[dt]) ├┤ Z ├┤ Delay(25[dt]) ├┤ X ├»
        ┌─────┴───┴─────┐┌─┴─┐└───────────────┘└───┘└───────────────┘└───┘»
   q_1: ┤ Delay(50[dt]) ├┤ X ├────────────────────────────────────────────»
        ├───────────────┤├───┤┌───────────────┐┌───┐┌───────────────┐┌───┐»
   q_2: ┤ Delay(50[dt]) ├┤ Z ├┤ Delay(50[dt]) ├┤ X ├┤ Delay(50[dt]) ├┤ Z ├»
        ├───────────────┤├───┤├───────────────┤├───┤├───────────────┤├───┤»
   q_3: ┤ Delay(75[dt]) ├┤ Z ├┤ Delay(75[dt]) ├┤ X ├┤ Delay(75[dt]) ├┤ Z ├»
        └───────────────┘└───┘└───────────────┘└───┘└───────────────┘└───┘»
meas: 4/══════════════════════════════════════════════════════════════════»
                                                                          »
«        ┌───────────────┐ ┌───┐┌───────────────┐┌───┐┌───────────────┐┌───┐»
«   q_0: ┤ Delay(25[dt]) ├─┤ Z ├┤ Delay(50[dt]) ├┤ Z ├┤ Delay(25[dt]) ├┤ X ├»
«        └───────────────┘ └───┘└───────────────┘└───┘└───────────────┘└───┘»
«   q_1: ───────────────────────────────────────────────────────────────────»
«        ┌────────────────┐┌───┐┌───────────────┐┌───┐┌───────────────┐┌───┐»
«   q_2: ┤ Delay(150[dt]) ├┤ Z ├┤ Delay(50[dt]) ├┤ X ├┤ Delay(50[dt]) ├┤ Z ├»
«        ├────────────────┤├───┤├───────────────┤├───┤├───────────────┤├───┤»
«   q_3: ┤ Delay(200[dt]) ├┤ Z ├┤ Delay(75[dt]) ├┤ X ├┤ Delay(75[dt]) ├┤ Z ├»
«        └────────────────┘└───┘└───────────────┘└───┘└───────────────┘└───┘»
«meas: 4/═══════════════════════════════════════════════════════════════════»
«                                                                           »
«        ┌───────────────┐┌───┐┌───────────────┐  ░ ┌─┐         
«   q_0: ┤ Delay(25[dt]) ├┤ Z ├┤ Delay(25[dt]) ├──░─┤M├─────────
«        └───────────────┘└───┘├───────────────┴┐ ░ └╥┘┌─┐      
«   q_1: ───────────────────■──┤ Delay(300[dt]) ├─░──╫─┤M├──────
«        ┌───────────────┐┌─┴─┐└────────────────┘ ░  ║ └╥┘┌─┐   
«   q_2: ┤ Delay(50[dt]) ├┤ X ├────────■──────────░──╫──╫─┤M├───
«        ├───────────────┤└───┘      ┌─┴─┐        ░  ║  ║ └╥┘┌─┐
«   q_3: ┤ Delay(75[dt]) ├───────────┤ X ├────────░──╫──╫──╫─┤M├
«        └───────────────┘           └───┘        ░  ║  ║  ║ └╥┘
«meas: 4/════════════════════════════════════════════╩══╩══╩══╩═
«                                                    0  1  2  3 
""".strip())
