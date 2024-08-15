from typing import Any
import math as m

from qiskit.circuit import QuantumCircuit, CircuitInstruction

from circuit_knitting.cutting.instructions import CutWire
from circuit_knitting.cutting.cutting_decomposition import cut_gates
from circuit_knitting.cutting.cut_finding.cco_utils import qc_to_cco_circuit


def cuts(
    circuit: QuantumCircuit,
    gate_cuts: list,
    wire_cuts: list,
) -> QuantumCircuit:

    circuit_cco = qc_to_cco_circuit(circuit)
    circ_out = cut_gates(circuit, gate_cuts)[0]
    gamma_gate = m.prod([circuit_cco[idx].gamma for idx in gate_cuts])
    # Insert all the wire cuts
    counter = 0
    for action in sorted(wire_cut_actions, key=lambda a: a[1][0]):
        inst_id = action.gate_spec.instruction_id
        # action.args[0][0] will be either 1 (control) or 2 (target)
        qubit_id = action.args[0][0] - 1
        circ_out.data.insert(
            inst_id + counter,
            CircuitInstruction(CutWire(), [circuit.data[inst_id].qubits[qubit_id]], []),
        )
        counter += 1

        if action.action.get_name() == "CutBothWires":
            # There should be two wires specified in the action in this case
            assert len(action.args) == 2
            qubit_id2 = action.args[1][0] - 1
            circ_out.data.insert(
                inst_id + counter,
                CircuitInstruction(
                    CutWire(), [circuit.data[inst_id].qubits[qubit_id2]], []
                ),
            )
            counter += 1

    # Return metadata describing the cut scheme
    metadata: dict[str, Any] = {"cuts": []}
    for i, inst in enumerate(circ_out.data):
        if inst.operation.name == "qpd_2q":
            metadata["cuts"].append(("Gate Cut", i))
        elif inst.operation.name == "cut_wire":
            metadata["cuts"].append(("Wire Cut", i))
    metadata["sampling_overhead"] = opt_out.upper_bound_gamma() ** 2

    return circ_out