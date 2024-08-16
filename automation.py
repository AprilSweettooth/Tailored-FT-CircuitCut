from typing import Any
import math as m
import numpy as np

from qiskit.circuit import QuantumCircuit, CircuitInstruction, Qubit, QuantumRegister, Instruction

from circuit_knitting.cutting.instructions import CutWire
from circuit_knitting.cutting.cutting_decomposition import cut_gates
from circuit_knitting.cutting.cut_finding.cco_utils import qc_to_cco_circuit
from circuit_knitting.cutting import partition_problem, cut_gates, cut_wires
from qiskit.circuit.library import Barrier
from qiskit.transpiler.passes import RemoveBarriers

def cuts(
    circuit: QuantumCircuit,
    gate_cuts: list,
    wire_cuts: dict[int,list],
) -> QuantumCircuit:
    
    """
    gate_cuts: list of cut positions [cut1, cut2, ... , cutN]
    wire_cuts: dict of cut positions at the specified instruction and wire, including multi-qubit gates cuts {inst1:[wire1,wire2], inst2:[wire3], ... , instM:[wireN]}
    where cut_i is located by the i-th instruction
    """

    circuit_cco = qc_to_cco_circuit(circuit)
    circ_out = cut_gates(circuit, gate_cuts)[0]
    gamma_gate = m.prod([circuit_cco[idx].gamma for idx in gate_cuts])
    gamma_wire = m.prod([4 for idx in wire_cuts.keys()])
    counter = 0
    for wire in wire_cuts.keys():
        inst_id = wire
        qubit_id = wire_cuts[wire][0]
        circ_out.data.insert(
            inst_id + counter,
            CircuitInstruction(CutWire(), [Qubit(QuantumRegister(circuit.num_qubits, 'q'), qubit_id)], []),
        )
        counter += 1

        if len(wire_cuts[wire]) > 1:
            # There should be two wires specified in the action in this case
            assert len(wire_cuts[wire]) == 2
            qubit_id2 = wire_cuts[wire][1]
            circ_out.data.insert(
                inst_id + counter,
                CircuitInstruction(
                    CutWire(), [Qubit(QuantumRegister(circuit.num_qubits, 'q'), qubit_id2)], []
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
    metadata["sampling_overhead"] = (gamma_gate*gamma_wire) ** 2

    return circ_out, metadata

def bell_cuts(
    circuit: QuantumCircuit,
    gate_cuts: list,
    wire_cuts: dict[int,list],
    ancilla: bool = False,
) -> QuantumCircuit:
    
    """
    gate_cuts: list of cut positions [cut1, cut2, ... , cutN]
    wire_cuts: dict of cut positions at the specified instruction and wire, including multi-qubit gates cuts {inst1:[wire1,wire2], inst2:[wire3], ... , instM:[wireN]}
    where cut_i is located by the i-th instruction
    """

    circuit_cco = qc_to_cco_circuit(circuit)
    circ_out = cut_gates(circuit, gate_cuts)[0]
    gamma_gate = m.prod([3 for _ in gate_cuts])
    gamma_wire = m.prod([4 for _ in wire_cuts.keys()])
    counter = 0
    for wire in wire_cuts.keys():
        inst_id = wire
        qubit_id = wire_cuts[wire][0]
        circ_out.data.insert(
            inst_id + counter,
            CircuitInstruction(CutWire(), [Qubit(QuantumRegister(circuit.num_qubits, 'q'), qubit_id)], []),
        )
        counter += 1

        if len(wire_cuts[wire]) > 1:
            # There should be two wires specified in the action in this case
            assert len(wire_cuts[wire]) == 2
            qubit_id2 = wire_cuts[wire][1]
            circ_out.data.insert(
                inst_id + counter,
                CircuitInstruction(
                    CutWire(), [Qubit(QuantumRegister(circuit.num_qubits, 'q'), qubit_id2)], []
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
    metadata["sampling_overhead"] = (gamma_gate*gamma_wire) ** 2
    print(
    f'Found solution using {len(metadata["cuts"])} cuts with a sampling '

    f'overhead of {metadata["sampling_overhead"]}.'
    )
    for cut in metadata["cuts"]:
        print(f"{cut[0]} at circuit instruction index {cut[1]}")
    f'overhead of {metadata["sampling_overhead"]}.'
    # circ_out.draw("mpl", scale=0.8, fold=-1)
    qc_w_ancilla = cut_wires(circ_out)
    # qc_w_ancilla.draw("mpl", scale=0.8)
    partitioned_problem = partition_problem(circuit=qc_w_ancilla)
    subcircuits = partitioned_problem.subcircuits
    for i,s in enumerate(subcircuits.values()):
        # s.draw("mpl", scale=0.8)
        counter = 0
        for d in s.data:
            if d.operation.name == 'qpd_1q':
                s.add_bits([Qubit(QuantumRegister(s.num_qubits, 'q'), 0)])
                # print(s.qubits[d.qubits[0]._index-1])
                del s.data[counter]
                s.data.insert(
                    counter,
                    CircuitInstruction(
                        Instruction(name='cx', num_qubits=2, num_clbits=0, params=[]),
                        [d.qubits[0], s.qubits[-1]], []
                    ),
                )
                s.data.insert(
                    counter+1,
                    CircuitInstruction(
                        Instruction(name='h', num_qubits=1, num_clbits=0, params=[]),
                        [d.qubits[0]], []
                    ),
                )  
            counter += 1
    if ancilla:
        return subcircuits, qc_w_ancilla
    else:
        return subcircuits
    
def map_subcirc(circ, remove=False):
    circuit = QuantumCircuit(circ.num_qubits)
    old_qubits = circ.qubits
    new_qubits = circuit.qubits
    for idx, ins in enumerate(circ.data):
        if ins.operation.name != 'qpd_1q' or remove == False:
            if len(list(ins.qubits)) == 1:
                qb = new_qubits[old_qubits.index(ins.qubits[0])]
                circuit.data.insert(
                    idx,
                    CircuitInstruction(
                        ins.operation,
                        [qb], []
                    ),
                    )  
            else:
                assert len(list(ins.qubits)) == 2
                qb1, qb2 = new_qubits[old_qubits.index(ins.qubits[0])], new_qubits[old_qubits.index(ins.qubits[1])]
                circuit.data.insert(
                    idx,
                    CircuitInstruction(
                        ins.operation,
                        [qb1,qb2], []
                    ),
                    )   
        else:
            qb = new_qubits[old_qubits.index(ins.qubits[0])]
            circuit.data.insert(
                idx,
                CircuitInstruction(
                    Barrier(1),
                    [qb], []
                ),
                )   
    return RemoveBarriers()(circuit)

def mix_cuts(
    circuit: QuantumCircuit,
    gate_cuts: list,
    wire_cuts: dict[int,list],
    gate_bell: bool = False,
    wire_bell: bool = False,
    ancilla: bool = False,
) -> QuantumCircuit:
    
    """
    gate_cuts: list of cut positions [cut1, cut2, ... , cutN]
    wire_cuts: dict of cut positions at the specified instruction and wire, including multi-qubit gates cuts {inst1:[wire1,wire2], inst2:[wire3], ... , instM:[wireN]}
    where cut_i is located by the i-th instruction
    """

    # circuit_cco = qc_to_cco_circuit(circuit)
    circ_out = cut_gates(circuit, gate_cuts)[0]
    gamma_gate = m.prod([3 for _ in gate_cuts])
    gamma_wire = m.prod([4 for _ in wire_cuts.keys()])

    counter = 0
    for wire in wire_cuts.keys():
        inst_id = wire
        qubit_id = wire_cuts[wire][0]
        circ_out.data.insert(
            inst_id + counter,
            CircuitInstruction(CutWire(), [Qubit(QuantumRegister(circuit.num_qubits, 'q'), qubit_id)], []),
        )
        counter += 1

        if len(wire_cuts[wire]) > 1:
            # There should be two wires specified in the action in this case
            assert len(wire_cuts[wire]) == 2
            qubit_id2 = wire_cuts[wire][1]
            circ_out.data.insert(
                inst_id + counter,
                CircuitInstruction(
                    CutWire(), [Qubit(QuantumRegister(circuit.num_qubits, 'q'), qubit_id2)], []
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

    qc_w_ancilla = cut_wires(circ_out)
    partitioned_problem = partition_problem(circuit=qc_w_ancilla)
    subcircuits = partitioned_problem.subcircuits

    if gate_bell:
        metadata["sampling_overhead"] = (gamma_gate*gamma_wire) ** 2
    else:
        metadata["sampling_overhead"] = np.prod([basis.overhead for basis in partitioned_problem.bases])

    print(
    f'Found solution using {len(metadata["cuts"])} cuts with a sampling '

    f'overhead of {metadata["sampling_overhead"]}.'
    )

    for cut in metadata["cuts"]:
        print(f"{cut[0]} at circuit instruction index {cut[1]}")

    for i,s in enumerate(subcircuits.values()):
        counter = 0
        for d in s.data:
            if d.operation.name == 'qpd_1q' and gate_bell and wire_bell:
                s.add_bits([Qubit(QuantumRegister(s.num_qubits, 'q'), 0)])
                del s.data[counter]
                s.data.insert(
                    counter,
                    CircuitInstruction(
                        Instruction(name='cx', num_qubits=2, num_clbits=0, params=[]),
                        [d.qubits[0], s.qubits[-1]], []
                    ),
                )
                s.data.insert(
                    counter+1,
                    CircuitInstruction(
                        Instruction(name='h', num_qubits=1, num_clbits=0, params=[]),
                        [d.qubits[0]], []
                    ),
                )
            elif d.operation.name == 'qpd_1q' and gate_bell:
                if 'move' not in str(d.operation.label):
                    # print(str(d.operation.label))
                    s.add_bits([Qubit(QuantumRegister(s.num_qubits, 'q'), 0)])
                    del s.data[counter]
                    s.data.insert(
                        counter,
                        CircuitInstruction(
                            Instruction(name='cx', num_qubits=2, num_clbits=0, params=[]),
                            [d.qubits[0], s.qubits[-1]], []
                        ),
                    )
                    s.data.insert(
                        counter+1,
                        CircuitInstruction(
                            Instruction(name='h', num_qubits=1, num_clbits=0, params=[]),
                            [d.qubits[0]], []
                        ),
                    ) 
            elif d.operation.name == 'qpd_1q' and wire_bell:
                if 'move' in str(d.operation.label):
                    s.add_bits([Qubit(QuantumRegister(s.num_qubits, 'q'), 0)])
                    del s.data[counter]
                    s.data.insert(
                        counter,
                        CircuitInstruction(
                            Instruction(name='cx', num_qubits=2, num_clbits=0, params=[]),
                            [d.qubits[0], s.qubits[-1]], []
                        ),
                    )
                    s.data.insert(
                        counter+1,
                        CircuitInstruction(
                            Instruction(name='h', num_qubits=1, num_clbits=0, params=[]),
                            [d.qubits[0]], []
                        ),
                    )  
                else:
                    s.data.insert(
                        counter+1,
                        CircuitInstruction(
                            Instruction(name='h', num_qubits=1, num_clbits=0, params=[]),
                            [d.qubits[0]], []
                        ),
                    )                      
            elif d.operation.name == 'qpd_1q':
                s.data.insert(
                    counter+1,
                    CircuitInstruction(
                        Instruction(name='h', num_qubits=1, num_clbits=0, params=[]),
                        [d.qubits[0]], []
                    ),
                )   
            counter += 1
    if len(subcircuits.values()) < 2:
        raise ValueError("Must cut the circuit into separate pieces !")
    final_sub = [map_subcirc(sub, True) for sub in subcircuits.values()]
    if ancilla:
        return final_sub, qc_w_ancilla
    else:
        return final_sub