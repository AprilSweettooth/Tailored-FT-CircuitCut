from __future__ import annotations

from typing import Any, cast
import math as m
import numpy as np
from dataclasses import dataclass

from qiskit.circuit import QuantumCircuit, CircuitInstruction, Qubit, QuantumRegister, Instruction
from qiskit.circuit.library import Barrier
from qiskit.transpiler.passes import RemoveBarriers

from .instructions import CutWire
from .cutting_decomposition import cut_gates, partition_problem
from .wire_cutting_transforms import cut_wires
from .cut_finding.optimization_settings import OptimizationSettings
from .cut_finding.disjoint_subcircuits_state import DisjointSubcircuitsState
# from .cut_finding.circuit_interface import SimpleGateList
# from .cut_finding.lo_cuts_optimizer import LOCutsOptimizer
# from .cut_finding.cco_utils import qc_to_cco_circuit
from .cut_finding.disjoint_subcircuits_state import DisjointSubcircuitsState

from azure.quantum import Workspace
from azure.quantum.qiskit import AzureQuantumProvider

def init_resource_estimator():
    workspace = Workspace (
    resource_id = "/subscriptions/e0d51919-2dda-4b7e-a8e7-77cc18580acb/resourceGroups/AzureQuantum/providers/Microsoft.Quantum/Workspaces/QECBenchmarking",
    location = "uksouth"
    )
    provider = AzureQuantumProvider(workspace)
    backend_est = provider.get_backend('microsoft.estimator')

    return backend_est


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

def resource_cost(
    circuit: QuantumCircuit,
    opt_out,
    backend,
    gate_bell: bool = False,
    wire_bell: bool = False,
) -> QuantumCircuit:
    
    """
    gate_cuts: list of cut positions [cut1, cut2, ... , cutN]
    wire_cuts: dict of cut positions at the specified instruction and wire, including multi-qubit gates cuts {inst1:[wire1,wire2], inst2:[wire3], ... , instM:[wireN]}
    where cut_i is located by the i-th instruction
    """

    # circuit_cco = qc_to_cco_circuit(circuit)

    # optimization_settings = OptimizationParameters(seed=111)
    # device_constraints = DeviceConstraints(qubits_per_subcircuit=4)
    # interface = SimpleGateList(circuit_cco)
    # opt_settings = OptimizationSettings(
    #     seed=optimization_settings.seed,
    #     max_gamma=optimization_settings.max_gamma,
    #     max_backjumps=optimization_settings.max_backjumps,
    #     gate_lo=optimization_settings.gate_lo,
    #     wire_lo=optimization_settings.wire_lo,
    # )
    # Hard-code the optimizer to an LO-only optimizer
    # optimizer = LOCutsOptimizer(interface, opt_settings, device_constraints)
    # Find cut locations
    # opt_out = optimizer.optimize()

    gate_cuts = []
    wire_cuts_action = []
    wire_cuts = {}

    opt_out = cast(DisjointSubcircuitsState, opt_out)
    opt_out.actions = cast(list, opt_out.actions)
    for action in opt_out.actions:
        if action.action.get_name() == "CutTwoQubitGate":
            gate_cuts.append(action.gate_spec.instruction_id)
        else:
            assert action.action.get_name() in (
                "CutLeftWire",
                "CutRightWire",
                "CutBothWires",
            )
            wire_cuts_action.append(action)

    circ_out = cut_gates(circuit, gate_cuts)[0]

    counter = 0
    for action in sorted(wire_cuts_action, key=lambda a: a[1][0]):
        inst_id = action.gate_spec.instruction_id
        # action.args[0][0] will be either 1 (control) or 2 (target)
        qubit_id = action.args[0][0] - 1
        circ_out.data.insert(
            inst_id + counter,
            CircuitInstruction(CutWire(), [circuit.data[inst_id].qubits[qubit_id]], []),
        )
        wire_cuts[inst_id] = [circuit.data[inst_id].qubits[qubit_id]._index]
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
            wire_cuts[inst_id].append([circuit.data[inst_id].qubits[qubit_id2]._index])
            counter += 1

    qc_w_ancilla = cut_wires(circ_out)
    partitioned_problem = partition_problem(circuit=qc_w_ancilla)
    subcircuits = partitioned_problem.subcircuits

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
    
    gamma_gate = m.prod([3 for _ in gate_cuts])
    gamma_wire = m.prod([4 for _ in wire_cuts.keys()])

    if gate_bell:
        sampling_overhead = (gamma_gate*gamma_wire) ** 2
    else:
        sampling_overhead = np.prod([basis.overhead for basis in partitioned_problem.bases])

    final_sub = [map_subcirc(sub, True) for sub in subcircuits.values()]

    # backend = init_resource_estimator()

    resources = []

    resource = {}
    job = backend.run(circuit, errorBudget=0.01)
    result = job.result()
    resource['physicalQubits'] = result.data()['physicalCounts']['physicalQubits']
    resource['runtime'] = result.data()['physicalCounts']['runtime']
    resource['rqops'] = result.data()['physicalCounts']['rqops']
    resource['number_of_T_state'] = result.data()['physicalCounts']['breakdown']['numTstates']
    resources.append(resource)

    for idx, s in enumerate(final_sub):
        sub_resource = {}
        job = backend.run(s, errorBudget=0.01/len(final_sub))
        result = job.result()
        sub_resource['physicalQubits'] = result.data()['physicalCounts']['physicalQubits']
        sub_resource['runtime'] = result.data()['physicalCounts']['runtime']
        sub_resource['rqops'] = result.data()['physicalCounts']['rqops']
        sub_resource['number_of_T_state'] = result.data()['physicalCounts']['breakdown']['numTstates']
        resources.append(sub_resource)

    return resources, sampling_overhead

@dataclass
class OptimizationParameters:
    """Specify parameters that control the optimization.

    If either of the constraints specified by ``max_backjumps`` or ``max_gamma`` are exceeded, the search terminates but
    nevertheless returns the result of a greedy best first search, which gives an *upper-bound* on gamma.
    """

    #: The seed to use when initializing Numpy random number generators in the best first search priority queue.
    seed: int | None = OptimizationSettings().seed

    #: Maximum allowed value of gamma which, if exceeded, forces the search to terminate.
    max_gamma: float = OptimizationSettings().max_gamma

    #: Maximum number of backjumps that can be performed before the search is forced to terminate; setting it to ``None`` implies that no such restriction is placed.
    max_backjumps: None | int = OptimizationSettings().max_backjumps

    #: Bool indicating whether or not to allow LO gate cuts while finding cuts.
    gate_lo: bool = OptimizationSettings().gate_lo

    #: Bool indicating whether or not to allow LO wire cuts while finding cuts.
    wire_lo: bool = OptimizationSettings().wire_lo


@dataclass
class DeviceConstraints:
    """Specify the constraints (qubits per subcircuit) that must be respected."""

    qubits_per_subcircuit: int

    def __post_init__(self):
        """Post-init method for data class."""
        if self.qubits_per_subcircuit < 1:
            raise ValueError(
                "qubits_per_subcircuit must be a positive definite integer."
            )

    def get_qpu_width(self) -> int:
        """Return the number of qubits per subcircuit."""
        return self.qubits_per_subcircuit
