from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.dagcircuit import DAGOpNode
from qiskit.converters import circuit_to_dag, dag_to_circuit
import math
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit, Instruction
# from resource_analysis import PhysicalParameters


def read_circ(circuit):
    dag = circuit_to_dag(circuit)
    gate_edges = []
    wire_edges = []
    # single_node_id = {}
    # multi_node_id = {}
    # single_id_node = {}
    # multi_id_node = {}
    node_name_ids = {}
    id_node_names = {}
    vertex_ids = {}
    curr_node_id = 0
    total_node_id = 0
    qubit_gate_counter = {}
    gate_set = {"rx", "ry", "rz", "t"}
    for qubit in dag.qubits:
        qubit_gate_counter[qubit] = 0
    for vertex in dag.topological_op_nodes():
        
        if len(vertex.qargs) == 2:
            arg0, arg1 = vertex.qargs

            vertex_name = "%s[%d]%d" % (
                arg0._register.name,
                arg0._index,
                qubit_gate_counter[arg0],
            )
            qubit_gate_counter[arg0] += 1

            if vertex_name not in node_name_ids and id(vertex) not in vertex_ids:
                node_name_ids[vertex_name] = [curr_node_id,'cnot0']
                id_node_names[curr_node_id] = vertex_name
                vertex_ids[id(vertex)] = curr_node_id
                gate_edges.append((curr_node_id,curr_node_id+1))
                curr_node_id += 1

                vertex_name = "%s[%d]%d" % (
                    arg1._register.name,
                    arg1._index,
                    qubit_gate_counter[arg1],
                )
                qubit_gate_counter[arg1] += 1

                if vertex_name not in node_name_ids:
                    node_name_ids[vertex_name] = [curr_node_id,'cnot1']
                    id_node_names[curr_node_id] = vertex_name
                    # vertex_ids[id(vertex)] = curr_node_id
                    curr_node_id += 1
                total_node_id += 1

        elif len(vertex.qargs) == 1:
            arg = vertex.qargs[0]

            vertex_name = "%s[%d]%d" % (
                arg._register.name,
                arg._index,
                qubit_gate_counter[arg],
            )
            qubit_gate_counter[arg] += 1
            if vertex_name not in node_name_ids and id(vertex) not in vertex_ids:
                if dag.op_nodes()[total_node_id].op.name in gate_set: 
                    node_name_ids[vertex_name] = [curr_node_id,dag.op_nodes()[total_node_id].op.name]
                    id_node_names[curr_node_id] = vertex_name
                    vertex_ids[id(vertex)] = curr_node_id
                    curr_node_id += 1
                    total_node_id += 1

        else:
            raise Exception("vertex does not have 1 or 2 qargs!")
        # print(node_name_ids)
    for u, v, _ in dag.edges():
        if isinstance(u, DAGOpNode) and isinstance(v, DAGOpNode):
            u_id = vertex_ids[id(u)]
            v_id = vertex_ids[id(v)]
            wire_edges.append((u_id, v_id))
            # print(edges)

    n_vertices = sum([len(node.split(' ')) for node in list(id_node_names.values())])

    return n_vertices, gate_edges, wire_edges, node_name_ids, id_node_names

def cuts_parser(cuts, circuit, vertex):
    dag = circuit_to_dag(circuit)
    positions = []
    for position in cuts:
        source, dest = position
        source_qargs = [
            (x.split("]")[0] + "]", int(x.split("]")[1])) for x in source.split(" ")
        ]
        # print(source_qargs)
        dest_qargs = [
            (x.split("]")[0] + "]", int(x.split("]")[1])) for x in dest.split(" ")
        ]
        qubit_cut_w = []
        qubit_cut_g = []
        for source_qarg in source_qargs:
            source_qubit, source_multi_Q_gate_idx = source_qarg
            for dest_qarg in dest_qargs:
                dest_qubit, dest_multi_Q_gate_idx = dest_qarg
                if (
                    vertex[source][0] == vertex[dest][0]-1 and len(vertex[source][1]) == 5 and len(vertex[dest][1]) == 5 
                ):
                    qubit_cut_g.append([source_qubit,dest_qubit])
                    for x in source.split(" "):
                        source_idx = 0
                        if x.split("]")[0] + "]" == qubit_cut_g[0][0]:
                            source_idx = int(x.split("]")[1])
                    for x in dest.split(" "):
                        dest_idx = 0
                        if x.split("]")[0] + "]" == qubit_cut_g[0][1]:
                            dest_idx = int(x.split("]")[1])
                    multi_Q_gate_idx = [source_idx+1, dest_idx+1]
                    # print(source_idx,dest_idx)
                    wire = None
                    for qubit in circuit.qubits:
                        if qubit._register.name == qubit_cut_g[0][0].split("[")[
                            0
                        ] and qubit._index == int(qubit_cut_g[0][0].split("[")[1].split("]")[0]):
                            wire = [qubit]
                    for qubit in circuit.qubits:
                        if qubit._register.name == qubit_cut_g[0][1].split("[")[
                            0
                        ] and qubit._index == int(qubit_cut_g[0][1].split("[")[1].split("]")[0]):
                            wire += [qubit]
                    tmp = 0
                    all_Q_gate_idx = None
                    p = []
                    for gate_idx, gate in enumerate(
                        list(dag.nodes_on_wire(wire=wire[0], only_ops=True))
                    ):
                        tmp += 1
                        if tmp == multi_Q_gate_idx[0]:
                            all_Q_gate_idx = gate_idx
                    p.append((wire[0], all_Q_gate_idx))
                    tmp = 0
                    all_Q_gate_idx = None
                    for gate_idx, gate in enumerate(
                        list(dag.nodes_on_wire(wire=wire[1], only_ops=True))
                    ):
                        tmp += 1
                        if tmp == multi_Q_gate_idx[1]:
                            all_Q_gate_idx = gate_idx
                    p.append((wire[1], all_Q_gate_idx))
                    positions.append(p)

                elif (vertex[source][1] == vertex[dest][1]):
                    if int(source_qubit.split("[")[1][0]) > int(dest_qubit.split("[")[1][0]):
                        qubit_cut_w.append(source_qubit)
                        for x in source.split(" "):
                            multi_Q_gate_idx = int(x.split("]")[1])+1
                    else:
                        qubit_cut_w.append(dest_qubit)
                        for x in dest.split(" "):
                            multi_Q_gate_idx = int(x.split("]")[1])
                    # for x in source.split(" "):
                    #     source_idx = 0
                    #     if x.split("]")[0] + "]" == qubit_cut_w[0]:
                    #         source_idx = int(x.split("]")[1])
                    # for x in dest.split(" "):
                    #     dest_idx = 0
                    #     if x.split("]")[0] + "]" == qubit_cut_w[0]:
                    #         dest_idx = int(x.split("]")[1])
                            # print(dest_idx)
                    # print(source_idx,dest_idx)
                    # multi_Q_gate_idx = max(source_idx, dest_idx)
                    # print('max:',multi_Q_gate_idx)
                    wire = None
                    for qubit in circuit.qubits:
                        if qubit._register.name == qubit_cut_w[0].split("[")[
                            0
                        ] and qubit._index == int(qubit_cut_w[0].split("[")[1].split("]")[0]):
                            wire = qubit
                    tmp = 0
                    all_Q_gate_idx = None
                    for gate_idx, gate in enumerate(
                        list(dag.nodes_on_wire(wire=wire, only_ops=True))
                    ):
                        tmp += 1
                        if tmp == multi_Q_gate_idx:
                            all_Q_gate_idx = gate_idx
                    positions.append((wire, all_Q_gate_idx))

                else:
                    qubit_cut_w.append(source_qubit)
                    for x in source.split(" "):
                        source_idx = 0
                        if x.split("]")[0] + "]" == qubit_cut_w[0]:
                            source_idx = int(x.split("]")[1])
                    for x in dest.split(" "):
                        dest_idx = 0
                        if x.split("]")[0] + "]" == qubit_cut_w[0]:
                            dest_idx = int(x.split("]")[1])
                    multi_Q_gate_idx = max(source_idx, dest_idx)+1
                    # print(multi_Q_gate_idx)
                    wire = None
                    for qubit in circuit.qubits:
                        if qubit._register.name == qubit_cut_w[0].split("[")[
                            0
                        ] and qubit._index == int(qubit_cut_w[0].split("[")[1].split("]")[0]):
                            wire = qubit
                    tmp = 0
                    all_Q_gate_idx = None
                    for gate_idx, gate in enumerate(
                        list(dag.nodes_on_wire(wire=wire, only_ops=True))
                    ):
                        tmp += 1
                        if tmp == multi_Q_gate_idx:
                            all_Q_gate_idx = gate_idx
                    positions.append((wire, all_Q_gate_idx))

    # positions = sorted(positions, reverse=True, key=lambda cut: cut[1])
    return positions

def subcircuits_parser(n_vertices, id_vertices, subcircuit_id, circuit):
    """
    complete_path_map[input circuit qubit] = [{subcircuit_idx,subcircuit_qubit}]
    """
    def calculate_distance_between_gate(gate_A, gate_B):
        if len(gate_A.split(" ")) >= len(gate_B.split(" ")):
            tmp_gate = gate_A
            gate_A = gate_B
            gate_B = tmp_gate
        distance = float("inf")
        for qarg_A in gate_A.split(" "):
            qubit_A = qarg_A.split("]")[0] + "]"
            qgate_A = int(qarg_A.split("]")[-1])
            for qarg_B in gate_B.split(" "):
                qubit_B = qarg_B.split("]")[0] + "]"
                qgate_B = int(qarg_B.split("]")[-1])
                # print('%s gate %d --> %s gate %d'%(qubit_A,qgate_A,qubit_B,qgate_B))
                if qubit_A == qubit_B:
                    distance = min(distance, abs(qgate_B - qgate_A))
        # print('Distance from %s to %s = %f'%(gate_A,gate_B,distance))
        return distance
    
    def construct_subcircuit_gates(subcircuit_id, n_vertices, id_vertices):
        subcircuits = []
        for i in range(len(subcircuit_id)):
            subcircuit = []
            for j in range(n_vertices):
                if abs(subcircuit_id[i][j]) > 1e-4:
                    subcircuit.append(id_vertices[j])
            subcircuits.append(subcircuit)
        assert (
                sum([len(subcircuit) for subcircuit in subcircuits])
                == n_vertices
            )
        return subcircuits

    subcircuit_gates = construct_subcircuit_gates(subcircuit_id, n_vertices, id_vertices)
    dag = circuit_to_dag(circuit)
    qubit_allGate_depths = {x: 0 for x in circuit.qubits}
    qubit_2qGate_depths = {x: 0 for x in circuit.qubits}
    gate_depth_encodings = {}
    # print('Before translation :',subcircuit_gates,flush=True)
    for op_node in dag.topological_op_nodes():
        gate_depth_encoding = ""
        for qarg in op_node.qargs:
            gate_depth_encoding += "%s[%d]%d " % (
                qarg._register.name,
                qarg._index,
                qubit_allGate_depths[qarg],
            )
        gate_depth_encoding = gate_depth_encoding[:-1]
        gate_depth_encodings[op_node] = gate_depth_encoding
        for qarg in op_node.qargs:
            qubit_allGate_depths[qarg] += 1
        if len(op_node.qargs) == 2:
            MIP_gate_depth_encoding = ""
            for qarg in op_node.qargs:
                MIP_gate_depth_encoding += "%s[%d]%d " % (
                    qarg._register.name,
                    qarg._index,
                    qubit_2qGate_depths[qarg],
                )
                qubit_2qGate_depths[qarg] += 1
            MIP_gate_depth_encoding = MIP_gate_depth_encoding[:-1]
            # print('gate_depth_encoding = %s, MIP_gate_depth_encoding = %s'%(gate_depth_encoding,MIP_gate_depth_encoding))
            for subcircuit_idx in range(len(subcircuit_gates)):
                for gate_idx in range(len(subcircuit_gates[subcircuit_idx])):
                    if (
                        subcircuit_gates[subcircuit_idx][gate_idx]
                        == MIP_gate_depth_encoding
                    ):
                        subcircuit_gates[subcircuit_idx][gate_idx] = gate_depth_encoding
                        break
    # print('After translation :',subcircuit_gates,flush=True)

    subcircuit_op_nodes = {x: [] for x in range(len(subcircuit_gates))}
    subcircuit_sizes = [0 for x in range(len(subcircuit_gates))]
    complete_path_map = {}
    for circuit_qubit in dag.qubits:
        complete_path_map[circuit_qubit] = []
        qubit_ops = dag.nodes_on_wire(wire=circuit_qubit, only_ops=True)
        for qubit_op_idx, qubit_op in enumerate(qubit_ops):
            gate_depth_encoding = gate_depth_encodings[qubit_op]
            # print(gate_depth_encoding)
            nearest_subcircuit_idx = -1
            min_distance = float("inf")
            for subcircuit_idx in range(len(subcircuit_gates)):
                distance = float("inf")
                for gate in subcircuit_gates[subcircuit_idx]:
    #                 if len(gate.split(" ")) == 1:
    # #                     # Do not compare against single qubit gates
    #                     continue
    #                 else:
                    distance = min(
                        distance,
                        calculate_distance_between_gate(
                            gate_A=gate_depth_encoding, gate_B=gate
                        ),
                    )
                # print('Distance from %s to subcircuit %d = %f'%(gate_depth_encoding,subcircuit_idx,distance))
                if distance < min_distance:
                    min_distance = distance
                    nearest_subcircuit_idx = subcircuit_idx
            assert nearest_subcircuit_idx != -1
            path_element = {
                "subcircuit_idx": nearest_subcircuit_idx,
                "subcircuit_qubit": subcircuit_sizes[nearest_subcircuit_idx],
            }
            if (
                len(complete_path_map[circuit_qubit]) == 0
                or nearest_subcircuit_idx
                != complete_path_map[circuit_qubit][-1]["subcircuit_idx"]
            ):
                # print('{} op #{:d} {:s} encoding = {:s}'.format(circuit_qubit,qubit_op_idx,qubit_op.name,gate_depth_encoding),
                # 'belongs in subcircuit %d'%nearest_subcircuit_idx)
                complete_path_map[circuit_qubit].append(path_element)
                subcircuit_sizes[nearest_subcircuit_idx] += 1

            subcircuit_op_nodes[nearest_subcircuit_idx].append(qubit_op)
    for circuit_qubit in complete_path_map:
        # print(circuit_qubit,'-->')
        for path_element in complete_path_map[circuit_qubit]:
            path_element_qubit = QuantumRegister(
                size=subcircuit_sizes[path_element["subcircuit_idx"]], name="q"
            )[path_element["subcircuit_qubit"]]
            path_element["subcircuit_qubit"] = path_element_qubit
            # print(path_element)
    subcircuits = generate_subcircuits(
        subcircuit_op_nodes=subcircuit_op_nodes,
        complete_path_map=complete_path_map,
        subcircuit_sizes=subcircuit_sizes,
        dag=dag,
    )
    return subcircuits


def generate_subcircuits(subcircuit_op_nodes, complete_path_map, subcircuit_sizes, dag):
    qubit_pointers = {x: 0 for x in complete_path_map}
    subcircuits = [QuantumCircuit(x, name="q") for x in subcircuit_sizes]
    # print(subcircuit_op_nodes)
    for op_node in dag.topological_op_nodes():
        # print(op_node.op)
        subcircuit_idx = list(
            filter(
                lambda x: op_node in subcircuit_op_nodes[x], subcircuit_op_nodes.keys()
            )
        )
        assert len(subcircuit_idx) == 1
        subcircuit_idx = subcircuit_idx[0]
        # print('{} belongs in subcircuit {:d}'.format(op_node.qargs,subcircuit_idx))
        subcircuit_qargs = []
        for op_node_qarg in op_node.qargs:
            if (
                complete_path_map[op_node_qarg][qubit_pointers[op_node_qarg]][
                    "subcircuit_idx"
                ]
                != subcircuit_idx
            ):
                qubit_pointers[op_node_qarg] += 1
            path_element = complete_path_map[op_node_qarg][qubit_pointers[op_node_qarg]]
            assert path_element["subcircuit_idx"] == subcircuit_idx
            subcircuit_qargs.append(path_element["subcircuit_qubit"])
        # print('-->',subcircuit_qargs)
        subcircuits[subcircuit_idx].append(
            instruction=op_node.op, qargs=subcircuit_qargs, cargs=None
        )
    return subcircuits


def circuit_stripping(circuit):
    # Remove all barriers in the circuit
    dag = circuit_to_dag(circuit)
    stripped_dag = DAGCircuit()
    [stripped_dag.add_qreg(x) for x in circuit.qregs]
    for vertex in dag.topological_op_nodes():
        if len(vertex.qargs) in {1,2}  and vertex.op.name != "barrier":
            stripped_dag.apply_operation_back(op=vertex.op, qargs=vertex.qargs)
    return dag_to_circuit(stripped_dag)

def translate_cnot(circuit, location, wire):
    # op = Instruction(name='barrier', num_qubits=1, num_clbits=0, params=[])
    seq_cnot = 0
    # for ct in location[0]:
    #     qc_c = QuantumCircuit(1, name='Control%d'%(seq_cnot))
    #     qc_c.barrier(0)
    #     op_c = qc_c.to_instruction()
    #     qc_t = QuantumCircuit(1, name='Target%d'%(seq_cnot))
    #     qc_t.barrier(0)
    #     op_t = qc_t.to_instruction()
    #     source, dest = ct
    #     source_qargs = [
    #         [x.split("]")[0] + "]", int(x.split("]")[1])] for x in source.split(" ")
    #     ]
    #     dest_qargs = [
    #         [x.split("]")[0] + "]", int(x.split("]")[1])] for x in dest.split(" ")
    #     ]
    #     qb = [int(source_qargs[0][0][2]),int(dest_qargs[0][0][2])]
    #     assert qb[0] == qb[1]-1
    #     idx_node = [source_qargs[0][1],dest_qargs[0][1]]
    #     counter = 0
    #     instruction = idx_node[0]
    #     for i in circuit.data:
    #         if qb[0] in {q._index for q in list(i.qubits)}:
    #             instruction -= 1
    #         if instruction < 0:
    #             break
    #         counter += 1

    #     del circuit.data[counter]
    #     circuit.data.insert(counter+seq_cnot,(op_c,[Qubit(QuantumRegister(circuit.num_qubits, 'q'), qb[0])],[]))
    #     circuit.data.insert(counter+seq_cnot+1,(op_t,[Qubit(QuantumRegister(circuit.num_qubits, 'q'), qb[1])],[]))
    #     seq_cnot += 1
    
    seq_wire = 0
    for ct in location[1]:
        qc_c = QuantumCircuit(1, name='Meas%d'%(seq_wire))
        qc_c.barrier(0)
        op_c = qc_c.to_instruction()
        qc_t = QuantumCircuit(1, name='Init%d'%(seq_wire))
        qc_t.barrier(0)
        op_t = qc_t.to_instruction()
        source, dest = ct
        source_qargs = [
            [x.split("]")[0] + "]", int(x.split("]")[1])] for x in source.split(" ")
        ]
        dest_qargs = [
            [x.split("]")[0] + "]", int(x.split("]")[1])] for x in dest.split(" ")
        ]
        qb = list(wire[seq_wire])[0]._index
        # assert qb[0] == qb[1]-1
        idx_node = [source_qargs[0][1],dest_qargs[0][1]]
        counter = 0
        instruction = idx_node[0]
        qbs = [int(source_qargs[0][0][2]),int(dest_qargs[0][0][2])]
        # print(qb,qbs)
        for i in circuit.data:
            if qbs[0] in {q._index for q in list(i.qubits)}:
                instruction -= 1
            counter += 1
            if instruction < 0:
                break

        circuit.data.insert(counter+seq_cnot+seq_wire*2+1,(op_c,[Qubit(QuantumRegister(circuit.num_qubits, 'q'), qb)],[]))
        # counter = 0
        # instruction = idx_node[1]
        # for i in circuit.data:
        #     if qb in {q._index for q in list(i.qubits)}:
        #         instruction -= 1
        #     counter += 1
        #     if instruction < 0:
        #         break

        circuit.data.insert(counter+seq_cnot+seq_wire*2+2,(op_t,[Qubit(QuantumRegister(circuit.num_qubits, 'q'), qb)],[]))
        seq_wire += 1
    return circuit

def find_cuts(
    circuit,
    subcircuits_id,
    location,
):
    stripped_circ = circuit_stripping(circuit=circuit)
    n_vertices, gedges, wedges, vertex_ids, id_vertices = read_circ(circuit=stripped_circ)
    num_qubits = circuit.num_qubits
    positions = cuts_parser(location[1], circuit, vertex_ids)
    # print(list(positions[0])[0]._index,list(positions[0])[1])
    # subcircuits = subcircuits_parser(
    #     n_vertices=n_vertices, id_vertices=id_vertices, subcircuit_id=subcircuits_id, circuit=translate_cnot(circuit,location,positions), 
    # )
    print(translate_cnot(circuit,location, positions))
    # cut_solution = {
    # "subcircuits": subcircuits,
    # "num_cuts": len(location[0])+len(location[1]),
    # "num_qubits": num_qubits
    # }
    # resources = []
    # print("Cutter result:")
    # print("%d subcircuits, %d cuts" % (len(subcircuits), len(location[0])+len(location[1])))

    # for subcircuit_idx in range(len(subcircuits)):
    #     resource = PhysicalParameters.make_beverland_et_al(subcircuits[subcircuit_idx])
    #     resources.append(resource)
    #     print("subcircuit %d" % subcircuit_idx)
    #     print(
    #         r"physical qubits = %d, logical time step (in $\mu$s) = %d, number of magic state = %d"
    #         % (
    #             resource.Q,
    #             resource.logical_time_step*1e6,
    #             resource.number_of_T_gates,
    #         )
    #     )
    #     print('Operations needed:',  circuit_to_dag(subcircuits[subcircuit_idx])._op_names)
    #     print(subcircuits[subcircuit_idx])
    # resource_original = PhysicalParameters.make_beverland_et_al(circuit)
    # print(
    #     r"Original physical qubits = %d, logical time step ($\mu$s)= %d, number of magic state = %d and the cutting sampling overhead = %d"
    #     % (
    #         resource_original.Q,
    #         resource_original.logical_time_step*1e6,
    #         resource_original.number_of_T_gates,
    #         9**len(location[0]) * 16**len(location[1]),
    #     )
    # )
    # return cut_solution



