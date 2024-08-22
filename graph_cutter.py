from qiskit.converters import circuit_to_dag
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit, CircuitInstruction
from qiskit.circuit.library import Barrier

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def read_circ(circuit):
    dag = circuit_to_dag(circuit)
    gate_edges = []
    wire_edges = []
    node_name_ids = {}
    id_node_names = {}
    curr_node_id = 0
    total_node_id = 0
    qubit_gate_counter = {}
    label_node = 0
    # gate_set = {"rx", "ry", "rz", "t", "h"}
    for qubit in dag.qubits:
        qubit_gate_counter[qubit] = 0
    # print(qubit_gate_counter)
    for ins in circuit.data:
        # print(vertex.qargs[0])
        if circuit.data[total_node_id].operation.name[:4] == 'Ctrl':
            arg = ins.qubits[0]
            vertex_name = "%s[%d]%d" % (
                arg._register.name,
                arg._index,
                qubit_gate_counter[arg],
            )
            # print(vertex_name)
            qubit_gate_counter[arg] += 1

            if vertex_name not in node_name_ids:
                node_name_ids[vertex_name] = [curr_node_id,'cx_c'+str(label_node)]
                id_node_names[curr_node_id] = vertex_name
                # vertex_ids[id(vertex)] = curr_node_id
                gate_edges.append((curr_node_id,curr_node_id+1))
                curr_node_id += 1
                total_node_id += 1

        elif circuit.data[total_node_id].operation.name[:4] == 'Tagt':
            arg = ins.qubits[0]
            vertex_name = "%s[%d]%d" % (
                arg._register.name,
                list(circuit.data[total_node_id].qubits)[0]._index,
                qubit_gate_counter[arg],
            )
            # print(vertex_name)
            qubit_gate_counter[Qubit(QuantumRegister(circuit.num_qubits, 'q'), list(circuit.data[total_node_id].qubits)[0]._index)] += 1
            if vertex_name not in node_name_ids:
                node_name_ids[vertex_name] = [curr_node_id,'cx_t'+str(label_node)]
                id_node_names[curr_node_id] = vertex_name
                # vertex_ids[id(vertex)] = curr_node_id
                curr_node_id += 1
                total_node_id += 1
                label_node += 1

        else:
            arg = ins.qubits[0]
            vertex_name = "%s[%d]%d" % (
                arg._register.name,
                arg._index,
                qubit_gate_counter[arg],
            )
            # print('else',vertex_name)
            qubit_gate_counter[arg] += 1

            if vertex_name not in node_name_ids:
                node_name_ids[vertex_name] = [curr_node_id,dag.op_nodes()[total_node_id].op.name+str(label_node)]
                id_node_names[curr_node_id] = vertex_name
                # vertex_ids[id(vertex)] = curr_node_id
                curr_node_id += 1
                total_node_id += 1
                label_node += 1
    # print(node_name_ids, gate_edges, id_node_names)
    for idx in range(len(list(node_name_ids.keys()))):
        for n_idx in range(idx+1, len(list(node_name_ids.keys()))):
            if list(node_name_ids.keys())[idx].split(']')[0].split('[')[1] == list(node_name_ids.keys())[n_idx].split(']')[0].split('[')[1]:
                wire_edges.append((idx,n_idx))
                break
            # print(edges)

    n_vertices = sum([len(node.split(' ')) for node in list(id_node_names.values())])
    # print(vertex_ids)
    return n_vertices, gate_edges, wire_edges, node_name_ids, id_node_names

def append_edge(partition, e1, e2):
    for i in range(len(partition)):
        if e1 in partition[i] and e2 in partition[i]:
            return i

def cutter(G, max_cut, p=5/12, seed=None):
    cut = True
    while cut:
        valid = False
        edge_cut_list = [] # Computed by listing edges between the 2 partitions
        while not valid:
            if seed is not None:
                cut_size, partition = nx.approximation.randomized_partitioning(G, p=p, seed=seed)
            else:
                cut_size, partition = nx.approximation.randomized_partitioning(G, p=p)
            subpartition = [p for p in partition]
            if all(len(p) > 0 for p in subpartition):
                # print(subpartition)
                valid = True
        for i in range(len(subpartition)-1):
            for p1_node in subpartition[i]:
                    for rest in subpartition[i+1:]:
                        for p2_node in rest:
                            if G.has_edge(p1_node,p2_node):
                                    if p1_node < p2_node:
                                        edge_cut_list.append((p1_node,p2_node))
                                    else:
                                        edge_cut_list.append((p2_node,p1_node))
        current_edges = [[]]
        for i in range(len(subpartition)-1):
            current_edges.append([])
        uncut_edge = list(G.edges())
        for eg in edge_cut_list:
            if eg in uncut_edge:
                uncut_edge.remove(eg)
            else:
                eg = (list(eg)[1], list(eg)[0])
                uncut_edge.remove(eg)
        
        for (e1,e2) in uncut_edge:
                # if e1 in subpartition[0] and e2 in subpartition[0]:
                #     current_edges[0].append((e1,e2))
                # elif e1 in subpartition[1] and e2 in subpartition[1]:
                #     current_edges[1].append((e1,e2))
                # else:
                #     print((e1,e2))
                #     raise Exception("Not enough cuts !")
                edge_id = append_edge(subpartition, e1, e2)
                current_edges[edge_id].append((e1,e2))
        G_cut = [nx.Graph()]
        for i in range(len(subpartition)-1):
            G_cut.append(nx.Graph())
        for i in range(len(G_cut)):
            G_cut[i].add_edges_from(current_edges[i])

        cluster = [p for part in partition for p in part]
        G_cluster = [p for part in G_cut for p in list(part.nodes())]
        single_op = set(cluster)-set(G_cluster)
        G_complete= [G_cut[i].subgraph(c) for i in range(len(G_cut)) for c in nx.connected_components(G_cut[i])]
        partition_complete = [list(g.nodes()) for g in G_complete]
        # print(set([p for i in range(len(subpartition)) for p in list(subpartition[i])]), set([p for i in range(len(partition_complete)) for p in partition_complete[i]]) )
        assert set([p for i in range(len(subpartition)) for p in list(subpartition[i])])-set([p for i in range(len(partition_complete)) for p in partition_complete[i]]) == single_op
        if cut_size < max_cut and cut_size > 0 and len(G_complete) > 1:
            cut = False

    return edge_cut_list, G_complete, subpartition, single_op

def disconnect_cnot(circuit):
    count = 0
    cnot = 0
    for d in circuit.data:
        if d.operation.name == 'cx':
            qc_c = QuantumCircuit(1, name='Ctrl%d'%(cnot))
            op_c = qc_c.to_instruction()
            # cnot += 1
            qc_t = QuantumCircuit(1, name='Tagt%d'%(cnot))
            op_t = qc_t.to_instruction()
            cnot += 1
            del circuit.data[count]
            circuit.data.insert(count,(op_c,[Qubit(QuantumRegister(circuit.num_qubits, 'q'), list(d.qubits)[0]._index)],[]))
            circuit.data.insert(count+1,(op_t,[Qubit(QuantumRegister(circuit.num_qubits, 'q'), list(d.qubits)[1]._index)],[]))
        count += 1
    return circuit

def op_split(s, h=False):
    head = s.rstrip('0123456789')
    tail = s[len(head):]
    if not h:
        return tail
    else:
        return head

def find_cnot(data, cnot_idx, idx):
    # print('in')
    if 'cx_c'+str(cnot_idx) in data[idx].operation.name and any('cx_t'+str(cnot_idx) in d.operation.name for d in data):
        # print('right')
        for i, d in enumerate(data):
            if 'cx_t'+str(cnot_idx)==d.operation.name:
                return True, i
        return False, 0
    else:
        # print('wrong')
        return False, 0

def cut_parser(node_name_ids, id_node_names, N, cuts=None, barrier=False):
    subcircuit = QuantumCircuit(int(max([id_node_names[n].split('[')[1].split(']')[0] for n in N]))+1)
    count = 0
    for n in N:
            op = node_name_ids[id_node_names[n]]
            # print(op[1])
            sq = QuantumCircuit(1, name=op[1])
            sq_ins = sq.to_instruction()
            qb = int(id_node_names[n].split('[')[1].split(']')[0])
            subcircuit.data.insert(count,(sq_ins,[Qubit(QuantumRegister(subcircuit.num_qubits, 'q'),qb)],[])) 
            count += 1
    # for i in subcircuit.data:
    #     print(i)
    # print(subcircuit.data)
    for idx, ins in enumerate(subcircuit.data):
        # print(op_split(ins.operation.name), idx)
        found, pos = find_cnot(subcircuit.data, op_split(ins.operation.name), idx)
        if found:
            # print('cnot',ins)
            qb_t = subcircuit.data[pos].qubits[0]._index
            qb_c = ins.qubits[0]._index
            del subcircuit.data[idx]
            cnot = QuantumCircuit(2, name='cnot'+op_split(ins.operation.name))
            cnot_ins = cnot.to_instruction()
            subcircuit.data.insert(idx,(cnot_ins,[Qubit(QuantumRegister(subcircuit.num_qubits, 'q'),qb_c), Qubit(QuantumRegister(subcircuit.num_qubits, 'q'),qb_t)],[]))
            del subcircuit.data[pos]
    # print('-------------------')
    if barrier:
        _len = len(subcircuit.data)
        for index, _instr in enumerate(reversed(subcircuit.data)):
            # if 'cx' in _instr.operation.name or 'cnot' in _instr.operation.name:
            subcircuit.data.insert(_len - index, CircuitInstruction(Barrier(subcircuit.num_qubits), range(subcircuit.num_qubits)))
        subcircuit.data.insert(0, CircuitInstruction(Barrier(subcircuit.num_qubits), range(subcircuit.num_qubits)))
        # for index, _instr in enumerate(subcircuit.data):
        #     if 'barrier' in _instr.operation.name:
        #         subcircuit.data.insert(index, CircuitInstruction(Barrier(subcircuit.num_qubits), range(subcircuit.num_qubits)))

    # implement the reset operation
    # init = QuantumCircuit(1, name='init')
    # init_ins = init.to_instruction()    
    # meas = QuantumCircuit(1, name='meas')
    # meas_ins = meas.to_instruction()
    # # reset_count = 0
    # for (u,v) in cuts:
    #     print(u,v)
    #     name_u = node_name_ids[id_node_names[u]][1]
    #     name_v = node_name_ids[id_node_names[v]][1]
    #     for idx, ins in enumerate(subcircuit.data):
    #         print(ins)  
    #         if ins.operation.name == name_u:
    #             qb = int(id_node_names[u].split('[')[1].split(']')[0])
    #             # del subcircuit.data[idx+1]
    #             subcircuit.data.insert(idx+1,(meas_ins,[Qubit(QuantumRegister(subcircuit.num_qubits, 'q'),qb)],[])) 
    #         elif ins.operation.name == name_v:
    #             qb = int(id_node_names[v].split('[')[1].split(']')[0])
    #             # del subcircuit.data[idx-1]
    #             subcircuit.data.insert(idx,(init_ins,[Qubit(QuantumRegister(subcircuit.num_qubits, 'q'),qb)],[]))   
    #         print(subcircuit.data)
    return subcircuit

def plot_partition(G_cut, label_dict):
    fig, axes = plt.subplots(nrows=1, ncols=len(G_cut), figsize=(17,5))
    ax = axes.flatten()

    for i in range(len(G_cut)):
        nx.draw_networkx(G_cut[i], ax=ax[i], pos=nx.circular_layout(G_cut[i]), node_color='r', edge_color='b', labels={node: label_dict[node] for node in G_cut[i].nodes()}, with_labels=True)
        ax[i].set_axis_off()

    plt.show()

def graph_to_circ(subcirc, circuit):
    for idx, _instruction in reversed(list(enumerate(subcirc.data))):
        if 'cx' in _instruction.operation.name:
            del subcirc.data[idx]
    for idx, ins in enumerate(subcirc.data):
        if 'cnot' in ins.operation.name:
            count = op_split(ins.operation.name)
            del subcirc.data[idx] 
            old_ins = circuit.data[int(count)]
            assert old_ins.operation.name == 'cx'
            subcirc.data.insert(idx,(old_ins.operation,[ins.qubits[0], ins.qubits[1]],[]))
        else:
            count = op_split(ins.operation.name)
            del subcirc.data[idx] 
            old_ins = circuit.data[int(count)]
            if old_ins.operation.name != op_split(ins.operation.name, True):
                print(old_ins.operation.name, op_split(ins.operation.name, True))
                raise Exception('Gate not match !')
            subcirc.data.insert(idx,(old_ins.operation,[ins.qubits[0]],[]))  
    return subcirc

def find_cuts(circuit, max_cut, seed=None, plot=False):
    n_vertices, gate_edges, wire_edges, node_name_ids, id_node_names = read_circ(disconnect_cnot(circuit.copy()))
    G = nx.Graph() 
    all_edges = gate_edges+wire_edges
    G.add_edges_from(all_edges)

    edge_cut_list, G_cut, partition, single_op = cutter(G, max_cut, p=5/12, seed=seed)

    if plot:
        label_dict = {}
        for i in list(node_name_ids.values()):
            label_dict[i[0]] = i[1]
        plot_partition(G_cut, label_dict)

    if single_op is not None:
        single_circs = []
    for op_s in list(single_op):
        subcircuit = QuantumCircuit(1)
        op = node_name_ids[id_node_names[op_s]]
        sq = QuantumCircuit(1, name=op[1])
        sq_ins = sq.to_instruction()
        subcircuit.data.insert(0,(sq_ins,[Qubit(QuantumRegister(subcircuit.num_qubits, 'q'),0)],[])) 
        single_circs.append(subcircuit)

    n = []
    s = []
    for i in range(len(G_cut)):
        n.append(list(G_cut[i].nodes()))
        # print(n)
        s.append(cut_parser(node_name_ids, id_node_names, sorted(n[-1])))

    gate_cuts = 0
    wire_cuts = 0
    for e in edge_cut_list:
        u,v = e
        if (u,v) in gate_edges or (v,u) in gate_edges:
            gate_cuts += 1
        elif (u,v) in wire_edges or (v,u) in wire_edges:
            wire_cuts += 1
        else:
            raise Exception('Cuts are interpreted wrongly !')

    assert gate_cuts+wire_cuts == len(edge_cut_list) and gate_cuts+wire_cuts < max_cut

    subcircuits = [graph_to_circ(sub, circuit) for sub in single_circs+s]

    return subcircuits, gate_cuts, wire_cuts

def check_estimation_validity(circ):
    valid = False
    while not valid:
        for i, sub in enumerate(reversed(circ)):
            if len(sub.data) == 0 or all([s.operation.name not in {'rx', 'ry', 'rz'} for s in sub.data]):
                circ.pop(len(circ)-1-i)
        if all(len(s.data) > 0 for s in circ) and not all([s.operation.name not in {'rx', 'ry', 'rz'} for s in sub.data]):
            valid = True
    return circ