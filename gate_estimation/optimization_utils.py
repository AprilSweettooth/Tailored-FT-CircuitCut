from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import CouplingMap
from qiskit.synthesis import SuzukiTrotter
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit import transpile

import numpy as np

def build_1D_hamiltonian(num_spins, t, num_timesteps=10, h=1):
    
    ### Your code goes here ###
    lattice_map = CouplingMap.from_line(num_spins, bidirectional=False)
    edgelist = lattice_map.graph.edge_list()
    even_hamlist = []
    odd_hamlist = []
    z_counter = 0
    x_counter = 0
    for edge in edgelist:
        if edge[0]%2 == 0:
            # even_hamlist.append(("XX", edge, 1.))
            # even_hamlist.append(("YY", edge, 1.))
            even_hamlist.append(("ZZ", edge, 1))
            z_counter += 1
        else:
            # odd_hamlist.append(("XX", edge, 1.))
            # odd_hamlist.append(("YY", edge, 1.))
            odd_hamlist.append(("ZZ", edge, 1))
            z_counter += 1
    for qubit in lattice_map.physical_qubits:
        if edge[0]%2 == 0:
            even_hamlist.append(("X", [qubit], h))
            x_counter += 1
            #even_hamlist.append(("Z", [qubit], h
        else:
            odd_hamlist.append(("X", [qubit], h))
            x_counter += 1
            #odd_hamlist.append(("Z", [qubit], h
    
    even_hamiltonian = SparsePauliOp.from_sparse_list(even_hamlist, 
                                                      num_qubits=num_spins)
    odd_hamiltonian =  SparsePauliOp.from_sparse_list(odd_hamlist, 
                                                      num_qubits=num_spins)
    hamiltonian = even_hamiltonian + odd_hamiltonian

    second_order_formula = SuzukiTrotter()

    epsilon = (t**(1+0.5) * (z_counter+h*x_counter)**(1+0.5) / num_timesteps)**2
    print('error: ', epsilon)

    dt = t/ num_timesteps

    trotter_step_second_order = PauliEvolutionGate(hamiltonian, dt, synthesis=second_order_formula)

    circuit = QuantumCircuit(num_spins)

    for _ in range(num_timesteps):
        circuit.append(trotter_step_second_order, range(num_spins))

    target_basis = ['rx', 'ry', 'rz', 'h', 'cx']
    circuit = transpile(circuit,
                        basis_gates=target_basis, 
                        optimization_level=1) 

    return circuit

def cuts_filter(circuit):
    _len = len(circuit.data)
    for index, _instr in enumerate(reversed(circuit.data)):
        if 'qpd' in _instr.operation.name:
            del circuit.data[_len - index-1]
    return circuit

def accel_asc(n):
    # https://stackoverflow.com/questions/10035752/elegant-python-code-for-integer-partitioning
    return set(accel_asc_yield(n))


def accel_asc_yield(n):
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield tuple(a[: k + 2])
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield tuple(a[: k + 1])

def error_budget_partition(overall_budget, num_subcirc):
    n = int(overall_budget * 5000)
    partition = list(accel_asc(n))
    sub_part = []
    for p in partition:
        if len(p) == num_subcirc:
            sub_part.append(list(p))
    return sub_part