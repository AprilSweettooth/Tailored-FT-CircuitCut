from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import CouplingMap
from qiskit.synthesis import SuzukiTrotter
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit import transpile
from qiskit_nature.second_q.hamiltonians import IsingModel
from qiskit_nature.second_q.hamiltonians.lattices import *
from qiskit.quantum_info import SparsePauliOp

from openfermion.ops import QubitOperator

import numpy as np

def qubitop_to_pauliop(qubit_operator):
    """Convert an openfermion QubitOperator to a qiskit WeightedPauliOperator.
    Args:
        qubit_operator ("QubitOperator"): Openfermion QubitOperator to convert to a qiskit.WeightedPauliOperator.
    Returns:
        paulis ("WeightedPauliOperator"): Qiskit WeightedPauliOperator.
    """
    if not isinstance(qubit_operator, QubitOperator):
        raise TypeError("qubit_operator must be an openFermion QubitOperator object.")
    ham = []
    coeffs = []
    temp_n = 1
    for qubit_terms, coefficient in qubit_operator.terms.items():
        for tensor_term in qubit_terms:
            # print(tensor_term[0])
            temp_n = max(tensor_term[0]+1, temp_n)
    n_qubits = temp_n
    for qubit_terms, coefficient in qubit_operator.terms.items():
        pauli_label = ['I' for _ in range(n_qubits)]
        coeffs.append(coefficient)
        
        for tensor_term in qubit_terms:
            pauli_label[tensor_term[0]] = tensor_term[1]

        ham.append(''.join(pauli_label))
    # print(paulis)
    return SparsePauliOp(ham, coeffs)


def build_hamiltonian(hamiltonian, t, num_timesteps=10):
    
    second_order_formula = SuzukiTrotter()

    dt = t/ num_timesteps

    trotter_step_second_order = PauliEvolutionGate(hamiltonian, dt, synthesis=second_order_formula)

    circuit = QuantumCircuit(hamiltonian[0].num_qubits)

    for _ in range(num_timesteps):
        circuit.append(trotter_step_second_order, range(hamiltonian[0].num_qubits))

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

def to_sparse_op(lattice):
        """Return the Hamiltonian of the Ising model in terms of ``SpinOp``.

        Returns:
            SpinOp: The Hamiltonian of the Ising model.
        """
        ham = []
        coeffs = []
        weighted_edge_list = lattice.weighted_edge_list
        # kinetic terms
        idt = list('I' * (max(lattice.node_indexes)  + 1))
        for node_a, node_b, weight in weighted_edge_list:
            if node_a == node_b:
                index = node_a
                h_temp = idt.copy()
                h_temp[index] = 'X'
                ham.append(''.join(h_temp))
                coeffs.append(weight)

        for node_a, node_b, weight in weighted_edge_list:
            if node_a != node_b:
                index_left = node_a
                index_right = node_b
                h_temp = idt.copy()
                h_temp[index_left] = 'Z'
                h_temp[index_right] = 'Z'
                ham.append(''.join(h_temp))
                coeffs.append(weight)

        return SparsePauliOp(ham, coeffs)

def build_Ising_hamiltonian(num_spins, t=0.1, h=-1, Dimension=1):
    
    if Dimension==1:
        lattice_map = CouplingMap.from_line(num_spins, bidirectional=False)
        edgelist = lattice_map.graph.edge_list()
        even_hamlist = []
        odd_hamlist = []
        z_counter = 0
        x_counter = 0
        for edge in edgelist:
            if edge[0]%2 == 0:
                even_hamlist.append(("ZZ", edge, -1))
                z_counter += 1
            else:
                odd_hamlist.append(("ZZ", edge, -1))
                z_counter += 1
        for qubit in lattice_map.physical_qubits:
            if edge[0]%2 == 0:
                even_hamlist.append(("X", [qubit], h))
                x_counter += 1
            else:
                odd_hamlist.append(("X", [qubit], h))
                x_counter += 1
        
        even_hamiltonian = SparsePauliOp.from_sparse_list(even_hamlist, 
                                                        num_qubits=num_spins)
        odd_hamiltonian =  SparsePauliOp.from_sparse_list(odd_hamlist, 
                                                        num_qubits=num_spins)
        hamiltonian = even_hamiltonian + odd_hamiltonian

    else:
        rows, cols = Dimension
        assert num_spins == rows*cols
        square_lattice = SquareLattice(rows=rows, cols=cols, edge_parameter=-1.0, onsite_parameter=h, boundary_condition=BoundaryCondition.PERIODIC)
        ising_model = IsingModel(square_lattice)
        hamiltonian = to_sparse_op(square_lattice)
        # print(hamiltonian)
        ham = ising_model.second_q_op()
        z_counter = len([z for z in list(ham) if 'Z' in z])
        x_counter = len([x for x in list(ham) if 'X' in x])

    # second_order_formula = SuzukiTrotter()
    fourth_order_formula = SuzukiTrotter(order=4)
    order = 4
    epsilon = 0.0025
    # num_timesteps = 80
    # print((t**(1+1/order) * (z_counter+h*x_counter)**(1+1/order)) / (epsilon)**(1/order))
    num_timesteps = int(np.ceil( (t**(1+1/order) * (z_counter+h*x_counter)**(1+1/order)) / (epsilon)**(1/order)))
    epsilon = (t**(1+1/order) * (z_counter+h*x_counter)**(1+1/order) / num_timesteps)**order
    # print('error: ', np.real(epsilon))
    print('number of time steps: ', num_timesteps)

    dt = t/ num_timesteps

    trotter_step_second_order = PauliEvolutionGate(hamiltonian, dt, synthesis=fourth_order_formula)

    circuit = QuantumCircuit(num_spins)

    for _ in range(num_timesteps):
        circuit.append(trotter_step_second_order, range(num_spins))

    target_basis = ['rx', 'ry', 'rz', 'h', 'cx']
    circuit = transpile(circuit,
                        basis_gates=target_basis, 
                        optimization_level=1) 

    return circuit