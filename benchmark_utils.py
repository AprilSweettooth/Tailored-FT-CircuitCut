from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import CouplingMap
from qiskit.synthesis import SuzukiTrotter, QDrift
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit import transpile
from qiskit_nature.second_q.hamiltonians.lattices import *
from qiskit.quantum_info import SparsePauliOp
import qiskit.circuit.library as library
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.problems import LatticeModelProblem
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel, HeisenbergModel, IsingModel
from qiskit.circuit.library import QAOAAnsatz
# from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.circuit.random import random_circuit
from openfermion.ops import QubitOperator
from qiskit.circuit.library import *
from qiskit.circuit import CircuitError

import numpy as np
import rustworkx as rx
import math, random

from uccsd_ansatz import *
from ripple_carry_adder import *
from Shor_Normal_QFT import *
from fable_custom import fable

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

def spinop_to_sparse_op(spinop):
        """Return the Hamiltonian of the Ising model in terms of ``SpinOp``.

        Returns:
            SpinOp: The Hamiltonian of the Ising model.
        """
        ham = []
        coeffs = []

        idt = list('I' * spinop.num_spins)
        for hamilton, weight in spinop.terms():
            h_temp = idt.copy()
            for hm in hamilton:
                h, idx = hm
                h_temp[idx] = h
            ham.append(''.join(h_temp))
            coeffs.append(weight)
        return SparsePauliOp(ham, coeffs)

def get_Ising_operators(lattice, J, g) -> tuple[SparsePauliOp, dict[str, SparsePauliOp] | None]:
        ising = IsingModel(    
            lattice.uniform_parameters(
                uniform_interaction=J,
                uniform_onsite_potential=g,
            ),)
        
        lmp = LatticeModelProblem(ising)
        main_second_q_op, aux_second_q_ops = lmp.second_q_ops()
        # for ham, coeff in main_second_q_op.terms():
        #     print(ham, coeff)
        main_operator = spinop_to_sparse_op(main_second_q_op)

        return main_operator


def get_FH_operators(lattice, t, v, u) -> tuple[SparsePauliOp, dict[str, SparsePauliOp] | None]:
        fhm = FermiHubbardModel(
            lattice.uniform_parameters(
                uniform_interaction=t,
                uniform_onsite_potential=v,
            ),
            onsite_interaction=u,
        )

        lmp = LatticeModelProblem(fhm)
        main_second_q_op, aux_second_q_ops = lmp.second_q_ops()

        main_operator = JordanWignerMapper().map(main_second_q_op)

        return main_operator

def get_HB_operators(lattice, J, h) -> tuple[SparsePauliOp, dict[str, SparsePauliOp] | None]:
        hbm = HeisenbergModel(lattice, J, h)
        lmp = LatticeModelProblem(hbm)
        main_second_q_op, aux_second_q_ops = lmp.second_q_ops()
        main_operator = spinop_to_sparse_op(main_second_q_op)

        return main_operator


def build_lattice_hamiltonian(num_spins=9, t=20, h=1, Dimension=1, method='trotter', model='Ising'):
    
    # assume basic 1d model to be ising
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

        # hamlist = []
        # for edge in edgelist:
        #         hamlist.append(("ZZ", edge, -1))
        # for qubit in lattice_map.physical_qubits:
        #         hamlist.append(("X", [qubit], h))
        
        # hamiltonian =  SparsePauliOp.from_sparse_list(hamlist, 
        #                                                 num_qubits=num_spins)

    else:
        rows, cols = Dimension
        # assert num_spins == rows*cols
        square_lattice = SquareLattice(rows=rows, cols=cols, boundary_condition=BoundaryCondition.OPEN)
        # ising_model = IsingModel(square_lattice)
        # print(square_lattice)
        if model=='Ising':
            # hamiltonian = to_sparse_op(square_lattice)
            # print(square_lattice.weighted_edge_list)
            hamiltonian = get_Ising_operators(square_lattice, J=-1, g=1)
            # print(hamiltonian)
        elif model=='FH':
            hamiltonian = get_FH_operators(square_lattice, t=-1, v=1, u=5)
            # print(hamiltonian)
        elif model=='Heisenberg':
            hamiltonian = get_HB_operators(square_lattice, (1.0, 1.0, 1.0), (0.0, 0.0, 1.0))
            # print(hamiltonian)
        # ham = ising_model.second_q_op()
        # z_counter = len([z for z in list(ham) if 'Z' in z])
        # x_counter = len([x for x in list(ham) if 'X' in x])

    # second_order_formula = SuzukiTrotter()
    epsilon = 0.078125
    # epsilon = 0.1
    if method=='trotter':
        fourth_order_formula = SuzukiTrotter(order=4)
        order = 4
        # num_timesteps = 10
        # print((t**(1+1/order) * (z_counter+h*x_counter)**(1+1/order)) / (epsilon)**(1/order))
        # num_timesteps = int(np.ceil( (t**(1+1/order) * (z_counter+h*x_counter)**(1+1/order)) / (epsilon)**(1/order)))
        # print((t**(1+1/order) / num_timesteps)**order)
        num_timesteps = int(np.ceil( t**(1+1/order) / (epsilon)**(1/order)))
        # epsilon = (t**(1+1/order) * (z_counter+h*x_counter)**(1+1/order) / num_timesteps)**order
        # print('error: ', np.real(epsilon))
        # print('number of time steps: ', num_timesteps)

        dt = t/ num_timesteps

        trotter_step = PauliEvolutionGate(hamiltonian, dt, synthesis=fourth_order_formula)

        circuit = QuantumCircuit(hamiltonian.num_qubits)

        for _ in range(1):
            circuit.append(trotter_step, range(hamiltonian.num_qubits))
    elif method=='qdrift':
        t = 0.01
        epsilon = 0.01
        num_timesteps = int(np.ceil( (t**(2)) / (epsilon)))
        print('number of time steps: ', num_timesteps)
        qdrift = QDrift(reps=1)
        dt = t/ num_timesteps
        qdrift_circ = PauliEvolutionGate(hamiltonian, dt, synthesis=qdrift)
        circuit = QuantumCircuit(num_spins)
        # circuit.append(qdrift_circ)
        for _ in range(num_timesteps):
            circuit.append(qdrift_circ, range(num_spins))


    target_basis = ['rx', 'ry', 'rz', 'h', 'rzz']
    circuit = transpile(circuit,
                        basis_gates=target_basis, 
                        optimization_level=3) 
    return circuit

def build_max_cut_paulis(graph: rx.PyGraph) -> list[tuple[str, float]]:
    """Convert the graph to Pauli list.

    This function does the inverse of `build_max_cut_graph`
    """
    pauli_list = []
    for edge in list(graph.edge_list()):
        paulis = ["I"] * len(graph)
        paulis[edge[0]], paulis[edge[1]] = "Z", "Z"

        weight = graph.get_edge_data(edge[0], edge[1])

        pauli_list.append(("".join(paulis)[::-1], weight))

    return pauli_list


def QAOA_ansatz(num_qubit, reps):
    n = num_qubit
    graph = rx.PyGraph()
    graph.add_nodes_from(np.arange(0, n, 1))
    elist = []
    # print(QiskitRuntimeService(channel='ibm_quantum').backend("ibm_brisbane").coupling_map)
    # QiskitRuntimeService(channel='ibm_quantum').backend("ibm_brisbane").coupling_map
    coupling = [[1, 0], [2, 1], [3, 2], [4, 3], [4, 5], [4, 15], [6, 5], [6, 7], [7, 8], [8, 9], [10, 9], [10, 11], [11, 12], [12, 17], [13, 12], [14, 0], [14, 18], [15, 22], [16, 8], [16, 26], [17, 30], [18, 19], [20, 19], [20, 33], [21, 20], [21, 22], [22, 23], [24, 23], [24, 34], [25, 24], [26, 25], [27, 26], [28, 27], [28, 29], [28, 35], [30, 29], [30, 31], [31, 32], [32, 36], [33, 39], [34, 43], [35, 47], [36, 51], [37, 38], [39, 38], [40, 39], [40, 41], [41, 53], [42, 41], [42, 43], [43, 44], [44, 45], [46, 45], [46, 47], [48, 47], [48, 49], [50, 49], [50, 51], [52, 37], [52, 56], [53, 60], [54, 45], [54, 64], [55, 49], [55, 68], [56, 57], [57, 58], [58, 59], [58, 71], [59, 60], [60, 61], [62, 61], [62, 63], [62, 72], [63, 64], [65, 64], [65, 66], [67, 66], [67, 68], [69, 68], [69, 70], [73, 66], [74, 70], [74, 89], [75, 90], [76, 75], [77, 71], [77, 76], [77, 78], [79, 78], [79, 80], [80, 81], [81, 72], [81, 82], [82, 83], [83, 92], [84, 83], [85, 73], [85, 84], [85, 86], [86, 87], [87, 88], [88, 89], [91, 79], [92, 102], [93, 87], [93, 106], [94, 90], [94, 95], [95, 96], [97, 96], [97, 98], [98, 91], [99, 98], [100, 99], [100, 110], [101, 100], [101, 102], [102, 103], [104, 103], [105, 104], [105, 106], [107, 106], [108, 107], [108, 112], [109, 96], [110, 118], [111, 104], [112, 126], [113, 114], [114, 109], [114, 115], [116, 115], [116, 117], [117, 118], [118, 119], [120, 119], [121, 120], [122, 111], [122, 121], [122, 123], [124, 123], [125, 124], [125, 126]]
    for edge in coupling:
        if edge[0] < n and edge[1] < n:
            elist.append((edge[0], edge[1], 1.0))
    graph.add_edges_from(elist)
    # from rustworkx.visualization import mpl_draw

    # mpl_draw(graph)
    max_cut_paulis = build_max_cut_paulis(graph)

    cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)

    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)

    np.random.seed(seed=42)
    gamma_initial = np.random.uniform(0, 2 * np.pi, reps).tolist()
    beta_initial = np.random.uniform(0, np.pi, reps).tolist()
    params0 = np.array(beta_initial + gamma_initial)
    # print(circuit.parameters.data)
    param_names = [circuit.parameters.data[r].name for r in range(len(params0))]
    for r in range(len(params0)):
        # print(circuit.parameters.data[r])
        # print(r)
        circuit.assign_parameters({param_names[r]:params0[r]}, inplace=True)

    target_basis = ['rx', 'ry', 'rz', 'h', 'rzz']
    circuit = transpile(circuit,
                        basis_gates=target_basis, 
                        optimization_level=1) 

    return circuit


def gen_uccsd(width, parameters="random", seed=None, barriers=False, regname=None):
    """
    Generate a UCCSD ansatz with the given width (number of qubits).
    """

    uccsd = UCCSD(
        width, parameters=parameters, seed=seed, barriers=barriers, regname=regname
    )

    circ = uccsd.gen_circuit()

    return circ


def gen_adder(
    nbits=None, a=0, b=0, use_toffoli=False, barriers=True, measure=False, regname=None
):
    """
    Generate an n-bit ripple-carry adder which performs a+b and stores the
    result in the b register.

    Based on the implementation of: https://arxiv.org/abs/quant-ph/0410184v1
    """

    adder = RCAdder(
        nbits=nbits,
        a=a,
        b=b,
        use_toffoli=use_toffoli,
        barriers=barriers,
        measure=measure,
        regname=regname,
    )

    circ = adder.gen_circuit()

    return circ


def generate_circ(num_qubits, circuit_type, seed=42):
    reg_name='q'
    random.seed(seed)
    full_circ = None
    num_trials = 100
    while num_trials:
        if circuit_type == "qft":
            full_circ = library.QFT(
                num_qubits=num_qubits, approximation_degree=0,do_swaps=False
            ).decompose()
        elif circuit_type == 'QPE':
            full_circ = library.PhaseEstimation(num_evaluation_qubits=num_qubits, unitary=random_circuit(num_qubits, depth=1, max_operands=1, seed=42), iqft=None, name='QPE').decompose(reps=3)
        elif circuit_type == 'uccd':
            full_circ = gen_uccsd(num_qubits, parameters="random", seed=seed, barriers=False, regname=reg_name)
        elif circuit_type == "aqft":
            approximation_degree = int(math.log(num_qubits, 2) + 2)
            full_circ = library.QFT(
                num_qubits=num_qubits,
                approximation_degree=num_qubits - approximation_degree,
                do_swaps=False,
            ).decompose()
        elif circuit_type == "adder":
            full_circ = gen_adder(
                nbits=int((num_qubits - 2) / 2), barriers=False, regname=reg_name
            )
        elif circuit_type == 'block_encoding':
            # generate a random matrix and block encode it
            np.random.seed(seed=42)
            A = np.random.randn(2**num_qubits, 2**num_qubits)
            full_circ, alpha = fable(A, 0)
        elif circuit_type == 'shor':
            full_circ = shor_factoring(num_qubits)
            # target_basis = ['u', 'swap', 'rz', 'h', 'cx']
            # full_circ = transpile(shor_circ,
            #                     basis_gates=target_basis, 
            #                     optimization_level=1) 
        else:
            raise Exception("Illegal circuit type:", circuit_type)

        if full_circ is not None and full_circ.num_tensor_factors() == 1:
            break
        elif full_circ is not None:
            break
        else:
            full_circ = None
            num_trials -= 1
    # print(full_circ.num_qubits)
    # assert full_circ is None or full_circ.num_qubits == num_qubits or full_circ.num_qubits == int(2*num_qubits)
    return full_circ



def random_circuit(num_qubits, depth, max_operands=3, measure=False, conditional=False, reset=False, seed=None):

    if max_operands < 1 or max_operands > 3:
        raise CircuitError("max_operands must be between 1 and 3")

    one_q_ops = [IGate, U1Gate, U2Gate, U3Gate, XGate, YGate, ZGate, HGate, SGate, SdgGate, TGate, TdgGate, RXGate, RYGate, RZGate]
    one_param = [U1Gate, RXGate, RYGate, RZGate, RZZGate, CU1Gate, CRZGate]
    two_param = [U2Gate]
    three_param = [U3Gate, CU3Gate]
    two_q_ops = [CXGate, RZZGate]
    three_q_ops = [CCXGate, CSwapGate]

    qr = QuantumRegister(num_qubits, "q")
    qc = QuantumCircuit(num_qubits)

    if measure or conditional:
        cr = ClassicalRegister(num_qubits, "c")
        qc.add_register(cr)

    if reset:
        one_q_ops += [Reset]

    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.default_rng(seed)

    # apply arbitrary random operations at every depth
    for _ in range(depth):
        for idx in range(qc.num_qubits):
            qc.t(idx)
        # choose either 1, 2, or 3 qubits for the operation
        remaining_qubits = list(range(num_qubits))
        rng.shuffle(remaining_qubits)
        while remaining_qubits:
            max_possible_operands = min(len(remaining_qubits), max_operands)
            if max_possible_operands < 2:
                break
            num_operands = 2
            operands = [remaining_qubits.pop() for _ in range(num_operands)]
            if num_operands == 1:
                operation = rng.choice(one_q_ops)
            elif num_operands == 2:
                operation = rng.choice(two_q_ops)
            elif num_operands == 3:
                operation = rng.choice(three_q_ops)
            if operation in one_param:
                num_angles = 1
            elif operation in two_param:
                num_angles = 2
            elif operation in three_param:
                num_angles = 3
            else:
                num_angles = 0
            angles = [rng.uniform(0, 2 * np.pi) for x in range(num_angles)]
            register_operands = [qr[i] for i in operands]
            op = operation(*angles)

            # with some low probability, condition on classical bit values
            if conditional and rng.choice(range(10)) == 0:
                value = rng.integers(0, np.power(2, num_qubits))
                op.condition = (cr, value)

            qc.append(op, register_operands)

    if measure:
        qc.measure(qr, cr)

    return qc
