from benchmark_dev import *

# lattice2D(lattice_size=range(3,11), lattice='Ising')
# lattice2D(lattice_size=range(3,11), lattice='Heisenberg')
# lattice2D(lattice_size=range(3,7), lattice='FH')
# algo_benchmark(num_qubits_list=range(5,21,5), circuit_type='qft')

# algo_benchmark(num_qubits_list=range(4,11,2), circuit_type='uccd')
# algo_benchmark(num_qubits_list=range(2,9,2), circuit_type='QPE')
# algo_benchmark(num_qubits_list=range(2,6,1), circuit_type='block_encoding')
# algo_benchmark(num_qubits_list=range(5,21,5), circuit_type='random')

# lattice2D(lattice_size=range(3,7), lattice='Heisenberg')
algo_benchmark(num_qubits_list=range(10,101,10), circuit_type='QAOA')
# lattice2D(lattice_size=range(2,5), lattice='FH')

# algo_benchmark(num_qubits_list=[35], circuit_type='shor')
# algo_benchmark(num_qubits_list=[15, 35, 303, 37*11, 41*37, 103*37], circuit_type='shor')