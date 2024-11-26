import numpy as np
import json
import sys
from qiskit import transpile
from qiskit_addon_cutting_custom.automated_cut_finding import *
from qiskit_addon_cutting_custom import *
from qiskit.converters import circuit_to_dag
from azure.quantum import Workspace
from azure.quantum.target.microsoft import MicrosoftEstimator
# from hamlib.hamlib_snippets import *
from benchmark_utils import *
# from gate_estimation.resource_analysis import *
from Shor_Normal_QFT import *
from qsharp.estimator import EstimatorParams
from qsharp.interop.qiskit import ResourceEstimatorBackend

workspace = Workspace (
   resource_id = "/subscriptions/e0d51919-2dda-4b7e-a8e7-77cc18580acb/resourceGroups/AzureQuantum/providers/Microsoft.Quantum/Workspaces/QECBenchmarking",
   location = "uksouth"
)

estimator = MicrosoftEstimator(workspace)

def lattice2D(lattice_size, lattice):
    big_data = {}
    for size in lattice_size:
        big_data[lattice+str(size)+' by '+str(size)] = {}
        if lattice=='Ising':
            circuit = build_lattice_hamiltonian(num_spins=size**2, Dimension=(size, size), model=lattice)
        elif lattice=='FH':
            if size%2 == 0:
                circuit = build_lattice_hamiltonian(num_spins=int(size//2)*size, Dimension=(int(size//2), size), model=lattice)
            elif size%2 != 0:
                circuit = build_lattice_hamiltonian(num_spins=int((1+size)//2)*size, Dimension=(int((1+size)//2), size), model=lattice)
        elif lattice=='Heisenberg':
            circuit = build_lattice_hamiltonian(num_spins=size**2, Dimension=(size, size), model=lattice)

        params = estimator.make_params(num_items=1)
        params.error_budget = 0.001
        params.constraints.max_t_factories = 1
        job = estimator.submit(circuit, input_params=params)
        r = job.get_results(timeout_secs=3000)
        dag = circuit_to_dag(circuit)
        big_data[lattice+str(size)+' by '+str(size)]['longest_path'] = dag.count_ops_longest_path()
        big_data[lattice+str(size)+' by '+str(size)]['two_qubit_gates'] = circuit.num_nonlocal_gates()
        big_data[lattice+str(size)+' by '+str(size)]['overall_physical_qubits'] = r['physicalCounts']['physicalQubits']
        big_data[lattice+str(size)+' by '+str(size)]['overall_runtime'] = r['physicalCounts']['runtime']*1e-9
        big_data[lattice+str(size)+' by '+str(size)]['logical_depth_qubits'] = r['physicalCounts']['breakdown']['logicalDepth']
        big_data[lattice+str(size)+' by '+str(size)]['algorithmic_depth_qubits'] = r['physicalCounts']['breakdown']['algorithmicLogicalDepth']
        big_data[lattice+str(size)+' by '+str(size)]['logical_qubits'] = r['logicalQubit']['physicalQubits']
        print('\n', 'total physical qubits required: ', r['physicalCounts']['physicalQubits'], '\n', 'total runtime', r['physicalCounts']['runtime']*1e-9)
        # print('\n', 'logical depth', r['physicalCounts']['breakdown']['logicalDepth'], 'algo depth', r['physicalCounts']['breakdown']['algorithmicLogicalDepth'], 'physical qubits', r['logicalQubit']['physicalQubits'])

        for ratio in range(1,size**2, size):
            big_data[lattice+str(size)+' by '+str(size)][str(ratio)] = {}
            optimization_settings = OptimizationParameters(seed=111) 
            device_constraints = DeviceConstraints(qubits_per_subcircuit=ratio)
            # print('max no. of qubit per subcircuit: ', ratio)
            cut_circuit, metadata = find_cuts(circuit, optimization_settings, device_constraints)
            # print(
            #     f'Found {size}x{size} Ising model solution using {len(metadata["cuts"])} cuts ({len(metadata["gate_cuts"])} gate {len(metadata["wire_cuts"])} wire cuts)'
            # )
            big_data[lattice+str(size)+' by '+str(size)][str(ratio)]['gate_cuts'] = len(metadata["gate_cuts"])
            big_data[lattice+str(size)+' by '+str(size)][str(ratio)]['wire_cuts'] = len(metadata["wire_cuts"])  
            qc_w_ancilla = cut_wires(cut_circuit)
            partitioned_problem = partition_problem(circuit=qc_w_ancilla)
            subcircuit = partitioned_problem.subcircuits
            subcircuits = [cuts_filter(s) for s in subcircuit.values()]

            quantum_runtime = []
            physical_qubits = []
            total_partition = [c.num_qubits for c in subcircuits]
            error_buget = [0.001*p/sum(total_partition) for p in total_partition]
            for idx, s in enumerate(subcircuits):
                params = estimator.make_params(num_items=1)
                params.constraints.max_t_factories = 1
                params.error_budget = error_buget[idx]
                job = estimator.submit(s, input_params=params)
                try:
                    r = job.get_results(timeout_secs=3000)
                    quantum_runtime.append(r['physicalCounts']['runtime'])
                    physical_qubits.append(r['physicalCounts']['physicalQubits'])
                    big_data[lattice+str(size)+' by '+str(size)][str(ratio)]['two_qubit_gates'] = s.num_nonlocal_gates() 
                    big_data[lattice+str(size)+' by '+str(size)][str(ratio)]['physical_qubits_subcircuit-'+str(idx+1)] = r['physicalCounts']['physicalQubits']
                    big_data[lattice+str(size)+' by '+str(size)][str(ratio)]['logical_depth_qubits_subcircuit-'+str(idx+1)] = r['physicalCounts']['breakdown']['logicalDepth']
                    big_data[lattice+str(size)+' by '+str(size)][str(ratio)]['algorithmic_depth_qubits_subcircuit-'+str(idx+1)] = r['physicalCounts']['breakdown']['algorithmicLogicalDepth']
                    big_data[lattice+str(size)+' by '+str(size)][str(ratio)]['logical_qubits_subcircuit-'+str(idx+1)] = r['logicalQubit']['physicalQubits']
                    big_data[lattice+str(size)+' by '+str(size)][str(ratio)]['runtime_subcircuit-'+str(idx+1)] = r['physicalCounts']['runtime']*1e-9
                except:
                    print('no magic state in subcirc'+str(idx+1))
                # print('\n', 'subcircuit-', idx+1, ': physical qubits--', r['physicalCounts']['physicalQubits'], 'runtime (in microsec)--', r['physicalCounts']['runtime']*1e-3)
                # print('\n', 'logical depth', r['physicalCounts']['breakdown']['logicalDepth'], 'algo depth', r['physicalCounts']['breakdown']['algorithmicLogicalDepth'], 'physical qubits', r['logicalQubit']['physicalQubits'])
            big_data[lattice+str(size)+' by '+str(size)][str(ratio)]['runtime_after_cutting'] = sum(quantum_runtime)*1e-9
            big_data[lattice+str(size)+' by '+str(size)][str(ratio)]['qubits_after_cutting'] = max(physical_qubits)
            print('\n', 'quantum runtime total: ', sum(quantum_runtime)*1e-9, 'number of physical qubits: ', max(physical_qubits))
        
    file_name = '2D_'+lattice+'_data.json'
    if file_name is not None:
        with open(file_name, 'w') as fp:
            json.dump(big_data, fp)
    return big_data


def algo_benchmark(num_qubits_list, circuit_type):
    big_data = {}
    for num_qubits in num_qubits_list:
        big_data[circuit_type+str(num_qubits)] = {}
        if circuit_type=='qft':
            circuit = generate_circ(num_qubits, circuit_type='qft', seed=42)
        elif circuit_type=='shor':
            circuit = generate_circ(num_qubits, circuit_type='shor', seed=42)
        elif circuit_type=='uccd':
            circuit = generate_circ(num_qubits, circuit_type='uccd', seed=42)
        elif circuit_type=='QPE':
            circuit = generate_circ(num_qubits, circuit_type='QPE', seed=42)
        elif circuit_type=='block_encoding':
            circuit = generate_circ(num_qubits, circuit_type='block_encoding', seed=42)
        elif circuit_type=='random':
            circuit = random_circuit(num_qubits, depth = num_qubits**2, seed=42)
        elif circuit_type=='QAOA':
            circuit = QAOA_ansatz(num_qubits, reps = 10)

        params = EstimatorParams()
        params.error_budget = 0.001
        params.constraints.max_t_factories = 1
        backend = ResourceEstimatorBackend()
        job = backend.run(circuit, params)
        r = job.result()
        # params = estimator.make_params(num_items=1)
        # params.error_budget = 0.001
        # params.constraints.max_t_factories = 1
        # job = estimator.submit(circuit, input_params=params)
        # r = job.get_results(timeout_secs=3000)
        dag = circuit_to_dag(circuit)
        big_data[circuit_type+str(num_qubits)]['longest_path'] = dag.count_ops_longest_path()
        big_data[circuit_type+str(num_qubits)]['two_qubit_gates'] = circuit.num_nonlocal_gates()
        big_data[circuit_type+str(num_qubits)]['overall_physical_qubits'] = r['physicalCounts']['physicalQubits']
        big_data[circuit_type+str(num_qubits)]['overall_runtime'] = r['physicalCounts']['runtime']*1e-9
        big_data[circuit_type+str(num_qubits)]['logical_depth_qubits'] = r['physicalCounts']['breakdown']['logicalDepth']
        big_data[circuit_type+str(num_qubits)]['algorithmic_depth_qubits'] = r['physicalCounts']['breakdown']['algorithmicLogicalDepth']
        big_data[circuit_type+str(num_qubits)]['logical_qubits'] = r['logicalQubit']['physicalQubits']
        print('\n', 'total physical qubits required: ', r['physicalCounts']['physicalQubits'], '\n', 'total runtime', r['physicalCounts']['runtime']*1e-9)
        # print('\n', 'logical depth', r['physicalCounts']['breakdown']['logicalDepth'], 'algo depth', r['physicalCounts']['breakdown']['algorithmicLogicalDepth'], 'physical qubits', r['logicalQubit']['physicalQubits'])

        for ratio in range(1,circuit.num_qubits,5):
            big_data[circuit_type+str(num_qubits)][str(ratio)] = {}
            optimization_settings = OptimizationParameters(seed=111) 
            device_constraints = DeviceConstraints(qubits_per_subcircuit=ratio)
            # print('max no. of qubit per subcircuit: ', ratio)
            cut_circuit, metadata = find_cuts(circuit, optimization_settings, device_constraints)
            print(
                f'Found  {len(metadata["cuts"])} cuts ({len(metadata["gate_cuts"])} gate {len(metadata["wire_cuts"])} wire cuts)'
            )
            big_data[circuit_type+str(num_qubits)][str(ratio)]['gate_cuts'] = len(metadata["gate_cuts"])
            big_data[circuit_type+str(num_qubits)][str(ratio)]['wire_cuts'] = len(metadata["wire_cuts"])  
            qc_w_ancilla = cut_wires(cut_circuit)
            partitioned_problem = partition_problem(circuit=qc_w_ancilla)
            subcircuit = partitioned_problem.subcircuits
            subcircuits = [cuts_filter(s) for s in subcircuit.values()]

            quantum_runtime = []
            physical_qubits = []
            total_partition = [c.num_qubits for c in subcircuits]
            error_buget = [0.001*p/sum(total_partition) for p in total_partition]
            for idx, s in enumerate(subcircuits):
                # params = estimator.make_params(num_items=1)
                # params.constraints.max_t_factories = 1
                # params.error_budget = error_buget[idx]
                # job = estimator.submit(s, input_params=params)

                params = EstimatorParams()
                params.error_budget = error_buget[idx]
                params.constraints.max_t_factories = 1
                backend = ResourceEstimatorBackend()
                job = backend.run(s, params)

                try:
                    # print(idx)
                    r = job.result()
                    # r = job.get_results(timeout_secs=3000)
                    quantum_runtime.append(r['physicalCounts']['runtime'])
                    physical_qubits.append(r['physicalCounts']['physicalQubits'])
                    big_data[circuit_type+str(num_qubits)][str(ratio)]['two_qubit_gates'] = s.num_nonlocal_gates()
                    big_data[circuit_type+str(num_qubits)][str(ratio)]['physical_qubits_subcircuit-'+str(idx+1)] = r['physicalCounts']['physicalQubits']
                    big_data[circuit_type+str(num_qubits)][str(ratio)]['logical_depth_qubits_subcircuit-'+str(idx+1)] = r['physicalCounts']['breakdown']['logicalDepth']
                    big_data[circuit_type+str(num_qubits)][str(ratio)]['algorithmic_depth_qubits_subcircuit-'+str(idx+1)] = r['physicalCounts']['breakdown']['algorithmicLogicalDepth']
                    big_data[circuit_type+str(num_qubits)][str(ratio)]['logical_qubits_subcircuit-'+str(idx+1)] = r['logicalQubit']['physicalQubits']
                    big_data[circuit_type+str(num_qubits)][str(ratio)]['runtime_subcircuit-'+str(idx+1)] = r['physicalCounts']['runtime']*1e-9
                except:
                    print('no magic state in subcirc'+str(idx+1))
                # print('\n', 'subcircuit-', idx+1, ': physical qubits--', r['physicalCounts']['physicalQubits'], 'runtime (in microsec)--', r['physicalCounts']['runtime']*1e-3)
                # print('\n', 'logical depth', r['physicalCounts']['breakdown']['logicalDepth'], 'algo depth', r['physicalCounts']['breakdown']['algorithmicLogicalDepth'], 'physical qubits', r['logicalQubit']['physicalQubits'])
            try:
                big_data[circuit_type+str(num_qubits)][str(ratio)]['runtime_after_cutting'] = sum(quantum_runtime)*1e-9
                big_data[circuit_type+str(num_qubits)][str(ratio)]['qubits_after_cutting'] = max(physical_qubits)
                print('\n', 'quantum runtime total: ', sum(quantum_runtime)*1e-9, 'number of physical qubits: ', max(physical_qubits))
            except:
                big_data[circuit_type+str(num_qubits)][str(ratio)]['runtime_after_cutting'] = 0
                big_data[circuit_type+str(num_qubits)][str(ratio)]['qubits_after_cutting'] = 0
                print('\n', 'no mgaic state in any of the subcircuits')        
            
    file_name = circuit_type+'_data.json'
    if file_name is not None:
        with open(file_name, 'w') as fp:
            json.dump(big_data, fp)
    return big_data