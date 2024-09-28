import numpy as np
import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit.circuit.library import TwoLocal
from qiskit_addon_cutting_custom.automated_cut_finding import *
from qiskit_addon_cutting_custom import *
from azure.quantum import Workspace
from azure.quantum.target.microsoft import MicrosoftEstimator, ErrorBudgetPartition
from hamlib.hamlib_snippets import *
from benchmark_utils import *
from gate_estimation.resource_analysis import *
from QASMBench.interface.qiskit import QASMBenchmark
import json

workspace = Workspace (
   resource_id = "/subscriptions/e0d51919-2dda-4b7e-a8e7-77cc18580acb/resourceGroups/AzureQuantum/providers/Microsoft.Quantum/Workspaces/QECBenchmarking",
   location = "uksouth"
)

estimator = MicrosoftEstimator(workspace)

def run(ratio_list=[0.5, 0.7, 0.9], lattice_size=range(3,7), only_wire=False, record=True, file_name='2D_Ising_data.json', method='trotter'):

    big_data = {}
    for ratio in ratio_list:
        data = {}
        for size in lattice_size:
            data['Ising'+str(size)+' by '+str(size)] = {}
            circuit = build_Ising_hamiltonian(num_spins=size**2, h=-1, Dimension=(size,size), method=method)
            if only_wire:
                optimization_settings = OptimizationParameters(seed=111, gate_lo=False)
            else:
                optimization_settings = OptimizationParameters(seed=111) 
            device_constraints = DeviceConstraints(qubits_per_subcircuit=int(np.floor(circuit.num_qubits*ratio)))
            print('max no. of qubit per subcircuit: ', int(np.floor(circuit.num_qubits*ratio)))
            cut_circuit, metadata = find_cuts(circuit, optimization_settings, device_constraints)
            print(
                f'Found {size}x{size} Ising model solution using {len(metadata["cuts"])} cuts ({len(metadata["gate_cuts"])} gate {len(metadata["wire_cuts"])} wire cuts)'
            )
            data['Ising'+str(size)+' by '+str(size)]['gate_cuts'] = len(metadata["gate_cuts"])
            data['Ising'+str(size)+' by '+str(size)]['wire_cuts'] = len(metadata["wire_cuts"])  
            qc_w_ancilla = cut_wires(cut_circuit)
            partitioned_problem = partition_problem(circuit=qc_w_ancilla)
            subcircuit= partitioned_problem.subcircuits
            subcircuits = [cuts_filter(s) for s in subcircuit.values()]

            params = estimator.make_params(num_items=1)
            params.error_budget = 0.01
            params.constraints.max_t_factories = 1
            job = estimator.submit(circuit, input_params=params)
            r = job.get_results()
            Q = r['physicalCounts']['physicalQubits']
            T = r['physicalCounts']['runtime']*1e-9
            data['Ising'+str(size)+' by '+str(size)]['overall_physical_qubits'] = Q
            data['Ising'+str(size)+' by '+str(size)]['overall_runtime'] = T
            data['Ising'+str(size)+' by '+str(size)]['logical_depth_qubits'] = r['physicalCounts']['breakdown']['logicalDepth']
            data['Ising'+str(size)+' by '+str(size)]['algorithmic_depth_qubits'] = r['physicalCounts']['breakdown']['algorithmicLogicalDepth']
            data['Ising'+str(size)+' by '+str(size)]['logical_qubits'] = r['logicalQubit']['physicalQubits']
            print('\n', 'total physical qubits required: ', Q, '\n', 'total runtime', T)
            print('\n', 'logical depth', r['physicalCounts']['breakdown']['logicalDepth'], 'algo depth', r['physicalCounts']['breakdown']['algorithmicLogicalDepth'], 'physical qubits', r['logicalQubit']['physicalQubits'])

            quantum_runtime = []
            physical_qubits = []
            for idx, s in enumerate(subcircuits):
                params = estimator.make_params(num_items=1)
                params.constraints.max_t_factories = 1
                job = estimator.submit(s, input_params=params)
                r = job.get_results()
                quantum_runtime.append(r['physicalCounts']['runtime'])
                physical_qubits.append(r['physicalCounts']['physicalQubits'])
                data['Ising'+str(size)+' by '+str(size)]['physical_qubits_subcircuit-'+str(idx+1)] = r['physicalCounts']['physicalQubits']
                data['Ising'+str(size)+' by '+str(size)]['logical_depth_qubits_subcircuit-'+str(idx+1)] = r['physicalCounts']['breakdown']['logicalDepth']
                data['Ising'+str(size)+' by '+str(size)]['algorithmic_depth_qubits_subcircuit-'+str(idx+1)] = r['physicalCounts']['breakdown']['algorithmicLogicalDepth']
                data['Ising'+str(size)+' by '+str(size)]['logical_qubits_subcircuit-'+str(idx+1)] = r['logicalQubit']['physicalQubits']
                data['Ising'+str(size)+' by '+str(size)]['runtime_subcircuit-'+str(idx+1)] = r['physicalCounts']['runtime']*1e-9
                print('\n', 'subcircuit-', idx+1, ': physical qubits--', r['physicalCounts']['physicalQubits'], 'runtime (in microsec)--', r['physicalCounts']['runtime']*1e-3)
                print('\n', 'logical depth', r['physicalCounts']['breakdown']['logicalDepth'], 'algo depth', r['physicalCounts']['breakdown']['algorithmicLogicalDepth'], 'physical qubits', r['logicalQubit']['physicalQubits'])
            data['Ising'+str(size)+' by '+str(size)]['runtime_after_cutting'] = sum(quantum_runtime)*1e-9
            data['Ising'+str(size)+' by '+str(size)]['qubits_after_cutting'] = max(physical_qubits)
            print('\n', 'quantum runtime total: ', sum(quantum_runtime)*1e-9, 'number of physical qubits: ', max(physical_qubits))
        big_data[str(ratio)] = data
    if record:
        with open(file_name, 'w') as fp:
            json.dump(big_data, fp)
    return big_data

def run_T_factory(T_list=[1,2,3,4,5], lattice_size=range(3,7), only_wire=False, record=True, file_name='2D_Ising_data_max_T_factories.json', method='trotter'):

    big_data = {}
    for max_T in T_list:
        data = {}
        for size in lattice_size:
            data['Ising'+str(size)+' by '+str(size)] = {}
            circuit = build_Ising_hamiltonian(num_spins=size**2, h=-1, Dimension=(size,size), method=method)
            if only_wire:
                optimization_settings = OptimizationParameters(seed=111, gate_lo=False)
            else:
                optimization_settings = OptimizationParameters(seed=111) 
            device_constraints = DeviceConstraints(qubits_per_subcircuit=int(np.floor(circuit.num_qubits*0.5)))
            print('max no. of qubit per subcircuit: ', int(np.floor(circuit.num_qubits*0.5)))
            cut_circuit, metadata = find_cuts(circuit, optimization_settings, device_constraints)
            print(
                f'Found {size}x{size} Ising model solution using {len(metadata["cuts"])} cuts ({len(metadata["gate_cuts"])} gate {len(metadata["wire_cuts"])} wire cuts)'
            )
            data['Ising'+str(size)+' by '+str(size)]['gate_cuts'] = len(metadata["gate_cuts"])
            data['Ising'+str(size)+' by '+str(size)]['wire_cuts'] = len(metadata["wire_cuts"])  
            qc_w_ancilla = cut_wires(cut_circuit)
            partitioned_problem = partition_problem(circuit=qc_w_ancilla)
            subcircuit= partitioned_problem.subcircuits
            subcircuits = [cuts_filter(s) for s in subcircuit.values()]

            params = estimator.make_params(num_items=1)
            params.error_budget = 0.01
            params.constraints.max_t_factories = max_T
            job = estimator.submit(circuit, input_params=params)
            r = job.get_results()
            Q = r['physicalCounts']['physicalQubits']
            T = r['physicalCounts']['runtime']*1e-9
            data['Ising'+str(size)+' by '+str(size)]['overall_physical_qubits'] = Q
            data['Ising'+str(size)+' by '+str(size)]['overall_runtime'] = T
            data['Ising'+str(size)+' by '+str(size)]['logical_depth_qubits'] = r['physicalCounts']['breakdown']['logicalDepth']
            data['Ising'+str(size)+' by '+str(size)]['algorithmic_depth_qubits'] = r['physicalCounts']['breakdown']['algorithmicLogicalDepth']
            data['Ising'+str(size)+' by '+str(size)]['logical_qubits'] = r['logicalQubit']['physicalQubits']
            print('\n', 'total physical qubits required: ', Q, '\n', 'total runtime', T)
            print('\n', 'logical depth', r['physicalCounts']['breakdown']['logicalDepth'], 'algo depth', r['physicalCounts']['breakdown']['algorithmicLogicalDepth'], 'physical qubits', r['logicalQubit']['physicalQubits'])

            quantum_runtime = []
            physical_qubits = []
            for idx, s in enumerate(subcircuits):
                params = estimator.make_params(num_items=1)
                params.constraints.max_t_factories = max_T
                job = estimator.submit(s, input_params=params)
                r = job.get_results()
                quantum_runtime.append(r['physicalCounts']['runtime'])
                physical_qubits.append(r['physicalCounts']['physicalQubits'])
                data['Ising'+str(size)+' by '+str(size)]['physical_qubits_subcircuit-'+str(idx+1)] = r['physicalCounts']['physicalQubits']
                data['Ising'+str(size)+' by '+str(size)]['logical_depth_qubits_subcircuit-'+str(idx+1)] = r['physicalCounts']['breakdown']['logicalDepth']
                data['Ising'+str(size)+' by '+str(size)]['algorithmic_depth_qubits_subcircuit-'+str(idx+1)] = r['physicalCounts']['breakdown']['algorithmicLogicalDepth']
                data['Ising'+str(size)+' by '+str(size)]['logical_qubits_subcircuit-'+str(idx+1)] = r['logicalQubit']['physicalQubits']
                data['Ising'+str(size)+' by '+str(size)]['runtime_subcircuit-'+str(idx+1)] = r['physicalCounts']['runtime']*1e-9
                print('\n', 'subcircuit-', idx+1, ': physical qubits--', r['physicalCounts']['physicalQubits'], 'runtime (in microsec)--', r['physicalCounts']['runtime']*1e-3)
                print('\n', 'logical depth', r['physicalCounts']['breakdown']['logicalDepth'], 'algo depth', r['physicalCounts']['breakdown']['algorithmicLogicalDepth'], 'physical qubits', r['logicalQubit']['physicalQubits'])
            data['Ising'+str(size)+' by '+str(size)]['runtime_after_cutting'] = sum(quantum_runtime)*1e-9
            data['Ising'+str(size)+' by '+str(size)]['qubits_after_cutting'] = max(physical_qubits)
            print('\n', 'quantum runtime total: ', sum(quantum_runtime)*1e-9, 'number of physical qubits: ', max(physical_qubits))
        big_data[str(max_T)] = data
    if record:
        with open(file_name, 'w') as fp:
            json.dump(big_data, fp)
    return big_data


def all_ratio_run(lattice_size=range(3,7), only_wire=False, record=True, file_name='all_data_trotter_2D_Ising.json', method='trotter'):

    big_data = {}
    for size in lattice_size:
        big_data['Ising'+str(size)+' by '+str(size)] = {}
        for ratio in range(1,size**2):
            big_data['Ising'+str(size)+' by '+str(size)][str(ratio)] = {}
            circuit = build_Ising_hamiltonian(num_spins=size**2, h=-1, Dimension=(size,size), method=method)
            if only_wire:
                optimization_settings = OptimizationParameters(seed=111, gate_lo=False)
            else:
                optimization_settings = OptimizationParameters(seed=111) 
            device_constraints = DeviceConstraints(qubits_per_subcircuit=ratio)
            print('max no. of qubit per subcircuit: ', ratio)
            cut_circuit, metadata = find_cuts(circuit, optimization_settings, device_constraints)
            print(
                f'Found {size}x{size} Ising model solution using {len(metadata["cuts"])} cuts ({len(metadata["gate_cuts"])} gate {len(metadata["wire_cuts"])} wire cuts)'
            )
            big_data['Ising'+str(size)+' by '+str(size)][str(ratio)]['gate_cuts'] = len(metadata["gate_cuts"])
            big_data['Ising'+str(size)+' by '+str(size)][str(ratio)]['wire_cuts'] = len(metadata["wire_cuts"])  
            qc_w_ancilla = cut_wires(cut_circuit)
            partitioned_problem = partition_problem(circuit=qc_w_ancilla)
            subcircuit= partitioned_problem.subcircuits
            subcircuits = [cuts_filter(s) for s in subcircuit.values()]

            params = estimator.make_params(num_items=1)
            params.error_budget = 0.01
            params.constraints.max_t_factories = 1
            job = estimator.submit(circuit, input_params=params)
            r = job.get_results()
            Q = r['physicalCounts']['physicalQubits']
            T = r['physicalCounts']['runtime']*1e-9
            big_data['Ising'+str(size)+' by '+str(size)][str(ratio)]['overall_physical_qubits'] = Q
            big_data['Ising'+str(size)+' by '+str(size)][str(ratio)]['overall_runtime'] = T
            big_data['Ising'+str(size)+' by '+str(size)][str(ratio)]['logical_depth_qubits'] = r['physicalCounts']['breakdown']['logicalDepth']
            big_data['Ising'+str(size)+' by '+str(size)][str(ratio)]['algorithmic_depth_qubits'] = r['physicalCounts']['breakdown']['algorithmicLogicalDepth']
            big_data['Ising'+str(size)+' by '+str(size)][str(ratio)]['logical_qubits'] = r['logicalQubit']['physicalQubits']
            print('\n', 'total physical qubits required: ', Q, '\n', 'total runtime', T)
            print('\n', 'logical depth', r['physicalCounts']['breakdown']['logicalDepth'], 'algo depth', r['physicalCounts']['breakdown']['algorithmicLogicalDepth'], 'physical qubits', r['logicalQubit']['physicalQubits'])

            quantum_runtime = []
            physical_qubits = []
            for idx, s in enumerate(subcircuits):
                params = estimator.make_params(num_items=1)
                params.constraints.max_t_factories = 1
                job = estimator.submit(s, input_params=params)
                r = job.get_results()
                quantum_runtime.append(r['physicalCounts']['runtime'])
                physical_qubits.append(r['physicalCounts']['physicalQubits'])
                big_data['Ising'+str(size)+' by '+str(size)][str(ratio)]['physical_qubits_subcircuit-'+str(idx+1)] = r['physicalCounts']['physicalQubits']
                big_data['Ising'+str(size)+' by '+str(size)][str(ratio)]['logical_depth_qubits_subcircuit-'+str(idx+1)] = r['physicalCounts']['breakdown']['logicalDepth']
                big_data['Ising'+str(size)+' by '+str(size)][str(ratio)]['algorithmic_depth_qubits_subcircuit-'+str(idx+1)] = r['physicalCounts']['breakdown']['algorithmicLogicalDepth']
                big_data['Ising'+str(size)+' by '+str(size)][str(ratio)]['logical_qubits_subcircuit-'+str(idx+1)] = r['logicalQubit']['physicalQubits']
                big_data['Ising'+str(size)+' by '+str(size)][str(ratio)]['runtime_subcircuit-'+str(idx+1)] = r['physicalCounts']['runtime']*1e-9
                print('\n', 'subcircuit-', idx+1, ': physical qubits--', r['physicalCounts']['physicalQubits'], 'runtime (in microsec)--', r['physicalCounts']['runtime']*1e-3)
                print('\n', 'logical depth', r['physicalCounts']['breakdown']['logicalDepth'], 'algo depth', r['physicalCounts']['breakdown']['algorithmicLogicalDepth'], 'physical qubits', r['logicalQubit']['physicalQubits'])
            big_data['Ising'+str(size)+' by '+str(size)][str(ratio)]['runtime_after_cutting'] = sum(quantum_runtime)*1e-9
            big_data['Ising'+str(size)+' by '+str(size)][str(ratio)]['qubits_after_cutting'] = max(physical_qubits)
            print('\n', 'quantum runtime total: ', sum(quantum_runtime)*1e-9, 'number of physical qubits: ', max(physical_qubits))
        
    if record:
        with open(file_name, 'w') as fp:
            json.dump(big_data, fp)
    return big_data


def run_model(circuit, file_name, ratio_list=[0.5, 0.7, 0.9], lattice_size=range(3,7), only_wire=False, record=True):

    big_data = {}
    for ratio in ratio_list:
        data = {}
        for size in lattice_size:
            data['Ising'+str(size)+' by '+str(size)] = {}
            if only_wire:
                optimization_settings = OptimizationParameters(seed=111, gate_lo=False)
            else:
                optimization_settings = OptimizationParameters(seed=111) 
            device_constraints = DeviceConstraints(qubits_per_subcircuit=int(np.floor(circuit.num_qubits*ratio)))
            print('max no. of qubit per subcircuit: ', int(np.floor(circuit.num_qubits*ratio)))
            cut_circuit, metadata = find_cuts(circuit, optimization_settings, device_constraints)
            print(
                f'Found {size}x{size} Ising model solution using {len(metadata["cuts"])} cuts ({len(metadata["gate_cuts"])} gate {len(metadata["wire_cuts"])} wire cuts)'
            )
            data['Ising'+str(size)+' by '+str(size)]['gate_cuts'] = len(metadata["gate_cuts"])
            data['Ising'+str(size)+' by '+str(size)]['wire_cuts'] = len(metadata["wire_cuts"])  
            qc_w_ancilla = cut_wires(cut_circuit)
            partitioned_problem = partition_problem(circuit=qc_w_ancilla)
            subcircuit= partitioned_problem.subcircuits
            subcircuits = [cuts_filter(s) for s in subcircuit.values()]

            params = estimator.make_params(num_items=1)
            params.error_budget = 0.01
            params.constraints.max_t_factories = 1
            job = estimator.submit(circuit, input_params=params)
            r = job.get_results()
            Q = r['physicalCounts']['physicalQubits']
            T = r['physicalCounts']['runtime']*1e-9
            data['Ising'+str(size)+' by '+str(size)]['overall_physical_qubits'] = Q
            data['Ising'+str(size)+' by '+str(size)]['overall_runtime'] = T
            data['Ising'+str(size)+' by '+str(size)]['logical_depth_qubits'] = r['physicalCounts']['breakdown']['logicalDepth']
            data['Ising'+str(size)+' by '+str(size)]['algorithmic_depth_qubits'] = r['physicalCounts']['breakdown']['algorithmicLogicalDepth']
            data['Ising'+str(size)+' by '+str(size)]['logical_qubits'] = r['logicalQubit']['physicalQubits']
            print('\n', 'total physical qubits required: ', Q, '\n', 'total runtime', T)
            print('\n', 'logical depth', r['physicalCounts']['breakdown']['logicalDepth'], 'algo depth', r['physicalCounts']['breakdown']['algorithmicLogicalDepth'], 'physical qubits', r['logicalQubit']['physicalQubits'])

            quantum_runtime = []
            physical_qubits = []
            for idx, s in enumerate(subcircuits):
                params = estimator.make_params(num_items=1)
                params.constraints.max_t_factories = 1
                job = estimator.submit(s, input_params=params)
                r = job.get_results()
                quantum_runtime.append(r['physicalCounts']['runtime'])
                physical_qubits.append(r['physicalCounts']['physicalQubits'])
                data['Ising'+str(size)+' by '+str(size)]['physical_qubits_subcircuit-'+str(idx+1)] = r['physicalCounts']['physicalQubits']
                data['Ising'+str(size)+' by '+str(size)]['logical_depth_qubits_subcircuit-'+str(idx+1)] = r['physicalCounts']['breakdown']['logicalDepth']
                data['Ising'+str(size)+' by '+str(size)]['algorithmic_depth_qubits_subcircuit-'+str(idx+1)] = r['physicalCounts']['breakdown']['algorithmicLogicalDepth']
                data['Ising'+str(size)+' by '+str(size)]['logical_qubits_subcircuit-'+str(idx+1)] = r['logicalQubit']['physicalQubits']
                data['Ising'+str(size)+' by '+str(size)]['runtime_subcircuit-'+str(idx+1)] = r['physicalCounts']['runtime']*1e-9
                print('\n', 'subcircuit-', idx+1, ': physical qubits--', r['physicalCounts']['physicalQubits'], 'runtime (in microsec)--', r['physicalCounts']['runtime']*1e-3)
                print('\n', 'logical depth', r['physicalCounts']['breakdown']['logicalDepth'], 'algo depth', r['physicalCounts']['breakdown']['algorithmicLogicalDepth'], 'physical qubits', r['logicalQubit']['physicalQubits'])
            data['Ising'+str(size)+' by '+str(size)]['runtime_after_cutting'] = sum(quantum_runtime)*1e-9
            data['Ising'+str(size)+' by '+str(size)]['qubits_after_cutting'] = max(physical_qubits)
            print('\n', 'quantum runtime total: ', sum(quantum_runtime)*1e-9, 'number of physical qubits: ', max(physical_qubits))
        big_data[str(ratio)] = data
    if record:
        with open(file_name, 'w') as fp:
            json.dump(big_data, fp)
    return big_data