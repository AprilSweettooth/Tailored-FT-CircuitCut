import numpy    as  np
import cirq     as  cirq
from qualtran   import cirq_interop
from typing import Dict, Any
from attrs import frozen

def estimate_resources(circuit_element: Any, Toffoli: bool = False) -> Dict:
    """
    Keyword arguments:
    circuit_element -- element to estimate resources of. Can be circuit, gate, or operation.
    rotation_gate_precision -- maximum approximation error for each rotation gate decomposition
    circuit_precision -- If not None, the approximation error for each rotation gate will be bounded by `circuit_precision` divided by the number of rotation gates in `circuit_element`
    """
    try:
        resource_dict = {'LogicalQubits':cirq.num_qubits(circuit_element)}
    except:
        resource_dict={}
    t_cliff_rot_resources = cirq_interop.t_complexity_protocol.t_complexity(circuit_element)

    resource_dict["T"] = t_cliff_rot_resources.t 
    resource_dict["Clifford"] = t_cliff_rot_resources.clifford
    resource_dict["Rotations"] = t_cliff_rot_resources.rotations
    if not Toffoli:
        resource_dict["Toffoli"] = 0
    resource_dict["Rotation_layer"] = np.ceil(t_cliff_rot_resources.rotations/cirq.num_qubits(circuit_element))
    return resource_dict

@frozen
class PhysicalParameters:
    """The physical properties of a quantum computer.

    Attributes:
        physical_error: The error rate of the underlying physical qubits.
        cycle_time_us: The number of microseconds it takes to do one cycle of error correction.

    """
    Q: int = 1
    logical_time_step: float = 1.0
    number_of_T_gates: int = 1

    def code_distance_from_budget(physical_error: float, budget: float) -> int:
        """Get the code distance that keeps one below the logical error `budget`."""

        # See: `logical_error_rate()`. p_l = a Λ^(-r) where r = (d+1)/2
        # Which we invert: r = ln(p_l/a) / ln(1/Λ)
        error_rate_scaler: float = 0.03
        error_rate_threshold: float = 0.01
        r = np.log(budget / error_rate_scaler) / np.log(
            physical_error / error_rate_threshold
        )
        d = 2 * np.ceil(r) - 1
        if d < 3:
            return 3
        return d

    def physical_qubits(code_distance: int) -> int:
        """The number of physical qubits per logical qubit used by the error detection circuit."""
        return 2 * code_distance**2

    @classmethod
    def make_beverland_et_al(
        cls, circuit_element: Any, qubit_modality: str = 'superconducting', optimistic_err_rate: bool = False, 
        optimize_synthesis: str = 'Bev', error_budget: list = [1e-3,1e-3,1e-3,1e-3], runtime_ratio: int = 1
    ):
        """
        The physical parameters considered in the Beverland et. al. reference.
        Args:
            circuit_element: The input circuit for resource analysis.
            qubit_modality: One of "superconducting" or "ion". This sets the
                cycle time, with ions being considerably slower.
            optimistic_err_rate: In the reference, the authors consider two error rates, which
                they term "realistic" and "optimistic". Set this to `True` to use optimistic
                error rates.
            optimize_synthesis: This gives different A and B for the synthesis gate count.
            error_budget: [epsilon_syn, epsilon_dis, epsilon_log, epsilon_alg]

        References:
            [Assessing requirements to scale to practical quantum advantage](https://arxiv.org/abs/2211.07629).
            Beverland et. al. (2022).
        """
        if optimistic_err_rate:
            phys_err_rate = 1e-4
        else:
            phys_err_rate = 1e-3
        if qubit_modality == 'ion':
            t_gate_ns = 100_000
            t_meas_ns = 100_000
        elif qubit_modality == 'superconducting':
            t_gate_ns = 50
            t_meas_ns = 100
        else:
            raise ValueError(
                f"Unknown qubit modality {qubit_modality}. Must be one "
                f"of 'ion', 'superconducting', or 'majorana'."
            )

        cycle_time_ns = 4 * t_gate_ns + 2 * t_meas_ns
        circ_param = estimate_resources(circuit_element)
        Q_alg = circ_param['LogicalQubits']
        M_R = circ_param['Rotations']
        M_Tof = circ_param['Toffoli']
        M_T = circ_param['T']
        M_meas = 1
        D_R = circ_param['Rotation_layer']
        if optimize_synthesis == 'Bev':
            rotation_cost = np.ceil(0.53 * np.log2(M_R/error_budget[0]) + 5.3) 
        else:
            rotation_cost = 1
        T_min = M_meas+M_R+M_T + rotation_cost*D_R + 3*M_Tof
        M = rotation_cost*M_R + 4*M_Tof + M_T
        alg_qubits = 2*Q_alg + np.ceil(8*Q_alg) + 1
        code_distance = cls.code_distance_from_budget(phys_err_rate, error_budget[2] / (alg_qubits * runtime_ratio * T_min))
        t = code_distance * cycle_time_ns * 1e-9 * T_min * runtime_ratio
        P_t = error_budget[1]/M
        d_dist = 2
        distillation_round = 1
        distillation_unit_per_round = 2
        tau_dist = distillation_round * cycle_time_ns * 1e-9 * d_dist * 13
        M_D = 1
        num_factories = np.ceil(M*tau_dist/(M_D*t))
        if not t > tau_dist:
            raise ValueError(f"runtime ratio must be larger.")
        # assume each round n is same
        distillation_qubits = num_factories * distillation_unit_per_round * 20 * cls.physical_qubits(d_dist)
        return PhysicalParameters(
            Q=alg_qubits*cls.physical_qubits(code_distance)+distillation_qubits, logical_time_step=t, number_of_T_gates=M
        )