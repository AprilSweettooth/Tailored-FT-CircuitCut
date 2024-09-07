from qiskit.quantum_info import SparsePauliOp

import numpy as np

def get_hamiltonian(L, J, h, alpha=0):

    # List of Hamiltonian terms as 3-tuples containing
    # (1) the Pauli string,
    # (2) the qubit indices corresponding to the Pauli string,
    # (3) the coefficient.
    ZZ_tuples = [("ZZ", [i, i + 1], -J) for i in range(0, L - 1)]
    Z_tuples = [("Z", [i], -h * np.sin(alpha)) for i in range(0, L)]
    X_tuples = [("X", [i], -h * np.cos(alpha)) for i in range(0, L)]

    # We create the Hamiltonian as a SparsePauliOp, via the method
    # `from_sparse_list`, and multiply by the interaction term.
    hamiltonian = SparsePauliOp.from_sparse_list([*ZZ_tuples, *Z_tuples, *X_tuples], num_qubits=L)
    return hamiltonian.simplify()

def plot_trotter_info(circuit):
    print(
        f"""
    Trotter step with Suzuki Trotter (2nd order)
    --------------------------------------------

                      Depth: {circuit.depth()}
                 Gate count: {len(circuit)}
        Nonlocal gate count: {circuit.num_nonlocal_gates()}
             Gate breakdown: {", ".join([f"{k.upper()}: {v}" for k, v in circuit.count_ops().items()])}

    """
    )

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
    n = int(overall_budget * 1000)
    partition = list(accel_asc(n))
    sub_part = []
    for p in partition:
        if len(p) == num_subcirc:
            sub_part.append(list(p))
    return sub_part
