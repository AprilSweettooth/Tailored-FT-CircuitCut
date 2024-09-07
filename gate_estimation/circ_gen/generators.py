from .hw_efficient_ansatz import *
from .uccsd_ansatz import *
from .bernstein_vazirani import *
from .ripple_carry_adder import *

def gen_hwea(
    width,
    depth,
    parameters="optimal",
    seed=None,
    barriers=False,
    measure=False,
    regname=None,
):
    """
    Create a quantum circuit implementing a hardware efficient
    ansatz with the given width (number of qubits) and
    depth (number of repetitions of the basic ansatz).
    """

    hwea = HWEA(
        width,
        depth,
        parameters=parameters,
        seed=seed,
        barriers=barriers,
        measure=measure,
        regname=regname,
    )

    circ = hwea.gen_circuit()

    return circ


def gen_uccsd(width, parameters="random", seed=None, barriers=False, regname=None):
    """
    Generate a UCCSD ansatz with the given width (number of qubits).
    """

    uccsd = UCCSD(
        width, parameters=parameters, seed=seed, barriers=barriers, regname=regname
    )

    circ = uccsd.gen_circuit()

    return circ


def gen_BV(secret=None, barriers=True, measure=False, regname=None):
    """
    Generate an instance of the Bernstein-Vazirani algorithm which queries a
    black-box oracle once to discover the secret key in:

    f(x) = x . secret (mod 2)

    The user must specify the secret bitstring to use: e.g. 00111001
    (It can be given as a string or integer)
    """

    bv = BV(
        secret=secret, barriers=barriers, measure=measure, regname=regname
    )

    circ = bv.gen_circuit()

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
