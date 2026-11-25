"""A libm call over a rank-2 temporary read by two nests is one table."""

import re
import warnings

from fortran_tools import GFORTRAN

import pytest

import sympy as sp

from test_extra_dimensions import (
    SIZES,
    TEMPERATURE,
    _check,
    _expected,
    _parameters,
)

from thermodiff.model import compile_model

pytestmark = pytest.mark.skipif(
    GFORTRAN is None, reason="gfortran is not available"
)

#: UNIFAC-like residual: ``S_lk`` is a rank-2 temporary whose ``log`` is read
#: by the reduction over ``l`` and by the gradient (same table, other names).
RESIDUAL = """model pairlog
target fortran

variables
    n   nc
    T

parameters
    v   nc ng
    q   ng
    a   ng ng

expressions
    psi_mk = exp(-a_mk/T)
    N_k = sum_s n_s*v_sk
    S_lk = sum_m[1:ng] v_lm*q_m*psi_mk
    Y_k = sum_m[1:ng] N_m*q_m*psi_mk

function
    sum_k[1:ng] q_k*(sum_l n_l*v_lk*log(S_lk) - N_k*log(Y_k))
"""

PAIR_LOG = re.compile(r"log\(t\d+\(\w+, \w+\)\)")


def _generate(**options) -> str:
    """Return the kernel of ``RESIDUAL`` for ``dn,dndn,dT``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compile_model(RESIDUAL, derivatives="dn,dndn,dT", **options)


def _reference(nc: int, ng: int, values):
    """Return the moles and the function of ``RESIDUAL`` in SymPy."""
    moles = sp.symbols(f"n0:{nc}")
    temperature = sp.Symbol("T")
    v, q, a = values["v"], values["q"], values["a"]
    psi = [
        [sp.exp(-a[m, k] / temperature) for k in range(ng)] for m in range(ng)
    ]
    amounts = [sum(moles[s] * v[s, k] for s in range(nc)) for k in range(ng)]
    total = 0
    for k in range(ng):
        pure = sum(
            moles[li]
            * v[li, k]
            * sp.log(sum(v[li, m] * q[m] * psi[m][k] for m in range(ng)))
            for li in range(nc)
        )
        mixed = amounts[k] * sp.log(
            sum(amounts[m] * q[m] * psi[m][k] for m in range(ng))
        )
        total += q[k] * (pure - mixed)
    return moles, total


@pytest.mark.parametrize("preset", ["minimal", "standard", "fastest"])
def test_pair_libm_table_only_in_fastest(preset):
    """Only the fastest preset builds the shared table."""
    code = _generate(preset=preset, optional_derivatives=False)
    expected_reads = 1 if preset == "fastest" else 2
    assert len(PAIR_LOG.findall(code)) == expected_reads


@pytest.mark.slow
@pytest.mark.parametrize("sizes", SIZES)
def test_shared_pair_libm_matches_sympy(sizes):
    """The kernel with the shared table agrees with SymPy."""
    nc, ng = sizes
    values = _parameters(nc, ng)
    moles, function = _reference(nc, ng, values)
    expected = _expected(moles, function, values, True)
    code = _generate(preset="fastest", optional_derivatives=False)
    assert TEMPERATURE > 0
    _check(code, nc, ng, expected, True)
