"""Chain rule with placeholder functions that carry index arguments.

``tau(l, k, T)`` must resolve to array inputs (``tau(l, k)``,
``dtaudT(l, k)``, ...) in the chain-rule kernels exactly as without the
chain rule (issue ipqa-research/thermodiff#6).
"""

import pathlib
import re
import tempfile

from fortran_tools import (
    DEBUG_FREE_FORM_FLAGS,
    GFORTRAN,
    compile_and_run,
)

from model_harness import OUTS, driver

import numpy as np

import pytest

import sympy as sp

from thermodiff import (
    DiffPlz,
    T,
    idx_function,
    k,
    l,
    n,
    sum_components,
)
from thermodiff.codegen.options import CodegenOptions
from thermodiff.fortran import to_fortran

pytestmark = pytest.mark.skipif(GFORTRAN is None, reason="needs gfortran")

ALPHA = sp.Rational(3, 10)
SUBROUTINE_HEADER = re.compile(r"subroutine \w+\(.*?\)\n", re.S)


def _tau_sum() -> DiffPlz:
    """Return ``sum_k n_k sum_l n_l tau(l, k, T)`` (the issue example)."""
    tau = idx_function("tau")(l, k, T)
    expr = sum_components(n[k] * sum_components(n[l] * tau, l), k)
    return DiffPlz(expr, internal_functions=[tau], indexes=[k, l], name="f")


def _nrtl_like() -> DiffPlz:
    """Return an NRTL-like excess energy with ``G = exp(-alpha tau)``."""
    tau = idx_function("tau")(l, k, T)
    gamma = sp.exp(-ALPHA * tau)
    numerator = sum_components(n[l] * tau * gamma, l)
    denominator = sum_components(n[l] * gamma, l)
    expr = sum_components(n[k] * numerator / denominator, k)
    return DiffPlz(expr, internal_functions=[tau], indexes=[k, l], name="f")


def _uniquac_like() -> DiffPlz:
    """Return a UNIQUAC-like residual term with ``exp(-tau)`` weights."""
    tau = idx_function("tau")(l, k, T)
    theta = sum_components(n[l] * sp.exp(-tau), l)
    expr = -sum_components(n[k] * sp.log(theta), k)
    return DiffPlz(expr, internal_functions=[tau], indexes=[k, l], name="f")


def _kernel(diff: DiffPlz, preset: str, chain_rule: bool) -> str:
    """Return the kernel wrapped in a module, for the test driver."""
    options = CodegenOptions.from_preset(
        preset, chain_rule=chain_rule, optional_derivatives=False
    )
    code = to_fortran(diff, options=options)
    return "module kernel_mod\ncontains\n" + code + "\nend module\n"


def _signature(code: str) -> str:
    """Return the argument list of the generated subroutine."""
    return re.sub(r"\s+", "", SUBROUTINE_HEADER.search(code).group(0))


def _evaluate(code: str) -> np.ndarray:
    """Compile ``code`` with a driver (nc = 3) and return all outputs."""
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        source = workdir / "drive.f90"
        source.write_text(driver(code, {}, OUTS, ncs=(3,)))
        compile_and_run([source], workdir, DEBUG_FREE_FORM_FLAGS)
        return np.loadtxt(workdir / "out.txt")


BUILDERS = {
    "tau_sum": _tau_sum,
    "nrtl_like": _nrtl_like,
    "uniquac_like": _uniquac_like,
}


@pytest.mark.slow
@pytest.mark.parametrize("preset", ["standard", "fastest"])
@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_chain_rule_matches_plain_kernel(name, preset):
    """Chain-rule kernels share the signature and values of the plain one."""
    diff = BUILDERS[name]()
    plain = _kernel(diff, preset, chain_rule=False)
    chained = _kernel(diff, preset, chain_rule=True)
    assert _signature(chained) == _signature(plain)
    np.testing.assert_allclose(
        _evaluate(chained), _evaluate(plain), rtol=1e-12
    )


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_chain_rule_exposes_tau_inputs(name):
    """The placeholder becomes array inputs; none is left unresolved."""
    code = _kernel(BUILDERS[name](), "standard", chain_rule=True)
    assert "tau(:, :)" in code
