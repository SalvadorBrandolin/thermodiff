"""``chain_factor_sums``: free-index factors stay outside the reductions.

With ``chain_rule`` and ``fuse_temps`` the pair array ``qs(l)*tau(l, k)`` is
inlined into the derivative reductions, which then read
``sum_k qs(i)*qs(j)*w(k)*tau(i, k)*tau(j, k)``.  The option moves the factors
of the free indices in front of the sum, as the plain path does.
"""

import pathlib
import re
import tempfile
import warnings

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
    m,
    n,
    sum_components,
)
from thermodiff.codegen.options import CodegenOptions
from thermodiff.fortran import to_fortran

ALPHA = sp.Rational(3, 10)
FACTORED_ACCUMULATION = re.compile(
    r"^\s*(\w+)\(i, j\) = \1\(i, j\) \+ (?P<term>.*\(k\).*)$", re.M
)
FREE_FACTOR_PAIR = "qs(i)*qs(j)*"


def _uniquac_like() -> DiffPlz:
    """Return the residual term of UNIQUAC with a ``tau(l, k, T)`` input."""
    qs = sp.IndexedBase("qs")
    tau = idx_function("tau")(l, k, T)
    total_q = sum_components(n[m] * qs[m], m)
    theta = n[l] * qs[l] / total_q
    residual = sum_components(theta * tau, l)
    expr = -sum_components(n[k] * qs[k] * sp.log(residual), k)
    return DiffPlz(expr, internal_functions=[tau], indexes=[k, l, m], name="f")


def _nrtl_like() -> DiffPlz:
    """Return an NRTL-like excess energy with ``G = exp(-alpha tau)``."""
    tau = idx_function("tau")(l, k, T)
    gamma = sp.exp(-ALPHA * tau)
    numerator = sum_components(n[l] * tau * gamma, l)
    denominator = sum_components(n[l] * gamma, l)
    expr = sum_components(n[k] * numerator / denominator, k)
    return DiffPlz(expr, internal_functions=[tau], indexes=[k, l], name="f")


BUILDERS = {"uniquac_like": _uniquac_like, "nrtl_like": _nrtl_like}


def _kernel(diff: DiffPlz, preset: str, factor: bool, wrap: bool) -> str:
    """Return the chain-rule kernel (module-wrapped for the driver)."""
    options = CodegenOptions.from_preset(
        preset,
        chain_rule=True,
        optional_derivatives=False,
        chain_factor_sums=factor,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        code = to_fortran(diff, options=options)
    if wrap:
        return "module kernel_mod\ncontains\n" + code + "\nend module\n"
    return code


def _evaluate(code: str) -> np.ndarray:
    """Compile ``code`` with a driver (nc = 3 and 7) and return the outputs."""
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        source = workdir / "drive.f90"
        source.write_text(driver(code, {}, OUTS))
        compile_and_run([source], workdir, DEBUG_FREE_FORM_FLAGS)
        return np.loadtxt(workdir / "out.txt")


def _reduction_terms(code: str) -> list[str]:
    """Return the right-hand sides accumulated into rank-2 temporaries."""
    return [
        found.group("term") for found in FACTORED_ACCUMULATION.finditer(code)
    ]


class TestStructure:
    """The free-index factors leave the innermost reduction."""

    def test_default_keeps_the_factors_inside(self):
        code = _kernel(_uniquac_like(), "fastest", False, wrap=False)
        assert any(FREE_FACTOR_PAIR in t for t in _reduction_terms(code))

    def test_option_moves_the_factors_out(self):
        code = _kernel(_uniquac_like(), "fastest", True, wrap=False)
        terms = _reduction_terms(code)
        assert terms
        assert not any("qs(i)" in term or "qs(j)" in term for term in terms)

    def test_fewer_multiplications_in_the_reduction(self):
        off = _reduction_terms(
            _kernel(_uniquac_like(), "fastest", False, False)
        )
        on = _reduction_terms(_kernel(_uniquac_like(), "fastest", True, False))
        assert sum(t.count("*") for t in on) < sum(t.count("*") for t in off)

    def test_inert_without_chain_rule(self):
        options = CodegenOptions(chain_factor_sums=True)
        with pytest.warns(UserWarning, match="chain_factor_sums"):
            normalised, ignored = options.validate()
        assert not normalised.chain_factor_sums
        assert "chain_factor_sums" in ignored

    def test_default_is_off_in_every_preset(self):
        for preset in ("minimal", "standard", "fastest"):
            assert not CodegenOptions.from_preset(preset).chain_factor_sums


@pytest.mark.slow
@pytest.mark.skipif(GFORTRAN is None, reason="needs gfortran")
@pytest.mark.parametrize("preset", ["standard", "fastest"])
@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_option_matches_default_numerically(name, preset):
    """Every output agrees with the option off at ``rel=1e-12``."""
    diff = BUILDERS[name]()
    reference = _evaluate(_kernel(diff, preset, False, wrap=True))
    values = _evaluate(_kernel(diff, preset, True, wrap=True))
    assert np.all(np.isfinite(values))
    np.testing.assert_allclose(values, reference, rtol=1e-12)
