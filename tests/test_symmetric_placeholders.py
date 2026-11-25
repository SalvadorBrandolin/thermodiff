"""Symmetric placeholders: ``a(k, l, T) == a(l, k, T)`` (issue #14).

Declared with ``symmetric_params=("a",)`` or ``idx_function("a",
symmetric=(0, 1))``, the symmetry reaches the derivative inputs ``dadT``,
``dadTdT`` so ``dadT(k, i) n(k)`` is merged with ``dadT(i, k) n(k)``.
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
from thermodiff.codegen.resolve import _PlaceholderResolver
from thermodiff.fortran import to_fortran

DERIVATIVES = ["dn", "dndT", "dndn", "dT", "dT2"]
PRESETS = ["minimal", "standard", "fastest"]


def _mixing(symmetric_attr: bool) -> DiffPlz:
    """Return ``sum_k n_k sum_l n_l exp(-a(k,l,T)) a(k,l,T)``."""
    a = idx_function("a", symmetric=(0, 1) if symmetric_attr else None)(
        k, l, T
    )
    body = n[k] * sum_components(n[l] * a * sp.exp(-a), l)
    return DiffPlz(
        sum_components(body, k),
        internal_functions=[a],
        indexes=[k, l],
        name="f",
    )


def _kernel(
    diff: DiffPlz, preset: str, chain_rule: bool, symmetric_params=()
) -> str:
    """Return the kernel wrapped in a module for the test driver."""
    options = CodegenOptions.from_preset(
        preset,
        chain_rule=chain_rule,
        optional_derivatives=False,
    )
    code = to_fortran(
        diff,
        derivatives=DERIVATIVES,
        options=options,
        symmetric_params=symmetric_params,
    )
    return "module kernel_mod\ncontains\n" + code + "\nend module\n"


def _evaluate(code: str) -> np.ndarray:
    """Compile ``code`` with a driver (nc = 3); inputs are symmetric."""
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        source = workdir / "drive.f90"
        source.write_text(driver(code, {}, OUTS, ncs=(3,)))
        compile_and_run([source], workdir, DEBUG_FREE_FORM_FLAGS)
        return np.loadtxt(workdir / "out.txt")


def _loop_count(code: str) -> int:
    """Return the number of ``do`` statements of the kernel."""
    return len(re.findall(r"^\s*do\b", code, re.M))


@pytest.mark.parametrize("chain_rule", [False, True])
def test_symmetry_removes_transposed_loops(chain_rule):
    """Declaring the symmetry never adds loops; plain kernels lose some."""
    plain = _kernel(_mixing(False), "fastest", chain_rule)
    by_param = _kernel(_mixing(False), "fastest", chain_rule, ("a",))
    by_attr = _kernel(_mixing(True), "fastest", chain_rule)
    if chain_rule:
        assert _loop_count(by_param) <= _loop_count(plain)
    else:
        assert _loop_count(by_param) < _loop_count(plain)
    assert _loop_count(by_attr) == _loop_count(by_param)


def test_resolver_marks_derivatives_without_extra_indices():
    """Only derivatives with the placeholder's own index pair are symmetric."""
    a = idx_function("a", symmetric=(0, 1))(k, l, T)
    resolver = _PlaceholderResolver(canonical=True)
    resolver.resolve(sp.Derivative(a, T) + sp.Derivative(a, (T, 2)) + a)
    resolver.resolve(sp.Derivative(a, n[k]))
    assert resolver.symmetric == {"a", "dadT", "dadTdT"}


def test_non_symmetric_placeholder_is_unchanged():
    """No declaration, no symmetric input."""
    a = idx_function("a")(k, l, T)
    resolver = _PlaceholderResolver(canonical=True)
    resolver.resolve(sp.Derivative(a, T) + a)
    assert resolver.symmetric == set()


def test_invalid_symmetric_pair_is_rejected():
    """Only the first two arguments can be declared symmetric."""
    with pytest.raises(ValueError, match="symmetric"):
        idx_function("a", symmetric=(1, 2))


@pytest.mark.slow
@pytest.mark.skipif(GFORTRAN is None, reason="needs gfortran")
@pytest.mark.parametrize("chain_rule", [False, True])
@pytest.mark.parametrize("preset", PRESETS)
def test_symmetric_kernels_match_plain_kernel(preset, chain_rule):
    """With symmetric inputs, symmetric kernels agree with the plain one."""
    reference = _evaluate(_kernel(_mixing(False), "minimal", False))
    for diff, params in ((_mixing(False), ("a",)), (_mixing(True), ())):
        code = _kernel(diff, preset, chain_rule, params)
        np.testing.assert_allclose(_evaluate(code), reference, rtol=1e-11)
