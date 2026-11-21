"""Tests for the targeted Kronecker-delta reduction inside summations."""

from pathlib import Path

import sympy as sp

from thermodiff import n, nc
from thermodiff.core.kronecker_handling import (
    handle_free_kronecker,
    handle_sum_kronecker,
)
from thermodiff.model.ir import build_ir
from thermodiff.model.parser import read_sections
from thermodiff.thermovars import i, j, k, l


class TestReduceDeltaSums:
    """handle_sum_kronecker reduces only the delta summations."""

    def test_reduces_single_sum(self):
        expr = sp.Sum(n[k] * sp.KroneckerDelta(i, k), (k, 1, nc))
        assert sp.simplify(handle_sum_kronecker(expr, i) - n[i]) == 0

    def test_reduces_single_sum_reversed_delta(self):
        expr = sp.Sum(n[k] * sp.KroneckerDelta(k, i), (k, 1, nc))
        assert sp.simplify(handle_sum_kronecker(expr, i) - n[i]) == 0

    def test_keeps_delta_free_sum(self):
        expr = sp.Sum(n[k] ** 2, (k, 1, nc))
        assert handle_sum_kronecker(expr, i) == expr

    def test_mixed_terms_split(self):
        expr = sp.Sum(n[k] + n[k] * sp.KroneckerDelta(i, k), (k, 1, nc))
        reduced = handle_sum_kronecker(expr, i)
        expected = n[i] + sp.Sum(n[k], (k, 1, nc))
        assert sp.simplify(reduced - expected) == 0

    def test_reduces_double_sum_flattened(self):
        body = n[k] * n[l] * sp.KroneckerDelta(i, k)
        expr = sp.Sum(sp.Sum(body, (l, 1, nc)), (k, 1, nc))
        reduced = handle_sum_kronecker(expr, i)
        expected = sp.Sum(n[i] * n[l], (l, 1, nc))
        assert sp.simplify(reduced - expected) == 0

    def test_reduces_nested_sum_under_function(self):
        inner = sp.Sum(n[l] * sp.KroneckerDelta(i, l), (l, 1, nc))
        expr = sp.Sum(n[k] * sp.log(1 + inner), (k, 1, nc))
        reduced = handle_sum_kronecker(expr, i)
        expected = sp.Sum(n[k] * sp.log(1 + n[i]), (k, 1, nc))
        assert sp.simplify(reduced - expected) == 0

    def test_reduces_bound_bound_delta(self):
        body = n[k] * n[l] * sp.KroneckerDelta(k, l)
        expr = sp.Sum(sp.Sum(body, (l, 1, nc)), (k, 1, nc))
        reduced = handle_sum_kronecker(expr, i)
        expected = sp.Sum(n[k] ** 2, (k, 1, nc))
        assert sp.simplify(reduced - expected) == 0

    def test_free_delta_outside_sum_is_untouched(self):
        delta = sp.KroneckerDelta(i, j)
        assert handle_sum_kronecker(delta, i) == delta

    def test_free_delta_inside_sum_is_kept(self):
        expr = sp.Sum(n[k] * sp.KroneckerDelta(i, j), (k, 1, nc))
        reduced = handle_sum_kronecker(expr, i)
        assert reduced.has(sp.KroneckerDelta)
        assert reduced.has(sp.Sum)

    def test_range_assumptions_are_applied(self):
        # The legacy cleanup turns the doit() range guards into True.
        condition = sp.And(nc >= i, i >= 1)
        expr = sp.Piecewise((n[i], condition), (0, True))
        assert handle_sum_kronecker(expr, i) == n[i]

    def test_xreplace_matches_subs_on_range_guards(self):
        guards = {i >= 1: True, nc >= i: True}
        conditions = [
            sp.And(nc >= i, i >= 1),
            sp.And(i <= nc, 1 <= i),
            sp.And(i > 0, i <= nc),
            i >= 1,
        ]
        for condition in conditions:
            piecewise = sp.Piecewise((n[i], condition), (0, True))
            nested = sp.Sum(2 * piecewise + n[k], (k, 1, nc)) * n[j]
            for expr in (piecewise, piecewise + n[j], nested):
                assert handle_sum_kronecker(expr, i) == expr.subs(guards)

    def test_delta_inside_add_factor_is_distributed(self):
        # The delta is not a factor of the whole term: only its own
        # summand may be substituted out of the summation.
        body = (n[k] + sp.KroneckerDelta(i, k) * n[l]) * n[k]
        expr = sp.Sum(body, (k, 1, nc))
        reduced = handle_sum_kronecker(expr, i)
        expected = n[i] * n[l] + sp.Sum(n[k] ** 2, (k, 1, nc))
        assert sp.simplify(reduced - expected) == 0

    def test_result_matches_doit_reference(self):
        # Equivalence with the previous doit()-based behaviour on a case
        # where doit() terminates.
        body = n[k] ** 2 + sp.KroneckerDelta(i, k) * n[l] * n[k]
        expr = sp.Sum(body, (k, 1, nc))
        expected = expr.doit().subs({i >= 1: True, nc >= i: True})
        assert sp.simplify(handle_sum_kronecker(expr, i) - expected) == 0

    def test_handle_free_kronecker_unchanged(self):
        expr = n[k] * sp.KroneckerDelta(i, k)
        piecewise = handle_free_kronecker(expr, k, i)
        assert piecewise.is_Piecewise
        assert len(piecewise.args) == 2


_CAPTURE_MODEL_OK = """
model ok
variables
    n nc
expressions
    ntot = sum_m n_m
    bmix = sum_i sum_j n_i*n_j/ntot
function
    bmix
"""

_CAPTURE_MODEL_BAD = """
model bad
variables
    n nc
expressions
    ntot = sum_i n_i
    bmix = sum_i sum_j n_i*n_j/ntot
function
    bmix
"""


def _nested_binding_violations(expr):
    """Return (outer, inner) Sum pairs that bind the same index."""
    violations = []
    for outer in expr.atoms(sp.Sum):
        outer_bound = {limit[0] for limit in outer.limits}
        for inner in outer.function.atoms(sp.Sum):
            inner_bound = {limit[0] for limit in inner.limits}
            if outer_bound & inner_bound:
                violations.append((outer, inner))
    return violations


class TestSumLetterCapture:
    """`ntot = sum_m n_m` avoids binder capture after definition inlining."""

    def test_sum_m_model_has_no_capture(self):
        raw = read_sections(_CAPTURE_MODEL_OK, "ok.model")
        ir = build_ir(raw, "ok.model")
        assert _nested_binding_violations(ir.function) == []

    def test_sum_i_sketch_no_longer_captures(self):
        # Before the capture-avoiding inlining (alpha renaming of bound
        # indices, thesis extension) `ntot = sum_i n_i` inlined into
        # `sum_i sum_j ...` bound `i` twice.  It must not any more.
        raw = read_sections(_CAPTURE_MODEL_BAD, "bad.model")
        ir = build_ir(raw, "bad.model")
        assert _nested_binding_violations(ir.function) == []

    def test_pr76_model_file_uses_sum_m(self):
        path = Path(__file__).resolve().parent / "data/models/pr76/pr76.model"
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        assert "ntot    = sum_m n_m" in text
        raw = read_sections(text, "pr76.model")
        ir = build_ir(raw, "pr76.model")
        assert _nested_binding_violations(ir.function) == []
