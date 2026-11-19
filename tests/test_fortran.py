"""Tests for the Fortran code generation module."""

import os
import shutil
import subprocess

import pytest

import sympy as sp

from thermodiff import (
    DiffPlz,
    R,
    T,
    V,
    idx_function,
    k,
    l,
    m,
    n,
    nc,
    sum_components,
    to_fortran,
)
from thermodiff.fortran import (
    _index_set,
    _Namer,
    _best_indexed_candidate,
    _best_scalar_candidate,
    _common_scalar_cse,
    _free_indices,
    _indexed_cse,
    _preorder_entries,
)


def _find_gfortran():
    """Return the gfortran executable to use, honoring ``FC``."""
    compiler = os.environ.get("FC")
    if compiler and "gfortran" in os.path.basename(compiler).lower():
        return compiler
    return shutil.which("gfortran")


GFORTRAN = _find_gfortran()


def explicit_diff():
    """Return a DiffPlz for f = T * sum(n_k**2) + V * sum(n_k)."""
    expr = T * sum_components(n[k] ** 2, k) + V * sum_components(n[k], k)
    return DiffPlz(expr, internal_functions=None, indexes=[k], name="f")


def mole_fraction_diff():
    """Return a DiffPlz for x_k = n[k] / sum(n_l)."""
    nt = sum_components(n[l], l)
    return DiffPlz(
        n[k] / nt, internal_functions=None, indexes=[k, l], name="x_k"
    )


def indexed_sum_diff():
    """Return a DiffPlz with indexed placeholder functions."""
    phi_k = idx_function(r"\phi")(n[k])
    tau_lk = idx_function(r"\tau")(l, k, T)
    expr = sum_components(n[k] * phi_k * tau_lk, k)
    return DiffPlz(
        expr, internal_functions=[phi_k, tau_lk], indexes=[k, l], name="f"
    )


class TestSignature:
    """Generated signature and declarations."""

    def test_subroutine_name_avoids_value_name(self):
        code = to_fortran(explicit_diff())
        assert code.startswith("subroutine f_derivs(")
        assert "end subroutine f_derivs" in code

    def test_has_implicit_none(self):
        assert "implicit none" in to_fortran(explicit_diff())

    def test_component_count_argument(self):
        code = to_fortran(explicit_diff())
        assert "integer, intent(in) :: nc" in code
        assert "real(8), intent(in) :: n(:)" in code

    def test_derivative_outputs(self):
        code = to_fortran(explicit_diff())
        assert "real(8), intent(out) :: f" in code
        assert "real(8), intent(out) :: dfdT" in code
        assert "real(8), intent(out) :: dfdV" in code
        assert "real(8), intent(out) :: dfdn(:)" in code
        assert "real(8), intent(out) :: dfdndn(:, :)" in code

    def test_indices_are_not_parameters(self):
        code = to_fortran(explicit_diff())
        assert "intent(in) :: i" not in code
        assert "intent(in) :: k" not in code

    def test_parameters_are_scalar_arguments(self):
        q = sp.Symbol("q")
        expr = q * T * sum_components(n[k], k)
        diff = DiffPlz(expr, internal_functions=None, indexes=[k], name="g")
        code = to_fortran(diff)
        assert "real(8), intent(in) :: q" in code

    def test_parameter_array(self):
        b = sp.IndexedBase("b", shape=(sp.Symbol("N_c", integer=True),))
        expr = sum_components(b[k] * n[k], k)
        diff = DiffPlz(expr, internal_functions=None, indexes=[k], name="g")
        code = to_fortran(diff)
        assert "real(8), intent(in) :: b(:)" in code

    def test_custom_subroutine_name(self):
        code = to_fortran(explicit_diff(), subroutine_name="my_routine")
        assert "subroutine my_routine(" in code


class TestBody:
    """Generated statements."""

    def test_sum_becomes_loop(self):
        code = to_fortran(explicit_diff())
        assert "do k = 1, nc" in code
        assert "end do" in code

    def test_piecewise_becomes_if_else(self):
        diff = mole_fraction_diff()
        code = to_fortran(diff, derivatives=["dni"])
        assert "if (i == k) then" in code
        assert "else" in code
        assert "dx_kdn(i, k) = res" in code

    def test_zero_derivatives_are_skipped(self):
        code = to_fortran(mole_fraction_diff())
        # x_k does not depend on T/V/P.
        assert "dfdT" not in code
        assert "intent(out) :: dx_kdn" in code


class TestPlaceholders:
    """Internal functions exposed as arguments."""

    def test_arrays_are_exposed(self):
        code = to_fortran(indexed_sum_diff(), derivatives=["dni"])
        assert "real(8), intent(in) :: phi(:)" in code
        assert "real(8), intent(in) :: tau(:, :)" in code
        assert "real(8), intent(in) :: dphidi(:, :)" in code

    def test_latex_names_are_sanitized(self):
        code = to_fortran(indexed_sum_diff(), derivatives=["dt"])
        assert r"\phi" not in code
        assert r"\tau" not in code
        assert "dtaudT(:, :)" in code

    def test_strict_raises(self):
        with pytest.raises(ValueError, match="strict mode"):
            to_fortran(indexed_sum_diff(), strict=True)

    def test_strict_ok_without_placeholders(self):
        # Should not raise.
        to_fortran(explicit_diff(), strict=True)


class TestOptions:
    """Selection helpers and method delegation."""

    def test_derivative_subset(self):
        code = to_fortran(explicit_diff(), derivatives=["dt", "dni"])
        assert "dfdT" in code
        assert "dfdn(:)" in code
        assert "dfdV" not in code
        assert "dfdndn" not in code

    def test_unknown_derivative_raises(self):
        with pytest.raises(ValueError, match="Unknown derivatives"):
            to_fortran(explicit_diff(), derivatives=["nope"])

    def test_friendly_derivative_names(self):
        code = to_fortran(explicit_diff(), derivatives=["dT", "dn"])
        assert "dfdT" in code
        assert "dfdn(:)" in code
        assert "dfdV" not in code

    def test_diffplz_skips_unselected_derivatives(self):
        diff = DiffPlz(
            T * sum_components(n[k] ** 2, k),
            indexes=[k],
            name="f",
            derivatives=["dn"],
        )
        assert diff.dt is None
        assert diff.dni is not None
        code = to_fortran(diff)
        assert "dfdn" in code
        assert "dfdT" not in code

    def test_mismatched_selection_raises(self):
        diff = DiffPlz(
            T * sum_components(n[k] ** 2, k),
            indexes=[k],
            name="f",
            derivatives=["dn"],
        )
        with pytest.raises(ValueError, match="not computed"):
            to_fortran(diff, derivatives=["dt"])

    def test_method_delegates(self):
        diff = explicit_diff()
        assert diff.to_fortran() == to_fortran(diff)

    def test_optional_outputs_are_declared(self):
        code = to_fortran(explicit_diff(), optional_derivatives=True)
        assert "real(8), optional, intent(out) :: f" in code
        assert "real(8), optional, intent(out) :: dfdT" in code
        assert "real(8), optional, intent(out) :: dfdn(:)" in code
        assert "real(8), optional, intent(out) :: dfdndn(:, :)" in code

    def test_optional_outputs_are_guarded(self):
        code = to_fortran(explicit_diff(), optional_derivatives=True)
        assert "if (present(f)) then" in code
        assert "if (present(dfdT)) then" in code
        assert "if (present(dfdn)) then" in code

    def test_optional_is_off_by_default(self):
        code = to_fortran(explicit_diff())
        assert "optional" not in code
        assert "present(" not in code

    def test_optional_guards_follow_derivative_selection(self):
        code = to_fortran(
            explicit_diff(),
            derivatives=["dt", "dn"],
            optional_derivatives=True,
        )
        assert "present(dfdV)" not in code
        assert "present(dfdndn)" not in code
        assert "if (present(dfdT)) then" in code


class TestFreeIndices:
    """Free-index detection on hand-built expressions."""

    NC = sp.Symbol("nc", integer=True)

    def test_free_and_bound_indices(self):
        total = sp.Sum(n[k], (k, 1, self.NC))
        assert _free_indices(n[k] + 1) == {k}
        assert _free_indices(total + 1) == set()
        assert _free_indices(sp.Sum(n[k] * n[l], (k, 1, self.NC))) == {l}

    def test_nested_sums(self):
        inner = sp.Sum(n[l], (l, 1, self.NC))
        nested = sp.Sum(n[k] * inner, (k, 1, self.NC))
        assert _free_indices(nested) == set()

    def test_idx_under_sum_limit_stays_free(self):
        assert _free_indices(sp.Sum(n[k], (k, 1, l))) == {l}

    def test_bound_index_is_bound_everywhere(self):
        # ``k`` is a summation index somewhere, so it is not reported
        # as free even where it appears outside that sum.
        first = sp.Sum(n[k], (k, 1, self.NC))
        second = sp.Sum(n[k] * n[l], (k, 1, self.NC))
        assert _free_indices(n[k] * first + n[l] * second) == {l}

    def test_shared_memo_matches_unmemoised_on_every_subtree(self):
        first = sp.Sum(n[k], (k, 1, self.NC))
        second = sp.Sum(n[k] * n[l], (k, 1, self.NC))
        expressions = [
            n[k] * first + n[l] * second,
            sp.Piecewise((n[k], sp.Eq(k, l)), (first, True)),
            sp.KroneckerDelta(k, l) * second,
            sp.Sum(sp.KroneckerDelta(k, l) * n[k], (k, 1, self.NC)),
            first * sp.Sum(n[k], (k, 1, l)),
            sp.Sum(
                n[k] * sp.Sum(n[l] * n[k], (l, 1, self.NC)), (k, 1, self.NC)
            )
            + n[l],
        ]
        memo = {}
        for expr in expressions:
            for sub in sp.preorder_traversal(expr):
                assert _free_indices(sub, memo=memo) == _free_indices(sub)


class TestSharing:
    """Shared summations and common sub-expressions."""

    def test_repeated_sums_are_computed_once(self):
        code = to_fortran(explicit_diff())
        # ``sum(n)`` and ``sum(n**2)`` feed several derivatives but are
        # expanded a single time each.
        assert code.count("do k = 1, nc") == 2
        assert code.count("n(k)**2") == 1

    def test_common_scalars_are_hoisted(self):
        namer = _Namer()
        arrays = {}
        x, y = sp.Symbol("x"), sp.Symbol("y")
        product = x * y
        exprs = [product + x, product + y, product + 1]
        residuals, definitions = _common_scalar_cse(exprs, namer, arrays)
        assert len(definitions) == 1
        target = definitions[0]["target"]
        assert all(target in expr.free_symbols for expr in residuals)

    def test_common_scalars_pinned_order(self):
        namer = _Namer()
        arrays = {}
        a, b, c = sp.symbols("a b c")
        inner = sp.exp(a * b)
        shared = sp.sqrt(a * b + c)
        exprs = [
            inner * T + shared,
            inner * V + shared,
            inner + shared * T,
            shared * inner * c,
            sp.log(inner) + shared**2,
        ]
        residuals, definitions = _common_scalar_cse(exprs, namer, arrays)
        assert [str(expr) for expr in residuals] == [
            "T*t2 + t3",
            "V*t2 + t3",
            "T*t3 + t2",
            "c*t2*t3",
            "c + t1 + log(t2)",
        ]
        assert [
            (str(item["target"]), str(item["body"])) for item in definitions
        ] == [
            ("t1", "a*b"),
            ("t2", "exp(t1)"),
            ("t3", "sqrt(c + t1)"),
        ]
        assert [str(name) for name in namer.created] == ["t1", "t2", "t3"]
        assert arrays == {}

    def test_common_scalars_sort_key_breaks_ties(self):
        namer = _Namer()
        arrays = {}
        a, c = sp.symbols("a c")
        # ``sin(a)`` and ``cos(a)`` share count and ``count_ops``, so the
        # larger ``default_sort_key`` (``cos``) is hoisted first.
        first, second = sp.sin(a), sp.cos(a)
        assert sp.count_ops(first) == sp.count_ops(second)
        exprs = [
            first * T + second * V,
            first * V + second * T,
            first + second + c,
        ]
        residuals, definitions = _common_scalar_cse(exprs, namer, arrays)
        assert [str(expr) for expr in residuals] == [
            "T*t2 + V*t1",
            "T*t1 + V*t2",
            "c + t1 + t2",
        ]
        assert [
            (str(item["target"]), str(item["body"])) for item in definitions
        ] == [("t1", "cos(a)"), ("t2", "sin(a)")]
        assert [str(name) for name in namer.created] == ["t1", "t2"]

    def test_common_scalars_count_ops_breaks_count_ties(self):
        namer = _Namer()
        arrays = {}
        a, b, c = sp.symbols("a b c")
        # Both candidates occur twice; ``sqrt(a*b + c)`` has more
        # operations, so it is hoisted first.  The last expression holds
        # no candidate and must come back untouched.
        big, small = sp.sqrt(a * b + c), sp.sin(a)
        assert sp.count_ops(big) > sp.count_ops(small)
        exprs = [big * T + small * V, big * V + small * T, c + 5]
        residuals, definitions = _common_scalar_cse(exprs, namer, arrays)
        assert [str(expr) for expr in residuals] == [
            "T*t1 + V*t2",
            "T*t2 + V*t1",
            "c + 5",
        ]
        assert residuals[2] is exprs[2]
        assert [
            (str(item["target"]), str(item["body"])) for item in definitions
        ] == [("t1", "sqrt(a*b + c)"), ("t2", "sin(a)")]

    def test_best_scalar_candidate_ranks_only_top_count(self):
        a, b = sp.symbols("a b")
        cheap, costly, rare = sp.sin(a), sp.exp(a * b), sp.log(a)
        counts = {cheap: 3, rare: 2, costly: 3}
        ranked = {}
        best = _best_scalar_candidate(list(counts), counts, ranked)
        assert best == costly
        assert rare not in ranked

    def test_best_indexed_candidate_falls_back_to_lower_count(self):
        ac = sp.IndexedBase("ac", shape=(sp.Symbol("nc", integer=True),))
        cheap = sp.log(ac[k])
        costly = sp.sqrt(ac[k] ** 2 + 1)
        # The count-three group holds only the cheap factor, which is
        # kept (three uses); the pair at count two is never examined.
        counts = {costly: 2, cheap: 3}
        ops_memo = {}
        assert _best_indexed_candidate(counts, False, ops_memo, {}) == cheap
        assert costly not in ops_memo
        # With the cheap factor used twice it is filtered out.
        counts = {costly: 2, cheap: 2}
        assert _best_indexed_candidate(counts, False, {}, {}) == costly
        assert _best_indexed_candidate({cheap: 2}, False, {}, {}) is None

    def test_index_dependent_terms_are_not_hoisted(self):
        namer = _Namer()
        arrays = {}
        b = sp.IndexedBase("b", shape=(sp.Symbol("nc", integer=True),))
        exprs = [b[k] + 1, b[k] + 2]
        residuals, definitions = _common_scalar_cse(exprs, namer, arrays)
        assert definitions == []
        assert residuals == exprs

    def test_indexed_cse_hoists_shared_factors(self):
        namer = _Namer()
        arrays = {}
        root = sp.sqrt(n[k] ** 2 + 1)
        exprs = [root * T, root * V, root + 1]
        residuals, definitions = _indexed_cse(exprs, namer, arrays)
        assert len(definitions) == 1
        definition = definitions[0]
        assert len(definition["free"]) == 1
        base = definition["target"].base
        assert str(base) in arrays
        assert arrays[str(base)] == 1
        assert all(
            any(indexed.base == base for indexed in expr.atoms(sp.Indexed))
            for expr in residuals
        )

    def test_indexed_cse_hoists_rank_two(self):
        namer = _Namer()
        arrays = {}
        root = sp.sqrt(n[k] * n[l] + 1)
        exprs = [root * T, root * V, root + 1]
        residuals, definitions = _indexed_cse(exprs, namer, arrays)
        assert len(definitions) == 1
        base = definitions[0]["target"].base
        assert arrays[str(base)] == 2
        assert all(
            any(indexed.base == base for indexed in expr.atoms(sp.Indexed))
            for expr in residuals
        )

    @staticmethod
    def _pinned_cse_expressions():
        ac = sp.IndexedBase("ac", shape=(sp.Symbol("nc", integer=True),))
        pair = sp.sqrt(ac[k] * ac[l])
        root = sp.sqrt(n[k] ** 2 + 1)
        logn = sp.log(n[k] + ac[k])
        return [
            pair * T + root * V,
            pair * V + root + logn,
            pair + logn * T,
            root * logn + 3,
        ]

    @staticmethod
    def _describe_cse(residuals, definitions, arrays):
        return (
            [str(expr) for expr in residuals],
            [
                (
                    str(item["target"]),
                    [str(index) for index in item["free"]],
                    str(item["body"]),
                )
                for item in definitions
            ],
            arrays,
        )

    def test_indexed_cse_output_is_pinned(self):
        arrays = {}
        residuals, definitions = _indexed_cse(
            self._pinned_cse_expressions(), _Namer(), arrays
        )
        assert self._describe_cse(residuals, definitions, arrays) == (
            [
                "T*t2[k, l] + V*t1[k]",
                "V*t2[k, l] + t1[k] + t3[k]",
                "T*t3[k] + t2[k, l]",
                "t1[k]*t3[k] + 3",
            ],
            [
                ("t1[p]", ["p"], "sqrt(n[p]**2 + 1)"),
                ("t2[p, q]", ["p", "q"], "sqrt(ac[p]*ac[q])"),
                ("t3[p]", ["p"], "log(ac[p] + n[p])"),
            ],
            {"t1": 1, "t2": 2, "t3": 1},
        )

    def test_indexed_cse_output_is_pinned_with_counted(self):
        # Only positions 1 and 2 are counted, but every position is
        # rewritten once a shared factor has been chosen.
        arrays = {}
        residuals, definitions = _indexed_cse(
            self._pinned_cse_expressions(),
            _Namer(),
            arrays,
            counted=[False, True, True, False],
        )
        assert self._describe_cse(residuals, definitions, arrays) == (
            [
                "T*t1[k, l] + V*sqrt(n[k]**2 + 1)",
                "V*t1[k, l] + sqrt(n[k]**2 + 1) + t2[k]",
                "T*t2[k] + t1[k, l]",
                "sqrt(n[k]**2 + 1)*t2[k] + 3",
            ],
            [
                ("t1[p, q]", ["p", "q"], "sqrt(ac[p]*ac[q])"),
                ("t2[p]", ["p"], "log(ac[p] + n[p])"),
            ],
            {"t1": 2, "t2": 1},
        )

    def test_indexed_cse_cheap_filter_with_counted(self):
        # ``log(ac[k])`` is counted twice with a single operation, so the
        # ``ops < 2 and count < 3`` filter keeps it inline; the third
        # position is not counted although it also holds the factor.
        ac = sp.IndexedBase("ac", shape=(sp.Symbol("nc", integer=True),))
        cheap = sp.log(ac[k])
        root = sp.sqrt(n[k] ** 2 + 1)
        exprs = [
            cheap * T + root * V,
            cheap * V + root + 3,
            cheap + root * T,
            cheap * 5,
        ]
        arrays = {}
        residuals, definitions = _indexed_cse(
            exprs, _Namer(), arrays, counted=[True, True, False, False]
        )
        assert self._describe_cse(residuals, definitions, arrays) == (
            [
                "T*log(ac[k]) + V*t1[k]",
                "V*log(ac[k]) + t1[k] + 3",
                "T*t1[k] + log(ac[k])",
                "5*log(ac[k])",
            ],
            [("t1[p]", ["p"], "sqrt(n[p]**2 + 1)")],
            {"t1": 1},
        )

    def test_index_set_matches_atoms(self):
        total = sp.Sum(n[k], (k, 1, 5))
        exprs = [
            sp.sqrt(n[k] * n[l]) + T,
            total + n[l],
            sp.KroneckerDelta(k, l) * V,
        ]
        memo = {}
        for expr in exprs:
            for sub in sp.preorder_traversal(expr):
                assert _index_set(sub, memo) == frozenset(sub.atoms(sp.Idx))

    def test_indexed_cse_skips_rank_three(self):
        # A rank three temporary is ``nc**3`` entries; benchmarks show it
        # costs more than it saves, so it must never be created.
        namer = _Namer()
        arrays = {}
        w = sp.IndexedBase("w", shape=(sp.Symbol("nc", integer=True),) * 3)
        root = sp.sqrt(w[k, l, m] ** 2 + 1)
        exprs = [root * T, root * V, root + 1]
        residuals, definitions = _indexed_cse(exprs, namer, arrays)
        assert definitions == []
        assert arrays == {}
        assert residuals == exprs

    def test_rank_three_model_has_no_rank_three_temporaries(self):
        # Regression: materialising a rank three factor made the generated
        # code slower (``nc**3`` entries fall out of cache).  A rank three
        # model must still share its rank two factors, but never allocate a
        # rank three temporary.
        ac = sp.IndexedBase("ac", shape=(sp.Symbol("nc", integer=True),))
        w = sp.IndexedBase("w", shape=(sp.Symbol("nc", integer=True),) * 3)
        term = n[k] * n[l] * n[m] * sp.sqrt(ac[k] * ac[l] * ac[m]) * w[k, l, m]
        expr = T * sum_components(
            sum_components(sum_components(term, m), l), k
        )
        diff = DiffPlz(
            expr, internal_functions=None, indexes=[k, l, m], name="f"
        )
        code = to_fortran(diff)
        assert "(nc, nc, nc)" not in code
        assert "(nc, nc)" in code

    def test_indexed_cse_ignores_conditions(self):
        # ``Eq`` conditions and ``Piecewise`` must never be turned into a
        # real array temporary.
        namer = _Namer()
        arrays = {}
        condition = sp.Eq(k, l)
        exprs = [
            sp.Piecewise((n[k], condition), (0, True)),
            sp.Piecewise((n[k] * T, condition), (0, True)),
        ]
        residuals, definitions = _indexed_cse(exprs, namer, arrays)
        assert definitions == []
        assert residuals == exprs


def _nested_expressions():
    """Build expressions that cover the node types of the CSE passes."""
    f = sp.Function("f")
    body = sp.sqrt(n[k] * n[l]) + sp.exp(-T / (R * n[k]))
    return [
        sp.Sum(body * f(T, n[k]), (k, 1, nc)) ** 2 + T,
        sp.Piecewise((body, sp.Eq(k, l)), (n[k] ** 3 / T, True)),
        sp.KroneckerDelta(k, l) * sp.log(T) * f(T, T) + sp.Max(T, 2, R),
        sp.And(sp.Eq(k, l), sp.Lt(T, 3)),
    ]


class TestPreorderEntries:
    """The memoised walk reproduces ``sp.preorder_traversal``."""

    @staticmethod
    def _node_itself(node):
        return (node,)

    def test_matches_preorder_traversal(self):
        memo = {}
        for expr in _nested_expressions():
            walked = _preorder_entries(expr, self._node_itself, memo)
            assert list(walked) == list(sp.preorder_traversal(expr))

    def test_reuses_memo_after_xreplace(self):
        memo = {}
        exprs = _nested_expressions()
        for expr in exprs:
            _preorder_entries(expr, self._node_itself, memo)
        rule = {sp.exp(-T / (R * n[k])): sp.Symbol("tmp")}
        for expr in exprs:
            rewritten = expr.xreplace(rule)
            walked = _preorder_entries(rewritten, self._node_itself, memo)
            assert list(walked) == list(sp.preorder_traversal(rewritten))

    def test_filtered_entries(self):
        def only_functions(node):
            return (node,) if isinstance(node, sp.Function) else ()

        for expr in _nested_expressions():
            walked = _preorder_entries(expr, only_functions, {})
            expected = [
                node
                for node in sp.preorder_traversal(expr)
                if isinstance(node, sp.Function)
            ]
            assert list(walked) == expected


@pytest.mark.skipif(not GFORTRAN, reason="gfortran not available")
class TestCompile:
    """Compile and run the generated code with gfortran."""

    def _compile_and_run(self, tmp_path, code, driver):
        module = "module thermo_mod\ncontains\n" + code
        module += "\nend module thermo_mod\n"
        source = tmp_path / "main.f90"
        source.write_text(module + driver)
        exe = tmp_path / ("a.exe" if os.name == "nt" else "a.out")
        compile_result = subprocess.run(
            [GFORTRAN, "-O2", "-o", str(exe), str(source)],
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        assert compile_result.returncode == 0, compile_result.stderr
        run_result = subprocess.run(
            [str(exe)], capture_output=True, text=True, cwd=tmp_path
        )
        assert run_result.returncode == 0, run_result.stderr
        return run_result.stdout

    def test_explicit_model(self, tmp_path):
        code = to_fortran(explicit_diff())
        driver = (
            "program main\n"
            "  use thermo_mod\n"
            "  implicit none\n"
            "  integer, parameter :: nc = 3\n"
            "  real(8) :: n(nc), V, T\n"
            "  real(8) :: f, dfdT, dfdV, dfdndV\n"
            "  real(8) :: dfdn(nc), dfdndT(nc), dfdndn(nc, nc)\n"
            "  n = [1.0d0, 2.0d0, 3.0d0]\n"
            "  V = 5.0d0\n"
            "  T = 2.0d0\n"
            "  call f_derivs(nc, V, T, n, f, dfdT, dfdV, dfdn, "
            "dfdndn, dfdndT, dfdndV)\n"
            "  write(*,*) f, dfdT, dfdV\n"
            "  write(*,*) dfdn\n"
            "  write(*,*) dfdndT\n"
            "  write(*,*) dfdndV\n"
            "  write(*,*) dfdndn(1, 1), dfdndn(1, 2)\n"
            "end program main\n"
        )
        output = self._compile_and_run(tmp_path, code, driver)
        values = [float(v) for v in output.split()]
        assert values[0] == pytest.approx(58.0)  # f
        assert values[1] == pytest.approx(14.0)  # dfdT
        assert values[2] == pytest.approx(6.0)  # dfdV
        assert values[3:6] == pytest.approx([9.0, 13.0, 17.0])  # dfdn
        assert values[6:9] == pytest.approx([2.0, 4.0, 6.0])  # dfdndT
        assert values[9] == pytest.approx(1.0)  # dfdndV
        assert values[10] == pytest.approx(4.0)  # dfdndn(1, 1)
        assert values[11] == pytest.approx(0.0)  # dfdndn(1, 2)

    def test_mole_fraction_model(self, tmp_path):
        diff = mole_fraction_diff()
        code = to_fortran(diff, derivatives=["dni"])
        driver = (
            "program main\n"
            "  use thermo_mod\n"
            "  implicit none\n"
            "  integer, parameter :: nc = 3\n"
            "  real(8) :: n(nc)\n"
            "  real(8) :: x_k(nc), dx_kdn(nc, nc)\n"
            "  n = [1.0d0, 2.0d0, 3.0d0]\n"
            "  call x_k_derivs(nc, n, x_k, dx_kdn)\n"
            "  write(*,*) x_k\n"
            "  write(*,*) dx_kdn(1, 1), dx_kdn(1, 2)\n"
            "end program main\n"
        )
        output = self._compile_and_run(tmp_path, code, driver)
        values = [float(v) for v in output.split()]
        # x_k = n_k / 6
        assert values[0:3] == pytest.approx([1 / 6, 2 / 6, 3 / 6])
        # dx_1/dn_1 = 1/6 - 1/36 ; dx_1/dn_2 = -2/36
        assert values[3] == pytest.approx(1 / 6 - 1 / 36)
        assert values[4] == pytest.approx(-2 / 36)

    def test_optional_derivatives_subset(self, tmp_path):
        code = to_fortran(explicit_diff(), optional_derivatives=True)
        driver = (
            "program main\n"
            "  use thermo_mod\n"
            "  implicit none\n"
            "  integer, parameter :: nc = 3\n"
            "  real(8) :: n(nc), V, T\n"
            "  real(8) :: f, full_f, full_t, only_t\n"
            "  n = [1.0d0, 2.0d0, 3.0d0]\n"
            "  V = 5.0d0\n"
            "  T = 2.0d0\n"
            "  call f_derivs(nc, V, T, n, f=f)\n"
            "  call f_derivs(nc, V, T, n, f=full_f, dfdT=full_t)\n"
            "  call f_derivs(nc, V, T, n, dfdT=only_t)\n"
            "  write(*,*) f, full_f, full_t, only_t\n"
            "end program main\n"
        )
        output = self._compile_and_run(tmp_path, code, driver)
        values = [float(v) for v in output.split()]
        # f = T*sum(n**2) + V*sum(n) ; dfdT = sum(n**2)
        assert values[0] == pytest.approx(58.0)  # only f
        assert values[1] == pytest.approx(58.0)  # f alongside dfdT
        assert values[2] == pytest.approx(14.0)  # dfdT alongside f
        assert values[3] == pytest.approx(14.0)  # dfdT on its own


def test_scan_atoms_matches_separate_atoms_calls():
    """The merged walk returns the same sets as separate ``atoms`` calls."""
    from thermodiff.fortran import _scan_atoms

    idx = sp.Idx("a")
    base = sp.IndexedBase("w")
    other = sp.IndexedBase("z")
    x_sym = sp.Symbol("x")
    exprs = [
        x_sym,
        base[idx] * x_sym
        + sp.Sum(other[idx, idx] * sp.exp(x_sym), (idx, 1, 3)),
        sp.Integer(3),
        sp.Sum(sp.Sum(base[idx], (idx, 1, 2)) * x_sym, (idx, 1, 4)),
    ]
    for expr in exprs:
        scan = _scan_atoms(expr)
        assert scan.indexed == expr.atoms(sp.Indexed)
        assert scan.sums == expr.atoms(sp.Sum)
        assert scan.idxs == expr.atoms(sp.Idx)
        assert scan.bases == expr.atoms(sp.IndexedBase)
        assert scan.symbols == expr.atoms(sp.Symbol)
        assert (x_sym in scan.symbols) == expr.has(x_sym)
