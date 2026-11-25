"""Names that differ only in case must map to one Fortran identifier.

Fortran is case-insensitive, so ``R`` and ``r`` are the same variable.  The
generator renames the clashing parameters consistently in the signature,
the declarations and the body, keeps module symbols, outputs and the fixed
names (``n``, ``T``, ``nc``...), and raises ``ValueError`` when two names
that cannot be renamed clash.
"""

import re
from pathlib import Path

import pytest

import sympy as sp

from fortran_tools import DEBUG_FREE_FORM_FLAGS, GFORTRAN, compile_and_run

from thermodiff import DiffPlz, T, idx_function, k, m, n, sum_components
from thermodiff.fortran import to_fortran

NC = 3
MOLES = [1.0, 2.5, 0.7]
RELATIVE_TOLERANCE = 1e-12
SIGNATURE = re.compile(r"subroutine\s+\w+\((.*?)\)", re.IGNORECASE)


def argument_names(source: str) -> list[str]:
    """Return the dummy argument names of the generated subroutine."""
    match = SIGNATURE.search(source)
    assert match is not None
    return [arg.strip() for arg in match.group(1).split(",")]


def assert_case_unique(source: str) -> None:
    """Assert that no two dummy arguments differ only in case."""
    lowered = [arg.lower() for arg in argument_names(source)]
    assert len(lowered) == len(set(lowered))


def run_kernel(
    tmp_path: Path,
    source: str,
    call_values: list[str],
    module_lines: str = "",
) -> list[float]:
    """Compile ``source`` in a module, call it and return the printed floats.

    ``call_values`` are the actual arguments, in order, as Fortran text.
    """
    name = SIGNATURE.search(source).group(0).split()[1].split("(")[0]
    program = (
        f"module kernels\n   implicit none\n{module_lines}\ncontains\n"
        f"{source}\nend module kernels\n\n"
        "program drive\n   use kernels\n   implicit none\n"
        f"   integer, parameter :: nc = {NC}\n"
        "   real(8) :: n(nc), f, dfdn(nc)\n"
        f"   n = [{', '.join(f'{v}d0' for v in MOLES)}]\n"
        f"   call {name}({', '.join(call_values)})\n"
        "   print '(*(es24.16))', f, dfdn\nend program drive\n"
    )
    path = tmp_path / "drive.f90"
    path.write_text(program, encoding="utf-8")
    run = compile_and_run([path], tmp_path, DEBUG_FREE_FORM_FLAGS)
    return [float(token) for token in run.stdout.split()]


def reference(expression: sp.Expr, values: dict[sp.Symbol, float]) -> list:
    """Return ``[f, df/dn_1 ... df/dn_nc]`` of an explicit-sum expression."""
    moles = sp.symbols(f"x1:{NC + 1}")
    explicit = expression.subs(
        {n[idx]: moles[idx - 1] for idx in range(1, NC + 1)}
    ).doit()
    point = {**values, **dict(zip(moles, MOLES))}
    return [float(explicit.subs(point))] + [
        float(sp.diff(explicit, x).subs(point)) for x in moles
    ]


def explicit_sum(term, index, count=NC):
    """Expand ``Sum(term, (index, 1, nc))`` for ``nc = count``."""
    return sum(term.subs(index, idx) for idx in range(1, count + 1))


pytestmark = pytest.mark.skipif(GFORTRAN is None, reason="needs gfortran")


class TestCaseClash:
    """Case-only clashes are renamed consistently or rejected."""

    def test_module_symbol_and_array_parameter(self, tmp_path):
        """The issue reproducer: module ``R`` against array ``r``."""
        big_r = sp.Symbol("R")
        base = sp.IndexedBase("r")
        total = sum_components(n[m] * base[m], m)
        expr = big_r * sum_components(n[k] * sp.log(base[k] / total), k)
        diff = DiffPlz(expr, indexes=[k, m], name="f")
        source = to_fortran(diff, derivatives=["dni"], module_symbols=["R"])
        assert_case_unique(source)
        r_values = [1.5, 0.5, 2.0]
        explicit = big_r * explicit_sum(
            n[k] * sp.log(base[k] / explicit_sum(n[m] * base[m], m)), k
        )
        values = {big_r: 8.5}
        values.update({base[i + 1]: v for i, v in enumerate(r_values)})
        got = run_kernel(
            tmp_path,
            source,
            ["nc", "n", "[1.5d0, 0.5d0, 2.0d0]", "f", "dfdn"],
            "   real(8), parameter :: R = 8.5d0",
        )
        expected = reference(explicit, values)
        assert got == pytest.approx(expected, rel=RELATIVE_TOLERANCE)

    def test_parameter_against_parameter(self, tmp_path):
        """Scalars ``A`` and ``a`` stay two distinct arguments."""
        upper, lower = sp.symbols("A a")
        expr = sum_components(n[k] * (upper + 2 * lower * n[k]), k)
        source = to_fortran(
            DiffPlz(expr, indexes=[k], name="f"), derivatives=["dni"]
        )
        assert_case_unique(source)
        assert len(argument_names(source)) == 6
        got = run_kernel(
            tmp_path, source, ["nc", "0.5d0", "1.25d0", "n", "f", "dfdn"]
        )
        explicit = explicit_sum(n[k] * (upper + 2 * lower * n[k]), k)
        expected = reference(explicit, {upper: 0.5, lower: 1.25})
        assert got == pytest.approx(expected, rel=RELATIVE_TOLERANCE)

    def test_parameter_against_output(self, tmp_path):
        """Parameter ``F`` is renamed: the output ``f`` keeps its name."""
        param = sp.Symbol("F")
        expr = param * sum_components(n[k] ** 2, k)
        source = to_fortran(
            DiffPlz(expr, indexes=[k], name="f"), derivatives=["dni"]
        )
        assert_case_unique(source)
        assert "f" in argument_names(source)
        assert "F" not in argument_names(source)
        got = run_kernel(tmp_path, source, ["nc", "3.5d0", "n", "f", "dfdn"])
        explicit = param * explicit_sum(n[k] ** 2, k)
        expected = reference(explicit, {param: 3.5})
        assert got == pytest.approx(expected, rel=RELATIVE_TOLERANCE)

    def test_placeholder_against_parameter(self, tmp_path):
        """Placeholder ``g`` and parameter ``G`` do not share a name."""
        param = sp.Symbol("G")
        placeholder = idx_function("g")(k)
        expr = param * sum_components(n[k] * placeholder, k)
        diff = DiffPlz(expr, internal_functions=[placeholder], indexes=[k])
        source = to_fortran(diff, name="f", derivatives=["dni"])
        assert_case_unique(source)
        got = run_kernel(
            tmp_path,
            source,
            ["nc", "2.0d0", "n", "[0.5d0, 1.5d0, 4.0d0]", "f", "dfdn"],
        )
        g_values = [0.5, 1.5, 4.0]
        symbols = sp.symbols("g1:4")
        explicit = param * sum(n[idx + 1] * symbols[idx] for idx in range(NC))
        values = {param: 2.0, **dict(zip(symbols, g_values))}
        expected = reference(explicit, values)
        assert got == pytest.approx(expected, rel=RELATIVE_TOLERANCE)

    def test_module_symbols_clash_raises(self):
        """Two module symbols that differ in case cannot be renamed."""
        expr = sp.Symbol("R") * sp.Symbol("r") * sum_components(n[k], k)
        diff = DiffPlz(expr, indexes=[k], name="f")
        with pytest.raises(ValueError, match=r"'R'.*'r'|'r'.*'R'"):
            to_fortran(diff, module_symbols=["R", "r"])

    def test_module_symbol_against_output_raises(self):
        """A module symbol ``F`` cannot be renamed to free the output ``f``."""
        expr = sp.Symbol("F") * sum_components(n[k], k)
        diff = DiffPlz(expr, indexes=[k], name="f")
        with pytest.raises(ValueError, match="'F'.*'f'|'f'.*'F'"):
            to_fortran(diff, module_symbols=["F"])

    def test_derived_placeholder_name_clash_raises(self):
        """``dgdT`` (derivative of ``g``) against a parameter ``DGDT``."""
        placeholder = idx_function("g")(k, T)
        param = sp.Symbol("DGDT")
        expr = param * sum_components(n[k] * placeholder, k)
        diff = DiffPlz(expr, internal_functions=[placeholder], indexes=[k])
        with pytest.raises(ValueError, match="same identifier"):
            to_fortran(diff, name="f", derivatives=["dt", "dni"])
