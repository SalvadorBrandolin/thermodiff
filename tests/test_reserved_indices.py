"""Summations over the reserved indices ``i`` and ``j`` (issue #16)."""

import pytest
import sympy as sp
from fortran_tools import (
    DEBUG_FREE_FORM_FLAGS,
    GFORTRAN,
    compile_and_run,
)

from thermodiff import DiffPlz, T, i, idx_function, j, k, l, n, sum_components

X = sp.IndexedBase("x")


def _expression(index):
    return sum_components(n[index] * X[index], index) ** 2


@pytest.mark.parametrize("bound", [i, j])
def test_bound_reserved_index_matches_free_notation(bound):
    reference = DiffPlz(_expression(k), indexes=[k], name="f")
    renamed = DiffPlz(_expression(bound), indexes=[bound], name="f")
    assert renamed.expression == reference.expression
    assert renamed.indexes == reference.indexes
    for attribute in ("dt", "dni", "dnidnj", "dtdni"):
        assert getattr(renamed, attribute) == getattr(reference, attribute)


def test_fortran_matches_free_notation():
    reference = DiffPlz(_expression(k), indexes=[k], name="f")
    renamed = DiffPlz(_expression(i), indexes=[i], name="f")
    assert renamed.to_fortran() == reference.to_fortran()


def test_fresh_index_avoids_names_in_use():
    expression = sum_components(n[i] * X[i] * sum_components(n[k], k), i)
    diff = DiffPlz(expression, indexes=[k], name="f")
    assert not diff.expression.has(i)
    assert diff.expression.has(l)


def test_free_reserved_index_raises():
    with pytest.raises(ValueError, match="reserved"):
        DiffPlz(n[i] * X[i], indexes=[k], name="f")


def test_non_clashing_expression_is_untouched():
    expression = _expression(k)
    assert DiffPlz(expression, indexes=[k]).expression is expression


PHI = idx_function("phi")
ALPHA = idx_function("a")


def _placeholder_diff(bound, free):
    """Return a DiffPlz with indexed placeholders summed over ``bound``."""
    phi = PHI(n[bound])
    a_bound = ALPHA(bound, T)
    expression = sum_components(n[bound] * phi * a_bound, bound)
    return DiffPlz(
        expression,
        internal_functions=[phi, a_bound],
        indexes=[free],
        name="f",
        derivatives=["dt", "dni", "dnidnj"],
    )


def test_internal_functions_follow_the_renaming():
    renamed = _placeholder_diff(i, k)
    reference = _placeholder_diff(l, k)  # ``l`` is the fresh index chosen
    assert renamed.expression == reference.expression
    assert renamed.internal_functions == reference.internal_functions
    assert not any(f.has(i) for f in renamed.internal_functions)
    for attribute in ("dt", "dni", "dnidnj"):
        assert getattr(renamed, attribute) == getattr(reference, attribute)


def test_placeholders_fortran_and_latex_match_free_notation():
    renamed = _placeholder_diff(i, k)
    reference = _placeholder_diff(l, k)
    assert renamed.to_fortran(chain_rule=False) == reference.to_fortran(
        chain_rule=False
    )
    assert renamed.latex_readable_plz() == reference.latex_readable_plz()
    # The placeholders were really substituted, not left as raw functions.
    assert "\\phi_{l}" in renamed.latex_readable_plz()["dni"]


_DRIVER = """
program drive
   use kernels
   implicit none
   integer, parameter :: nc = 3
   real(8) :: n(nc), a(nc), dadT(nc), phi(nc), dphidn(nc, nc)
   real(8) :: dphidndn(nc, nc, nc)
   real(8) :: f, dfdT, dfdn(nc), dfdndn(nc, nc)
   integer :: p, q, r
   n = [0.3d0, 0.5d0, 0.9d0]
   a = [1.1d0, 0.7d0, 1.9d0]
   dadT = [0.2d0, -0.4d0, 0.6d0]
   phi = [1.5d0, 0.8d0, 2.1d0]
   do p = 1, nc
      do q = 1, nc
         dphidn(p, q) = 0.1d0 * p - 0.05d0 * q
         do r = 1, nc
            dphidndn(p, q, r) = 0.01d0 * (p + 2 * q - r)
         end do
      end do
   end do
   call {name}(nc, n, a, dadT, dphidn, dphidndn, phi, &
               f, dfdT, dfdn, dfdndn)
   print "(*(es24.16))", f, dfdT, dfdn, dfdndn
end program drive
"""


@pytest.mark.skipif(GFORTRAN is None, reason="gfortran not available")
def test_placeholder_kernel_matches_free_notation(tmp_path):
    diffs = {"inew": _placeholder_diff(i, k), "kold": _placeholder_diff(k, l)}
    body = "".join(
        diff.to_fortran(chain_rule=False, subroutine_name=f"{name}_derivs")
        + "\n"
        for name, diff in diffs.items()
    )
    (tmp_path / "kernels.f90").write_text(
        "module kernels\ncontains\n" + body + "end module kernels\n"
    )
    results = {}
    for name in diffs:
        (tmp_path / f"{name}.f90").write_text(
            _DRIVER.format(name=f"{name}_derivs")
        )
        run = compile_and_run(
            [tmp_path / "kernels.f90", tmp_path / f"{name}.f90"],
            tmp_path,
            DEBUG_FREE_FORM_FLAGS,
        )
        results[name] = [float(v) for v in run.stdout.split()]
    assert len(results["inew"]) == 2 + 3 + 9
    assert results["inew"] == pytest.approx(results["kold"], rel=1e-12)
