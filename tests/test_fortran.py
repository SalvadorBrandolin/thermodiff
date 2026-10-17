"""Tests for the Fortran code generation module."""

import os
import shutil
import subprocess

import pytest

import sympy as sp

from thermodiff import (
    DiffPlz,
    T,
    V,
    idx_function,
    k,
    l,
    n,
    sum_components,
    to_fortran,
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
