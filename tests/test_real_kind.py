"""Configurable real kind of the generated kernels (issue #9).

``real_kind`` and ``kind_module`` replace ``real(8)`` and the ``d``
literals; the default is the historical output.
"""

import re
import warnings

from fortran_tools import DEBUG_FREE_FORM_FLAGS, GFORTRAN, compile_and_run

import pytest

import sympy as sp

from thermodiff import DiffPlz, T, k, l, n, sum_components
from thermodiff.codegen.common import apply_real_kind
from thermodiff.codegen.options import CodegenOptions
from thermodiff.fortran import to_fortran

DERIVS = ["dn", "dndn"]
NC = 3
DOUBLE_DIGITS = 17
REAL128_TOLERANCE = 1e-13
CUSTOM_KIND = {"real_kind": "pr", "kind_module": "kmod"}


def log_model():
    """Return a model with float, rational and integer constants."""
    alpha = sp.IndexedBase("alpha")
    inner = sum_components(n[l] * alpha[k, l], l)
    expr = (
        sp.Symbol("R") * T * sum_components(n[k] * sp.log(n[k] / inner), k)
        + sp.Rational(1, 3) * T**2
        + 1.5e-10 * T**3
        + sp.Float(2.5) * sp.sqrt(T)
    )
    return DiffPlz(expr, indexes=[k, l], name="f")


def generate(**extra):
    """Generate the kernel ``kern`` with the given option overrides."""
    options = CodegenOptions(optional_derivatives=False, **extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return to_fortran(
            log_model(),
            derivatives=DERIVS,
            subroutine_name="kern",
            module_symbols=["R"],
            options=options,
        )


class TestOptions:
    def test_default_is_the_historical_output(self):
        text = generate()
        assert "real(8)" in text and "0.0d0" in text
        assert generate(real_kind="8") == text
        assert generate(real_kind=8) == text
        assert "use " not in text

    @pytest.mark.parametrize("value", ["", "a b", "1.5", "pr;", "-1"])
    def test_real_kind_rejects_non_kinds(self, value):
        with pytest.raises(ValueError, match="real_kind"):
            CodegenOptions(real_kind=value)

    def test_kind_module_rejects_non_identifiers(self):
        with pytest.raises(ValueError, match="kind_module"):
            CodegenOptions(real_kind="pr", kind_module="a-b")

    def test_kind_module_is_moot_for_an_integer_kind(self):
        options = CodegenOptions(real_kind="16", kind_module="kmod")
        with pytest.warns(UserWarning, match="kind_module"):
            normalised, names = options.validate()
        assert names == ("kind_module",) and normalised.kind_module is None

    def test_legacy_round_trip(self):
        options = CodegenOptions(**CUSTOM_KIND)
        assert CodegenOptions.from_legacy(**options.to_legacy()) == options


class TestText:
    @pytest.mark.parametrize(
        "extra", [CUSTOM_KIND, {**CUSTOM_KIND, "work_array": True}]
    )
    def test_no_real8_or_d_literal_remains(self, extra):
        text = generate(**extra)
        assert "real(8)" not in text
        assert not re.search(r"\d[dD][+-]?\d", text)
        assert "real(pr)" in text and "0.0_pr" in text
        assert "1.5e-10_pr" in text and "(1.0_pr/3.0_pr)" in text
        assert text.count("use kmod, only: pr") == 1

    def test_use_line_precedes_implicit_none(self):
        lines = generate(**CUSTOM_KIND).splitlines()
        assert lines[1] == "   use kmod, only: pr"
        assert lines[2] == "   implicit none"

    def test_kind_without_module_adds_no_use(self):
        text = generate(real_kind="16")
        assert "real(16)" in text and "0.0_16" in text
        assert "use " not in text

    def test_apply_real_kind_literals(self):
        text = "x = 0.5d0*y + 2.5d+20 + 1.5d-10 + a1.0d0"
        out = apply_real_kind(text, "pr")
        assert out == "x = 0.5_pr*y + 2.5e+20_pr + 1.5e-10_pr + a1.0d0"
        assert apply_real_kind(text, "8") == text


def _program(kind):
    """Return the driver program that prints the outputs of ``kern``."""
    return f"""program drive
   use kmod, only: pr
   use kern_mod, only: kern
   implicit none
   integer, parameter :: nc = {NC}
   real(pr) :: alpha(nc, nc), n(nc), T, f, dfdn(nc), dfdndn(nc, nc)
   integer :: i, j
   do i = 1, nc
      n(i) = 0.4_pr + 0.1_pr*i
      do j = 1, nc
         alpha(i, j) = 0.3_pr + 0.07_pr*(i + j) + 0.01_pr*i*j
      end do
   end do
   T = 310.0_pr
   call kern(nc, T, alpha, n, f, dfdn, dfdndn)
   write (*, '(es{DOUBLE_DIGITS + 8}.{DOUBLE_DIGITS}e3)') &
      real(f, 8), real(dfdn, 8), real(dfdndn, 8)
end program drive
"""


def _run(tmp_path, kernel, kind_name):
    """Compile ``kernel`` against ``pr = <kind_name>`` and print values."""
    kmod = (
        "module kmod\n"
        "   use iso_fortran_env, only: real64, real128\n"
        f"   integer, parameter :: pr = {kind_name}\n"
        "   real(pr), parameter :: R = 8.314_pr\n"
        "end module kmod\n"
    )
    wrapper = (
        "module kern_mod\n   use kmod, only: R\n   implicit none\n"
        "contains\n" + kernel + "end module kern_mod\n"
    )
    sources = []
    for name, text in (
        ("kmod", kmod),
        ("kern_mod", wrapper),
        ("drive", _program(kind_name)),
    ):
        path = tmp_path / f"{name}.f90"
        path.write_text(text)
        sources.append(path)
    result = compile_and_run(sources, tmp_path, DEBUG_FREE_FORM_FLAGS)
    return [float(token) for token in result.stdout.split()], result.stdout


@pytest.mark.slow
@pytest.mark.skipif(not GFORTRAN, reason="gfortran not available")
class TestNumerics:
    def test_real64_kind_equals_real8_bitwise(self, tmp_path):
        reference, ref_text = _run(tmp_path, generate(), "real64")
        custom, text = _run(tmp_path, generate(**CUSTOM_KIND), "real64")
        assert text == ref_text and custom == reference

    def test_real128_agrees_with_real64(self, tmp_path):
        reference, _ = _run(tmp_path, generate(), "real64")
        wide, _ = _run(tmp_path, generate(**CUSTOM_KIND), "real128")
        assert len(wide) == len(reference) == 1 + NC + NC * NC
        for got, expected in zip(wide, reference):
            assert got == pytest.approx(expected, rel=REAL128_TOLERANCE)

    def test_real128_with_work_array(self, tmp_path):
        extra = {**CUSTOM_KIND, "work_array": True}
        wide, _ = _run(tmp_path, generate(**CUSTOM_KIND), "real128")
        stored, _ = _run(tmp_path, generate(**extra), "real128")
        assert stored == wide


class TestModelFrontEnd:
    MODEL = (
        "model m\ntarget fortran\n\nvariables\n    n  nc\n    T\n\n"
        "parameters\n    a  nc\n\nfunction\n    T*sum_i a_i*n_i**2\n"
    )

    def test_compile_model_threads_the_kind(self):
        from thermodiff.model import compile_model

        code = compile_model(
            self.MODEL, real_kind="pr", kind_module="yaeos__constants"
        )
        assert "use yaeos__constants, only: pr" in code
        assert "real(8)" not in code and "d0" not in code

    def test_cli_flags(self, tmp_path, capsys):
        from thermodiff.model.cli import main

        model = tmp_path / "m.model"
        model.write_text(self.MODEL)
        argv = [str(model), "--real-kind", "pr", "--kind-module", "kmod"]
        assert main([*argv, "--work-array"]) == 0
        out = capsys.readouterr().out
        assert "use kmod, only: pr" in out and "real(8)" not in out
