"""Tests for the ``.model`` front-end (parser, IR, Fortran emission)."""

import io
import math
import os
import re
import shutil
import subprocess

import pytest

from thermodiff.model import ModelError, compile_model, compile_model_file
from thermodiff.model.cli import main

PR_MODEL = """
model pr
target fortran
use yaeos__constants, only: R

variables
    n   nc
    V
    T

parameters
    ac    nc
    Tc    nc
    k     nc nc
    b     nc

expressions
    alpha_i = (T/Tc_i)^2
    a_i     = ac_i*alpha_i
    amix    = sum_i sum_j n_i*n_j*sqrt(a_i*a_j)*(1 - k_ij)
    bmix    = sum_i n_i*b_i

function
    R*T/(V - bmix) - amix/(V*(V + bmix) + bmix*(V - bmix))
"""

MOLEFRAC_MODEL = """
model molefrac
variables
    n nc
expressions
    ntot = sum_i n_i
function
    n_k / ntot
"""

CUSTOM_DIM_MODEL = """
model custom
variables
    n   ncomp
parameters
    b   ncomp
expressions
    bmix = sum_i[1:ncomp] n_i*b_i
function
    bmix
"""

PRES_MODEL = """
model pres
variables
    n   nc
    V
    T
    P
parameters
    b   nc
function
    P*V + T + sum_i n_i*b_i
"""


def _find_gfortran():
    """Return the gfortran executable, honoring ``FC``."""
    compiler = os.environ.get("FC")
    if compiler and "gfortran" in os.path.basename(compiler).lower():
        return compiler
    return shutil.which("gfortran")


GFORTRAN = _find_gfortran()

_YAEOS_CONSTANTS_STUB = (
    "module yaeos__constants\n"
    "  implicit none\n"
    "  real(8), parameter :: R = 8.31446261815324d0\n"
    "end module yaeos__constants\n\n"
)


class TestParsing:
    """Structure of the generated module."""

    def code(self):
        return compile_model(PR_MODEL, filename="pr.model")

    def test_module_and_subroutine_names(self):
        code = self.code()
        assert "module pr_mod" in code
        assert "end module pr_mod" in code
        assert "subroutine pr_derivs(" in code

    def test_use_statement_is_preserved(self):
        assert "use yaeos__constants, only: R" in self.code()

    def test_r_is_not_an_argument(self):
        assert "intent(in) :: R" not in self.code()

    def test_dimension_argument(self):
        assert "integer, intent(in) :: nc" in self.code()

    def test_rank_two_parameter_array(self):
        assert "real(8), intent(in) :: k(:, :)" in self.code()

    def test_all_derivatives_are_emitted(self):
        code = self.code()
        for name in [
            "f",
            "dfdT",
            "dfdV",
            "dfdT2",
            "dfdV2",
            "dfdTdV",
            "dfdn",
            "dfdndn",
            "dfdndT",
            "dfdndV",
        ]:
            assert f"intent(out) :: {name}" in code

    def test_intermediates_are_inlined(self):
        code = self.code()
        assert "alpha" not in code
        assert "amix" not in code
        assert "bmix" not in code

    def test_free_index_gives_array_output(self):
        code = compile_model(MOLEFRAC_MODEL, filename="m.model")
        assert "real(8), intent(out) :: f(:)" in code
        assert "real(8), intent(out) :: dfdn(:, :)" in code

    def test_explicit_sum_limits_and_custom_dimension(self):
        code = compile_model(CUSTOM_DIM_MODEL, filename="c.model")
        assert "integer, intent(in) :: ncomp" in code
        assert "do k = 1, ncomp" in code

    def test_derivative_subset(self):
        code = compile_model(PR_MODEL, derivatives=["dt"])
        assert "intent(out) :: dfdT" in code
        assert "dfdV" not in code
        assert "dfdn" not in code

    def test_friendly_derivative_selection(self):
        code = compile_model(PR_MODEL, derivatives="dn,dndn")
        assert "intent(out) :: dfdn" in code
        assert "intent(out) :: dfdndn" in code
        for absent in ["dfdT", "dfdV", "dfdT2", "dfdndT", "dfdndV"]:
            assert absent not in code

    def test_derivative_groups(self):
        first = compile_model(PR_MODEL, derivatives="first")
        assert "intent(out) :: dfdT" in first
        assert "intent(out) :: dfdn" in first
        assert "dfdT2" not in first
        none = compile_model(PR_MODEL, derivatives="none")
        assert "intent(out) :: f" in none
        assert "intent(out) :: dfd" not in none

    def test_inapplicable_derivative_is_an_error(self):
        with pytest.raises(ModelError, match="requires the variable"):
            compile_model(PR_MODEL, derivatives=["dP"])

    def test_unknown_derivative_is_an_error(self):
        with pytest.raises(ModelError, match="Unknown derivatives"):
            compile_model(PR_MODEL, derivatives=["nope"])

    def test_shared_terms_are_computed_once(self):
        code = compile_model(PR_MODEL)
        # ``bmix = sum(n_i b_i)`` feeds every derivative but its summand
        # is materialised a single time (``t1(p) = b(p)*n(p)``).
        assert len(re.findall(r"b\(\w+\)\*n\(\w+\)", code)) == 1

    def test_repeated_indexed_factor_is_materialised(self):
        code = compile_model(PR_MODEL)
        # The ``sqrt`` of the attractive term is shared by every
        # derivative, so it must live in a rank-2 temporary computed once
        # instead of being repeated inside each sum.
        assert len(re.findall(r"sqrt\(T\*\*4\*ac\(\w+\)\*ac\(", code)) == 1
        assert "sqrt(T**4*ac" in code
        assert re.search(r"real\(8\) :: \w+\(nc, nc\)", code)

    def test_array_cse_can_be_disabled(self):
        code = compile_model(PR_MODEL, array_cse=False)
        assert not re.search(r"real\(8\) :: \w+\(nc, nc\)", code)
        assert "sqrt(T**4*ac" in code

    def test_pressure_variable_derivatives(self):
        code = compile_model(PRES_MODEL, filename="pres.model")
        for name in ["dfdP", "dfdP2", "dfdTdP", "dfdVdP", "dfdndP"]:
            assert f"intent(out) :: {name}" in code

    def test_optional_derivatives_are_declared(self):
        code = compile_model(PR_MODEL, optional_derivatives=True)
        for name in ["f", "dfdT", "dfdV", "dfdn", "dfdndn"]:
            assert f"real(8), optional, intent(out) :: {name}" in code
        assert "if (present(f)) then" in code
        assert "if (present(dfdndn)) then" in code

    def test_optional_derivatives_skip_shared_terms(self):
        code = compile_model(PR_MODEL, optional_derivatives=True)
        # A temporary that only the compositional second derivatives need
        # is skipped unless one of them is actually passed.
        assert "if (present(dfdn) .or. present(dfdndn)" in code

    def test_optional_is_off_by_default(self):
        code = self.code()
        assert "optional" not in code
        assert "present(" not in code

    def test_implicit_multiplication_and_powers(self):
        model = (
            "model m\nvariables\n n nc\nparameters\n b nc\n"
            "expressions\n q = sum_i 2 n_i b_i\n"
            "function\n q^2 + q**3 + sqrt(q)\n"
        )
        code = compile_model(model, filename="m.model")
        assert "module m_mod" in code


class TestErrors:
    """Invalid models must raise readable errors."""

    def test_undefined_name(self):
        with pytest.raises(ModelError, match="undefined name 'x_i'"):
            compile_model("model m\nvariables\n n nc\nfunction\n x_i*n_i\n")

    def test_wrong_subscript_count(self):
        model = (
            "model m\nvariables\n n nc\nparameters\n k nc nc\n"
            "function\n k_i\n"
        )
        with pytest.raises(ModelError, match="expects 2 subscript"):
            compile_model(model)

    def test_unbalanced_indices(self):
        model = (
            "model m\nvariables\n n nc\nexpressions\n a_i = n_i + n_j\n"
            "function\n a_i\n"
        )
        with pytest.raises(ModelError, match="index mismatch"):
            compile_model(model)

    def test_duplicate_declaration(self):
        model = "model m\nparameters\n b nc\n b nc\nfunction\n 1\n"
        with pytest.raises(ModelError, match="declared more than once"):
            compile_model(model)

    def test_missing_function_section(self):
        with pytest.raises(ModelError, match="function"):
            compile_model("model m\nvariables\n n nc\n")

    def test_bad_sum_index(self):
        model = "model m\nvariables\n n nc\nfunction\n sum_ij n_i\n"
        with pytest.raises(ModelError, match="single letter"):
            compile_model(model)

    def test_forward_reference_expression(self):
        model = (
            "model m\nvariables\n n nc\nexpressions\n a_i = b_i\n"
            " b_i = n_i\nfunction\n a_i\n"
        )
        with pytest.raises(ModelError, match="undefined name 'b_i'"):
            compile_model(model)

    def test_unsupported_target(self):
        model = "model m\ntarget c\nvariables\n n nc\nfunction\n n_i\n"
        with pytest.raises(ModelError, match="unsupported target"):
            compile_model(model)


def test_cli_writes_output(tmp_path):
    """The CLI compiles a file and writes the module."""
    model_path = tmp_path / "pr.model"
    model_path.write_text(PR_MODEL)
    out_path = tmp_path / "pr.f90"
    assert main([str(model_path), "-o", str(out_path)]) == 0
    assert "module pr_mod" in out_path.read_text()


def test_cli_selects_derivatives(tmp_path, capsys):
    """``--derivatives`` keeps only the requested outputs."""
    model_path = tmp_path / "pr.model"
    model_path.write_text(PR_MODEL)
    assert main([str(model_path), "--derivatives", "dn,dndn"]) == 0
    output = capsys.readouterr().out
    assert "intent(out) :: dfdn" in output
    assert "intent(out) :: dfdndn" in output
    assert "intent(out) :: dfdT" not in output


def test_cli_optional_derivatives(tmp_path, capsys):
    """``--optional-derivatives`` declares every output as optional."""
    model_path = tmp_path / "pr.model"
    model_path.write_text(PR_MODEL)
    assert main([str(model_path), "--optional-derivatives"]) == 0
    output = capsys.readouterr().out
    assert "real(8), optional, intent(out) :: f" in output
    assert "if (present(dfdT)) then" in output


def test_cli_reports_bad_derivative(tmp_path, capsys):
    """An inapplicable derivative is reported as an error."""
    model_path = tmp_path / "pr.model"
    model_path.write_text(PR_MODEL)
    assert main([str(model_path), "--derivatives", "dP"]) == 1
    assert "error:" in capsys.readouterr().err


def test_cli_reports_errors(tmp_path, capsys):
    """The CLI reports a model error and returns a non-zero code."""
    model_path = tmp_path / "bad.model"
    model_path.write_text("model m\nvariables\n n nc\nfunction\n q_i\n")
    assert main([str(model_path)]) == 1
    assert "error:" in capsys.readouterr().err


def test_cli_writes_to_stdout(tmp_path, capsys):
    """Without ``-o`` the module goes to standard output."""
    model_path = tmp_path / "pr.model"
    model_path.write_text(PR_MODEL)
    assert main([str(model_path)]) == 0
    assert "module pr_mod" in capsys.readouterr().out


def test_cli_warns_unused_parameter(tmp_path, capsys):
    """Unused parameters produce a warning on standard error."""
    model_path = tmp_path / "unused.model"
    model_path.write_text("model m\nparameters\n b nc\nfunction\n 1\n")
    assert main([str(model_path)]) == 0
    assert "warning: parameter 'b' is declared but never used" in (
        capsys.readouterr().err
    )


def test_cli_reads_stdin(monkeypatch, capsys):
    """``-`` reads the model from standard input."""
    monkeypatch.setattr("sys.stdin", io.StringIO(PR_MODEL))
    assert main(["-"]) == 0
    assert "module pr_mod" in capsys.readouterr().out


def test_cli_missing_file(tmp_path, capsys):
    """A missing file is reported as an error."""
    assert main([str(tmp_path / "nope.model")]) == 1
    assert "error:" in capsys.readouterr().err


def test_python_dash_m_entry_point(tmp_path):
    """``python -m thermodiff.model`` works."""
    model_path = tmp_path / "pr.model"
    model_path.write_text(PR_MODEL)
    result = subprocess.run(
        [os.sys.executable, "-m", "thermodiff.model", str(model_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "module pr_mod" in result.stdout


def test_compile_model_file(tmp_path):
    """``compile_model_file`` reads and compiles a file."""
    model_path = tmp_path / "pr.model"
    model_path.write_text(PR_MODEL)
    assert "module pr_mod" in compile_model_file(str(model_path))
    optional = compile_model_file(str(model_path), optional_derivatives=True)
    assert "optional, intent(out)" in optional


def test_too_many_indices():
    """A model with more distinct indices than names raises an error."""
    letters = "abcdefghijklmnopqrstuv"
    sums = " ".join(f"sum_{letter}" for letter in letters)
    model = "model m\nvariables\n n nc\nfunction\n" f" {sums} n_a\n"
    with pytest.raises(ModelError, match="too many distinct indices"):
        compile_model(model)


# ---------------------------------------------------------------------------
# Numerical verification
# ---------------------------------------------------------------------------

_GAS_CONSTANT = 8.31446261815324

_PR_DATA = {
    "nc": 3,
    "n": [1.2, 0.7, 2.1],
    "vol": 3.5,
    "temp": 320.0,
    "ac": [0.5, 0.8, 0.3],
    "tc": [300.0, 400.0, 500.0],
    "k": [[0.0, 0.02, 0.01], [0.02, 0.0, 0.03], [0.01, 0.03, 0.0]],
    "b": [0.04, 0.03, 0.05],
}


def _pr_reference(n, vol, temp):
    """Evaluate the PR model in Python."""
    data = _PR_DATA
    nc = data["nc"]
    a = [data["ac"][i] * (temp / data["tc"][i]) ** 2 for i in range(nc)]
    amix = 0.0
    for i in range(nc):
        for j in range(nc):
            amix += (
                n[i] * n[j] * math.sqrt(a[i] * a[j]) * (1 - data["k"][i][j])
            )
    bmix = sum(n[i] * data["b"][i] for i in range(nc))
    denominator = vol * (vol + bmix) + bmix * (vol - bmix)
    return _GAS_CONSTANT * temp / (vol - bmix) - amix / denominator


def _at(**changes):
    """Evaluate the reference model at the base point with changes."""
    data = _PR_DATA
    n = list(changes.get("n", data["n"]))
    vol = changes.get("vol", data["vol"])
    temp = changes.get("temp", data["temp"])
    return _pr_reference(n, vol, temp)


def _shift_n(index, delta, **changes):
    data = _PR_DATA
    n = list(changes.pop("n", data["n"]))
    n[index] += delta
    return _at(n=n, **changes)


def _second_difference(plus, minus, base, step):
    return (plus - 2 * base + minus) / step**2


def _finite_differences():
    """Return the reference derivatives of the PR model."""
    data = _PR_DATA
    nc = data["nc"]
    h1, h2 = 1e-5, 1e-3

    def d_temp():
        return (_at(temp=data["temp"] + h1) - _at(temp=data["temp"] - h1)) / (
            2 * h1
        )

    def d_vol():
        return (_at(vol=data["vol"] + h1) - _at(vol=data["vol"] - h1)) / (
            2 * h1
        )

    def d_ni(index):
        return (_shift_n(index, h1) - _shift_n(index, -h1)) / (2 * h1)

    def d2_temp():
        return _second_difference(
            _at(temp=data["temp"] + h2),
            _at(temp=data["temp"] - h2),
            _at(),
            h2,
        )

    def d2_vol():
        return _second_difference(
            _at(vol=data["vol"] + h2),
            _at(vol=data["vol"] - h2),
            _at(),
            h2,
        )

    def d2_temp_vol():
        return (
            _at(temp=data["temp"] + h2, vol=data["vol"] + h2)
            - _at(temp=data["temp"] - h2, vol=data["vol"] + h2)
            - _at(temp=data["temp"] + h2, vol=data["vol"] - h2)
            + _at(temp=data["temp"] - h2, vol=data["vol"] - h2)
        ) / (4 * h2**2)

    def d2_ni_temp(index):
        return (
            _shift_n(index, h2, temp=data["temp"] + h2)
            - _shift_n(index, h2, temp=data["temp"] - h2)
            - _shift_n(index, -h2, temp=data["temp"] + h2)
            + _shift_n(index, -h2, temp=data["temp"] - h2)
        ) / (4 * h2**2)

    def d2_ni_vol(index):
        return (
            _shift_n(index, h2, vol=data["vol"] + h2)
            - _shift_n(index, h2, vol=data["vol"] - h2)
            - _shift_n(index, -h2, vol=data["vol"] + h2)
            + _shift_n(index, -h2, vol=data["vol"] - h2)
        ) / (4 * h2**2)

    def d2_ni_nj(first, second):
        values = []
        for step_first in (h2, -h2):
            for step_second in (h2, -h2):
                n = list(data["n"])
                n[first] += step_first
                n[second] += step_second
                values.append(_at(n=n))
        return (values[0] - values[1] - values[2] + values[3]) / (4 * h2**2)

    reference = [_at(), d_temp(), d_vol(), d2_temp(), d2_vol(), d2_temp_vol()]
    reference.extend(d_ni(i) for i in range(nc))
    reference.extend(d2_ni_temp(i) for i in range(nc))
    reference.extend(d2_ni_vol(i) for i in range(nc))
    for i in range(nc):
        for j in range(nc):
            reference.append(d2_ni_nj(i, j))
    return reference


def _pr_driver():
    data = _PR_DATA
    literals = {
        name: ", ".join(f"{value:.17g}d0" for value in data[name])
        for name in ("n", "ac", "tc", "b")
    }
    kmat = ", ".join(f"{value:.17g}d0" for row in data["k"] for value in row)
    return (
        "program main\n"
        "  use yaeos__constants\n"
        "  use pr_mod\n"
        "  implicit none\n"
        f"  integer, parameter :: nc = {data['nc']}\n"
        "  real(8) :: n(nc), V, T\n"
        "  real(8) :: ac(nc), Tc(nc), k(nc, nc), b(nc)\n"
        "  real(8) :: fv, dfdT, dfdV, dfdT2, dfdV2, dfdTdV\n"
        "  real(8) :: dfdn(nc), dfdndn(nc, nc)\n"
        "  real(8) :: dfdndT(nc), dfdndV(nc)\n"
        "  integer :: i, j\n"
        f"  n = [{literals['n']}]\n"
        f"  ac = [{literals['ac']}]\n"
        f"  Tc = [{literals['tc']}]\n"
        f"  b = [{literals['b']}]\n"
        f"  k = reshape([{kmat}], [nc, nc])\n"
        f"  V = {data['vol']:.17g}d0\n"
        f"  T = {data['temp']:.17g}d0\n"
        "  call pr_derivs(nc, V, T, Tc, ac, b, k, n, fv, dfdT, dfdV, "
        "dfdn, dfdT2, dfdV2, dfdndn, dfdTdV, dfdndT, dfdndV)\n"
        "  write(*,*) fv, dfdT, dfdV, dfdT2, dfdV2, dfdTdV\n"
        "  write(*,*) (dfdn(i), i=1,nc)\n"
        "  write(*,*) (dfdndT(i), i=1,nc)\n"
        "  write(*,*) (dfdndV(i), i=1,nc)\n"
        "  do i=1,nc\n"
        "    write(*,*) (dfdndn(i,j), j=1,nc)\n"
        "  end do\n"
        "end program main\n"
    )


def _pr_optional_driver():
    """Return a driver that calls the PR module with subsets of outputs."""
    data = _PR_DATA
    literals = {
        name: ", ".join(f"{value:.17g}d0" for value in data[name])
        for name in ("n", "ac", "tc", "b")
    }
    kmat = ", ".join(f"{value:.17g}d0" for row in data["k"] for value in row)
    return (
        "program main\n"
        "  use yaeos__constants\n"
        "  use pr_mod\n"
        "  implicit none\n"
        f"  integer, parameter :: nc = {data['nc']}\n"
        "  real(8) :: n(nc), V, T\n"
        "  real(8) :: ac(nc), Tc(nc), k(nc, nc), b(nc)\n"
        "  real(8) :: fv, dfdT\n"
        "  real(8) :: dfdn(nc), dfdndn(nc, nc)\n"
        "  integer :: i, j\n"
        f"  n = [{literals['n']}]\n"
        f"  ac = [{literals['ac']}]\n"
        f"  Tc = [{literals['tc']}]\n"
        f"  b = [{literals['b']}]\n"
        f"  k = reshape([{kmat}], [nc, nc])\n"
        f"  V = {data['vol']:.17g}d0\n"
        f"  T = {data['temp']:.17g}d0\n"
        "  call pr_derivs(nc, V, T, Tc, ac, b, k, n, f=fv)\n"
        "  write(*,*) fv\n"
        "  call pr_derivs(nc, V, T, Tc, ac, b, k, n, dfdT=dfdT, dfdn=dfdn)\n"
        "  write(*,*) dfdT\n"
        "  write(*,*) (dfdn(i), i=1,nc)\n"
        "  call pr_derivs(nc, V, T, Tc, ac, b, k, n, dfdndn=dfdndn)\n"
        "  do i=1,nc\n"
        "    write(*,*) (dfdndn(i,j), j=1,nc)\n"
        "  end do\n"
        "end program main\n"
    )


@pytest.mark.skipif(not GFORTRAN, reason="gfortran not available")
class TestCompile:
    """Compile and run the generated modules."""

    def _compile_and_run(self, tmp_path, code, driver, stub=""):
        source = tmp_path / "main.f90"
        source.write_text(stub + code + driver)
        exe = tmp_path / ("a.exe" if os.name == "nt" else "a.out")
        result = subprocess.run(
            [GFORTRAN, "-O2", "-o", str(exe), str(source)],
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stderr
        run = subprocess.run(
            [str(exe)], capture_output=True, text=True, cwd=tmp_path
        )
        assert run.returncode == 0, run.stderr
        return [float(value) for value in run.stdout.split()]

    def test_pr_matches_finite_differences(self, tmp_path):
        code = compile_model(PR_MODEL, filename="pr.model")
        values = self._compile_and_run(
            tmp_path, code, _pr_driver(), stub=_YAEOS_CONSTANTS_STUB
        )
        reference = _finite_differences()
        # Central differences are limited by round-off for the tiny second
        # order derivatives, hence the relatively loose absolute tolerance.
        assert values == pytest.approx(reference, rel=1e-5, abs=1e-6)

    def test_optional_derivatives_subset(self, tmp_path):
        full = self._compile_and_run(
            tmp_path,
            compile_model(PR_MODEL, filename="pr.model"),
            _pr_driver(),
            stub=_YAEOS_CONSTANTS_STUB,
        )
        code = compile_model(
            PR_MODEL, filename="pr.model", optional_derivatives=True
        )
        values = self._compile_and_run(
            tmp_path,
            code,
            _pr_optional_driver(),
            stub=_YAEOS_CONSTANTS_STUB,
        )
        # f on its own; dfdT and dfdn without asking for f; dfdndn on its
        # own.  Every value must agree with the full evaluation.
        assert values[0] == pytest.approx(full[0], rel=1e-12)
        assert values[1] == pytest.approx(full[1], rel=1e-12)
        assert values[2:5] == pytest.approx(full[6:9], rel=1e-12)
        assert values[5:14] == pytest.approx(full[15:24], rel=1e-12)

    def test_molefrac_compiles(self, tmp_path):
        code = compile_model(MOLEFRAC_MODEL, filename="m.model")
        driver = (
            "program main\n"
            "  use molefrac_mod\n"
            "  implicit none\n"
            "  integer, parameter :: nc = 2\n"
            "  real(8) :: n(nc)\n"
            "  real(8) :: f(nc), dfdn(nc, nc), dfdndn(nc, nc, nc)\n"
            "  n = [1.0d0, 3.0d0]\n"
            "  call molefrac_derivs(nc, n, f, dfdn, dfdndn)\n"
            "  write(*,*) f\n"
            "end program main\n"
        )
        values = self._compile_and_run(tmp_path, code, driver)
        assert values == pytest.approx([0.25, 0.75])
