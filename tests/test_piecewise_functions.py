"""Derivatives of ``max``, ``min`` and ``abs`` in generated kernels.

SymPy differentiates ``Max``/``Min`` with ``Heaviside`` (and, at second
order, ``DiracDelta``) and ``Abs`` of a symbol that is not declared real
with ``re``/``im`` (issue ipqa-research/thermodiff#19).  The kernels must
compile and match SymPy away from the kink; at the kink itself the value
of ``Heaviside`` is ``1/2``, SymPy's own convention.
"""

from fortran_tools import DEBUG_FREE_FORM_FLAGS, GFORTRAN, compile_and_run

import pytest

import sympy as sp

from thermodiff.codegen.common import fortran_code
from thermodiff.model import compile_model

pytestmark = pytest.mark.skipif(GFORTRAN is None, reason="needs gfortran")

PRESETS = ("minimal", "standard", "fastest")
DERIVATIVES = "dT,dT2,dn,dndT"
MOLES = (0.3, 0.5, 0.9)
CRITICAL = (400.0, 500.0, 650.0)
SHIFT = (310.0, 330.0, 345.0)

MODEL = """\
model kink
variables
    n   nc
    T
parameters
    Tc  nc
    a   nc
function
    {function}
"""

DRIVER = """\
program drive
   use kink_mod
   implicit none
   real(8) :: f, dfdT, dfdT2, dfdn(3), dfdndT(3), temperature
   real(8) :: moles(3) = [{moles}], tc(3) = [{tc}], a(3) = [{a}]
   integer :: u, ipoint
   real(8) :: points({npoints}) = [{points}]
   open(newunit=u, file='out.txt')
   do ipoint = 1, {npoints}
      temperature = points(ipoint)
      call kink_derivs(nc=3, T=temperature, {inputs}n=moles, f=f, &
                       dfdT=dfdT, dfdT2=dfdT2, dfdn=dfdn, dfdndT=dfdndT)
      write(u, '(9es30.17e3)') f, dfdT, dfdT2, dfdn, dfdndT
   end do
   close(u)
end program
"""

TEMPERATURE = sp.Symbol("T", real=True)
MOLE_SYMBOLS = sp.symbols("n0:3", real=True)

CASES = {
    "max_squared": (
        "sum_k n_k * max(1 - sqrt(T/Tc_k), 0)**2",
        lambda n, tc, a: n * sp.Max(1 - sp.sqrt(TEMPERATURE / tc), 0) ** 2,
        (300.0, 420.0, 520.0, 700.0),
    ),
    "min_linear": (
        "sum_k n_k * min(T - a_k, 20)",
        lambda n, tc, a: n * sp.Min(TEMPERATURE - a, 20),
        (300.0, 335.0, 360.0),
    ),
    "abs_squared": (
        "sum_k n_k * abs(T - a_k)**2",
        lambda n, tc, a: n * sp.Abs(TEMPERATURE - a) ** 2,
        (300.0, 320.0, 340.0, 360.0),
    ),
    "abs_linear": (
        "sum_k n_k * abs(T - a_k)",
        lambda n, tc, a: n * sp.Abs(TEMPERATURE - a),
        (300.0, 320.0, 360.0),
    ),
}


def expected_row(build, temperature):
    """Return ``f, dT, dT2, dn, dndT`` of the explicit three-term sum."""
    terms = [
        build(n, tc, a) for n, tc, a in zip(MOLE_SYMBOLS, CRITICAL, SHIFT)
    ]
    total = sum(terms)
    values = {symbol: value for symbol, value in zip(MOLE_SYMBOLS, MOLES)} | {
        TEMPERATURE: temperature
    }
    outputs = [total, total.diff(TEMPERATURE), total.diff(TEMPERATURE, 2)]
    outputs += [total.diff(n) for n in MOLE_SYMBOLS]
    outputs += [total.diff(n, TEMPERATURE) for n in MOLE_SYMBOLS]
    return [float(sp.N(out.subs(values))) for out in outputs]


def used_inputs(code):
    """Return the keyword arguments of the parameters the kernel takes."""
    signature = code.split("subroutine kink_derivs(", 1)[1].split(")", 1)[0]
    names = [name.strip() for name in signature.replace("&", "").split(",")]
    pairs = {"Tc": "tc", "a": "a"}
    return "".join(
        f"{name}={value}, " for name, value in pairs.items() if name in names
    )


def run_kernel(function, points, preset, tmp_path):
    """Compile ``function`` as a model and return its rows at ``points``."""
    code = compile_model(
        MODEL.format(function=function),
        derivatives=DERIVATIVES,
        optional_derivatives=False,
        preset=preset,
    )
    source = tmp_path / "kink.f90"
    source.write_text(code)
    driver = tmp_path / "drive.f90"
    driver.write_text(
        DRIVER.format(
            moles=", ".join(f"{x!r}d0" for x in MOLES),
            tc=", ".join(f"{x!r}d0" for x in CRITICAL),
            a=", ".join(f"{x!r}d0" for x in SHIFT),
            inputs=used_inputs(code),
            npoints=len(points),
            points=", ".join(f"{x!r}d0" for x in points),
        )
    )
    compile_and_run([source, driver], tmp_path, DEBUG_FREE_FORM_FLAGS)
    rows = (tmp_path / "out.txt").read_text().splitlines()
    return [[float(token) for token in row.split()] for row in rows]


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("case", sorted(CASES))
def test_kernel_matches_sympy_away_from_kink(case, preset, tmp_path):
    function, build, points = CASES[case]
    rows = run_kernel(function, points, preset, tmp_path)
    for temperature, row in zip(points, rows):
        assert row == pytest.approx(
            expected_row(build, temperature), rel=1e-10, abs=1e-12
        )


def test_value_at_the_kink_is_one_half(tmp_path):
    """``d max(T - a, 0)/dT`` at ``T = a`` is ``Heaviside(0) = 1/2``."""
    function = "sum_k n_k * max(T - a_k, 0)"
    kink = SHIFT[0]
    rows = run_kernel(function, (kink,), "minimal", tmp_path)
    dfdt = rows[0][1]
    assert dfdt == pytest.approx(0.5 * MOLES[0])


def test_printer_uses_real_literals():
    """Heaviside, DiracDelta, Max and Min print without integer literals."""
    x = sp.Symbol("x")
    assert fortran_code(sp.Heaviside(x)) == (
        "(merge(0.0d0, merge(0.5d0, 1.0d0, x == 0.0d0), x < 0.0d0))"
    )
    assert fortran_code(sp.DiracDelta(x)) == "0.0d0"
    assert fortran_code(sp.Max(x, 0)) == "max(0.0d0, x)"
    assert fortran_code(sp.Min(x, 2)) == "min(2.0d0, x)"
