"""Helper names of the generated code never clash with the model's names.

The work array, its internal procedure, the dispatcher's mask variable and
the ``nc_in`` dummy of the fixed-nc copies take another name when a module
symbol or an input has the same name; a ``real_kind`` constant equal to a
model name is rejected.
"""

import re
import warnings

from fortran_tools import GFORTRAN, compile_and_run

import numpy as np

import pytest

import sympy as sp

from thermodiff import DiffPlz, T, k, l, n, sum_components, to_fortran
from thermodiff.codegen.common import unique_name
from thermodiff.codegen.options import CodegenOptions
from thermodiff.codegen.workload import workload_module

FLAGS = ("-O1", "-ffree-line-length-0", "-fcheck=all")
SCALE = 1.5
NC = 2
TOLERANCE = 1e-13

DRIVER = """
program drive
   use sym_mod
   use grp_mod
   implicit none
   integer, parameter :: nc = {nc}
   real(8) :: alpha(nc, nc), n(nc), f(2), dfdn(nc, 2), dfdndn(nc, nc, 2)
   integer :: unit, i, j
{decls}
   do i = 1, nc
      n(i) = 0.5d0 + 0.2d0*i
      do j = 1, nc
         alpha(i, j) = 0.3d0 + 0.1d0*i*j
      end do
   end do
   call kern(nc=nc, T=310d0, alpha=alpha, n=n, {scalars} f=f(1), &
      dfdn=dfdn(:, 1), dfdndn=dfdndn(:, :, 1))
   call kern_fallback(nc=nc, T=310d0, alpha=alpha, n=n, {scalars} f=f(2), &
      dfdn=dfdn(:, 2), dfdndn=dfdndn(:, :, 2))
   open(newunit=unit, file='out.txt')
   do i = 1, 2
      write(unit, '(es30.17e3)') f(i), dfdn(:, i), dfdndn(:, :, i)
   end do
   close(unit)
end program
"""


def _model(symbol_name: str) -> DiffPlz:
    alpha = sp.IndexedBase("alpha")
    inner = sum_components(n[l] * alpha[k, l], l)
    expr = (
        sp.Symbol(symbol_name)
        * T
        * sum_components(n[k] * sp.log(n[k] / inner), k)
    )
    return DiffPlz(expr, indexes=[k, l], name="f")


def _symbol_module(name: str) -> str:
    return (
        "module sym_mod\n   implicit none\n"
        f"   real(8), parameter :: {name} = {SCALE}d0\nend module\n"
    )


def _run_values(tmp_path, code: str, symbol_module: str, scalars: str, decls):
    (tmp_path / "sym.f90").write_text(symbol_module)
    (tmp_path / "grp.f90").write_text(code)
    (tmp_path / "drive.f90").write_text(
        DRIVER.format(nc=NC, scalars=scalars, decls=decls)
    )
    sources = [tmp_path / name for name in ("sym", "grp", "drive")]
    compile_and_run([s.with_suffix(".f90") for s in sources], tmp_path, FLAGS)
    return np.loadtxt(tmp_path / "out.txt").reshape(-1)


def _workload_code(symbol: str, module_symbols, **options) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return workload_module(
            _model(symbol),
            ["dn,dndn"],
            module_name="grp_mod",
            uses=["sym_mod"],
            subroutine_name="kern",
            module_symbols=module_symbols,
            preset="minimal",
            **options,
        )


@pytest.mark.skipif(GFORTRAN is None, reason="gfortran is not available")
@pytest.mark.parametrize("symbol", ["nc_in", "present_mask_"])
def test_module_symbol_is_not_shadowed_by_dispatcher_names(symbol, tmp_path):
    code = _workload_code(symbol, [symbol], fixed_nc=[NC])
    values = _run_values(tmp_path, code, _symbol_module(symbol), "", "")
    half = len(values) // 2
    assert np.allclose(values[:half], values[half:], rtol=TOLERANCE)
    assert values[0] != 0.0


@pytest.mark.skipif(GFORTRAN is None, reason="gfortran is not available")
@pytest.mark.parametrize("symbol", ["nc_in", "present_mask_"])
def test_input_named_like_a_dispatcher_name(symbol, tmp_path):
    code = _workload_code(symbol, None, fixed_nc=[NC])
    decls = f"   real(8) :: {symbol} = {SCALE}d0"
    values = _run_values(
        tmp_path,
        code,
        "module sym_mod\nend module\n",
        f"{symbol}={symbol},",
        decls,
    )
    half = len(values) // 2
    assert np.allclose(values[:half], values[half:], rtol=TOLERANCE)
    assert values[0] != 0.0


@pytest.mark.parametrize("symbol", ["work_", "kern_body", "WORK_"])
def test_work_array_names_avoid_module_symbols(symbol):
    code = to_fortran(
        _model(symbol),
        derivatives=["dn", "dndn"],
        subroutine_name="kern",
        module_symbols=[symbol],
        options=CodegenOptions(optional_derivatives=False, work_array=True),
    )
    declared = re.findall(r"real\(8\) :: (\w+)\(", code)
    internal = re.findall(r"subroutine (\w+)\(", code)
    assert symbol.lower() not in {name.lower() for name in declared}
    assert symbol.lower() not in {name.lower() for name in internal}


@pytest.mark.parametrize("symbol", ["pr", "PR"])
def test_real_kind_name_must_not_be_a_model_symbol(symbol):
    options = CodegenOptions(
        optional_derivatives=False, real_kind="pr", kind_module="kmod"
    )
    with pytest.raises(ValueError, match="real_kind"):
        to_fortran(
            _model(symbol),
            derivatives=["dn"],
            module_symbols=[symbol],
            options=options,
        )
    # Another kind name is fine.
    ok = CodegenOptions(
        optional_derivatives=False, real_kind="dp", kind_module="kmod"
    )
    to_fortran(
        _model(symbol), derivatives=["dn"], module_symbols=[symbol], options=ok
    )


def test_unique_name_ignores_case_and_appends_underscores():
    assert unique_name("work_", ["x"]) == "work_"
    assert unique_name("work_", ["WORK_", "work__"]) == "work___"
