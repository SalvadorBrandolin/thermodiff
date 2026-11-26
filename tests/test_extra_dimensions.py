"""Intermediates over an extra dimension (``ng``) keep their own extent.

The generated kernels are compiled with ``-fcheck=bounds`` and compared with
SymPy values, for ``nc < ng`` and ``nc > ng``, in every preset and with the
derivatives optional or not.
"""

import itertools
import re
import tempfile
import warnings
from pathlib import Path

from fortran_tools import GFORTRAN, compile_and_run

import numpy as np

import pytest

import sympy as sp

from thermodiff import DiffPlz, l, m, n, sum_components, sum_custom, to_fortran
from thermodiff.codegen.options import resolve_codegen_options
from thermodiff.model import compile_model

pytestmark = pytest.mark.skipif(
    GFORTRAN is None, reason="gfortran is not available"
)

BOUNDS_FLAGS = ("-O0", "-fcheck=bounds", "-ffree-line-length-0")
SIZES = [(2, 3), (4, 2)]
PRESETS = ["minimal", "standard", "fastest"]
RELATIVE_TOLERANCE = 1e-8
TEMPERATURE = 310.0

GROUP_SUM = """model grp
target fortran

variables
    n   nc
    T

parameters
    v   nc ng
    x   ng

expressions
    N_m = sum_s n_s*v_sm

function
    sum_m[1:ng] x_m*log(N_m)
"""

GROUP_TABLE = """model grp2
target fortran

variables
    n   nc
    T

parameters
    v   nc ng
    q   ng
    a   ng ng

expressions
    psi_km = exp(-a_km/T)
    X_m = sum_s n_s*v_sm
    sq = sum_m[1:ng] q_m*X_m
    theta_m = q_m*X_m/sq
    S_m = sum_k[1:ng] theta_k*psi_km

function
    sum_m[1:ng] q_m*X_m*log(S_m)
"""

GROUP_MIXED = """model grp3
target fortran

variables
    n   nc
    T

parameters
    v   nc ng
    q   ng
    r   nc

expressions
    sr = sum_s n_s*r_s
    phi_i = r_i*n_i/sr
    X_m = sum_s n_s*v_sm
    sq = sum_m[1:ng] q_m*X_m
    theta_m = q_m*X_m/sq

function
    T*(sum_i n_i*log(phi_i/n_i) + sum_i n_i*sum_m[1:ng] v_im*log(theta_m))
"""

#: name -> shape (dimension names) of every parameter of the models.
SHAPES = {
    "n": ("nc",),
    "v": ("nc", "ng"),
    "x": ("ng",),
    "q": ("ng",),
    "a": ("ng", "ng"),
    "r": ("nc",),
}


def _parameters(nc: int, ng: int) -> dict[str, np.ndarray]:
    """Return deterministic, well conditioned parameter values."""
    sizes = {"nc": nc, "ng": ng}
    values = {}
    for position, (name, shape) in enumerate(SHAPES.items()):
        grid = np.indices([sizes[dim] for dim in shape]).sum(axis=0)
        values[name] = 0.5 + 0.1 * ((grid * (position + 2)) % 5) + 0.05 * grid
    values["a"] = 200.0 * values["a"] - 100.0
    return values


def _reference_group_sum(nc, ng, values):
    """Return f, dfdn, dfdndn of ``GROUP_SUM`` from explicit SymPy."""
    moles = sp.symbols(f"n0:{nc}")
    amounts = [
        sum(moles[s] * values["v"][s, m] for s in range(nc)) for m in range(ng)
    ]
    return moles, sum(values["x"][m] * sp.log(amounts[m]) for m in range(ng))


def _reference_group_table(nc, ng, values):
    """Return the moles and f of ``GROUP_TABLE`` from explicit SymPy."""
    moles = sp.symbols(f"n0:{nc}")
    temperature = sp.Symbol("T")
    amounts = [
        sum(moles[s] * values["v"][s, m] for s in range(nc)) for m in range(ng)
    ]
    total = sum(values["q"][m] * amounts[m] for m in range(ng))
    theta = [values["q"][m] * amounts[m] / total for m in range(ng)]
    surface = [
        sum(
            theta[k] * sp.exp(-values["a"][k, m] / temperature)
            for k in range(ng)
        )
        for m in range(ng)
    ]
    return moles, sum(
        values["q"][m] * amounts[m] * sp.log(surface[m]) for m in range(ng)
    )


def _reference_group_mixed(nc, ng, values):
    """Return the moles and f of ``GROUP_MIXED`` from explicit SymPy."""
    moles = sp.symbols(f"n0:{nc}")
    temperature = sp.Symbol("T")
    ratio = sum(moles[s] * values["r"][s] for s in range(nc))
    amounts = [
        sum(moles[s] * values["v"][s, m] for s in range(nc)) for m in range(ng)
    ]
    total = sum(values["q"][m] * amounts[m] for m in range(ng))
    theta = [values["q"][m] * amounts[m] / total for m in range(ng)]
    combinatorial = sum(
        moles[i] * sp.log(values["r"][i] / ratio) for i in range(nc)
    )
    residual = sum(
        moles[i] * sum(values["v"][i, m] * sp.log(theta[m]) for m in range(ng))
        for i in range(nc)
    )
    return moles, temperature * (combinatorial + residual)


def _expected(moles, function, values, with_dt):
    """Evaluate f and its derivatives at the test point."""
    temperature = sp.Symbol("T")
    point = {temperature: TEMPERATURE}
    point.update({mole: values["n"][pos] for pos, mole in enumerate(moles)})
    gradient = [sp.diff(function, mole) for mole in moles]
    expected = {
        "f": float(function.subs(point)),
        "dfdn": np.array([float(g.subs(point)) for g in gradient]),
        "dfdndn": np.array(
            [
                [float(sp.diff(g, other).subs(point)) for other in moles]
                for g in gradient
            ]
        ),
    }
    if with_dt:
        expected["dfdT"] = float(sp.diff(function, temperature).subs(point))
    return expected


def _split_arguments(code: str) -> list[str]:
    """Return the dummy arguments of the generated subroutine."""
    header = re.search(r"subroutine \w+\((.*?)\)\n", code.replace("&\n", ""))
    assert header is not None
    return [name.strip() for name in header.group(1).split(",")]


def _driver(code, nc, ng, values, outputs):
    """Return the Fortran program that calls the kernel and dumps outputs."""
    arguments = _split_arguments(code)
    module = re.search(r"^module (\w+)", code, re.M).group(1)
    name = re.search(r"subroutine (\w+)\(", code).group(1)
    lines = [f"x_T = {TEMPERATURE!r}d0"]
    decls = []
    for arg in arguments:
        if arg in values:
            shape = ",".join(SHAPES[arg])
            decls.append(f"real(8) :: x_{arg}({shape})")
            lines.append(f"read(u, *) x_{arg}")
    for out, rank in outputs.items():
        shape = ",".join(["nc"] * rank)
        decls.append(f"real(8) :: y_{out}" + (f"({shape})" if rank else ""))
    call = []
    for arg in arguments:
        if arg in ("nc", "ng"):
            call.append(f"{arg}={arg}")
        elif arg == "T":
            call.append("T=x_T")
        elif arg in outputs:
            call.append(f"{arg}=y_{arg}")
        else:
            call.append(f"{arg}=x_{arg}")
    lines.append(f"call {name}({', '.join(call)})")
    lines += [f"write(w, '(es30.17e3)') y_{out}" for out in outputs]
    return (
        code
        + f"\nprogram drive\n use {module}\n implicit none\n"
        + f" integer, parameter :: nc = {nc}, ng = {ng}\n"
        + " integer :: u, w\n real(8) :: x_T\n"
        + "\n".join(decls)
        + "\n open(newunit=u, file='in.txt')\n"
        + " open(newunit=w, file='out.txt')\n"
        + "\n".join(lines)
        + "\n close(u); close(w)\nend program\n"
    )


def _run_kernel(code, nc, ng, values, outputs):
    """Compile with bounds checking, run, and return the outputs."""
    with tempfile.TemporaryDirectory() as folder:
        workdir = Path(folder)
        names = [a for a in _split_arguments(code) if a in values]
        text = "\n".join(
            " ".join(f"{v:.17e}" for v in values[a].ravel(order="F"))
            for a in names
        )
        (workdir / "in.txt").write_text(text + "\n")
        source = workdir / "drive.f90"
        source.write_text(_driver(code, nc, ng, values, outputs))
        compile_and_run([source], workdir, BOUNDS_FLAGS)
        flat = np.loadtxt(workdir / "out.txt").ravel()
    results, start = {}, 0
    for out, rank in outputs.items():
        size = nc**rank
        chunk = flat[start : start + size]
        results[out] = (
            float(chunk[0])
            if rank == 0
            else chunk.reshape([nc] * rank, order="F")
        )
        start += size
    return results


def _check(code, nc, ng, expected, with_dt):
    """Run ``code`` and compare every output with ``expected``."""
    values = _parameters(nc, ng)
    outputs = {"f": 0, "dfdn": 1, "dfdndn": 2}
    if with_dt:
        outputs["dfdT"] = 0
    found = _run_kernel(code, nc, ng, values, outputs)
    for name in outputs:
        scale = np.max(np.abs(expected[name]))
        np.testing.assert_allclose(
            found[name],
            expected[name],
            rtol=RELATIVE_TOLERANCE,
            atol=RELATIVE_TOLERANCE * scale,
        )


_MODELS = {
    "group_sum": (GROUP_SUM, _reference_group_sum),
    "group_table": (GROUP_TABLE, _reference_group_table),
    "group_mixed": (GROUP_MIXED, _reference_group_mixed),
}

CASES = list(
    itertools.product(
        PRESETS,
        [False, True],
        SIZES,
        ["group_sum", "group_table", "group_mixed"],
    )
)


@pytest.mark.slow
@pytest.mark.parametrize("preset, optional, sizes, model", CASES)
def test_model_with_extra_dimension(preset, optional, sizes, model):
    """Kernels over ``ng`` match SymPy for both orders of size."""
    nc, ng = sizes
    values = _parameters(nc, ng)
    text, reference = _MODELS[model]
    with_dt = model != "group_sum"
    derivatives = "dn,dndn" + (",dT" if with_dt else "")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        code = compile_model(
            text,
            derivatives=derivatives,
            preset=preset,
            optional_derivatives=optional,
        )
    moles, function = reference(nc, ng, values)
    expected = _expected(moles, function, values, with_dt)
    _check(code, nc, ng, expected, with_dt)


STORAGE_OPTIONS = {
    "work_array": {"work_array": True},
    "max_nc": {"max_nc": 8},
}


@pytest.mark.slow
@pytest.mark.parametrize("storage", sorted(STORAGE_OPTIONS))
@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("sizes", SIZES)
@pytest.mark.parametrize("model", ["group_sum", "group_table", "group_mixed"])
def test_storage_options_keep_extra_extents(storage, preset, sizes, model):
    """``work_array`` sizes temporaries by their extents; ``max_nc`` only
    fixes the ``nc`` ones and leaves ``ng`` automatic."""
    nc, ng = sizes
    with_dt = model != "group_sum"
    values = _parameters(nc, ng)
    text, reference = _MODELS[model]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        code = compile_model(
            text,
            derivatives="dn,dndn" + (",dT" if with_dt else ""),
            preset=preset,
            optional_derivatives=False,
            **STORAGE_OPTIONS[storage],
        )
    if storage == "work_array":
        assert "work_(" in code
    moles, function = reference(nc, ng, values)
    expected = _expected(moles, function, values, with_dt)
    _check(code, nc, ng, expected, with_dt)


def _api_group_sum_code(preset, optional):
    """Generate ``GROUP_SUM`` through the Python API, wrapped in a module."""
    groups = sp.Symbol("ng", integer=True, positive=True)
    weights = sp.IndexedBase("v")
    fractions = sp.IndexedBase("x")
    amount = sum_components(n[l] * weights[l, m], l)
    expression = sum_custom(fractions[m] * sp.log(amount), m, 1, groups)
    diff = DiffPlz(expression, indexes=[l, m], name="grp")
    options = resolve_codegen_options(preset, optional_derivatives=optional)
    code = to_fortran(diff, ["dn", "dndn"], name="f", options=options)
    return "module api_mod\n implicit none\ncontains\n" + code + "end module\n"


@pytest.mark.slow
@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("optional", [False, True])
@pytest.mark.parametrize("sizes", SIZES)
def test_python_api_dimension_is_integer(preset, optional, sizes):
    """A dimension symbol of ``sum_custom`` is an integer argument."""
    nc, ng = sizes
    values = _parameters(nc, ng)
    code = _api_group_sum_code(preset, optional)
    assert "integer, intent(in) :: ng" in code
    assert "_S" not in code
    moles, function = _reference_group_sum(nc, ng, values)
    _check(code, nc, ng, _expected(moles, function, values, False), False)


def test_storage_declarations_follow_shapes():
    """``max_nc`` fixes only the ``nc`` extents; the work array sums them."""
    from thermodiff.codegen.storage import (
        array_declarations,
        work_array_kernel,
    )

    shapes = {"t1": ("ng",), "t2": ("nc", "ng"), "t3": ("nc", "nc")}
    fixed = " ".join(array_declarations(shapes, 0, 8))
    assert "t1(ng)" in fixed
    assert "t2(8, ng)" in fixed
    assert "t3(8, 8)" in fixed
    lines = work_array_kernel("k", ["nc", "ng"], [], [], [], shapes, 0)
    assert "work_(ng + nc*nc + nc*ng)" in "\n".join(lines)
