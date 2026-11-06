"""Tests for the thesis extensions of thermodiff.

* capture-avoiding inlining of ``expressions`` (alpha renaming),
* the ``symmetric`` parameter attribute,
* the fine-grained ablation flags (``share_sums``, ``scalar_cse``),
* parameter and Hessian symmetry, and the SymPy CSE backend.

The numerical tests generate the full W6 kernel of a PR76 model for every
extended variant, compile it with gfortran and evaluate it with two
different numbers of components using the *same* compiled subroutine.  All
variants must agree with the reference variant ``a0`` (no optimisation).
"""

import os
import shutil
import subprocess

import numpy as np
import pytest
import sympy as sp

from thermodiff.bench_hooks import EXTENDED_VARIANTS
from thermodiff.fortran import (
    _canonical_symmetric,
    _fold_symmetric_sums,
    _indexed_cse,
    _Namer,
)
from thermodiff.model import compile_model
from thermodiff.model.errors import ModelError
from thermodiff.model.ir import build_ir
from thermodiff.model.parser import read_sections

GFORTRAN = os.environ.get("FC") or shutil.which("gfortran")

RKPR_PATH = os.path.join(os.path.dirname(__file__), "data", "rkpr.model")

PR76_SYM = """
model pr76
target fortran
use yaeos__constants, only: R

variables
    n   nc
    V
    T

parameters
    Tc    nc
    Pc    nc
    w     nc
    kij   nc nc symmetric
    lij   nc nc symmetric

expressions
    kappa_i = 0.37464 + 1.54226*w_i - 0.26993*w_i^2
    ac_i    = 0.45723553*R^2*Tc_i^2/Pc_i
    b_i     = 0.07779607*R*Tc_i/Pc_i
    alpha_i = (1 + kappa_i*(1 - sqrt(T/Tc_i)))^2
    a_i     = ac_i*alpha_i
    ntot    = sum_i n_i
    bmix    = sum_i sum_j n_i*n_j*((b_i + b_j)/2.0)*(1.0 - lij_ij)/ntot
    D       = sum_i sum_j n_i*n_j*sqrt(a_i*a_j)*(1.0 - kij_ij)
    d1      = 1.0 + sqrt(2.0)
    d2      = 1.0 - sqrt(2.0)
    f       = log((V + d1*bmix)/(V + d2*bmix))/(bmix*(d1 - d2))
    g       = R*log(1.0 - bmix/V)

function
    -ntot*g*T - D*f
"""


def _nested_binding_violations(expr):
    violations = []
    for outer in expr.atoms(sp.Sum):
        outer_bound = {limit[0] for limit in outer.limits}
        for inner in outer.function.atoms(sp.Sum):
            if outer_bound & {limit[0] for limit in inner.limits}:
                violations.append((outer, inner))
    return violations


def _free(expr):
    bound = {lim[0] for s in expr.atoms(sp.Sum) for lim in s.limits}
    return {i for i in expr.atoms(sp.Idx) if i not in bound}


class TestAlphaRenaming:
    def test_sum_i_no_longer_captures(self):
        ir = build_ir(read_sections(PR76_SYM, "x"), "x")
        assert _nested_binding_violations(ir.function) == []

    def test_free_index_not_shadowed(self):
        text = """
model cap
variables
    n nc
parameters
    b nc
expressions
    ntot = sum_i n_i
    x_i  = n_i*ntot*b_i
function
    sum_i x_i
"""
        ir = build_ir(read_sections(text, "x"), "x")
        assert _nested_binding_violations(ir.function) == []
        assert _free(ir.function) == set()

    def test_definition_used_with_its_bound_letter(self):
        # x_i = sum_j a_ij used as x_j: the bound j must be renamed.
        text = """
model cap2
variables
    n nc
parameters
    a nc nc
expressions
    x_i = sum_j a_ij*n_j
function
    sum_j n_j*x_j
"""
        ir = build_ir(read_sections(text, "x"), "x")
        assert _nested_binding_violations(ir.function) == []
        (total,) = [
            s for s in ir.function.atoms(sp.Sum) if s.function.has(sp.Sum)
        ]
        inner = [s for s in total.function.atoms(sp.Sum)][0]
        # the matrix keeps two distinct subscripts
        (a_occ,) = [
            o for o in inner.function.atoms(sp.Indexed) if str(o.base) == "a"
        ]
        assert a_occ.indices[0] != a_occ.indices[1]


class TestSymmetricAttribute:
    def test_parsed(self):
        ir = build_ir(read_sections(PR76_SYM, "x"), "x")
        assert ir.symmetric == ["kij", "lij"]

    def test_rejects_vector(self):
        text = PR76_SYM.replace("w     nc", "w     nc symmetric")
        with pytest.raises(ModelError):
            build_ir(read_sections(text, "x"), "x")

    def test_rejects_dimension_after_attribute(self):
        text = PR76_SYM.replace(
            "kij   nc nc symmetric", "kij   nc symmetric nc"
        )
        with pytest.raises(ModelError):
            read_sections(text, "x")

    def test_canonical_and_fold(self):
        p, q = sp.Idx("p"), sp.Idx("q")
        k = sp.IndexedBase("k")
        x = sp.IndexedBase("x")
        nc = sp.Symbol("nc")
        body = x[p] * x[q] * (1 - k[q, p])
        canon = _canonical_symmetric(body, ["k"])
        assert canon == x[p] * x[q] * (1 - k[p, q])
        folded = _fold_symmetric_sums(
            sp.Sum(body, (q, 1, nc), (p, 1, nc)), ["k"]
        )
        assert folded != sp.Sum(body, (q, 1, nc), (p, 1, nc))
        # a non-symmetric body is left untouched
        plain = sp.Sum(x[p] * k[p, q], (q, 1, nc), (p, 1, nc))
        assert _fold_symmetric_sums(plain, ["k"]) == plain


def test_default_output_unchanged_by_symmetric_attribute():
    """Declaring ``symmetric`` alone does not alter the generated code."""
    plain = PR76_SYM.replace(" symmetric", "")
    assert compile_model(PR76_SYM, derivatives="dn,dndn") == compile_model(
        plain, derivatives="dn,dndn"
    )


# ---------------------------------------------------------------------------
# numerical equivalence of every extended variant
# ---------------------------------------------------------------------------

W6 = "dT,dV,dn,dT2,dV2,dndn,dTdV,dndT,dndV"

DRIVER = """
module yaeos__constants
   implicit none
   real(8), parameter :: R = 0.0831446261815324d0
end module yaeos__constants

program drive
   use pr76_mod
   implicit none
   integer :: nc, i, j, u
   real(8), allocatable :: tc(:), pc(:), w(:), kij(:,:), lij(:,:), n(:)
   real(8), allocatable :: dn(:), dnv(:), dnt(:), dnn(:,:)
   real(8) :: v, t, f, fv, ft, fvv, ftt, ftv
   integer :: ncs(2), m
   ncs = [3, 7]
   open(newunit=u, file='out.txt')
   do m = 1, 2
      nc = ncs(m)
      allocate(tc(nc), pc(nc), w(nc), kij(nc,nc), lij(nc,nc), n(nc))
      allocate(dn(nc), dnv(nc), dnt(nc), dnn(nc,nc))
      do i = 1, nc
         tc(i) = 190d0 + 37d0*i
         pc(i) = 46d0 - 1.7d0*i
         w(i) = 0.01d0 + 0.04d0*i
         n(i) = 0.3d0 + 0.11d0*i
      end do
      do i = 1, nc
         do j = 1, nc
            kij(i,j) = 0.01d0*abs(i - mod(3*j + i, nc))
         end do
      end do
      kij = 0.5d0*(kij + transpose(kij))
      lij = 0.3d0*kij
      do i = 1, nc
         kij(i,i) = 0d0; lij(i,i) = 0d0
      end do
      v = 0.9d0*sum(n); t = 320d0
      call pr76_derivs(nc=nc, V=v, T=t, Pc=pc, Tc=tc, kij=kij, lij=lij, &
         n=n, w=w, f=f, dfdV=fv, dfdT=ft, dfdTdV=ftv, dfdV2=fvv, &
         dfdT2=ftt, dfdn=dn, dfdndV=dnv, dfdndT=dnt, dfdndn=dnn)
      write(u, '(es25.17)') f, fv, ft, ftv, fvv, ftt, dn, dnv, dnt, dnn
      deallocate(tc, pc, w, kij, lij, n, dn, dnv, dnt, dnn)
   end do
   close(u)
end program drive
"""


def _run_variant(tmp_path, settings):
    code = compile_model(
        PR76_SYM, derivatives=W6, optional_derivatives=True, **settings
    )
    work = tmp_path
    (work / "drive.f90").write_text(
        DRIVER.split("program drive")[0]
        + code
        + "\nprogram drive"
        + DRIVER.split("program drive")[1]
    )
    subprocess.run(
        [GFORTRAN, "-O0", "-ffree-line-length-0", "drive.f90", "-o", "drive"],
        cwd=work,
        check=True,
        capture_output=True,
    )
    subprocess.run(["./drive"], cwd=work, check=True)
    return np.loadtxt(work / "out.txt"), code


@pytest.mark.skipif(not GFORTRAN, reason="gfortran not available")
@pytest.mark.parametrize("variant", sorted(EXTENDED_VARIANTS))
def test_variant_matches_reference(tmp_path, variant):
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    reference, _ = _run_variant(ref_dir, EXTENDED_VARIANTS["a0"])
    var_dir = tmp_path / variant
    var_dir.mkdir()
    values, code = _run_variant(var_dir, EXTENDED_VARIANTS[variant])
    scale = np.maximum(np.abs(reference), 1e-300)
    rel = np.abs(values - reference) / np.maximum(scale, 1e-8)
    assert rel.max() < 1e-11, (variant, rel.max())
    if EXTENDED_VARIANTS[variant].get("hessian_symmetry"):
        assert "do j = i, nc" in code
    if EXTENDED_VARIANTS[variant].get("param_symmetry"):
        assert " + 1, nc" in code


def test_extended_variants_are_deterministic():
    for settings in (EXTENDED_VARIANTS["a6"], EXTENDED_VARIANTS["a7"]):
        first = compile_model(PR76_SYM, derivatives="dn,dndn", **settings)
        second = compile_model(PR76_SYM, derivatives="dn,dndn", **settings)
        assert first == second


def test_aux_index_never_clashes_case_insensitively():
    """Fortran is case insensitive: no loop index may be named like ``R``."""
    import re

    with open(RKPR_PATH) as fh:
        text = fh.read()
    code = compile_model(text, derivatives="dn,dndn", param_symmetry=True)
    loop_vars = set(re.findall(r"do (\w+) = ", code))
    assert "r" not in {v.lower() for v in loop_vars}


def test_array_cse_slots_avoid_symbols_case_insensitively():
    """Array CSE slot indices never coincide with a symbol, ignoring case."""
    i = sp.Idx("i")
    x = sp.IndexedBase("x")
    P, Q = sp.Symbol("P"), sp.Symbol("Q")
    sub = sp.log(x[i] + x[i]**2)
    exprs = [P * sub + Q, Q * sub - P]
    _, defs = _indexed_cse(exprs, _Namer(), {})
    assert defs
    names = {str(idx).lower() for d in defs for idx in d["free"]}
    assert not names & {"p", "q"}, names


def test_loop_names_avoid_module_constant_case_insensitively():
    """The friendly loop names of ``emit`` skip ``r`` when ``R`` is used."""
    import re

    text = """
model six
target fortran
use yaeos__constants, only: R
variables
    n nc
parameters
    a nc nc
expressions
    s1 = sum_i sum_j n_i*n_j*a_ij
    s2 = sum_k sum_l n_k*n_l*a_kl^2
    s3 = sum_m sum_o n_m*n_o*a_mo^3
function
    R*s1*s2*s3
"""
    code = compile_model(text, derivatives="dn")
    loop_vars = {v.lower() for v in re.findall(r"do (\w+) = ", code)}
    assert "r" not in loop_vars


def test_multiline_entry_closes_on_accumulated_balance():
    """A parenthesis closed on a continuation line ends the entry."""
    text = """model x
variables
    T
expressions
    x = (T +
         2*T)
    y = 3*T
function
    x*y
"""
    raw = read_sections(text, "x")
    assert [(e.lhs, e.rhs) for e in raw.expressions] == [
        ("x", "(T + 2*T)"),
        ("y", "3*T"),
    ]


# ---------------------------------------------------------------------------
# loop-invariant hoisting (hoist_invariants)
# ---------------------------------------------------------------------------


RKPR_DRIVER = """
module yaeos__constants
   implicit none
   real(8), parameter :: R = 0.0831446261815324d0
end module yaeos__constants

program drive
   use rkpr_mod
   implicit none
   integer :: nc, i, j, u, m, ncs(2)
   real(8), allocatable :: tc(:), ac(:), b(:), k(:), d1(:), kij(:,:), lij(:,:), n(:)
   real(8), allocatable :: dn(:), dnv(:), dnt(:), dnn(:,:)
   real(8) :: v, t, f, fv, ft, fvv, ftt, ftv
   ncs = [3, 7]
   open(newunit=u, file='out.txt')
   do m = 1, 2
      nc = ncs(m)
      allocate(tc(nc), ac(nc), b(nc), k(nc), d1(nc), kij(nc,nc), lij(nc,nc), n(nc))
      allocate(dn(nc), dnv(nc), dnt(nc), dnn(nc,nc))
      do i = 1, nc
         tc(i) = 190d0 + 37d0*i
         ac(i) = 2.5d0 + 0.8d0*i
         b(i) = 0.03d0 + 0.006d0*i
         k(i) = 1.9d0 + 0.13d0*i
         d1(i) = 1.6d0 + 0.07d0*i
         n(i) = 0.3d0 + 0.11d0*i
      end do
      do i = 1, nc
         do j = 1, nc
            kij(i,j) = 0.01d0*abs(i - mod(3*j + i, nc))
         end do
      end do
      kij = 0.5d0*(kij + transpose(kij))
      lij = 0.3d0*kij
      v = 0.9d0*sum(n); t = 320d0
      call rkpr_derivs(nc=nc, V=v, T=t, Tc=tc, ac=ac, b=b, k=k, delta1=d1, &
         kij=kij, lij=lij, n=n, f=f, dfdV=fv, dfdT=ft, dfdTdV=ftv, dfdV2=fvv, &
         dfdT2=ftt, dfdn=dn, dfdndV=dnv, dfdndT=dnt, dfdndn=dnn)
      write(u, '(es25.17)') f, fv, ft, ftv, fvv, ftt, dn, dnv, dnt, dnn
      deallocate(tc, ac, b, k, d1, kij, lij, n, dn, dnv, dnt, dnn)
   end do
   close(u)
end program drive
"""


def _rkpr_text():
    with open(RKPR_PATH) as fh:
        return fh.read()


def _run_rkpr(work, settings):
    code = compile_model(
        _rkpr_text(), derivatives=W6, optional_derivatives=True, **settings
    )
    head, tail = RKPR_DRIVER.split("program drive", 1)
    (work / "drive.f90").write_text(head + code + "\nprogram drive" + tail)
    subprocess.run(
        [GFORTRAN, "-O0", "-ffree-line-length-0", "drive.f90", "-o", "drive"],
        cwd=work, check=True, capture_output=True,
    )
    subprocess.run(["./drive"], cwd=work, check=True)
    return np.loadtxt(work / "out.txt"), code


@pytest.mark.skipif(not GFORTRAN, reason="gfortran not available")
@pytest.mark.parametrize("variant", ["a3h", "a6h", "a10h"])
def test_rkpr_hoisted_variant_matches_reference(tmp_path, variant):
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    reference, _ = _run_rkpr(ref_dir, EXTENDED_VARIANTS["a0"])
    var_dir = tmp_path / variant
    var_dir.mkdir()
    values, _ = _run_rkpr(var_dir, EXTENDED_VARIANTS[variant])
    rel = np.abs(values - reference) / np.maximum(np.abs(reference), 1e-8)
    assert rel.max() < 1e-11, (variant, rel.max())


def _nested_loop_bodies(code):
    """Statements located inside at least two nested ``do`` loops."""
    depth = 0
    inner = []
    for line in code.splitlines():
        stripped = line.strip().lower()
        if stripped.startswith("do "):
            depth += 1
        elif stripped == "end do":
            depth -= 1
        elif depth >= 2:
            inner.append(stripped)
    return inner


def test_hoisting_moves_rkpr_power_out_of_double_loop():
    """RKPR W0 with folded symmetry: ``alpha**k`` leaves the double loop."""
    settings = dict(EXTENDED_VARIANTS["a6"])
    plain = compile_model(_rkpr_text(), derivatives="none", **settings)
    hoisted = compile_model(
        _rkpr_text(), derivatives="none", hoist_invariants=True, **settings
    )
    assert any("**k(" in s for s in _nested_loop_bodies(plain))
    assert not any("**k(" in s for s in _nested_loop_bodies(hoisted))
    # no division by a loop-invariant scalar remains inside a double loop
    assert not any("/t" in s for s in _nested_loop_bodies(hoisted))


def test_hoisting_is_deterministic_and_off_by_default():
    text = _rkpr_text()
    first = compile_model(text, derivatives="dn,dndn", **EXTENDED_VARIANTS["a6h"])
    second = compile_model(text, derivatives="dn,dndn", **EXTENDED_VARIANTS["a6h"])
    assert first == second
    assert compile_model(text, derivatives="dn,dndn") == compile_model(
        text, derivatives="dn,dndn", hoist_invariants=False
    )


def test_hoisted_slot_index_never_clashes_with_an_argument():
    """A scalar argument ``P`` must not be reused as a hoisting loop index.

    Regression found in review: the second hoisting call (on summation
    bodies) did not see the scalar arguments that appear only outside sums.
    """
    import re

    text = (
        _rkpr_text()
        .replace("    k       nc\n", "    k       nc\n    P\n")
        .replace("    -ntot*g*T - D*f", "    -ntot*g*T - D*f + P*G")
        .replace(
            "    g       = R",
            "    G       = sum_i sum_j n_i*n_j*log(Tc_i + ntot)\n    g       = R",
        )
    )
    for settings in (dict(), dict(param_symmetry=True, hessian_symmetry=True)):
        code = compile_model(
            text, derivatives="dn,dndn", optional_derivatives=True,
            hoist_invariants=True, **settings,
        )
        loop_vars = {v.lower() for v in re.findall(r"do (\w+) = ", code)}
        args = {a.lower() for a in re.findall(r"intent\(in\) :: (\w+)", code)}
        assert not loop_vars & args, loop_vars & args


# ---------------------------------------------------------------------------
# chain-rule differentiation and loop passes (chain_rule, hoist_loops,
# sum_free_alpha, fuse_temps, nest_sums)
# ---------------------------------------------------------------------------

from model_harness import CHAIN_SETTINGS  # noqa: E402

CHAIN_VARIANTS = ["c3", "c6", "c6l", "c6f", "c6n", "c6s"]


@pytest.mark.skipif(not GFORTRAN, reason="gfortran not available")
@pytest.mark.parametrize("variant", CHAIN_VARIANTS)
def test_chain_variant_matches_reference_rkpr(tmp_path, variant):
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    reference, _ = _run_rkpr(ref_dir, EXTENDED_VARIANTS["a0"])
    var_dir = tmp_path / variant
    var_dir.mkdir()
    values, _ = _run_rkpr(var_dir, CHAIN_SETTINGS[variant])
    rel = np.abs(values - reference) / np.maximum(np.abs(reference), 1e-8)
    assert rel.max() < 1e-11, (variant, rel.max())


def test_chain_rule_nodes_of_pr76():
    """PR76 has three summation nodes: sum n, the b rule and the a rule."""
    from thermodiff.chain import factor_sums, summation_nodes
    from thermodiff.model.emit import _rename_indices

    ir = build_ir(read_sections(PR76_SYM, "pr76"), "pr76")
    function, _ = _rename_indices(ir, ["dni"])
    outer, nodes = summation_nodes(factor_sums(function))
    assert len(nodes) == 3
    assert not outer.has(sp.Sum)


def _double_loop_statements(code):
    return _nested_loop_bodies(code)


def test_c6n_hessian_has_analytic_structure():
    """With every loop pass the W6 kernel has few O(nc^2) passes and no
    transcendental function inside a double loop except the a_ij matrix."""
    code = compile_model(PR76_SYM, derivatives=W6, **CHAIN_SETTINGS["c6n"])
    inner = _double_loop_statements(code)
    sqrt_lines = [s for s in inner if "sqrt(" in s]
    assert len(sqrt_lines) == 1  # the a_ij pair matrix
    assert not any("log(" in s or "exp(" in s for s in inner)
    hess = [s for s in inner if s.startswith("res =")]
    assert hess, "Hessian loop not found"


def test_loop_passes_are_deterministic_and_off_by_default():
    first = compile_model(PR76_SYM, derivatives="dn,dndn", **CHAIN_SETTINGS["c6n"])
    second = compile_model(PR76_SYM, derivatives="dn,dndn", **CHAIN_SETTINGS["c6n"])
    assert first == second
    plain = compile_model(PR76_SYM, derivatives="dn,dndn")
    assert plain == compile_model(
        PR76_SYM, derivatives="dn,dndn", chain_rule=False, hoist_loops=False,
        sum_free_alpha=False, fuse_temps=False, nest_sums=False,
    )


# regression tests for the issues found by the independent review of the
# chain-rule path (V-dependent per-component factors, zero component
# functions, T only inside abstracted arrays, name clashes)

_REVIEW_MODELS = {
    "tv_factor": """
model tv
variables
    n nc
    V
    T
parameters
    Tc nc
    b nc
expressions
    alpha_i = exp(1 - T/Tc_i)
function
    sum_i n_i*alpha_i*log(1 - b_i/V) + (sum_i n_i)*log(V)*T
""",
    "zero_comp": """
model zc
variables
    n nc
    V
    T
parameters
    Tc nc
    ac nc
    kij nc nc symmetric
expressions
    a_i = ac_i*(1 + 0.5*(1 - sqrt(T/Tc_i)))^2
function
    sum_i sum_j n_i*n_j*a_i*a_j*(1 - kij_ij)/V + T*(sum_m n_m)*0.001
""",
    "names": """
model nm
variables
    n nc
    V
    T
parameters
    Tc nc
    S0
    ia nc
    cmp0 nc
    mat1 nc nc
    kij nc nc symmetric
expressions
    a_i = (1 + 0.5*(1 - sqrt(T/Tc_i)))^2
    ntot = sum_m n_m
    D = sum_i sum_j n_i*n_j*sqrt(a_i*a_j)*(1 - kij_ij)
function
    -ntot*log(1 - 0.1*ntot/V)*T - D/(V + S0) + (sum_m n_m*ia_m + sum_m n_m*cmp0_m + sum_m sum_l n_m*n_l*mat1_ml/ntot)*T
""",
}
_REVIEW_INITS = {
    "tv_factor": {"n": "0.3d0+0.11d0*i", "Tc": "190d0+37d0*i", "b": "0.02d0+0.003d0*i", "V": "2d0*sum(x_n)"},
    "zero_comp": {"n": "0.3d0+0.11d0*i", "Tc": "190d0+37d0*i", "ac": "merge(0d0, 1d0+0.1d0*i, i == 2)"},
    "names": {"n": "0.3d0+0.11d0*i", "Tc": "190d0+37d0*i", "kij": "0.01d0*(i+j)*merge(0d0,1d0,i==j)"},
}


@pytest.mark.skipif(not GFORTRAN, reason="gfortran not available")
@pytest.mark.parametrize("model", sorted(_REVIEW_MODELS))
def test_chain_review_regressions(model):
    import model_harness as mh

    text = _REVIEW_MODELS[model]
    ref, _ = mh.run(text, EXTENDED_VARIANTS["a0"], _REVIEW_INITS[model])
    for variant in ("c3", "c6n"):
        values, _ = mh.run(text, CHAIN_SETTINGS[variant], _REVIEW_INITS[model])
        assert np.all(np.isfinite(values)), (model, variant)
        rel = np.abs(values - ref) / np.maximum(np.abs(ref), 1e-8)
        assert rel.max() < 1e-11, (model, variant, rel.max())
