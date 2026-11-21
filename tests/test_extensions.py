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
import warnings

from benchmarks.td_bench import EXTENDED_VARIANTS

from fortran_tools import DEBUG_FREE_FORM_FLAGS, GFORTRAN, compile_and_run

import numpy as np

import pytest

import sympy as sp

from thermodiff.fortran import _canonical_symmetric, _fold_symmetric_sums
from thermodiff.model import compile_model
from thermodiff.model.errors import ModelError
from thermodiff.model.ir import build_ir
from thermodiff.model.parser import read_sections

# Variants are applied with optional_derivatives=True, so validate() is
# expected to report inert options; silence only that message here.
INERT_OPTIONS_MESSAGE = "options ignored by the current combination"

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
            sp.Sum(body, (q, 1, nc), (p, 1, nc)), ["k"], sym_rank3=False
        )
        assert folded != sp.Sum(body, (q, 1, nc), (p, 1, nc))
        # a non-symmetric body is left untouched
        plain = sp.Sum(x[p] * k[p, q], (q, 1, nc), (p, 1, nc))
        assert _fold_symmetric_sums(plain, ["k"], sym_rank3=False) == plain


def test_default_output_unchanged_by_symmetric_attribute():
    """Declaring ``symmetric`` alone does not alter the generated code."""
    plain = PR76_SYM.replace(" symmetric", "")
    assert compile_model(
        PR76_SYM, derivatives="dn,dndn", preset="minimal"
    ) == compile_model(plain, derivatives="dn,dndn", preset="minimal")


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
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=INERT_OPTIONS_MESSAGE, category=UserWarning
        )
        code = compile_model(
            PR76_SYM,
            derivatives=W6,
            preset="minimal",
            optional_derivatives=True,
            **settings,
        )
    work = tmp_path
    (work / "drive.f90").write_text(
        DRIVER.split("program drive")[0]
        + code
        + "\nprogram drive"
        + DRIVER.split("program drive")[1]
    )
    compile_and_run([work / "drive.f90"], work, DEBUG_FREE_FORM_FLAGS)
    return np.loadtxt(work / "out.txt"), code


@pytest.mark.slow
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
    if EXTENDED_VARIANTS[variant].get("param_symmetry") and not (
        EXTENDED_VARIANTS[variant].get("split_sums")
    ):
        # with split_sums every double sum may become a dot product, so a
        # triangular loop is no longer guaranteed
        assert " + 1, nc" in code


@pytest.mark.slow
def test_extended_variants_are_deterministic():
    for settings in (EXTENDED_VARIANTS["a6"], EXTENDED_VARIANTS["a7"]):
        first = compile_model(
            PR76_SYM, derivatives="dn,dndn", preset="minimal", **settings
        )
        second = compile_model(
            PR76_SYM, derivatives="dn,dndn", preset="minimal", **settings
        )
        assert first == second


@pytest.mark.slow
def test_aux_index_never_clashes_case_insensitively():
    """Fortran is case insensitive: no loop index may be named like ``R``."""
    import re

    with open(
        os.path.join(os.path.dirname(__file__), "data/models/rkpr/rkpr.model")
    ) as fh:
        text = fh.read()
    code = compile_model(
        text, derivatives="dn,dndn", preset="minimal", param_symmetry=True
    )
    loop_vars = set(re.findall(r"do (\w+) = ", code))
    assert "r" not in {v.lower() for v in loop_vars}


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

RKPR_PATH = os.path.join(
    os.path.dirname(__file__), "data/models/rkpr/rkpr.model"
)

RKPR_DRIVER = """
module yaeos__constants
   implicit none
   real(8), parameter :: R = 0.0831446261815324d0
end module yaeos__constants

program drive
   use rkpr_mod
   implicit none
   integer :: nc, i, j, u, m, ncs(2)
   real(8), allocatable :: tc(:), ac(:), b(:), k(:), d1(:), &
      kij(:,:), lij(:,:), n(:)
   real(8), allocatable :: dn(:), dnv(:), dnt(:), dnn(:,:)
   real(8) :: v, t, f, fv, ft, fvv, ftt, ftv
   ncs = [3, 7]
   open(newunit=u, file='out.txt')
   do m = 1, 2
      nc = ncs(m)
      allocate(tc(nc), ac(nc), b(nc), k(nc), d1(nc), &
         kij(nc,nc), lij(nc,nc), n(nc))
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
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=INERT_OPTIONS_MESSAGE, category=UserWarning
        )
        code = compile_model(
            _rkpr_text(),
            derivatives=W6,
            preset="minimal",
            optional_derivatives=True,
            **settings,
        )
    head, tail = RKPR_DRIVER.split("program drive", 1)
    (work / "drive.f90").write_text(head + code + "\nprogram drive" + tail)
    compile_and_run([work / "drive.f90"], work, DEBUG_FREE_FORM_FLAGS)
    return np.loadtxt(work / "out.txt"), code


@pytest.mark.slow
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
    plain = compile_model(
        _rkpr_text(), derivatives="none", preset="minimal", **settings
    )
    hoisted = compile_model(
        _rkpr_text(),
        derivatives="none",
        preset="minimal",
        hoist_invariants=True,
        **settings,
    )
    assert any("**k(" in s for s in _nested_loop_bodies(plain))
    assert not any("**k(" in s for s in _nested_loop_bodies(hoisted))
    # no division by a loop-invariant scalar remains inside a double loop
    assert not any("/t" in s for s in _nested_loop_bodies(hoisted))


@pytest.mark.slow
def test_hoisting_is_deterministic_and_off_by_default():
    text = _rkpr_text()
    first = compile_model(
        text,
        derivatives="dn,dndn",
        preset="minimal",
        **EXTENDED_VARIANTS["a6h"],
    )
    second = compile_model(
        text,
        derivatives="dn,dndn",
        preset="minimal",
        **EXTENDED_VARIANTS["a6h"],
    )
    assert first == second
    assert compile_model(
        text, derivatives="dn,dndn", preset="minimal"
    ) == compile_model(
        text, derivatives="dn,dndn", preset="minimal", hoist_invariants=False
    )


@pytest.mark.slow
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
            "    G       = sum_i sum_j n_i*n_j*log(Tc_i + ntot)\n"
            "    g       = R",
        )
    )
    for settings in (dict(), dict(param_symmetry=True, hessian_symmetry=True)):
        code = compile_model(
            text,
            derivatives="dn,dndn",
            preset="minimal",
            optional_derivatives=True,
            hoist_invariants=True,
            **settings,
        )
        loop_vars = {v.lower() for v in re.findall(r"do (\w+) = ", code)}
        args = {a.lower() for a in re.findall(r"intent\(in\) :: (\w+)", code)}
        assert not loop_vars & args, loop_vars & args


# ---------------------------------------------------------------------------
# chain-rule differentiation and loop passes (chain_rule, hoist_loops,
# sum_free_alpha, fuse_temps, nest_sums)
# ---------------------------------------------------------------------------

CHAIN_VARIANTS = [
    "c3",
    "c6",
    "c6l",
    "c6f",
    "c6n",
    "c6s",
    "c6x",
    "c6r",
    "c6m",
    "it12",
    "it13",
    "it20",
    "it22",
]


@pytest.mark.slow
@pytest.mark.skipif(not GFORTRAN, reason="gfortran not available")
@pytest.mark.parametrize("variant", CHAIN_VARIANTS)
def test_chain_variant_matches_reference_rkpr(tmp_path, variant):
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    reference, _ = _run_rkpr(ref_dir, EXTENDED_VARIANTS["a0"])
    var_dir = tmp_path / variant
    var_dir.mkdir()
    values, _ = _run_rkpr(var_dir, EXTENDED_VARIANTS[variant])
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


def test_nested_nodes_extracts_inner_sums():
    """sum_i n_i f_i(sum_k ...) gets the inner index-free sum as its own
    node."""
    from thermodiff.chain import _nest_inner_nodes

    i, k = sp.Idx("i", (1, 3)), sp.Idx("k", (1, 3))
    n, d = sp.IndexedBase("n"), sp.IndexedBase("d")
    inner = sp.Sum(n[k] * d[k] ** 3, (k, 1, 3))
    outer = sp.Sum(n[i] * sp.log(1 - d[i] * inner), (i, 1, 3))
    outer_dummy, inner_dummy = sp.Dummy("S0"), sp.Dummy("S1")
    nodes, inner_map = _nest_inner_nodes(
        [(outer_dummy, outer), (inner_dummy, inner)]
    )
    assert len(nodes) == 2  # the inner sum is the existing node S1
    assert inner_map[outer_dummy] == {inner_dummy}
    assert not dict(nodes)[outer_dummy].function.has(sp.Sum)


def _double_loop_statements(code):
    return _nested_loop_bodies(code)


@pytest.mark.slow
def test_c6n_hessian_has_analytic_structure():
    """With every loop pass the W6 kernel has few O(nc^2) passes and no
    transcendental function inside a double loop except the a_ij matrix."""
    code = compile_model(
        PR76_SYM, derivatives=W6, preset="minimal", **EXTENDED_VARIANTS["c6n"]
    )
    inner = _double_loop_statements(code)
    sqrt_lines = [s for s in inner if "sqrt(" in s]
    assert len(sqrt_lines) == 1  # the a_ij pair matrix
    assert not any("log(" in s or "exp(" in s for s in inner)
    hess = [s for s in inner if s.startswith("res =")]
    assert hess, "Hessian loop not found"


@pytest.mark.slow
def test_loop_passes_are_deterministic_and_off_by_default():
    first = compile_model(
        PR76_SYM,
        derivatives="dn,dndn",
        preset="minimal",
        **EXTENDED_VARIANTS["c6n"],
    )
    second = compile_model(
        PR76_SYM,
        derivatives="dn,dndn",
        preset="minimal",
        **EXTENDED_VARIANTS["c6n"],
    )
    assert first == second
    plain = compile_model(PR76_SYM, derivatives="dn,dndn", preset="minimal")
    assert plain == compile_model(
        PR76_SYM,
        derivatives="dn,dndn",
        preset="minimal",
        chain_rule=False,
        hoist_loops=False,
        sum_free_alpha=False,
        fuse_temps=False,
        nest_sums=False,
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
""",  # noqa: E501 (model text is one expression per line)
    # index-free sums nested inside other sums (PC-SAFT hard chain, UNIQUAC)
    "nested": """
model ns
variables
    n nc
    V
    T
parameters
    m nc
    d nc
    q nc
expressions
    z3 = sum_k n_k*m_k*d_k^3/V
    z2 = sum_k n_k*m_k*d_k^2/V
    g_i = 1/(1 - z3) + d_i*z2/(1 - z3)^2
function
    T*sum_i n_i*(m_i - 1)*log(g_i) + T*sum_i n_i*q_i*log(q_i*(sum_j n_j)/(sum_j n_j*q_j))
""",  # noqa: E501 (model text is one expression per line)
}
_REVIEW_INITS = {
    "tv_factor": {
        "n": "0.3d0+0.11d0*i",
        "Tc": "190d0+37d0*i",
        "b": "0.02d0+0.003d0*i",
        "V": "2d0*sum(x_n)",
    },
    "zero_comp": {
        "n": "0.3d0+0.11d0*i",
        "Tc": "190d0+37d0*i",
        "ac": "merge(0d0, 1d0+0.1d0*i, i == 2)",
    },
    "names": {
        "n": "0.3d0+0.11d0*i",
        "Tc": "190d0+37d0*i",
        "kij": "0.01d0*(i+j)*merge(0d0,1d0,i==j)",
    },
    "nested": {
        "n": "0.3d0+0.11d0*i",
        "m": "1d0+0.2d0*i",
        "d": "0.3d0+0.02d0*i",
        "q": "0.8d0+0.3d0*i",
        "V": "4d0*sum(x_n)",
    },
}


_SLOW_REVIEW_MODELS = ("names", "nested")


@pytest.mark.skipif(not GFORTRAN, reason="gfortran not available")
@pytest.mark.parametrize(
    "model",
    [
        (
            pytest.param(name, marks=pytest.mark.slow)
            if name in _SLOW_REVIEW_MODELS
            else name
        )
        for name in sorted(_REVIEW_MODELS)
    ],
)
def test_chain_review_regressions(model):
    import model_harness as mh

    text = _REVIEW_MODELS[model]
    ref, _ = mh.run(text, EXTENDED_VARIANTS["a0"], _REVIEW_INITS[model])
    for variant in ("c3", "c6n", "c6m", "it12", "it13"):
        values, _ = mh.run(
            text, EXTENDED_VARIANTS[variant], _REVIEW_INITS[model]
        )
        assert np.all(np.isfinite(values)), (model, variant)
        rel = np.abs(values - ref) / np.maximum(np.abs(ref), 1e-8)
        assert rel.max() < 1e-11, (model, variant, rel.max())


# The kernels measured in the campaigns must regenerate byte for byte: any
# change of the generator that alters them has to be behind a new option.
_ROOT = os.path.join(os.path.dirname(__file__), "data")
_MEASURED = [
    ("pr76", "models/pr76/pr76_sym.model", v, w)
    for v in ("a3", "a6", "a10", "a6h", "a10h", "c6n", "c6m")
    for w in ("w3", "w6")
] + [("rkpr", "models/rkpr/rkpr.model", v, "w6") for v in ("a6", "c6n", "c6m")]


@pytest.mark.parametrize(
    "model,path,variant,workload",
    [
        (
            pytest.param(*case, marks=pytest.mark.slow)
            if case[3] == "w6"
            else case
        )
        for case in _MEASURED
    ],
)
def test_measured_kernels_regenerate(model, path, variant, workload):
    from benchmarks.td_bench import generate_workload

    stored = os.path.join(
        _ROOT,
        "bench2",
        "kernels",
        model,
        variant,
        f"{model}_{variant}_{workload}.f90",
    )
    if not os.path.exists(stored):
        pytest.skip("kernel not generated in this checkout")
    meta = generate_workload(
        os.path.join(_ROOT, path),
        workload,
        variant,
        ".",
        optional_derivatives=False,
        write=False,
    )
    with open(stored) as fh:
        assert fh.read() == meta["source"]


def test_imported_symbol_differing_only_in_case_is_renamed():
    """A parameter ``r`` and the imported constant ``R`` would clash in
    Fortran."""
    text = """model cl
target fortran
use yaeos__constants, only: R

variables
    n   nc
    T

parameters
    r   nc

function
    R*T*sum_i n_i*r_i
"""
    code = compile_model(text, derivatives="dn", preset="minimal")
    assert "only: R_use => R" in code
    assert "r_" not in code.replace("R_use", "")


# ---------------------------------------------------------------------------
# it14-it20 (emission passes that only act without optional outputs) and the
# ``positive`` attribute of the language
# ---------------------------------------------------------------------------

_ROOT_MODELS = os.path.join(os.path.dirname(__file__), "data", "models")
_NOOPT_VARIANTS = [
    "it13",
    "it14",
    "it15",
    "it16",
    "it17",
    "it18",
    "it19",
    "it20",
    "it21",
    "it22",
]


@pytest.mark.slow
@pytest.mark.skipif(not GFORTRAN, reason="gfortran not available")
@pytest.mark.parametrize("variant", _NOOPT_VARIANTS)
def test_cmr_rank3_variants_without_optional_outputs(variant):
    import model_harness as mh

    path = os.path.join(_ROOT_MODELS, "cmr", "cmr.model")
    text = open(path).read()
    inits = {
        "n": "0.3d0+0.11d0*i",
        "tc": "190d0+37d0*i",
        "ac": "1d0+0.2d0*i",
        "bi": "0.03d0+0.004d0*i",
        "mk": "0.4d0+0.05d0*i",
        "V": "4d0*sum(x_n)",
    }
    ref, _ = mh.run(text, EXTENDED_VARIANTS["a3"], inits, opt=False)
    values, _ = mh.run(text, EXTENDED_VARIANTS[variant], inits, opt=False)
    rel = np.abs(values - ref) / np.maximum(np.abs(ref), 1e-8)
    assert rel.max() < 1e-11, (variant, rel.max())


@pytest.mark.skipif(not GFORTRAN, reason="gfortran not available")
@pytest.mark.parametrize("variant", ["it17", "it18", "it20"])
def test_cubic_fusion_variants_without_optional_outputs(variant):
    import model_harness as mh

    text = _REVIEW_MODELS["names"]
    ref, _ = mh.run(
        text, EXTENDED_VARIANTS["a3"], _REVIEW_INITS["names"], opt=False
    )
    values, _ = mh.run(
        text, EXTENDED_VARIANTS[variant], _REVIEW_INITS["names"], opt=False
    )
    rel = np.abs(values - ref) / np.maximum(np.abs(ref), 1e-8)
    assert rel.max() < 1e-11, (variant, rel.max())


def test_sym_perm_puts_the_innermost_index_first():
    path = os.path.join(_ROOT_MODELS, "cmr", "cmr.model")
    code = compile_model(
        open(path).read(),
        derivatives="dn",
        preset="minimal",
        optional_derivatives=False,
        **EXTENDED_VARIANTS["it14"],
    )
    reads = [
        line
        for line in code.splitlines()
        if "kijk(" in line and "intent" not in line
    ]
    assert reads and all("kijk(m," in line for line in reads), reads


def test_fuse_guard_keeps_one_exponential_per_pair():
    path = os.path.join(_ROOT_MODELS, "uniquac", "uniquac.model")
    text = open(path).read()
    before = compile_model(
        text,
        derivatives="dn",
        preset="minimal",
        optional_derivatives=False,
        **EXTENDED_VARIANTS["it14"],
    )
    after = compile_model(
        text,
        derivatives="dn",
        preset="minimal",
        optional_derivatives=False,
        **EXTENDED_VARIANTS["it15"],
    )
    assert after.count("exp(") < before.count("exp(")


def test_positive_parameter_splits_roots():
    text = """model pos
variables
    n nc
    T
parameters
    e nc positive
    k nc nc symmetric
function
    T*sum_i sum_j n_i*n_j*sqrt(e_i*e_j)*(1 - k_ij)
"""
    code = compile_model(
        text, derivatives="dn", preset="minimal", **EXTENDED_VARIANTS["a3"]
    )
    assert "sqrt(e(" in code and "sqrt(e(i)*e(j))" not in code.replace(" ", "")


@pytest.mark.skipif(not GFORTRAN, reason="gfortran not available")
def test_positive_parameter_split_matches_closed_form():
    """The split roots reproduce ``f`` and ``df/dn`` of the pair sum."""
    from model_harness import run

    text = """model pos
variables
    n nc
    T
parameters
    e nc positive
    k nc nc symmetric
function
    T*sum_i sum_j n_i*n_j*sqrt(e_i*e_j)*(1 - k_ij)
"""
    values, _ = run(
        text,
        EXTENDED_VARIANTS["a3"],
        {"T": "320d0", "k": "0.01d0*(i+j)+0.003d0*i*j"},
        derivs="dn",
        opt=False,
    )
    temperature = 320.0
    offset = 0
    for count in (3, 7):  # the driver evaluates nc = 3 and nc = 7
        idx = np.arange(1, count + 1, dtype=float)
        moles = 0.5 + 0.13 * idx + 0.02 * idx * idx
        positive = moles  # ``e`` shares the default rank-one initialiser
        pair = np.sqrt(np.outer(positive, positive)) * (
            1.0 - (0.01 * np.add.outer(idx, idx) + 0.003 * np.outer(idx, idx))
        )
        expected_f = temperature * moles @ pair @ moles
        expected_dn = 2.0 * temperature * pair @ moles
        got = values[offset : offset + 1 + count]
        offset += 1 + count
        assert got[0] == pytest.approx(expected_f, rel=1e-12)
        assert got[1:] == pytest.approx(expected_dn, rel=1e-12)


def test_positive_only_for_parameters():
    from thermodiff.model.errors import ModelError

    text = """model pos
variables
    n nc positive
function
    sum_i n_i
"""
    with pytest.raises(ModelError):
        compile_model(text, derivatives="dn")


# ---------------------------------------------------------------------------
# Standard names (NAMING.md): each stage/step key generates exactly the code
# of its legacy key, except for the routine and module names.
# ---------------------------------------------------------------------------


def test_stage_keys_cover_legacy_keys_one_to_one():
    from benchmarks.td_bench import LEGACY_KEYS, STAGE_VARIANTS

    assert len(set(LEGACY_KEYS.values())) == len(LEGACY_KEYS)
    for key, old in LEGACY_KEYS.items():
        assert old in EXTENDED_VARIANTS, old
        assert STAGE_VARIANTS[key]["id"] == f"td-{key}"
        assert len(STAGE_VARIANTS[key]["id"]) <= 16


@pytest.mark.parametrize(
    "key", ["e0", "e2", "e3", "e5", "e7", "e8", "e2h", "p13", "p22"]
)
def test_stage_key_generates_legacy_code(key):
    from benchmarks.td_bench import LEGACY_KEYS, generate_workload

    path = os.path.join(_ROOT_MODELS, "pr76", "pr76_sym.model")
    new = generate_workload(
        path, "w3", key, ".", optional_derivatives=False, write=False
    )["source"]
    old_key = LEGACY_KEYS[key]
    old = generate_workload(
        path, "w3", old_key, ".", optional_derivatives=False, write=False
    )["source"]
    assert new.replace(f"_{key}_", "_X_") == old.replace(f"_{old_key}_", "_X_")


# ---------------------------------------------------------------------------
# Second optimisation round: p23 = p22 + contiguous_args.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_contiguous_args_only_adds_the_attribute():
    from benchmarks.td_bench import generate_workload

    path = os.path.join(_ROOT_MODELS, "pr76", "pr76_sym.model")
    base = generate_workload(
        path, "w6", "p22", ".", optional_derivatives=False, write=False
    )["source"]
    new = generate_workload(
        path, "w6", "p23", ".", optional_derivatives=False, write=False
    )["source"]
    new = new.replace("_p23_", "_p22_")
    assert "contiguous" not in base
    decls = [stmt for stmt in new.splitlines() if "contiguous" in stmt]
    assert decls and all(("(:" in stmt) for stmt in decls)
    assert any("intent(in)" in stmt for stmt in decls) and any(
        "intent(out)" in stmt for stmt in decls
    )
    assert new.replace("real(8), contiguous, ", "real(8), ") == base


def _loops_with_nested_loop(source):
    """Bodies (as text) of the outer loops that contain a nested ``do``."""
    out, depth, cur = [], 0, []
    for line in source.splitlines():
        s = line.strip()
        if s.startswith("do "):
            depth += 1
        if depth:
            cur.append(s)
        if s == "end do":
            depth -= 1
            if depth == 0:
                if sum(stmt.startswith("do ") for stmt in cur) > 1:
                    out.append("\n".join(cur))
                cur = []
    return out


def test_fuse_libm_guard_keeps_libm_maps_out_of_double_reductions():
    from benchmarks.td_bench import generate_workload

    path = os.path.join(_ROOT_MODELS, "rkpr", "rkpr.model")
    old = generate_workload(
        path, "w0", "p23", ".", optional_derivatives=False, write=False
    )["source"]
    new = generate_workload(
        path, "w0", "p24", ".", optional_derivatives=False, write=False
    )["source"]
    assert any(
        "**" in body and "log" not in body
        for body in _loops_with_nested_loop(old)
    )
    for body in _loops_with_nested_loop(new):
        assert "**k(" not in body and "sqrt" not in body and "exp(" not in body


def test_keep_libm_tables_moves_exp_out_of_reductions():
    import re
    from benchmarks.td_bench import generate_workload

    path = os.path.join(_ROOT_MODELS, "uniquac", "uniquac.model")
    acc = re.compile(r"^\s*(\w+)(\(\w+\))? = \1(\(\w+\))? \+ .*exp\(")
    old = generate_workload(
        path, "g0", "p24", ".", optional_derivatives=False, write=False
    )["source"]
    new = generate_workload(
        path, "g0", "p25", ".", optional_derivatives=False, write=False
    )["source"]
    assert any(acc.match(stmt) for stmt in old.splitlines())
    assert not any(acc.match(stmt) for stmt in new.splitlines())
    assert any("= exp(" in stmt for stmt in new.splitlines())


@pytest.mark.slow
def test_table_layout_follows_parameter_subscripts():
    import re
    from benchmarks.td_bench import generate_workload

    path = os.path.join(_ROOT_MODELS, "nrtl", "nrtl.model")
    old = generate_workload(
        path, "g4", "p25", ".", optional_derivatives=False, write=False
    )["source"]
    new = generate_workload(
        path, "g4", "p26", ".", optional_derivatives=False, write=False
    )["source"]
    table = re.compile(r"mat0\(ia, ib\) = .*\bb\((\w+), (\w+)\)")
    assert table.search(old).groups() == ("ib", "ia")
    assert table.search(new).groups() == ("ia", "ib")
    lines = [stmt.strip() for stmt in new.splitlines()]
    k = next(i for i, stmt in enumerate(lines) if table.search(stmt))
    assert lines[k - 2 : k] == ["do ib = 1, nc", "do ia = 1, nc"]
    # the O(N^3) products read the tables with the contiguous inner index
    acc = [
        i
        for i, stmt in enumerate(lines)
        if re.match(r"t\d+\(i, j\) = t\d+\(i, j\) \+ mat0\(i, k\)", stmt)
    ]
    assert acc and all(
        lines[i - 1].startswith("do i = 1, ")
        and lines[i - 2] == "do k = 1, nc"
        for i in acc
    )
    # symmetric pair tables (cubic) are unchanged
    cub = os.path.join(_ROOT_MODELS, "pr76", "pr76_sym.model")
    a = generate_workload(
        cub, "w5c", "p25", ".", optional_derivatives=False, write=False
    )["source"]
    b = generate_workload(
        cub, "w5c", "p26", ".", optional_derivatives=False, write=False
    )["source"]
    assert a.replace("_p25_", "_X_") == b.replace("_p26_", "_X_")


@pytest.mark.skipif(not GFORTRAN, reason="gfortran not available")
def test_table_layout_gives_the_same_derivatives():
    from model_harness import run

    text = open(os.path.join(_ROOT_MODELS, "nrtl", "nrtl.model")).read()
    inits = {
        "a": "0.1d0*(i-j)",
        "b": "10d0*(i-j)+3d0*i*j",
        "c": "0.2d0+0.01d0*(i+j)",
    }
    base = dict(EXTENDED_VARIANTS["it25"])
    ref, ref_code = run(text, base, inits, derivs="dT,dn,dndn", opt=False)
    val, code = run(
        text,
        dict(base, table_layout=True),
        inits,
        derivs="dT,dn,dndn",
        opt=False,
    )
    assert code != ref_code  # the option changed the generated tables
    assert _max_relative_difference(val, ref) < 1e-12


def _hessian_nest(source):
    """Lines of the loop nest that assembles ``dfdndn``."""
    lines = [stmt.strip() for stmt in source.splitlines()]
    end = next(
        k for k, stmt in enumerate(lines) if stmt.startswith("dfdndn(i, j) = ")
    )
    start = max(
        k
        for k in range(end)
        if lines[k].startswith("do ") and lines[k + 1].startswith("do ")
    )
    return lines[start : end + 4]


def _max_relative_difference(values, reference):
    """Largest relative difference, floored at ``1e-8`` in the reference."""
    floor = np.maximum(np.abs(reference), 1e-8)
    return np.max(np.abs(values - reference) / floor)


def test_lowrank_hess_assembles_columns_from_vectors():
    from benchmarks.td_bench import generate_workload

    path = os.path.join(_ROOT_MODELS, "uniquac", "uniquac.model")
    old = generate_workload(
        path, "g3", "p26", ".", optional_derivatives=False, write=False
    )["source"]
    new = generate_workload(
        path, "g3", "p27", ".", optional_derivatives=False, write=False
    )["source"]
    old_nest, new_nest = _hessian_nest(old), _hessian_nest(new)
    assert (
        old_nest[:2] == ["do i = 1, nc", "do j = i, nc"]
        and "dfdndn(j, i) = res" in old_nest
    )
    assert (
        new_nest[:2] == ["do j = 1, nc", "do i = 1, nc"]
        and "dfdndn(j, i) = res" not in new_nest
    )
    body = " ".join(new_nest)
    # the separable terms q(i)*q(j), q(i)*r(j), r(i)*r(j) became two
    # products r_l(i)*g_l(j)
    assert "q(i)*q(j)" not in body and "r(i)*r(j)" not in body
    assert "q(j)*t" in body and "r(j)*t" in body
    # NRTL has only pair terms: unchanged
    nrtl = os.path.join(_ROOT_MODELS, "nrtl", "nrtl.model")
    a = generate_workload(
        nrtl, "g3", "p26", ".", optional_derivatives=False, write=False
    )["source"]
    b = generate_workload(
        nrtl, "g3", "p27", ".", optional_derivatives=False, write=False
    )["source"]
    assert a.replace("_p26_", "_X_") == b.replace("_p27_", "_X_")


def test_lowrank_hess_gives_the_same_hessian():
    from model_harness import run
    from benchmarks.td_bench import EXTENDED_VARIANTS

    text = open(os.path.join(_ROOT_MODELS, "pr76", "pr76_sym.model")).read()
    inits = {
        "kij": "0.01d0*(i+j)*merge(0d0,1d0,i==j)",
        "lij": "0.003d0*(i+j)",
        "Tc": "300d0+20d0*i",
        "Pc": "40d0+2d0*i",
        "w": "0.1d0+0.05d0*i",
        "V": "3d0*sum(x_n)",
    }
    base = dict(EXTENDED_VARIANTS["it26"])
    ref, _ = run(text, base, inits, derivs="dn,dndn", opt=False)
    val, code = run(
        text, dict(base, lowrank_hess=True), inits, derivs="dn,dndn", opt=False
    )
    assert "do j = i, nc" not in code
    assert np.max(np.abs(val - ref) / np.maximum(np.abs(ref), 1e-8)) < 1e-12


def test_drop_pair_tables_inlines_the_dispersion_table():
    import re
    from benchmarks.td_bench import generate_workload

    path = os.path.join(_ROOT_MODELS, "pcsaft", "pcsaft.model")
    old = generate_workload(
        path, "w0", "p27", ".", optional_derivatives=False, write=False
    )["source"]
    new = generate_workload(
        path, "w0", "p28", ".", optional_derivatives=False, write=False
    )["source"]
    assert "mat5(" in old and "mat" not in new.replace("_p28_", "")
    lines = [stmt.strip() for stmt in new.splitlines()]
    k = next(
        i for i, stmt in enumerate(lines) if stmt.startswith("do x = w + 1")
    )
    # inner accumulators, outer factor applied once per outer iteration
    accs = [stmt.split(" = ")[0] for stmt in lines[k - 2 : k]]
    assert lines[k - 2 : k] == [f"{a} = 0.0d0" for a in accs]
    assert [stmt.split(" = ")[0] for stmt in lines[k + 1 : k + 3]] == accs
    assert "n(x)" not in lines[k + 1] and "m(x)" not in lines[k + 2]
    assert lines[k + 3] == "end do"
    # each inner accumulator is folded once into an outer running total
    for line, acc in zip(lines[k + 4 : k + 6], accs):
        assert re.fullmatch(rf"(\w+) = \1 \+ {acc}\*.*", line), line


@pytest.mark.skipif(not GFORTRAN, reason="gfortran not available")
def test_drop_pair_tables_gives_the_same_value():
    from model_harness import run

    text = open(os.path.join(_ROOT_MODELS, "pcsaft", "pcsaft.model")).read()
    inits = {
        "m": "1d0+0.3d0*i",
        "sigma": "3d0+0.1d0*i",
        "epsk": "200d0+10d0*i",
        "kij": "0.01d0*(i+j)*merge(0d0,1d0,i==j)",
        "V": "2d0*sum(x_n)",
    }
    base = dict(EXTENDED_VARIANTS["it27"])
    ref, ref_code = run(text, base, inits, derivs="none", opt=False)
    val, code = run(
        text,
        dict(base, drop_pair_tables=True),
        inits,
        derivs="none",
        opt=False,
    )
    assert "mat5(" in ref_code and "mat5(" not in code
    assert np.all(np.isfinite(val)) and len(val) == 2  # nc = 3 and 7
    assert _max_relative_difference(val, ref) < 1e-12


def test_share_elementwise_reuses_logs_between_value_and_gradient():
    import re
    from benchmarks.td_bench import generate_workload

    path = os.path.join(_ROOT_MODELS, "uniquac", "uniquac.model")
    old = generate_workload(
        path, "g1", "p28", ".", optional_derivatives=False, write=False
    )["source"]
    new = generate_workload(
        path, "g1", "p29", ".", optional_derivatives=False, write=False
    )["source"]
    # three per-component logs, computed in the value reduction and again
    # in the gradient
    assert old.count("log(") == 7 and new.count("log(") == 4
    lines = [stmt.strip() for stmt in new.splitlines()]
    maps = [stmt for stmt in lines if re.match(r"t\d+\(k\) = log\(", stmt)]
    assert len(maps) == 3
    vecs = [m.split("(")[0] for m in maps]
    grad = new[new.index("do i = 1, nc\n      res =") :]
    assert all(f"{v}(i)" in grad for v in vecs) and "log(" not in grad
    # x/y and (1/y)*x prefactors are merged: one scalar per ratio
    assert sum(bool(re.match(r"t\d+ = 1d0/t7$", stmt)) for stmt in lines) == 1
    # NRTL has no such logs: unchanged
    nrtl = os.path.join(_ROOT_MODELS, "nrtl", "nrtl.model")
    a = generate_workload(
        nrtl, "g1", "p28", ".", optional_derivatives=False, write=False
    )["source"]
    b = generate_workload(
        nrtl, "g1", "p29", ".", optional_derivatives=False, write=False
    )["source"]
    assert a.replace("_p28_", "_X_") == b.replace("_p29_", "_X_")


def test_share_elementwise_gives_the_same_gradient():
    from model_harness import run
    from benchmarks.td_bench import EXTENDED_VARIANTS

    text = open(os.path.join(_ROOT_MODELS, "uniquac", "uniquac.model")).read()
    inits = {
        "ua": "0.1d0*(i-j)",
        "ub": "10d0*(i-j)+3d0*i*j",
        "uc": "0.001d0*(i+j)",
        "q": "0.8d0+0.3d0*i",
        "r": "0.9d0+0.2d0*i",
    }
    base = dict(EXTENDED_VARIANTS["it28"])
    ref, ref_code = run(text, base, inits, derivs="dn,dndn", opt=False)
    val, code = run(
        text,
        dict(base, share_elementwise=True),
        inits,
        derivs="dn,dndn",
        opt=False,
    )
    assert code.count("log(") < ref_code.count("log(")
    assert np.max(np.abs(val - ref) / np.maximum(np.abs(ref), 1e-8)) < 1e-12


@pytest.mark.slow
def test_scalar_chain_shrinks_the_scalar_block():
    from benchmarks.td_bench import generate_workload

    path = os.path.join(_ROOT_MODELS, "rkpr", "rkpr.model")
    old = generate_workload(
        path, "w5c", "p29", ".", optional_derivatives=False, write=False
    )
    new = generate_workload(
        path, "w5c", "p30", ".", optional_derivatives=False, write=False
    )
    again = generate_workload(
        path, "w5c", "p30", ".", optional_derivatives=False, write=False
    )
    assert new["source"] == again["source"]
    assert new["operation_count"] < 0.6 * old["operation_count"]
    # same loops as without the option: only the scalar block changes

    def loops(src):
        return [
            stmt.strip()
            for stmt in src.splitlines()
            if stmt.strip().startswith("do ")
        ]

    assert len(loops(new["source"])) == len(loops(old["source"]))
    # the value alone needs no derivative of F: unchanged
    a = generate_workload(
        path, "w0", "p29", ".", optional_derivatives=False, write=False
    )["source"]
    b = generate_workload(
        path, "w0", "p30", ".", optional_derivatives=False, write=False
    )["source"]
    assert a.replace("_p29_", "_X_") == b.replace("_p30_", "_X_")


def test_scalar_chain_falls_back_on_non_smooth_functions():
    import sympy as sp
    from thermodiff.chain import _scalar_chain

    a, b, x = sp.symbols("a b x")
    assert _scalar_chain(sp.Max(a, b) * x, [a, b], [a, b], [(a, a)]) is None
    defs, value, d1, d2 = _scalar_chain(
        sp.log(a / b) * a, [a, b], [a, b], [(a, a), (a, b)]
    )
    env = {}
    for symbol, expr in defs:
        env[symbol] = expr.xreplace(env)
    num = {a: 1.7, b: 0.6}
    f = sp.log(a / b) * a
    for got, ref in [
        (value, f),
        (d1[a], f.diff(a)),
        (d1[b], f.diff(b)),
        (d2[(a, a)], f.diff(a, a)),
        (d2[(a, b)], f.diff(a, b)),
    ]:
        assert (
            abs(
                float(sp.sympify(got).xreplace(env).subs(num))
                - float(ref.subs(num))
            )
            < 1e-12
        )


@pytest.mark.slow
@pytest.mark.skipif(not GFORTRAN, reason="gfortran not available")
def test_scalar_chain_gives_the_same_derivatives():
    from model_harness import run
    from benchmarks.td_bench import EXTENDED_VARIANTS

    text = open(os.path.join(_ROOT_MODELS, "rkpr", "rkpr.model")).read()
    inits = {
        "kij": "0.01d0*(i+j)*merge(0d0,1d0,i==j)",
        "lij": "0.003d0*(i+j)",
        "Tc": "300d0+20d0*i",
        "ac": "2d0+0.5d0*i",
        "b": "0.03d0+0.004d0*i",
        "k": "1.6d0+0.1d0*i",
        "delta1": "1.3d0+0.2d0*i",
        "V": "3d0*sum(x_n)",
    }
    base = dict(EXTENDED_VARIANTS["it29"])
    ref, ref_code = run(text, base, inits, opt=False)
    val, code = run(text, dict(base, scalar_chain=True), inits, opt=False)
    assert code != ref_code
    assert np.max(np.abs(val - ref) / np.maximum(np.abs(ref), 1e-8)) < 1e-11
    # the review models (nested nodes, zero components, T-dependent factors)
    import model_harness as mh

    for model, text in _REVIEW_MODELS.items():
        ref, _ = mh.run(
            text, EXTENDED_VARIANTS["a0"], _REVIEW_INITS[model], opt=False
        )
        val, _ = mh.run(
            text,
            dict(base, scalar_chain=True),
            _REVIEW_INITS[model],
            opt=False,
        )
        rel = np.abs(val - ref) / np.maximum(np.abs(ref), 1e-8)
        assert np.all(np.isfinite(val)) and rel.max() < 1e-11, (
            model,
            rel.max(),
        )


@pytest.mark.slow
def test_split_affine_leaves_one_multiply_add_in_the_cmr_nest():
    import re
    from benchmarks.td_bench import generate_workload

    path = os.path.join(_ROOT_MODELS, "cmr", "cmr.model")
    old = generate_workload(
        path, "w0", "p30", ".", optional_derivatives=False, write=False
    )["source"]
    new = generate_workload(
        path, "w0", "p31", ".", optional_derivatives=False, write=False
    )["source"]

    def inner(src):
        return [
            stmt.strip() for stmt in src.splitlines() if "ijk(m, k, l)" in stmt
        ]

    assert all("(1" in stmt for stmt in inner(old))
    body = inner(new)
    assert len(body) == 2 and not any("(1" in stmt for stmt in body)
    # cmp0(m)*n(m) is read from a vector, and the readers subtract the table
    assert re.search(r"kijk\(m, k, l\)\*t\d+\(m\)", body[1])
    assert re.search(r"\(t\d+ - t\d+\(k, l\)\)", new)
    # the cubic Hessian reads kij next to the matrix-vector products: unchanged
    cub = os.path.join(_ROOT_MODELS, "pr76", "pr76_sym.model")
    a = generate_workload(
        cub, "w5c", "p30", ".", optional_derivatives=False, write=False
    )["source"]
    b = generate_workload(
        cub, "w5c", "p31", ".", optional_derivatives=False, write=False
    )["source"]
    assert a.replace("_p30_", "_X_") == b.replace("_p31_", "_X_")


@pytest.mark.slow
@pytest.mark.skipif(not GFORTRAN, reason="gfortran not available")
def test_split_affine_gives_the_same_derivatives():
    from model_harness import run
    from benchmarks.td_bench import EXTENDED_VARIANTS

    text = open(os.path.join(_ROOT_MODELS, "cmr", "cmr.model")).read()
    inits = {
        "tc": "300d0+20d0*i",
        "ac": "2d0+0.5d0*i",
        "bi": "0.03d0+0.004d0*i",
        "mk": "0.4d0+0.1d0*i",
        "V": "3d0*sum(x_n)",
    }
    base = dict(EXTENDED_VARIANTS["it30"])
    ref, ref_code = run(text, base, inits, opt=False)
    val, code = run(text, dict(base, split_affine=True), inits, opt=False)
    assert code != ref_code
    assert np.max(np.abs(val - ref) / np.maximum(np.abs(ref), 1e-8)) < 1e-11


# ---------------------------------------------------------------------------
# Third round: p32 = p31 with scalar_chain="estimate" (one pipeline pass).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "workload",
    ["w0", "w1", "w5b", pytest.param("w6", marks=pytest.mark.slow)],
)
def test_scalar_chain_estimate_reproduces_cost_choice(workload):
    from benchmarks.td_bench import generate_workload

    path = os.path.join(_ROOT_MODELS, "pr76", "pr76_sym.model")
    cost = generate_workload(
        path, workload, "p31", ".", optional_derivatives=False, write=False
    )
    est = generate_workload(
        path, workload, "p32", ".", optional_derivatives=False, write=False
    )
    assert est["source"].replace("_p32_", "_X_") == cost["source"].replace(
        "_p31_", "_X_"
    )
    assert est["scalar_chain_used"] == cost["scalar_chain_used"]


@pytest.mark.slow
def test_split_acc_uses_two_accumulators_in_the_cmr_nest():
    import re
    from benchmarks.td_bench import generate_workload

    path = os.path.join(_ROOT_MODELS, "cmr", "cmr.model")
    old = generate_workload(
        path, "w0", "p32", ".", optional_derivatives=False, write=False
    )["source"]
    new = generate_workload(
        path, "w0", "p33", ".", optional_derivatives=False, write=False
    )["source"]
    assert "do m = 1, nc - 1, 2" not in old
    lines = [stmt.strip() for stmt in new.splitlines()]
    k = lines.index("do m = 1, nc - 1, 2")
    updates = [
        re.fullmatch(r"(t\d+) = \1 \+ (.*)", line)
        for line in lines[k + 1 : k + 5]
    ]
    assert all(updates)
    accs = [update.group(1) for update in updates]
    assert len(set(accs)) == 4  # lijk and kijk, even and odd steps
    assert [re.sub(r"t\d+\(", "tN(", u.group(2)) for u in updates] == [
        "lijk(m, k, l)*n(m)",
        "lijk(m + 1, k, l)*n(m + 1)",
        "kijk(m, k, l)*tN(m)",
        "kijk(m + 1, k, l)*tN(m + 1)",
    ]
    # the remainder loop runs once, then the odd partial sums are folded
    assert lines[k + 6] == "do m = m, nc"
    assert lines[k + 10 : k + 12] == [
        f"{accs[0]} = {accs[0]} + {accs[1]}",
        f"{accs[2]} = {accs[2]} + {accs[3]}",
    ]
    # no rank-3 parameter elsewhere: the other models are unchanged
    cub = os.path.join(_ROOT_MODELS, "pr76", "pr76_sym.model")
    a = generate_workload(
        cub, "w6", "p32", ".", optional_derivatives=False, write=False
    )["source"]
    b = generate_workload(
        cub, "w6", "p33", ".", optional_derivatives=False, write=False
    )["source"]
    assert a.replace("_p32_", "_X_") == b.replace("_p33_", "_X_")


@pytest.mark.slow
@pytest.mark.skipif(not GFORTRAN, reason="gfortran not available")
def test_split_acc_gives_the_same_derivatives():
    from model_harness import run
    from benchmarks.td_bench import EXTENDED_VARIANTS

    text = open(os.path.join(_ROOT_MODELS, "cmr", "cmr.model")).read()
    inits = {
        "tc": "300d0+20d0*i",
        "ac": "2d0+0.5d0*i",
        "bi": "0.03d0+0.004d0*i",
        "mk": "0.4d0+0.1d0*i",
        "V": "3d0*sum(x_n)",
    }
    base = dict(EXTENDED_VARIANTS["it32"])
    ref, ref_code = run(
        text, base, inits, opt=False
    )  # nc = 3 and 7: remainder taken
    val, code = run(text, dict(base, split_acc=2), inits, opt=False)
    assert "nc - 1, 2" in code and "nc - 1, 2" not in ref_code
    assert np.max(np.abs(val - ref) / np.maximum(np.abs(ref), 1e-8)) < 1e-12
