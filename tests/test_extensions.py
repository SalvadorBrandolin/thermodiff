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

from thermodiff.fortran import _indexed_cse, _Namer
from thermodiff.model import compile_model
from thermodiff.model.errors import ModelError
from thermodiff.model.ir import build_ir
from thermodiff.model.parser import read_sections

GFORTRAN = os.environ.get("FC") or shutil.which("gfortran")

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
    kij   nc nc
    lij   nc nc

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
