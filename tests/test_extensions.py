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
