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

from thermodiff.model import compile_model
from thermodiff.model.errors import ModelError
from thermodiff.model.ir import build_ir
from thermodiff.model.parser import read_sections

GFORTRAN = os.environ.get("FC") or shutil.which("gfortran")

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
