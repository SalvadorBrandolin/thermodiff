"""Scalar chain when the outer function is a summation node itself."""

import re
import warnings

import sympy as sp

from thermodiff import DiffPlz, k, l, n, sum_components
from thermodiff.codegen.options import CodegenOptions
from thermodiff.fortran import to_fortran


def test_outer_function_that_is_a_node_has_no_stray_arguments():
    """``f = sum_k n_k sum_l n_l a_lk`` keeps its value and its signature."""
    param = sp.IndexedBase("a")
    expr = sum_components(n[k] * sum_components(n[l] * param[l, k], l), k)
    diff = DiffPlz(expr, indexes=[k, l], name="f")
    options = CodegenOptions.from_preset(
        "fastest", chain_rule=True, optional_derivatives=False
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        code = to_fortran(diff, options=options)
    header = re.search(r"subroutine \w+\((.*?)\)\n", code, re.S).group(1)
    assert "_S" not in header and "_plain" not in header
    assert "S0" not in code
    assert re.search(r"^\s*f = res$", code, re.M)
