"""Module containing functions to handle Kronecker deltas in expressions.

When differentiating thermodynamic expressions, Kronecker deltas can appear.
This module provides functions to handle these deltas, either by simplifying
them or replacing them with piecewise expressions.
"""

import sympy as sp

from thermodiff.thermovars import nc


def handle_sum_kronecker(expr: sp.Expr, idx: sp.Idx) -> sp.Expr:
    """Simplifies Kronecker deltas in a sum expression.

    The most of the times in thermodynamic expressions, we differentiate
    respect to mole numbers expressions that have summations of mole numbers.
    These expressions provoques Kronecker deltas that appear inside summations
    that can be simplified.

    Parameters
    ----------
    expr : sp.Expr
        Expression to simplify.
    idx : sp.Idx
        Index variable for the Kronecker delta. Normally this is either `i` or
        `j`.

    Returns
    -------
    sp.Expr
        Simplified expression.
    """
    return expr.doit().subs({idx >= 1: True, nc >= idx: True})


def handle_free_kronecker(
    expr: sp.Expr, kdx: sp.Idx, idx: sp.Idx
) -> sp.Piecewise:
    """Replace expression cotanining Kronecker delta(i, k) with Pieacewise.

    Parameters
    ----------
    expr : sp.Expr
        Expression to replace.
    kdx : sp.Idx
        Index variable of the Kronecker delta, usually `k`.
    idx : sp.Idx
        Index variable to replace the Kronecker delta, usually `i` or `j`.

    Returns
    -------
    sp.Piecewise
        Piecewise expression that replaces the Kronecker delta.
    """
    delta = sp.KroneckerDelta(idx, kdx)

    if expr.has(delta):
        i, k = delta.args

        return sp.Piecewise(
            (expr.subs(kdx, idx), sp.Eq(i, k)), (expr.subs(delta, 0), True)
        )

    return expr
