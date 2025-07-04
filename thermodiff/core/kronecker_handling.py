from thermodiff.thermovars import nc

import sympy as sp


def handle_sum_kronecker(expr, idx):
    return expr.doit().subs({idx >= 1: True, nc >= idx: True})


def handle_free_kronecker(expr, kdx, idx):
    """Reemplaza un delta de Kronecker por una expresión Piecewise."""
    delta = sp.KroneckerDelta(idx, kdx)

    if expr.has(delta):
        i, k = delta.args
        return sp.Piecewise(
            (expr.subs(kdx, idx), sp.Eq(i, k)), (expr.subs(delta, 0), True)
        )
    return expr
