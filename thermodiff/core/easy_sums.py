"""Easy sums for thermodynamic variables.

Normally in thermodynamics we sum over components of a system. this module
provides to easy-to-use funciton to generate SymPy sumatories, cause always
forget how to use the sp.Sum class (also it is too verbose).

Use the function as:

.. code-block:: python

    import thermodiff as td

    td.sum_components(expr, idx)

    # or

    td.sum_custom(expr, idx, start=1, end=None)
"""

import sympy as sp

from thermodiff.thermovars import nc


def sum_components(expr: sp.Expr, idx: sp.Idx) -> sp.Sum:
    """Sum over components of a system.

    The funciton will sum over the components of the system the expression
    given in the `expr` parameter. `idx` is the index of the component, so
    the index of the summation variable.

    the summation will go from 1 to `nc`. `nc` is the number of components
    variable defined in the `thermodiff.thermovars` module.

    Parameters
    ----------
    expr : SymPy expression
        Expression to be summed over the components.
    idx : sp.Idx
        Index of the component to be summed (index of summation variable).

    Returns
    -------
    SymPy Sum
        Basically sp.Sum(expr, (idx, 1, nc))
    """
    return sp.Sum(expr, (idx, 1, nc))


def sum_custom(expr, idx, start=1, end=None):
    """Sum over a custom range.

    Sometimes you need to sum over a custom range, this function provides a
    simple way to do it.

    Parameters
    ----------
    expr : Sympy expression
        Expression to be summed.
    idx : sp.Idx
        Index of the summation variable.
    start : int, optional
        Start if the summation, by default 1
    end : SymPy Integer, optional
        SymPy symbol to end the summation, by default None

    Returns
    -------
    SymPy Sum
        Bassically sp.Sum(expr, (idx, start, end))
    """
    return sp.Sum(expr, (idx, start, end))
