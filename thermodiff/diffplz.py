"""Differentiate Please module.

This module provides the `DiffPlz` class, which is used to obtain all the
derivatives of a thermodynamic expression with respect to temperature (T),
volume (V), and the number of moles of components (n[i], n[j]).
It handles both direct and cross derivatives, including second derivatives
and cross derivatives between temperature, volume, and number of moles.

Use the class as.

.. code-block:: python

    import thermodiff as td
    
    sol = td.DiffPlz(expression, internal_functions, indexes, name="f")    
"""

from typing import List

import sympy as sp

from thermodiff.thermovars import i, j, k, l, m, n, P, T, V
from thermodiff.core.kronecker_handling import (
    handle_free_kronecker,
    handle_sum_kronecker,
)


class DiffPlz:
    """Class to obtaine all the derivatives of a thermodynamic expression.

    The class will obtain the derivatives respect to:
        - Temperature (T)
        - Volume (V)
        - Number of moles `i` (n[i])
        - Number of moles `j` (n[j])

    For that, your expression must be defined in terms of these variables.
    Import them from `thermodiff.thermovars`:

    .. code-block:: python

        from thermodiff import n, T, V

    On the other hand, if you expression contains index-based variables. You
    can't use the indexes `i` and `j` directly since they are reserved for the
    derivatives. Instead, please use `k`, `l`, `m`. You can import them from
    `thermodiff.thermovars` as well:

    .. code-block:: python

        from thermodiff import k, l, m

    Or, you can define new symbols with different names, but make sure they are
    not `i` or `j`.

    .. code-block:: python

        from sympy import symbols

        o, p, q = symbols("o p q", cls=sp.Idx)

    Parameters
    ----------
    expression : sympy expression
        The thermodynamic expression to differentiate.
    internal_functions : list of sympy functions
        List of sympy functions that are used in the expression.
    indexes : list of sympy Idx
        List of sympy Idx that are used in the expression. These are usually
        the indexes for the number of moles, like [k, l, m].
    name : str, optional
        Name of the DiffClass instance, by default "f". It could be for example
        the name of the thermodynamic function you are working with, like "G^E"
        or "A^r".

    Attributes
    ----------
    name : str
        Name of the Function to be differentiated instance.
    expression : SymPy expression
        The thermodynamic expression to differentiate.
    internal_functions : list of SymPy functions
        List of SymPy functions that are used in the expression.
    indexes : list of SymPy Idx
        List of SymPy Idx that are used in the expression. These are usually
        the indexes for the number of moles, like [k, l, m].
    dt : SymPy expression
        First derivative of the expression with respect to temperature (T).
    dt2 : SymPy expression
        Second derivative of the expression with respect to temperature (T).
    dv : SymPy expression
        First derivative of the expression with respect to volume (V).
    dv2 : SymPy expression
        Second derivative of the expression with respect to volume (V).
    dv3 : SymPy expression
        Third derivative of the expression with respect to volume (V).
    dni : SymPy expression
        First derivative of the expression with respect to number of moles `i`
        (n[i]).
    dnidnj : SymPy expression
        Second derivative of the expression with respect to number of moles `i`
        (n[i]) and `j` (n[j]).
    dtdv : SymPy expression
        Cross second derivative respect to temperature(T) and volume (V).
    dtdp : SymPy expression
        Cross second derivative respect to temperature(T) and pressure (P).
    dtdni : SymPy expression
        Cross second derivative respect to temperature(T) and number of moles
        `i` (n[i]).
    dvdni : SymPy expression
        Cross second derivative volume (V) and number of moles `i` (n[i]).
    dvdp : SymPy expression
        Cross second derivative volume (V) and pressure (P).
    dpdni : SymPy expression
        Cross second derivative pressure (P) and number of moles `i` (n[i]).
    """

    def __init__(
        self,
        expression,
        internal_functions: List[sp.Function] = [],
        indexes: List[sp.Idx] = [k, l, m],
        name: str = "f",
    ):
        self.name = name
        self.expression = expression
        self.internal_functions = internal_functions
        self.indexes = indexes

        # Temperature derivatives
        self.dt = sp.diff(self.expression, T)
        self.dt2 = sp.diff(self.dt, T)

        # Volume derivatives
        self.dv = sp.diff(self.expression, V)
        self.dv2 = sp.diff(self.dv, V)
        self.dv3 = sp.diff(self.dv2, V)
        
        # Pressure derivatives
        self.dp = sp.diff(self.expression, P)
        self.dp2 = sp.diff(self.dp, P)

        # Derivatives with respect to number of moles
        self.dni = self.diff_ni(self.expression, i)
        self.dnidnj = self.diff_dnidnj(self.dni, j)

        # Cross second derivatives
        self.dtdv = sp.diff(self.dt, V)
        self.dtdp = sp.diff(self.dt, P)
        self.dtdni = self.diff_ni(self.dt, i)
        
        self.dvdni = self.diff_ni(self.dv, i)
        self.dvdp = sp.diff(self.dv, P)
        
        self.dpdni = self.diff_ni(self.dp, i)

    def diff_ni(self, expression: sp.Expr, index: sp.Idx) -> sp.Expr:
        """Obtain the compositional derivative dn[index].

        Parameters
        ----------
        expression : sympy expression
            The thermodynamic expression to differentiate.
        index : sp.Idx
            The index of the number of moles to differentiate with respect to.

        Returns
        -------
        sympy expression
            The compositional derivative dn[index].
        """
        # Directly use SymPy's diff function to obtain the derivative
        raw_diff = sp.diff(expression, n[index])

        # Clean the kronecker delta that appears in summation expressions
        diff_not_sumkron = handle_sum_kronecker(raw_diff, index)

        # Handle the free kronecker deltas in the expression. If there is
        # free kronecker delta in the expression you will obtain a piecewise
        # expression.
        diff_final = diff_not_sumkron
        for kdx in self.indexes:
            diff_final = handle_free_kronecker(diff_final, kdx, index)

        return diff_final

    def diff_dnidnj(self, expression: sp.Expr, index: sp.Idx) -> sp.Expr:
        """Obtain the second compositional derivative.

        Parameters
        ----------
        expression : sympy expression
            The thermodynamic expression of first compositional derivative.
        index : sp.Idx
            The index of the number of moles to differentiate with respect to.

        Returns
        -------
        sympy expression
            The second compositional derivative.
        """
        # If the expression is not a Piecewise, we can use the diff_ni method
        if not expression.is_Piecewise:
            return self.diff_ni(expression, index)

        # If the expression is a Piecewise, we need to handle each case
        # separately and combine the results.

        pieces = []  # (list of tuples with (expression, condition))

        for case in expression.args:
            expr = case[0]
            cond = case[1]

            # Differentiate the expression and handle the kronecker deltas
            # as in the diff_ni method.
            raw_diff = sp.diff(expr, n[index])
            diff_not_sumkron = handle_sum_kronecker(raw_diff, index)

            diff_final_c = diff_not_sumkron
            for kdx in self.indexes:
                diff_final_c = handle_free_kronecker(diff_final_c, kdx, index)

            # Here appears new kronecker deltas. delta(j, i)
            diff_final_c = handle_free_kronecker(diff_final_c, index, i)

            # The derivative of an expression of the original piecewise,
            # differentiated respect n[index] again could be a Piecewise
            # expression.
            if diff_final_c.is_Piecewise:
                for dcase in diff_final_c.args:
                    d_expr = dcase[0]
                    d_cond = dcase[1]

                    # We concatenate the conditions of the original piecewise
                    # with the conditions of the derivative piecewise.
                    if d_cond == True and cond != True:
                        pieces.append((d_expr, cond & sp.Ne(i, index)))
                    elif d_cond != True and cond == True:
                        pieces.append((d_expr, d_cond & sp.Ne(i, index)))
                    else:
                        pieces.append((d_expr, cond & d_cond))
            else:
                pieces.append((diff_final_c, cond))

        return sp.Piecewise(*pieces)
