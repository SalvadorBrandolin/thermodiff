from typing import List

import sympy as sp

from thermodiff.thermovars import i, j, k, l, m, n, T, V
from thermodiff.core.kronecker_handling import (
    handle_free_kronecker,
    handle_sum_kronecker,
)


class DiffClass:
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
    can't use the indexes `i` and `j` directly since they are reserved for
    the derivatives. Instead, please use `k`, `l`, `m`. You can import them
    from `thermodiff.thermovars` as well:

    .. code-block:: python

        from thermodiff import k, l, m

    Or, you can define new symbols with different names, but make sure
    they are not `i` or `j`.

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
        
        # Derivatives with respect to number of moles
        self.dni = self.diff_ni(self.expression, i)
        self.dnidnj = self.diff_dnidnj(self.dni, j)
        
        # Cross second derivatives
        self.dtdv = sp.diff(self.dt, V)
        self.dtdni = self.diff_ni(self.dt, i)
        self.dvdni = self.diff_ni(self.dv, i)

    def diff_ni(self, expression: sp.Expr, index: sp.Idx):
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

        if not expression.is_Piecewise:
            raw_diff = sp.diff(expression, n[index])
            diff_not_sumkron = handle_sum_kronecker(raw_diff, index)
            
            diff_final = diff_not_sumkron
            
            for kdx in self.indexes:
                diff_final = handle_free_kronecker(diff_final, kdx, index)
            
            return diff_final

    def diff_dnidnj(self, expression: sp.Expr, index: sp.Idx):
        if not expression.is_Piecewise:
            return self.diff_ni(expression, index)

        pieces = []
        for case in expression.args:
            expr = case[0]
            cond = case[1]
            
            raw_diff = sp.diff(expr, n[index])
            diff_not_sumkron = handle_sum_kronecker(raw_diff, index)
            
            diff_final_c = diff_not_sumkron
            for kdx in self.indexes:
                diff_final_c = handle_free_kronecker(diff_final_c, kdx, index)
                
            diff_final_c = handle_free_kronecker(diff_final_c, index, i)

            if diff_final_c.is_Piecewise:
                for dcase in diff_final_c.args:
                    d_expr = dcase[0]
                    d_cond = dcase[1]
                    
                    if d_cond == True and cond != True:
                        pieces.append((d_expr, cond & sp.Ne(i, index)))
                    elif d_cond != True and cond == True:
                        pieces.append((d_expr, d_cond & sp.Ne(i, index)))
                    else:
                        pieces.append((d_expr, cond & d_cond))
            else:
                pieces.append((diff_final_c, cond))

        return sp.Piecewise(*pieces)
