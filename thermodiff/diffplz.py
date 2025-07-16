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

from thermodiff.core.kronecker_handling import (
    handle_free_kronecker,
    handle_sum_kronecker,
)
from thermodiff.thermovars import P, T, V, i, j, k, l, m, n

import copy


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
        indexes: List[sp.Idx] = [k, l, m],  # noqa E741)
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

        # Arguments
        self.arguments = self._detect_arguments()

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
                    if (d_cond is True) and not (cond is True):
                        pieces.append((d_expr, cond & sp.Ne(i, index)))
                    elif not (d_cond is True) and (cond is True):
                        pieces.append((d_expr, d_cond & sp.Ne(i, index)))
                    else:
                        pieces.append((d_expr, cond & d_cond))
            else:
                pieces.append((diff_final_c, cond))

        return sp.Piecewise(*pieces)

    def clean_plz(self):
        """Clean the DiffPlz instance.

        This method applies known simplifications patterns to the
        differentiated expressions in the DiffPlz instance.
        """
        sym = sp.Function(self.name)(*self.arguments)

        # =====================================================================
        # First diffs has the expression?
        # =====================================================================
        # dT
        dt = copy.copy(self.dt)
        dtsym = sp.Derivative(sym, T)
        if self.dt.has(self.expression):
            self.dt = self.dt.subs(self.expression, sym)

        if self.dt.has(self.expression / T):
            self.dt = self.dt.subs(self.expression / T, sym / T)

        # dP
        dp = copy.copy(self.dp)
        dpsym = sp.Derivative(sym, P)
        if self.dp.has(self.expression):
            self.dp = self.dp.subs(self.expression, sym)

        # dV
        dv = copy.copy(self.dv)
        dvsym = sp.Derivative(sym, V)
        if self.dv.has(self.expression):
            self.dv = self.dv.subs(self.expression, sym)

        # dni
        dni = copy.copy(self.dni)
        dnisym = sp.Derivative(sym, n[i])
        if self.dni.has(self.expression):
            self.dni = self.dni.subs(self.expression, sym)

        # =====================================================================
        # Second diffs has the expression?
        # =====================================================================
        # dT2
        if self.dt2.has(self.expression):
            self.dt2 = self.dt2.subs(self.expression, sym)

        if self.dt2.has(self.expression / T):
            self.dt2 = self.dt2.subs(self.expression / T, sym / T)

        if self.dt2.has(self.dt) and self.dt != 0:
            self.dt2 = self.dt2.subs(self.dt, dtsym)

        if self.dt2.has(self.dt / T) and self.dt != 0:
            self.dt2 = self.dt2.subs(self.dt / T, dtsym / T)

        # dV2
        if self.dv2.has(self.expression):
            self.dv2 = self.dv2.subs(self.expression, sym)

        if self.dv2.has(dv) and self.dv != 0:
            self.dv2 = self.dv2.subs(self.dv, dvsym)

        # dP2
        if self.dp2.has(self.expression):
            self.dp2 = self.dp2.subs(self.expression, sym)

        if self.dp2.has(dp) and self.dp != 0:
            self.dp2 = self.dp2.subs(self.dp, dpsym)

        # dnidnj
        if self.dnidnj.has(self.expression):
            self.dnidnj = self.dnidnj.subs(self.expression, sym)

        if self.dnidnj.has(dni) and self.dni != 0:
            self.dnidnj = self.dnidnj.subs(self.dni, dnisym)

        # dtdv
        if self.dtdv.has(self.expression):
            self.dtdv = self.dtdv.subs(self.expression, sym)

        if self.dtdv.has(dt) and self.dt != 0:
            self.dtdv = self.dtdv.subs(self.dt, dtsym)

        if self.dtdv.has(dv) and self.dv != 0:
            self.dtdv = self.dtdv.subs(self.dv, dvsym)

        if self.dtdv.has(dt / T) and self.dt != 0:
            self.dtdv = self.dtdv.subs(self.dt / T, dtsym / T)

        if self.dtdv.has(dv / T) and self.dv != 0:
            self.dtdv = self.dtdv.subs(self.dv / T, dvsym / T)

        # dtdp
        if self.dtdp.has(self.expression):
            self.dtdp = self.dtdp.subs(self.expression, sym)

        if self.dtdp.has(dt) and self.dt != 0:
            self.dtdp = self.dtdp.subs(self.dt, dtsym)

        if self.dtdp.has(dp) and self.dp != 0:
            self.dtdp = self.dtdp.subs(self.dp, dpsym)

        if self.dtdp.has(dt / T) and self.dt != 0:
            self.dtdp = self.dtdp.subs(self.dt / T, dtsym / T)

        if self.dtdp.has(dp / T) and self.dp != 0:
            self.dtdp = self.dtdp.subs(self.dp / T, dpsym / T)

        # dtdni
        if self.dtdni.has(self.expression):
            self.dtdni = self.dtdni.subs(self.expression, sym)

        if self.dtdni.has(dt) and self.dt != 0:
            self.dtdni = self.dtdni.subs(self.dt, dtsym)

        if self.dtdni.has(dni) and self.dni != 0:
            self.dtdni = self.dtdni.subs(self.dni, dnisym)

        if self.dtdni.has(dt / T) and self.dt != 0:
            self.dtdni = self.dtdni.subs(self.dt / T, dtsym / T)

        if self.dtdni.has(dni / T) and self.dni != 0:
            self.dtdni = self.dtdni.subs(self.dni / T, dnisym / T)

        # dvdni
        if self.dvdni.has(self.expression):
            self.dvdni = self.dvdni.subs(self.expression, sym)

        if self.dvdni.has(dv) and self.dv != 0:
            self.dvdni = self.dvdni.subs(self.dv, dvsym)

        if self.dvdni.has(dni) and self.dni != 0:
            self.dvdni = self.dvdni.subs(self.dni, dnisym)

        # dvdp
        if self.dvdp.has(self.expression):
            self.dvdp = self.dvdp.subs(self.expression, sym)

        if self.dvdp.has(dv) and self.dv != 0:
            self.dvdp = self.dvdp.subs(self.dv, dvsym)

        if self.dvdp.has(dp) and self.dp != 0:
            self.dvdp = self.dvdp.subs(self.dp, dpsym)

    def latex_readable_plz(self) -> str:
        latex_finals = {}

        # =====================================================================
        # Clean first derivatives
        # =====================================================================
        expresions = {
            "T": copy.copy(self.dt),
            "V": copy.copy(self.dv),
            "P": copy.copy(self.dp),
            "n_i": copy.copy(self.dni),
        }

        for diff, expr in expresions.items():
            for function in self.internal_functions:
                expr = clean_first_deriv(expr, function, diff)
                expr = expr.replace(function, function.func)

            expr = sp.latex(expr).replace("\\\\", "\\").replace("()", "")
            expr = expr.replace(r"\left( \right)", "")

            latex_finals["d" + diff] = expr

        # =====================================================================
        # Clean second derivatives
        # =====================================================================
        expresions = {
            "T2": copy.copy(self.dt2),
            "V2": copy.copy(self.dv2),
            "P2": copy.copy(self.dp2),
            "n2": copy.copy(self.dnidnj),
            "Tn": copy.copy(self.dtdni),
            "Vn": copy.copy(self.dvdni),
            "Pn": copy.copy(self.dpdni),
            "TV": copy.copy(self.dtdv),
            "TP": copy.copy(self.dtdp),
            "VP": copy.copy(self.dvdp),
        }

        for diff, expr in expresions.items():
            for function in self.internal_functions:
                expr = clean_second_deriv(expr, function, diff)

                for d1 in ["T", "V", "P", "n_i", "n_j"]:
                    expr = clean_first_deriv(expr, function, d1)

                expr = expr.replace(function, function.func)

            expr = sp.latex(expr).replace("\\\\", "\\").replace("()", "")
            expr = expr.replace(r"\left( \right)", "")

            latex_finals["d" + diff] = expr

        return latex_finals

    def _detect_arguments(self) -> List[sp.Symbol]:
        """Check the thermodynamic variables used in the expression."""
        arguments = []

        if self.expression.has(n):
            arguments.append(n)

        if self.expression.has(V):
            arguments.append(V)

        if self.expression.has(P):
            arguments.append(P)

        if self.expression.has(T):
            arguments.append(T)

        return arguments


def clean_first_deriv(
    expresion: sp.Expr, function: sp.Function, differential: str
) -> sp.Expr:
    derivs = {
        "T": sp.Derivative(function, T),
        "V": sp.Derivative(function, V),
        "P": sp.Derivative(function, P),
        "n_i": sp.Derivative(function, n[i]),
        "n_j": sp.Derivative(function, n[j]),
    }

    deriv = derivs[differential]

    if expresion.has(deriv):
        pretty_deriv = sp.Symbol(
            rf"\frac{{\partial {sp.latex(function.func)}}}{{\partial {differential}}}",
            commutative=True,
        )

        expresion = expresion.replace(deriv, pretty_deriv)

    return expresion


def clean_second_deriv(
    expresion: sp.Expr, function: sp.Function, differential: str
) -> sp.Expr:
    derivs = {
        "T2": sp.Derivative(function, T, T),
        "V2": sp.Derivative(function, V, V),
        "P2": sp.Derivative(function, P, P),
        "n2": sp.Derivative(function, n[i], n[j]),
        "Tn": sp.Derivative(function, T, n[i]),
        "Vn": sp.Derivative(function, V, n[i]),
        "Pn": sp.Derivative(function, P, n[i]),
        "TV": sp.Derivative(function, T, V),
        "TP": sp.Derivative(function, T, P),
        "VP": sp.Derivative(function, V, P),
    }

    deriv = derivs[differential]

    if expresion.has(deriv):
        if "2" in differential:
            if "n" in differential:
                pretty_deriv = sp.Symbol(
                    rf"\frac{{\partial^2 {sp.latex(function.func)}}}{{\partial n_i \partial n_j}}", # noqa
                    commutative=True,
                )
            else:
                pretty_deriv = sp.Symbol(
                    rf"\frac{{\partial^2 {sp.latex(function.func)}}}{{\partial {differential}^2}}", # noqa
                    commutative=True,
                )
        else:
            if "n" in differential:
                pretty_deriv = sp.Symbol(
                    rf"\frac{{\partial^2 {sp.latex(function.func)}}}{{\partial {differential[0]} \partial n_i}}", # noqa
                    commutative=True,
                )
            else:
                pretty_deriv = sp.Symbol(
                    rf"\frac{{\partial^2 {sp.latex(function.func)}}}{{\partial {differential[0]} \partial {differential[1]}}}", # noqa
                    commutative=True,
                )

        expresion = expresion.replace(deriv, pretty_deriv)

    return expresion
