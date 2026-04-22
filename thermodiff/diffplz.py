"""Differentiate Please module.

This module provides the `DiffPlz` class, which is used to obtain all the
derivatives of a thermodynamic expression with respect to temperature (T),
volume (V) (could be pressure (P) too), and the number of moles of components
(n[i], n[j]). It handles both direct and cross derivatives, including second
derivatives and cross derivatives between temperature, volume, and number of
moles.

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


class DiffPlz:
    """Class to obtaine all the derivatives of a thermodynamic expression.

    The class will obtain the derivatives respect to:
        - Temperature (T)
        - Volume (V) or pressure (P)
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
    dp : SymPy expression
        First derivative of the expression with respect to pressure (P).
    dp2 : SymPy expression
        Second derivative of the expression with respect to pressure (P).
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
    arguments : List[sp.Symbol]
        List of symbols (variables) that appear in the expression.
    """

    def __init__(
        self,
        expression,
        internal_functions: List[sp.Function] | None = None,
        indexes: List[sp.Idx] | None = None,  # noqa E741)
        name: str = "f",
    ):
        self.name = name
        self.expression = expression
        self.internal_functions = (
            internal_functions if internal_functions is not None else []
        )
        self.indexes = indexes if indexes is not None else [k, l, m]

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

    def clean_plz(self, derivs_to_clean: list[str] | None = None):
        """Clean the DiffPlz instance.

        This method applies known simplification patterns to the differentiated
        expressions in the DiffPlz instance.

        Parameters
        ----------
        derivs_to_clean : list[str], optional
            List of derivative keys to clean. If None, all derivatives will be
            cleaned. The possible keys are: "dT", "dV", "dP", "dni", "dT2",
            "dV2", "dP2", "dnidnj", "dTn", "dVn", "dPn", "dTV", "dTP", "dVP".
        """
        sym = sp.Function(self.name)(*self.arguments)

        def _safe_subs(expr, old, new):
            return expr.subs(old, new) if expr.has(old) else expr

        # Normalize user input
        if derivs_to_clean is not None:
            derivs_to_clean = {k.lower() for k in derivs_to_clean}

        # =====================================================================
        # First derivatives
        # =====================================================================
        first = {
            "dt": ("dt", self.dt, sp.Derivative(sym, T)),
            "dv": ("dv", self.dv, sp.Derivative(sym, V)),
            "dp": ("dp", self.dp, sp.Derivative(sym, P)),
            "dni": ("dni", self.dni, sp.Derivative(sym, n[i])),
        }

        if derivs_to_clean is not None:
            first = {k: v for k, v in first.items() if k in derivs_to_clean}

        for _, (attr_name, expr, deriv_sym) in first.items():
            expr = _safe_subs(expr, self.expression, sym)
            expr = _safe_subs(expr, self.expression / T, sym / T)
            setattr(self, attr_name, expr)

        # =====================================================================
        # Second derivatives
        # =====================================================================
        second = {
            "dt2": ("dt2", self.dt2),
            "dv2": ("dv2", self.dv2),
            "dp2": ("dp2", self.dp2),
            "dnidnj": ("dnidnj", self.dnidnj),
            "dtdv": ("dtdv", self.dtdv),
            "dtdp": ("dtdp", self.dtdp),
            "dtdni": ("dtdni", self.dtdni),
            "dvdni": ("dvdni", self.dvdni),
            "dvdp": ("dvdp", self.dvdp),
        }

        if derivs_to_clean is not None:
            second = {k: v for k, v in second.items() if k in derivs_to_clean}

        # Updated base refs AFTER cleaning first derivatives
        base = {
            "dt": self.dt,
            "dv": self.dv,
            "dp": self.dp,
            "dni": self.dni,
        }

        deriv_symbols = {
            "dt": sp.Derivative(sym, T),
            "dv": sp.Derivative(sym, V),
            "dp": sp.Derivative(sym, P),
            "dni": sp.Derivative(sym, n[i]),
        }

        for _, (attr_name, expr) in second.items():
            expr = _safe_subs(expr, self.expression, sym)

            for name, base_expr in base.items():
                if base_expr != 0:
                    expr = _safe_subs(expr, base_expr, deriv_symbols[name])
                    expr = _safe_subs(
                        expr, base_expr / T, deriv_symbols[name] / T
                    )

            setattr(self, attr_name, expr)

    def latex_readable_plz(self) -> dict:
        """Return a clean latex representation of derivatives.

        Heavily vibecoded method, could easily not work. Ad hocs at the end.

        """

        def _extract_subscript(arg) -> str | None:
            """Extract the LaTeX subscript string from a function argument."""
            if isinstance(arg, sp.Idx):
                return sp.latex(arg)
            if isinstance(arg, sp.Indexed):
                return sp.latex(arg.indices[0])
            return None

        def _build_pretty_symbol(instance) -> sp.Symbol:
            r"""Build a pretty LaTeX symbol for a function instance.

            tau(l, i, T) -> \tau_{li}
            phi(n[i])    -> \phi_i
            f(T)         -> \f  (no subscript)
            """
            base = sp.latex(instance.func)
            subscripts = [
                sub
                for arg in instance.args
                if (sub := _extract_subscript(arg)) is not None
            ]
            if subscripts:
                subscript_str = "".join(subscripts)
                name = base + "_{" + subscript_str + "}"
            else:
                name = base
            return sp.Symbol(name, commutative=True)

        def _make_partial(func_name: str, wrt: str) -> sp.Symbol:
            r"""Build \frac{\partial func_name}{\partial wrt} symbol."""
            return sp.Symbol(
                rf"\frac{{\partial {func_name}}}{{\partial {wrt}}}",
                commutative=True,
            )

        def _make_partial2(func_name: str, wrt1: str, wrt2: str) -> sp.Symbol:
            r"""Build the symbol.

            \frac{\partial^2 func_name}{\partial wrt1 \partial wrt2}
            """
            return sp.Symbol(
                rf"\frac{{\partial^2 {func_name}}}{{\partial {wrt1} \partial {wrt2}}}",  # noqa
                commutative=True,
            )

        def _process_expr(expr: sp.Expr, order: int, diff_key: str) -> sp.Expr:
            """Apply all pretty replacements for a derivative expression."""
            wrt1_map = {
                "T": (T, "T"),
                "V": (V, "V"),
                "P": (P, "P"),
                "n_i": (n[i], "n_i"),
                "n_j": (n[j], "n_j"),
            }

            wrt2_map = {
                "T2": (T, T, "T", "T"),
                "V2": (V, V, "V", "V"),
                "P2": (P, P, "P", "P"),
                "n2": (n[i], n[j], "n_i", "n_j"),
                "Tn": (T, n[i], "T", "n_i"),
                "Vn": (V, n[i], "V", "n_i"),
                "Pn": (P, n[i], "P", "n_i"),
                "TV": (T, V, "T", "V"),
                "TP": (T, P, "T", "P"),
                "VP": (V, P, "V", "P"),
            }

            for function in self.internal_functions:
                func_type = type(function)

                if order == 1:
                    w, lbl = wrt1_map[diff_key]

                    # Replace Derivative(instance, w) nodes using
                    # replace+predicate
                    def match_deriv1(e, _w=w, _ft=func_type):
                        return (
                            isinstance(e, sp.Derivative)
                            and isinstance(e.expr, _ft)
                            and e.variables == (_w,)
                        )

                    def sub_deriv1(e, _lbl=lbl):
                        fname = _build_pretty_symbol(e.expr).name
                        return _make_partial(fname, _lbl)

                    expr = expr.replace(match_deriv1, sub_deriv1)

                else:  # order == 2
                    w1, w2, l1, l2 = wrt2_map[diff_key]

                    # Replace second-order Derivative nodes
                    def match_deriv2(e, _w1=w1, _w2=w2, _ft=func_type):
                        return (
                            isinstance(e, sp.Derivative)
                            and isinstance(e.expr, _ft)
                            and e.variables == (_w1, _w2)
                        )

                    def sub_deriv2(e, _l1=l1, _l2=l2):
                        fname = _build_pretty_symbol(e.expr).name
                        return _make_partial2(fname, _l1, _l2)

                    expr = expr.replace(match_deriv2, sub_deriv2)

                    # Replace first-order Derivative nodes that survive inside
                    # second-order expressions (e.g. chain rule terms)
                    for w, lbl in [(w1, l1), (w2, l2)]:

                        def match_deriv1_in2(e, _w=w, _ft=func_type):
                            return (
                                isinstance(e, sp.Derivative)
                                and isinstance(e.expr, _ft)
                                and e.variables == (_w,)
                            )

                        def sub_deriv1_in2(e, _lbl=lbl):
                            fname = _build_pretty_symbol(e.expr).name
                            return _make_partial(fname, _lbl)

                        expr = expr.replace(match_deriv1_in2, sub_deriv1_in2)

                # Replace all remaining free instances (penetrates Sum,
                # Piecewise, etc.)
                def match_free(e, _ft=func_type):
                    return isinstance(e, _ft)

                def sub_free(e):
                    return _build_pretty_symbol(e)

                expr = expr.replace(match_free, sub_free)

            return expr

        latex_finals = {}

        first_exprs = {
            "T": self.dt,
            "V": self.dv,
            "P": self.dp,
            "n_i": self.dni,
        }

        for diff_key, expr in first_exprs.items():
            expr = _process_expr(expr, order=1, diff_key=diff_key)
            result = (
                sp.latex(expr).replace("()", "").replace(r"\left( \right)", "")
            )
            latex_finals["d" + diff_key] = result

        second_exprs = {
            "T2": self.dt2,
            "V2": self.dv2,
            "P2": self.dp2,
            "n2": self.dnidnj,
            "Tn": self.dtdni,
            "Vn": self.dvdni,
            "Pn": self.dpdni,
            "TV": self.dtdv,
            "TP": self.dtdp,
            "VP": self.dvdp,
        }

        for diff_key, expr in second_exprs.items():
            expr = _process_expr(expr, order=2, diff_key=diff_key)
            result = (
                sp.latex(expr).replace("()", "").replace(r"\left( \right)", "")
            )
            latex_finals["d" + diff_key] = result

        # =====================================================================
        # Dumb adhoc things
        # =====================================================================
        latex_finals["dT2"] = latex_finals["dT2"].replace(
            r"\partial T \partial T", r"\partial T^2"
        )

        latex_finals["dni"] = latex_finals.pop("dn_i")

        return latex_finals

    def _detect_arguments(self) -> List[sp.Symbol]:
        """Check the thermodynamic variables used in the expression."""
        arguments = []

        if any(
            isinstance(a, sp.Indexed) and a.base == n
            for a in self.expression.atoms(sp.Indexed)
        ):
            arguments.append(n)

        if self.expression.has(V):
            arguments.append(V)

        if self.expression.has(P):
            arguments.append(P)

        if self.expression.has(T):
            arguments.append(T)

        return arguments

    def __repr__(self):
        """Return a string representation of the DiffPlz object."""
        return f"DiffPlz(name={self.name}, args={self.arguments})"
