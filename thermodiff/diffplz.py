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
from thermodiff.derivatives import DERIVATIVE_ATTRIBUTES, resolve_derivatives
from thermodiff.latex_utils import (
    clean_latex_string,
    finalize_latex_dict,
    process_derivative_expression,
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
    derivatives : str or iterable of str, optional
        Derivative attributes to compute.  By default every derivative is
        computed.  When a selection is given, only the selected derivatives
        (and the lower order ones they are built from) are differentiated;
        the attributes that were not requested are set to ``None``.  The
        names understood by
        :func:`thermodiff.derivatives.resolve_derivatives` are accepted.

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
    derivatives : list of str or None
        The derivative attributes that were selected for computation, or
        ``None`` when every derivative was computed.
    """

    def __init__(
        self,
        expression,
        internal_functions: List[sp.Function] | None = None,
        indexes: List[sp.Idx] | None = None,  # noqa E741)
        name: str = "f",
        derivatives=None,
    ):
        self.name = name
        self.expression = expression
        self.internal_functions = (
            internal_functions if internal_functions is not None else []
        )
        self.indexes = indexes if indexes is not None else [k, l, m]
        self._cache = {}

        if derivatives is None:
            self.derivatives = None
            selected = list(DERIVATIVE_ATTRIBUTES)
        else:
            selected = resolve_derivatives(derivatives)
            self.derivatives = list(selected)

        for attribute in DERIVATIVE_ATTRIBUTES:
            value = self._compute(attribute) if attribute in selected else None
            setattr(self, attribute, value)

        # Arguments
        self.arguments = self._detect_arguments()

    def _compute(self, attribute: str) -> sp.Expr:
        """Return a derivative, computing and caching it on demand.

        Parameters
        ----------
        attribute : str
            One of :data:`thermodiff.derivatives.DERIVATIVE_ATTRIBUTES`.

        Returns
        -------
        sympy expression
            The requested derivative.
        """
        if attribute in self._cache:
            return self._cache[attribute]

        expression = self.expression
        if attribute == "dt":
            value = sp.diff(expression, T)
        elif attribute == "dv":
            value = sp.diff(expression, V)
        elif attribute == "dp":
            value = sp.diff(expression, P)
        elif attribute == "dni":
            value = self.diff_ni(expression, i)
        elif attribute == "dt2":
            value = sp.diff(self._compute("dt"), T)
        elif attribute == "dv2":
            value = sp.diff(self._compute("dv"), V)
        elif attribute == "dp2":
            value = sp.diff(self._compute("dp"), P)
        elif attribute == "dnidnj":
            value = self.diff_dnidnj(self._compute("dni"), j)
        elif attribute == "dtdv":
            value = sp.diff(self._compute("dt"), V)
        elif attribute == "dtdp":
            value = sp.diff(self._compute("dt"), P)
        elif attribute == "dtdni":
            value = self.diff_ni(self._compute("dt"), i)
        elif attribute == "dvdni":
            value = self.diff_ni(self._compute("dv"), i)
        elif attribute == "dvdp":
            value = sp.diff(self._compute("dv"), P)
        elif attribute == "dpdni":
            value = self.diff_ni(self._compute("dp"), i)
        else:  # pragma: no cover - guarded by DERIVATIVE_ATTRIBUTES
            raise ValueError(f"Unknown derivative attribute {attribute!r}")

        self._cache[attribute] = value
        return value

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
            if expr is None:
                continue
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
            if expr is None:
                continue
            expr = _safe_subs(expr, self.expression, sym)

            for name, base_expr in base.items():
                if base_expr is not None and base_expr != 0:
                    expr = _safe_subs(expr, base_expr, deriv_symbols[name])
                    expr = _safe_subs(
                        expr, base_expr / T, deriv_symbols[name] / T
                    )

            setattr(self, attr_name, expr)

    def latex_readable_plz(self) -> dict:
        """Return a clean LaTeX representation of all derivatives.

        Uses latex_utils module for the heavy lifting. This method is now
        a thin orchestration layer.
        """
        latex_finals = {}

        # First derivatives
        first_exprs = {
            "T": self.dt,
            "V": self.dv,
            "P": self.dp,
            "n_i": self.dni,
        }

        for diff_key, expr in first_exprs.items():
            if expr is None:
                continue
            expr = process_derivative_expression(
                expr,
                order=1,
                diff_key=diff_key,
                internal_functions=self.internal_functions,
            )
            result = clean_latex_string(sp.latex(expr))
            latex_finals["d" + diff_key] = result

        # Second derivatives
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
            if expr is None:
                continue
            expr = process_derivative_expression(
                expr,
                order=2,
                diff_key=diff_key,
                internal_functions=self.internal_functions,
            )
            result = clean_latex_string(sp.latex(expr))
            latex_finals["d" + diff_key] = result

        return finalize_latex_dict(latex_finals)

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

    def to_fortran(self, *args, **kwargs):
        """Generate a Fortran subroutine for this instance.

        This is a thin wrapper around
        :func:`thermodiff.fortran.to_fortran`.  See that function for the
        accepted arguments.

        Returns
        -------
        str
            The free-form Fortran source code.
        """
        from thermodiff.fortran import to_fortran

        return to_fortran(self, *args, **kwargs)

    def __repr__(self):
        """Return a string representation of the DiffPlz object."""
        return f"DiffPlz(name={self.name}, args={self.arguments})"
