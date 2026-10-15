"""LaTeX formatting utilities for DiffPlz.

Extracted from DiffPlz.latex_readable_plz to improve readability
and testability.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import sympy as sp
from sympy import Derivative, Function, Idx, Indexed, Symbol

from thermodiff.thermovars import P, T, V, i, j, n

# -----------------------------------------------------------------------------
# Symbol building
# -----------------------------------------------------------------------------


def extract_subscript(arg: Union[Idx, Indexed, sp.Expr]) -> Optional[str]:
    """Extract LaTeX subscript string from a function argument.

    Parameters
    ----------
    arg : sympy expression
        Argument to extract subscript from. Typically an Idx or Indexed.

    Returns
    -------
    str or None
        LaTeX representation of the subscript, or None if not applicable.
    """
    if isinstance(arg, Idx):
        return sp.latex(arg)
    if isinstance(arg, Indexed):
        return sp.latex(arg.indices[0])
    return None


def build_pretty_symbol(instance: Function) -> Symbol:
    r"""Build a pretty LaTeX symbol for a function instance.

    Examples
    --------
    >>> tau = sp.Function(r"\tau")(l, k, T)
    >>> build_pretty_symbol(tau)  # -> Symbol(r"\tau_{lk}")

    >>> phi = sp.Function(r"\phi")(n[k])
    >>> build_pretty_symbol(phi)  # -> Symbol(r"\phi_{k}")

    >>> f = sp.Function("f")(T)
    >>> build_pretty_symbol(f)  # -> Symbol("f")
    """
    base = sp.latex(instance.func)
    subscripts = [
        sub
        for arg in instance.args
        if (sub := extract_subscript(arg)) is not None
    ]
    if subscripts:
        name = base + "_{" + "".join(subscripts) + "}"
    else:
        name = base
    return Symbol(name, commutative=True)


def make_partial(func_name: str, wrt: str) -> Symbol:
    r"""Build \frac{\partial func_name}{\partial wrt} symbol."""
    return Symbol(
        rf"\frac{{\partial {func_name}}}{{\partial {wrt}}}",
        commutative=True,
    )


def make_partial2(func_name: str, wrt1: str, wrt2: str) -> Symbol:
    r"""Build \frac{\partial^2 func_name}{\partial wrt1 \partial wrt2} symbol."""  # noqa: E501
    return Symbol(
        rf"\frac{{\partial^2 {func_name}}}{{\partial {wrt1} \partial {wrt2}}}",
        commutative=True,
    )


# -----------------------------------------------------------------------------
# Derivative replacement logic
# -----------------------------------------------------------------------------


class DerivativeReplacer:
    """Handles replacement of sympy Derivative nodes with pretty LaTeX symbols.

    This encapsulates the complex matching/replacement logic that was
    previously nested inside latex_readable_plz.
    """

    def __init__(self, internal_functions: List[Function]):
        self.internal_functions = internal_functions
        self._func_types = {type(f) for f in internal_functions}

    # --- First-order derivatives ---

    def make_first_order_matcher(
        self, wrt_var: sp.Symbol, func_type: type
    ) -> Callable:
        """Create a predicate matching Derivative(expr, wrt) for a specific func type."""  # noqa: E501
        def match(e: sp.Expr) -> bool:  # noqa: BLK100
            return (
                isinstance(e, Derivative)
                and isinstance(e.expr, func_type)
                and e.variables == (wrt_var,)
            )

        return match

    def make_first_order_subber(self, wrt_label: str) -> Callable:
        """Create a replacement function for first-order derivatives."""

        def sub(e: Derivative) -> Symbol:
            fname = build_pretty_symbol(e.expr).name
            return make_partial(fname, wrt_label)

        return sub

    # --- Second-order derivatives ---

    def make_second_order_matcher(
        self, wrt1: sp.Symbol, wrt2: sp.Symbol, func_type: type
    ) -> Callable:
        """Create a predicate matching Derivative(expr, wrt1, wrt2)."""

        def match(e: sp.Expr) -> bool:
            return (
                isinstance(e, Derivative)
                and isinstance(e.expr, func_type)
                and e.variables == (wrt1, wrt2)
            )

        return match

    def make_second_order_subber(
        self, wrt1_label: str, wrt2_label: str
    ) -> Callable:
        """Create a replacement function for second-order derivatives."""

        def sub(e: Derivative) -> Symbol:
            fname = build_pretty_symbol(e.expr).name
            return make_partial2(fname, wrt1_label, wrt2_label)

        return sub

    # --- Free function instances (non-derivative) ---

    def make_free_matcher(self, func_type: type) -> Callable:
        """Create a predicate matching bare function instances."""

        def match(e: sp.Expr) -> bool:
            return isinstance(e, func_type)

        return match

    def make_free_subber(self) -> Callable:
        """Create a replacement for bare function instances."""

        def sub(e: Function) -> Symbol:
            return build_pretty_symbol(e)

        return sub


# -----------------------------------------------------------------------------
# Variable/label mappings
# -----------------------------------------------------------------------------

# Maps derivative key -> (sympy variable, latex label)
WRT1_MAP: Dict[str, Tuple[sp.Symbol, str]] = {
    "T": (T, "T"),
    "V": (V, "V"),
    "P": (P, "P"),
    "n_i": (n[i], "n_i"),
}

# Maps derivative key -> (var1, var2, label1, label2)
WRT2_MAP: Dict[str, Tuple[sp.Symbol, sp.Symbol, str, str]] = {
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


# -----------------------------------------------------------------------------
# Main processing function
# -----------------------------------------------------------------------------


def process_derivative_expression(
    expr: sp.Expr,
    order: int,
    diff_key: str,
    internal_functions: List[Function],
) -> sp.Expr:
    """Apply all pretty replacements for a derivative expression.

    Parameters
    ----------
    expr : sympy expression
        The derivative expression to process.
    order : int
        1 for first derivatives, 2 for second derivatives.
    diff_key : str
        Key identifying which derivative (e.g., "T", "T2", "Tn").
    internal_functions : list of sympy Functions
        Functions to replace with pretty symbols.

    Returns
    -------
    sympy expression
        Expression with Derivative nodes and function instances replaced.
    """
    replacer = DerivativeReplacer(internal_functions)

    if order == 1:
        wrt_var, wrt_label = WRT1_MAP[diff_key]

        for func_type in replacer._func_types:
            # Replace Derivative(instance, wrt) nodes
            expr = expr.replace(
                replacer.make_first_order_matcher(wrt_var, func_type),
                replacer.make_first_order_subber(wrt_label),
            )

            # Replace remaining free function instances
            expr = expr.replace(
                replacer.make_free_matcher(func_type),
                replacer.make_free_subber(),
            )

    else:  # order == 2
        wrt1, wrt2, label1, label2 = WRT2_MAP[diff_key]

        for func_type in replacer._func_types:
            # Replace second-order Derivative nodes
            expr = expr.replace(
                replacer.make_second_order_matcher(wrt1, wrt2, func_type),
                replacer.make_second_order_subber(label1, label2),
            )

            # Replace first-order Derivative nodes that survive inside
            # second-order expressions (e.g., chain rule terms)
            for w, lbl in [(wrt1, label1), (wrt2, label2)]:
                expr = expr.replace(
                    replacer.make_first_order_matcher(w, func_type),
                    replacer.make_first_order_subber(lbl),
                )

            # Replace all remaining free instances
            expr = expr.replace(
                replacer.make_free_matcher(func_type),
                replacer.make_free_subber(),
            )

    return expr


# -----------------------------------------------------------------------------
# Post-processing cleanup
# -----------------------------------------------------------------------------


def clean_latex_string(latex_str: str) -> str:
    """Clean up common LaTeX artifacts from sympy output."""
    return (
        latex_str.replace("()", "")
        .replace(r"\left( \right)", "")
        .replace(r"\partial T \partial T", r"\partial T^2")
        .replace(r"\partial V \partial V", r"\partial V^2")
        .replace(r"\partial P \partial P", r"\partial P^2")
    )


def finalize_latex_dict(latex_dict: Dict[str, str]) -> Dict[str, str]:
    """Apply final key renames and cleanups to the latex output dict."""
    # Rename keys to match expected API
    if "dn_i" in latex_dict:
        latex_dict["dni"] = latex_dict.pop("dn_i")
    return latex_dict
