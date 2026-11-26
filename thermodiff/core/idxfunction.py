"""IndexBase function module.

SymPy doesn't provide a direct way to create symbolic functions with lazy
derivatives. This module provides a way to create such functions by extending
the `sp.Function` class and overriding the `_eval_derivative` method to return
a `sp.Derivative` object without evaluating it.
"""

import sympy as sp

#: Argument positions accepted by ``idx_function(..., symmetric=...)``.
SYMMETRIC_PAIR = (0, 1)


def idx_function(
    name: str, symmetric: tuple[int, int] | None = None
) -> sp.Function:
    """Create a symbolic function with lazy derivative.

    Parameters
    ----------
    name : str
        Name of the symbolic function to create.
    symmetric : tuple of int, optional
        Declare the function symmetric in its first two arguments,
        ``a(k, l, ...) == a(l, k, ...)``; only ``(0, 1)`` is supported.
        The Fortran generator then treats the placeholder and its
        derivatives that carry no extra index (``a``, ``dadT``,
        ``dadTdT``) as symmetric arrays, like ``symmetric_params``; the
        caller must pass symmetric arrays.

    Returns
    -------
    sp.Function
        A symbolic function class with a lazy derivative.

    Raises
    ------
    ValueError
        If ``symmetric`` is neither ``None`` nor ``(0, 1)``.
    """
    if symmetric is not None and tuple(symmetric) != SYMMETRIC_PAIR:
        raise ValueError(
            f"symmetric must be {SYMMETRIC_PAIR} or None, got {symmetric!r}"
        )
    lazy_function_class = type(
        name,
        (sp.Function,),
        {
            "_eval_derivative": lazy_derivative,
            "symmetric_args": None if symmetric is None else SYMMETRIC_PAIR,
            "__doc__": f"Simbolic function {name} with lazy derivative",
        },
    )

    return lazy_function_class


def lazy_derivative(self: sp.Function, s: sp.Symbol) -> sp.Derivative:
    """Return a lazy derivative of the symbolic function (self).

    Parameters
    ----------
    self : sp.Function
        lazy_function_class instance
    s : sp.Symbol
        Symbol with respect to which the derivative is taken.

    Returns
    -------
    sp.Derivative
        A simple SymPy derivative object without evaluation.
    """
    return sp.Derivative(self, s, evaluate=False)
