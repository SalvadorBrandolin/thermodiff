"""IndexBase function module.

SymPy doesn't provide a direct way to create symbolic functions with lazy
derivatives. This module provides a way to create such functions by extending
the `sp.Function` class and overriding the `_eval_derivative` method to return
a `sp.Derivative` object without evaluating it.
"""

import sympy as sp


def idx_function(name: str) -> sp.Function:
    """Create a symbolic function with lazy derivative.

    Parameters
    ----------
    name : str
        Name of the symbolic function to create.

    Returns
    -------
    sp.Function
        A symbolic function class with a lazy derivative.
    """
    LazyFunctionClass = type(
        name,
        (sp.Function,),
        {
            "_eval_derivative": lazy_derivative,
            "__doc__": f"Simbolic function {name} with lazy derivative",
        },
    )

    return LazyFunctionClass


def lazy_derivative(self: sp.Function, s: sp.Symbol) -> sp.Derivative:
    """Return a lazy derivative of the symbolic function (self).

    Parameters
    ----------
    self : sp.Function
        LazyFunctionClass instance
    s : sp.Symbol
        Symbol with respect to which the derivative is taken.

    Returns
    -------
    sp.Derivative
        A simple SymPy derivative object without evaluation.
    """
    return sp.Derivative(self, s, evaluate=False)
