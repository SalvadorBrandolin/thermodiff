"""Module containing functions to handle Kronecker deltas in expressions.

When differentiating thermodynamic expressions, Kronecker deltas may appear.
This module provides functions to handle these deltas, either by simplifying
them or replacing them with piecewise expressions.
"""

import itertools

import sympy as sp

from thermodiff.thermovars import nc


def handle_sum_kronecker(expr: sp.Expr, idx: sp.Idx) -> sp.Expr:
    """Simplifies Kronecker deltas in summations of an expression.

    Differentiating thermodynamic expressions with respect to mole numbers
    produces Kronecker deltas inside summations, for example
    ``Sum(f(k)*KroneckerDelta(i, k), (k, 1, nc))``.  This function reduces
    those summations exactly as
    ``Sum(f(k)*KroneckerDelta(i, k), (k, 1, nc)) = f(i)``, keeping every
    delta-free summation as a symbolic :class:`sympy.Sum` (so the code
    generator can turn it into a loop and ``N`` stays variable).

    The reduction is applied recursively to every ``Sum`` of ``expr``:

    * a term of a summation containing ``KroneckerDelta(x, u)`` (either
      argument order) with ``u`` the summation index is substituted
      ``u -> x`` and taken out of the summation, dropping the ``u`` limit;
    * a term containing ``KroneckerDelta(u, v)`` with both indices bound by
      the summation is substituted ``u -> v`` and the ``u`` limit is
      dropped;
    * terms without such deltas stay inside the (possibly reduced)
      summation.

    This is equivalent to evaluating the delta summations with
    ``Sum.doit()`` under the assumptions ``1 <= x <= nc``, but it never
    expands the delta-free summations (which made full mixture models
    impractical to differentiate).

    Parameters
    ----------
    expr : sp.Expr
        Expression to simplify.
    idx : sp.Idx
        Index variable for the Kronecker delta. Normally this is either `i` or
        `j`.

    Returns
    -------
    sp.Expr
        Simplified expression.
    """
    reduced = _reduce_delta_sums(expr)
    return reduced.subs({idx >= 1: True, nc >= idx: True})


def _reduce_delta_sums(expr):
    """Recursively reduce summations containing Kronecker deltas.

    Only :class:`sympy.Sum` nodes are rewritten; every other node is
    rebuilt from its (reduced) arguments, and nodes without Kronecker
    deltas are returned unchanged.

    Parameters
    ----------
    expr : sp.Expr
        Expression to rewrite.

    Returns
    -------
    sp.Expr
        Expression whose delta summations are reduced.
    """
    if not expr.has(sp.KroneckerDelta):
        return expr
    if isinstance(expr, sp.Sum):
        body = _reduce_delta_sums(expr.function)
        if body == expr.function:
            return _reduce_sum(expr)
        return _reduce_sum(sp.Sum(body, *expr.limits))
    if isinstance(expr, sp.Piecewise):
        pairs = tuple(
            sp.ExprCondPair(_reduce_delta_sums(piece.expr), piece.cond)
            for piece in expr.args
        )
        if pairs == tuple(expr.args):
            return expr
        return sp.Piecewise(*pairs)
    if isinstance(expr, (sp.Add, sp.Mul, sp.Pow)):
        args = tuple(_reduce_delta_sums(arg) for arg in expr.args)
        if args == tuple(expr.args):
            return expr
        return expr.func(*args)
    if isinstance(expr, (sp.Idx, sp.Indexed, sp.IndexedBase, sp.Symbol)):
        return expr
    if isinstance(expr, sp.Expr) and expr.args:
        args = tuple(_reduce_delta_sums(arg) for arg in expr.args)
        if args == tuple(expr.args):
            return expr
        return expr.func(*args)
    return expr


def _find_delta(term, index):
    """Return a delta of ``term`` involving ``index`` in exactly one argument.

    Parameters
    ----------
    term : sp.Expr
        Expression to inspect.
    index : sp.Idx
        Bound index the delta must involve.

    Returns
    -------
    sp.KroneckerDelta or None
        The delta to reduce, or ``None`` when there is none.
    """
    for delta in sorted(term.atoms(sp.KroneckerDelta), key=str):
        first, second = delta.args
        if (first == index) != (second == index):
            return delta
    return None


def _distribute_delta_factors(term):
    """Split ``term`` so that every Kronecker delta is a multiplicative factor.

    ``Sum.doit``-style reduction is only valid on terms of the shape
    ``delta(x, u) * rest``; a delta inside an additive factor (for example
    ``(n[k] + delta(i, k)*n[l])*n[k]``) must first be distributed.  Only
    delta-containing sums are distributed, so the plain thermodynamic
    products are left factored.

    Parameters
    ----------
    term : sp.Expr
        One additive component of a summation body.

    Returns
    -------
    list of sp.Expr
        Additive components whose deltas are all multiplicative factors.
    """
    if not term.has(sp.KroneckerDelta):
        return [term]
    if isinstance(term, sp.Add):
        pieces = []
        for sub in term.args:
            pieces.extend(_distribute_delta_factors(sub))
        return pieces
    if isinstance(term, sp.Mul):
        groups = [_distribute_delta_factors(arg) for arg in term.args]
        return [sp.Mul(*combo) for combo in itertools.product(*groups)]
    return [term]


def _reduce_sum(total):
    """Reduce the Kronecker deltas of a single summation.

    Parameters
    ----------
    total : sp.Sum
        Summation whose body may contain Kronecker deltas.

    Returns
    -------
    sp.Expr
        The reduced expression: substituted terms plus the summations of
        the terms that could not be reduced.
    """
    body = total.function
    if not body.has(sp.KroneckerDelta):
        return total

    limits = [tuple(limit) for limit in total.limits]
    reduced_terms = []
    kept = {}
    changed = False
    for term in _distribute_delta_factors(body):
        current = term
        remaining = list(limits)
        progress = True
        while progress:
            progress = False
            for limit in list(remaining):
                index = limit[0]
                delta = _find_delta(current, index)
                if delta is None:
                    continue
                first, second = delta.args
                other = second if first == index else first
                substituted = current.subs(index, other)
                if substituted.has(delta):
                    # The substitution did not take effect (for example the
                    # delta is bound by a nested summation): leave it alone.
                    continue
                current = substituted
                remaining.remove(limit)
                progress = True
                changed = True
        if remaining:
            kept.setdefault(tuple(remaining), []).append(current)
        else:
            reduced_terms.append(current)

    if not changed:
        return total

    parts = list(reduced_terms)
    for key in kept:
        parts.append(sp.Sum(sp.Add(*kept[key]), *key))
    return sp.Add(*parts)


def handle_free_kronecker(
    expr: sp.Expr, kdx: sp.Idx, idx: sp.Idx
) -> sp.Piecewise:
    """Replace expression cotanining Kronecker delta(i, k) with Pieacewise.

    Parameters
    ----------
    expr : sp.Expr
        Expression to replace.
    kdx : sp.Idx
        Index variable of the Kronecker delta, usually `k`.
    idx : sp.Idx
        Index variable to replace the Kronecker delta, usually `i` or `j`.

    Returns
    -------
    sp.Piecewise
        Piecewise expression that replaces the Kronecker delta.
    """
    delta = sp.KroneckerDelta(idx, kdx)

    if expr.has(delta):
        d_i, d_k = delta.args

        return sp.Piecewise(
            (expr.subs(kdx, idx), sp.Eq(d_i, d_k)), (expr.subs(delta, 0), True)
        )

    return expr
