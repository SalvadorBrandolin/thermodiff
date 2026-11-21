"""Symmetry helpers for the Fortran code generator."""

from collections.abc import Collection, Iterable
from itertools import permutations
from typing import Any, cast

import sympy as sp

from thermodiff.codegen.common import (
    _INDENT,
    _LoopStart,
    _free_indices,
    _order_free,
    _print_expr,
)
from thermodiff.codegen.ir import TempDef


def _alpha_key(total: sp.Sum) -> sp.Basic:
    """Canonical form of a summation modulo renaming of its bound indices.

    Two summations that differ only in the names of their dummy indices
    (``sum_l x(i,l)`` and ``sum_m x(i,m)``) get the same key.
    """
    mapping: dict[sp.Idx, sp.Idx] = {}
    for position, limit in enumerate(total.limits):
        mapping[limit[0]] = sp.Idx(f"_bound{position}")
    return total.xreplace(mapping)


def _free_alpha_key(
    total: sp.Sum,
    indexes: Iterable[sp.Idx],
    symmetric_params: Collection[str] = (),
    *,
    sym_rank3: bool,
) -> tuple[sp.Basic, list[sp.Idx]]:
    """Canonical form of a summation modulo renaming of all its indices.

    Bound indices are renamed by position, free indices by their order, so
    that ``sum_k x(i,k)`` and ``sum_l x(j,l)`` get the same key.
    """
    free = _order_free(_free_indices(total), indexes)
    mapping = {index: sp.Idx(f"_free{pos}") for pos, index in enumerate(free)}
    key = _alpha_key(total.xreplace(mapping))
    if symmetric_params:
        key = _canonical_symmetric(key, symmetric_params, sym_rank3)
    return key, free


def _canonical_image(
    term: sp.Basic,
    bound: list[sp.Idx],
    symmetric_params: Collection[str],
    sym_rank3: bool,
) -> sp.Basic:
    """Return the smallest image of ``term`` under permutations of ``bound``.

    Parameters
    ----------
    term : sympy expression
        One term of a summation body.
    bound : list
        The bound indices of the summation.
    symmetric_params : iterable of str
        Symmetric parameters, canonicalised in every image.
    sym_rank3 : bool
        Also canonicalise symmetric parameters of rank three.

    Returns
    -------
    sympy expression
        The image with the smallest ``sp.default_sort_key``; ties keep the
        first permutation that reached it.
    """
    best: tuple[Any, sp.Basic] | None = None
    for perm in permutations(bound):
        image = term.xreplace(dict(zip(bound, perm)))
        if symmetric_params:
            image = _canonical_symmetric(image, symmetric_params, sym_rank3)
        key = sp.default_sort_key(image)
        if best is None or key < best[0]:
            best = (key, image)
    return cast(tuple[Any, sp.Basic], best)[1]


def _merge_body_terms(
    body: sp.Basic,
    bound: list[sp.Idx],
    symmetric_params: Collection[str],
    sym_rank3: bool,
) -> sp.Basic:
    """Merge the terms of ``body`` that are images of each other.

    Parameters
    ----------
    body : sympy expression
        Body of a summation over ``bound``.
    bound : list
        The bound indices of the summation.
    symmetric_params : iterable of str
        Symmetric parameters, canonicalised in every image.
    sym_rank3 : bool
        Also canonicalise symmetric parameters of rank three.

    Returns
    -------
    sympy expression
        The merged body, or ``body`` itself when no terms merge.
    """
    expanded = sp.expand(body)
    terms = expanded.args if isinstance(expanded, sp.Add) else (expanded,)
    if len(terms) <= 1:
        return body
    merged: dict[sp.Basic, sp.Basic] = {}
    for term in terms:
        image = _canonical_image(term, bound, symmetric_params, sym_rank3)
        coeff, rest = image.as_coeff_Mul()
        merged[rest] = merged.get(rest, 0) + coeff
    if len(merged) < len(terms):
        return sp.factor_terms(
            sp.Add(*[coeff * rest for rest, coeff in merged.items()])
        )
    return body


def _merge_in_sum(
    total: sp.Sum,
    symmetric_params: Collection[str],
    shared_bases: Collection[str],
    sym_rank3: bool,
) -> sp.Sum:
    """Rebuild a summation with its dummy images merged, innermost first.

    Parameters
    ----------
    total : sympy.Sum
        The summation to process.
    symmetric_params : iterable of str
        Symmetric parameters, canonicalised in every image.
    shared_bases : frozenset of str
        Arrays that keep a summation from being merged.
    sym_rank3 : bool
        Also canonicalise symmetric parameters of rank three.

    Returns
    -------
    sympy.Sum
        The summation with the merged body.
    """
    body = _merge_visit(
        total.function, symmetric_params, shared_bases, sym_rank3
    )
    bound = [lim[0] for lim in total.limits]
    ranges = {(lim[1], lim[2]) for lim in total.limits}
    reads_shared = any(
        str(indexed.base) in shared_bases for indexed in body.atoms(sp.Indexed)
    )
    if 2 <= len(bound) <= 3 and len(ranges) == 1 and not reads_shared:
        body = _merge_body_terms(body, bound, symmetric_params, sym_rank3)
    return total.func(body, *total.limits)


def _merge_visit(
    sub: sp.Basic,
    symmetric_params: Collection[str],
    shared_bases: Collection[str],
    sym_rank3: bool,
) -> sp.Basic:
    """Apply the dummy-image merge to every summation inside ``sub``."""
    if not getattr(sub, "args", ()) or isinstance(sub, (sp.Idx, sp.Indexed)):
        return sub
    if isinstance(sub, sp.Sum):
        return _merge_in_sum(sub, symmetric_params, shared_bases, sym_rank3)
    return sub.func(
        *[
            _merge_visit(arg, symmetric_params, shared_bases, sym_rank3)
            for arg in sub.args
        ]
    )


def _merge_dummy_images(
    expr: sp.Basic,
    symmetric_params: Collection[str] = (),
    shared_bases: Collection[str] = frozenset(),
    *,
    sym_rank3: bool,
) -> sp.Basic:
    """Merge the terms of a summation that are images under a renaming of
    its bound indices.

    In ``sum_{k,l,m} F`` with all indices over the same range, renaming the
    dummies never changes the value, so
    ``sum (b_k + b_l + b_m) L_klm n_k n_l n_m``
    with ``L`` symmetric equals ``3 sum b_k L_klm n_k n_l n_m``.  Each term of
    the body is mapped to the smallest of its images under every permutation
    of the bound indices (symmetric parameters canonicalised), equal images
    are merged with their multiplicity.  Exact; applied innermost first.

    A summation that reads an array of ``shared_bases`` (arrays that other
    outputs of the same kernel read inside sums with free indices) is left
    as is: its unmerged form shares the inner tables with those outputs, and
    merging would add a second pass over the array.
    """
    return _merge_visit(expr, symmetric_params, shared_bases, sym_rank3)


def _symv_parts(
    definition: TempDef,
    symmetric_params: Collection[str],
    *,
    sym_rank3: bool,
) -> tuple[sp.Basic, sp.Basic] | None:
    """Split ``v(f) = sum_b P(b,f)*R(b)`` with ``P`` symmetric.

    Returns ``(pair, rest)`` when the body is a product of exactly one
    factor that depends on both indices and is symmetric under their swap,
    and factors that depend only on the bound index (or on none); else None.
    """
    free = definition.free
    limits = definition.limits
    if len(free) != 1 or len(limits) != 1:
        return None
    b, low, high = limits[0]
    if low != 1 or str(high) != "nc":
        return None
    f = free[0]
    body = definition.body
    factors = body.args if isinstance(body, sp.Mul) else (body,)
    pair = [x for x in factors if {f, b} <= x.atoms(sp.Idx)]
    rest = [x for x in factors if x not in pair]
    if len(pair) != 1 or any(f in x.atoms(sp.Idx) for x in rest):
        return None
    if body.has(sp.Piecewise) or any(
        isinstance(x, _LoopStart) for x in body.atoms(sp.Symbol)
    ):
        return None
    canon = _canonical_symmetric(pair[0], symmetric_params, sym_rank3)
    if _swap(canon, f, b, symmetric_params, sym_rank3) != canon:
        return None
    return pair[0], sp.Mul(*rest)


def _render_symv(definition: TempDef) -> list[str]:
    """Symmetric matrix-vector product reading the upper triangle only.

    For every column ``f``: an axpy into ``v(1:f-1)`` and a dot product for
    ``v(f)``, both vectorisable.
    """
    target = definition.target
    f = definition.free[0]
    b = definition.limits[0][0]
    pair, rest = cast(tuple[Any, Any], definition.symv)
    acc = definition.symv_acc
    base = target.base
    rest_f_expr = rest.xreplace({b: f})
    up_to_f = _print_expr(pair * rest_f_expr)  # P(b,f) R(f), b < f
    up_to_b = _print_expr(pair * rest)  # P(b,f) R(b)
    diag = _print_expr(pair.xreplace({b: f}) * rest_f_expr)
    lines = [
        f"do {f} = 1, nc",
        f"{_INDENT}{_print_expr(sp.Indexed(base, f))} = 0.0d0",
        "end do",
    ]
    lines.append(f"do {f} = 1, nc")
    lines.append(f"{_INDENT}{acc} = 0.0d0")
    lines.append(f"{_INDENT}do {b} = 1, {f} - 1")
    vb = _print_expr(sp.Indexed(base, b))
    lines.append(f"{_INDENT * 2}{vb} = {vb} + {up_to_f}")
    lines.append(f"{_INDENT * 2}{acc} = {acc} + {up_to_b}")
    lines.append(f"{_INDENT}end do")
    vf = _print_expr(sp.Indexed(base, f))
    lines.append(f"{_INDENT}{vf} = {vf} + {acc} + {diag}")
    lines.append("end do")
    return lines


def _canonical_symmetric(
    expr: sp.Basic,
    symmetric_params: Collection[str],
    sym_rank3: bool = False,
) -> sp.Basic:
    """Sort the subscripts of every symmetric parameter occurrence.

    For a parameter declared ``symmetric`` (``k(i,j) == k(j,i)``) the
    occurrence ``k(q,p)`` is rewritten as ``k(p,q)``.  The rewrite is exact
    and makes transposed occurrences syntactically identical, which lets
    the common sub-expression passes merge them and lets
    :func:`_fold_symmetric_sums` recognise symmetric summation bodies.

    Parameters
    ----------
    expr : sympy.Expr
        Expression to canonicalise.
    symmetric_params : iterable of str
        Names of the parameters declared ``symmetric``.
    sym_rank3 : bool, optional
        Also sort every subscript of symmetric parameters of rank three or
        more (fully symmetric tensors).  Off by default.
    """
    if not symmetric_params:
        return expr
    names = set(symmetric_params)
    replacements: dict[sp.Basic, sp.Basic] = {}
    for indexed in expr.atoms(sp.Indexed):
        if str(indexed.base) not in names:
            continue
        if len(indexed.indices) >= 3 and sym_rank3:
            # fully symmetric tensor (k_ijk): sort every subscript
            ordered = sorted(indexed.indices, key=sp.default_sort_key)
            if list(ordered) != list(indexed.indices):
                replacements[indexed] = sp.Indexed(indexed.base, *ordered)
            continue
        if len(indexed.indices) != 2:
            continue
        first, second = indexed.indices
        ordered = sorted((first, second), key=sp.default_sort_key)
        if list(ordered) != [first, second]:
            replacements[indexed] = sp.Indexed(indexed.base, *ordered)
    return expr.xreplace(replacements) if replacements else expr


def _swap(
    expr: sp.Basic,
    a: sp.Idx,
    b: sp.Idx,
    symmetric_params: Collection[str],
    sym_rank3: bool = False,
) -> sp.Basic:
    """Swap the indices ``a`` and ``b`` and re-canonicalise.

    ``sym_rank3`` keeps a default only because the chain-rule stage
    (``chain.py``) calls this helper before emission, without the option.
    """
    swapped = expr.xreplace({a: b, b: a})
    return _canonical_symmetric(swapped, symmetric_params, sym_rank3)


def _fold_symmetric_sums(
    expr: sp.Basic, symmetric_params: Collection[str], *, sym_rank3: bool
) -> sp.Basic:
    """Fold double summations whose body is symmetric in its two indices.

    ``sum_a sum_b F(a,b)`` with ``F(a,b) == F(b,a)`` is rewritten as
    ``sum_a F(a,a) + 2 sum_a sum_{b>a} F(a,b)``, which halves the number of
    evaluations of the body.  The symmetry is checked symbolically after
    :func:`_canonical_symmetric`; when it cannot be proven the summation is
    left untouched, so the rewrite is always exact.

    The folded double sum is stored with its limits in *loop order*
    (outer loop first), ``Sum(F, (a, 1, nc), (b, a + 1, nc))``, which is the
    order in which the code generator emits the ``do`` loops.  The object is
    only used for code generation, never evaluated by SymPy.
    """
    replacements: dict[sp.Basic, sp.Basic] = {}
    for total in expr.atoms(sp.Sum):
        if len(total.limits) != 2:
            continue
        (a, low_a, high_a), (b, low_b, high_b) = total.limits
        if low_a != 1 or low_b != 1 or high_a != high_b:
            continue
        body = total.function
        canon = _canonical_symmetric(body, symmetric_params, sym_rank3)
        if _swap(canon, a, b, symmetric_params, sym_rank3) != canon:
            continue
        outer, inner = sorted((a, b), key=sp.default_sort_key)
        diagonal = canon.xreplace({inner: outer})
        replacements[total] = sp.Sum(
            diagonal, (outer, 1, high_a)
        ) + sp.Integer(2) * sp.Sum(
            canon, (outer, 1, high_a), (inner, _LoopStart(outer), high_a)
        )
    return expr.xreplace(replacements) if replacements else expr
