"""Loop fusion and table rewrites for the Fortran code generator."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

import sympy as sp

from thermodiff.codegen.common import (
    _LoopStart,
    _calls_libm,
    _free_indices,
    _is_expensive,
    _new_shared_temp,
    _order_free,
    _temp_name,
)
from thermodiff.codegen.ir import (
    ComponentFusedMap,
    FusedNest,
    OuterFusedNest,
    SiblingNest,
    TempDef,
)
from thermodiff.codegen.symmetry import _alpha_key, _canonical_symmetric


class _NestContext(NamedTuple):
    """Inputs of the nesting of :func:`_nest_double_sums`.

    Attributes
    ----------
    indexes : sequence
        Indices of the model, to order the free indices of a sum.
    symmetric_params : tuple of str
        Symmetric parameters, whose subscripts are canonicalised.
    sym_rank3 : bool
        Whether rank-3 symmetric parameters are fully symmetric.
    general : bool
        Nest sums over three or more indices too.
    """

    indexes: Sequence
    symmetric_params: tuple
    sym_rank3: bool
    general: bool


class _NestChoice(NamedTuple):
    """Which inner sums are shared and how often each one appears.

    Attributes
    ----------
    vectors : set
        Keys of the vector sums some output needs anyway.
    shared : set
        Keys of the inner sums worth a vector.
    counts : dict
        Number of double sums that nest onto each inner-sum key.
    cost : bool
        Nest by cost when no inner sum is shared.
    """

    vectors: set
    shared: set
    counts: dict
    cost: bool


def _sum_key(total, context):
    """Return the canonical key of a sum, free indices renamed by order."""
    free = _order_free(_free_indices(total), context.indexes)
    mapping = {index: sp.Idx(f"_free{pos}") for pos, index in enumerate(free)}
    renamed = _alpha_key(total.xreplace(mapping))
    if context.symmetric_params:
        renamed = _canonical_symmetric(
            renamed, context.symmetric_params, context.sym_rank3
        )
    return renamed


def _split_product(total, outer_factors):
    """Return the numeric factors and the rest of the product of ``total``."""
    numbers = [f for f in total.function.args if f.is_Number]
    rest = [
        f
        for f in total.function.args
        if f not in outer_factors and not f.is_Number
    ]
    return numbers, rest


def _orientations_general(total):
    """Nestings of a sum over two or more indices.

    One bound index goes to an inner sum (rank 1 or 2 in its free indices),
    the rest stay outside.  Returns ``(outer limits, outer factors, inner
    sum, None)`` for each nesting.
    """
    if len(total.limits) < 2 or not isinstance(total.function, sp.Mul):
        return []
    ranges = {(lim[1], lim[2]) for lim in total.limits}
    if len(ranges) != 1:
        return []
    result = []
    for pos, (bound, low, high) in enumerate(total.limits):
        outer_factors = [
            f
            for f in total.function.args
            if bound not in f.atoms(sp.Idx) and f.atoms(sp.Idx)
        ]
        numbers, rest = _split_product(total, outer_factors)
        if not outer_factors or not rest:
            continue
        inner = sp.Sum(sp.Mul(*rest), (bound, low, high))
        if len(_free_indices(inner)) > 2:
            continue
        outer_limits = [lim for k, lim in enumerate(total.limits) if k != pos]
        result.append((outer_limits, numbers + outer_factors, inner, None))
    return result


def _orientations_pair(total):
    """Nestings of a double sum: ``(outer index, factors, inner, limits)``."""
    if len(total.limits) != 2 or not isinstance(total.function, sp.Mul):
        return []
    (first, low_a, high_a), (second, low_b, high_b) = total.limits
    if (low_a, high_a) != (low_b, high_b):
        return []
    result = []
    for outer, inner_index, lims in (
        (first, second, (low_a, high_a)),
        (second, first, (low_b, high_b)),
    ):
        outer_factors = [
            f
            for f in total.function.args
            if f.atoms(sp.Idx) <= {outer} and f.atoms(sp.Idx)
        ]
        numbers, rest = _split_product(total, outer_factors)
        if not outer_factors or not rest:
            continue
        inner = sp.Sum(sp.Mul(*rest), (inner_index, *lims))
        result.append((outer, numbers + outer_factors, inner, lims))
    return result


def _orientations(total, general):
    """Return the nestings of ``total`` (general or double-sum form)."""
    if general:
        return _orientations_general(total)
    return _orientations_pair(total)


def _vector_keys(staged, context):
    """Return the keys of the vector sums the outputs already contain."""
    vectors = set()
    for _, _, expr in staged:
        for total in expr.atoms(sp.Sum):
            if len(total.limits) == 1 and _free_indices(total):
                vectors.add(_sum_key(total, context))
    return vectors


def _inner_key_counts(staged, context):
    """Count, per inner-sum key, the sums that can nest onto it."""
    counts = {}
    for _, _, expr in staged:
        for total in expr.atoms(sp.Sum):
            options = _orientations(total, context.general)
            for key in {_sum_key(o[2], context) for o in options}:
                counts[key] = counts.get(key, 0) + 1
    return counts


def _nest_by_cost(total, context):
    """Nest by cost: fewest free indices inside, most factors outside."""
    options = sorted(
        _orientations(total, context.general),
        key=lambda o: (len(_free_indices(o[2])), -len(o[1])),
    )
    if not options:
        return None
    outer_limits, outer_factors, inner, _ = options[0]
    return sp.Sum(sp.Mul(*outer_factors) * inner, *outer_limits)


def _nest_sum(total, context, choice):
    """Return ``total`` nested onto a shared inner sum, or ``None``."""
    options = [
        o
        for o in _orientations(total, context.general)
        if _sum_key(o[2], context) in choice.shared
    ]
    if not options and choice.cost and context.general:
        if len(total.limits) >= 3:
            nested = _nest_by_cost(total, context)
            if nested is not None:
                return nested
    if not options:
        return None
    # prefer an inner sum that some output needs anyway
    options.sort(
        key=lambda o: (
            _sum_key(o[2], context) not in choice.vectors,
            -choice.counts.get(_sum_key(o[2], context), 0),
        )
    )
    outer, outer_factors, inner, lims = options[0]
    if context.general:
        return sp.Sum(sp.Mul(*outer_factors) * inner, *outer)
    return sp.Sum(sp.Mul(*outer_factors) * inner, (outer, *lims))


def _nest_expression(expr, context, choice):
    """Nest the sums of ``expr`` repeatedly (up to three rounds if general)."""
    for _ in range(3 if context.general else 1):
        replacements = {}
        for total in expr.atoms(sp.Sum):
            nested = _nest_sum(total, context, choice)
            if nested is not None:
                replacements[total] = nested
        if not replacements:
            break
        expr = expr.xreplace(replacements)
    return expr


def _nest_double_sums(
    staged,
    indexes,
    symmetric_params=(),
    shared_inner=False,
    general=False,
    cost=False,
    *,
    sym_rank3,
):
    """Rewrite double sums as dot products with existing vector sums.

    ``sum_k sum_l n(k)*n(l)*m(k,l)`` equals ``sum_k n(k)*v(k)`` with
    ``v(k) = sum_l n(l)*m(k,l)``.  When some output already needs that
    vector -- the composition gradient does -- the nested form lets
    :func:`_share_sums` compute ``v`` once and the double sum becomes an
    O(nc) dot product.  Double sums whose inner sum matches no vector sum
    are left untouched (they may still be folded into a triangle).

    With ``cost`` (and ``general``) a sum over three or more indices is
    nested even when no other output shares its inner sum: the inner sum
    becomes a rank-2 table (computed once, triangle when symmetric) and
    the factors of the outer indices leave the innermost loop.
    """
    from thermodiff.chain import factor_sums

    context = _NestContext(indexes, symmetric_params, sym_rank3, general)
    # factors that do not depend on the bound indices leave the sums first
    staged = [
        (attr, out_name, factor_sums(expr)) for attr, out_name, expr in staged
    ]
    vectors = _vector_keys(staged, context)
    # inner sums shared by several double sums are worth a vector too
    # (e.g. the temperature derivatives of sum_kl n_k n_l s_k s_l K_kl)
    counts = _inner_key_counts(staged, context)
    shared = vectors | (
        {k for k, c in counts.items() if c >= 2} if shared_inner else set()
    )
    if not shared and not (cost and general):
        return staged
    choice = _NestChoice(vectors, shared, counts, cost)
    return [
        (attr, out_name, _nest_expression(expr, context, choice))
        for attr, out_name, expr in staged
    ]


class _FuseSettings(NamedTuple):
    """Switches of :func:`_fuse_temporaries`.

    Attributes
    ----------
    dedup : bool
        Count equal summations once when weighing the reads of a table.
    guard : bool
        Keep a costly table read by two or more different summations.
    keep_libm : bool
        Keep a pair table that calls libm and is read as a matrix-vector.
    """

    dedup: bool
    guard: bool
    keep_libm: bool


def _temp_refs(expr, name) -> list:
    """Return the reads of the array ``name`` inside ``expr``."""
    return [x for x in expr.atoms(sp.Indexed) if str(x.base) == name]


def _read_outside_sums(expr, name) -> bool:
    """Whether ``expr`` reads the array ``name`` outside every summation."""
    if isinstance(expr, sp.Sum):
        return False
    if isinstance(expr, sp.Indexed):
        return str(expr.base) == name
    return any(
        _read_outside_sums(arg, name) for arg in getattr(expr, "args", ())
    )


def _free_alpha_key(total):
    """Return the alpha key of a summation with its free indices renamed."""
    free = sorted(_free_indices(total), key=str)
    mapping = {index: sp.Idx(f"_free{pos}") for pos, index in enumerate(free)}
    return _alpha_key(total.xreplace(mapping))


def _direct_reads(total, name) -> list:
    """Return the reads of ``name`` in a summation outside its inner sums."""
    body = total.function
    return [
        x
        for x in _temp_refs(body, name)
        if not any(
            x in inner.atoms(sp.Indexed) for inner in body.atoms(sp.Sum)
        )
    ]


def _read_weight(exprs, name, dedup) -> float:
    """Return how many times each element of ``name`` is read.

    Diagonal reads count 0, a triangular (folded) summation 1/2 and a full
    summation 1.  With ``dedup`` equal summations count once.
    """
    weight = 0.0
    seen_keys = set()
    for expr in exprs:
        for total in expr.atoms(sp.Sum):
            key = _free_alpha_key(total)
            if dedup and key in seen_keys:
                continue
            seen_keys.add(key)
            inner_refs = _temp_refs(total.function, name)
            if not inner_refs or total.function.atoms(sp.Sum):
                continue
            if all(len(set(x.indices)) < len(x.indices) for x in inner_refs):
                continue
            folded = any(
                isinstance(lim[1], _LoopStart) for lim in total.limits
            )
            weight += 0.5 if folded else 1.0
    return weight


def _direct_sum_count(exprs, name) -> int:
    """Count the distinct summations that read ``name`` in their own body.

    Distinct means by alpha key; nested summations are included.
    """
    keys = set()
    for expr in exprs:
        for total in expr.atoms(sp.Sum):
            if _direct_reads(total, name):
                keys.add(_free_alpha_key(total))
    return len(keys)


def _has_pair_libm(body, free) -> bool:
    """Whether ``body`` has a libm call that depends on both pair indices.

    A separable ``sqrt(e_i)*sqrt(e_j)`` is hoisted into vectors later
    anyway, so it does not count.
    """
    both = set(free)
    for sub in sp.preorder_traversal(body):
        call = (
            isinstance(sub, sp.Function) and not isinstance(sub, sp.Piecewise)
        ) or (isinstance(sub, sp.Pow) and not sub.exp.is_Integer)
        if call and both <= {
            idx for x in sub.atoms(sp.Indexed) for idx in x.indices
        }:
            return True
    return False


def _only_vector_reads(exprs, name) -> bool:
    """Whether every summation reading ``name`` directly keeps a free index.

    A table read by scalar double sums is better inlined
    (reportes/brechas/ronda2-pcsaft.md, T1).
    """
    found = False
    for expr in exprs:
        for total in expr.atoms(sp.Sum):
            if not _direct_reads(total, name):
                continue
            found = True
            if not _free_indices(total):
                return False
    return found


def _inline_read(expr, name, body, free):
    """Substitute the body of the table ``name`` into its reads in ``expr``."""
    mapping = {
        x: body.xreplace(dict(zip(free, x.indices)))
        for x in _temp_refs(expr, name)
    }
    return expr.xreplace(mapping) if mapping else expr


def _keeps_for_cost(definition, name, exprs, settings) -> bool:
    """Whether the cost of the table ``name`` argues against inlining it."""
    body = definition.body
    expensive = _is_expensive(body)
    ops = sp.count_ops(body)
    trivial = not expensive and ops <= 3
    if not trivial and any(_read_outside_sums(e, name) for e in exprs):
        return True
    if (
        settings.keep_libm
        and _has_pair_libm(body, definition.free)
        and _direct_sum_count(exprs, name) >= 1
        and _only_vector_reads(exprs, name)
    ):
        return True
    if settings.guard and expensive and _direct_sum_count(exprs, name) >= 2:
        # an expensive body (exp, log, root, ...) read by two or more
        # different summations, at any nesting depth, is computed once
        return True
    cheap = not expensive and ops <= 8
    # costly body read more than once per element: keep it
    return not cheap and _read_weight(exprs, name, settings.dedup) > 1


def _try_inline_temporary(definition, array_defs, exprs, settings):
    """Inline one rank-2 temporary if its reads allow it.

    Parameters
    ----------
    definition : TempDef
        Candidate; only a rank-2 table without limits qualifies.
    array_defs : list of TempDef
        The temporaries; the candidate is removed from it (in place) and
        the bodies of the ones that read it are rewritten.
    exprs : list of sympy expressions
        Expressions to rewrite.
    settings : _FuseSettings
        Switches of the pass.

    Returns
    -------
    tuple
        ``(exprs, name)``: the rewritten expressions and the removed array
        name, or ``(exprs, None)`` when the temporary stays.
    """
    free = definition.free
    if len(free) != 2 or definition.limits:
        return exprs, None
    name = _temp_name(definition.target)
    if _keeps_for_cost(definition, name, exprs, settings):
        return exprs, None
    readers = [
        d
        for d in array_defs
        if d is not definition and _temp_refs(d.body, name)
    ]
    if any(
        len(set(x.indices)) == len(x.indices)
        for d in readers
        for x in _temp_refs(d.body, name)
    ):
        # another temporary reads it off the diagonal: keep it
        return exprs, None
    body = definition.body
    exprs = [_inline_read(e, name, body, free) for e in exprs]
    for reader in readers:
        reader.body = _inline_read(reader.body, name, body, free)
    array_defs.remove(definition)
    return exprs, name


def _fuse_temporaries(
    array_defs, exprs, dedup=False, guard=False, keep_libm=False
):
    """Inline cheap rank-2 array temporaries read only inside summations.

    The indexed CSE materialises ``t(p,q) = n(q)*m(p,q)`` when the product
    appears several times; when every use is inside the body of a
    summation, storing the matrix and reading it back costs more than the
    multiplication it saves, and it hides from :func:`_share_sums` that
    ``sum_k n(k)*m(i,k)`` and ``sum_k n(k)*m(k,l)`` are the same vector.
    Such a temporary is removed and its body substituted into the sums.
    Runs before :func:`_share_sums`.

    With ``keep_libm`` a rank-2 temporary that calls a libm routine (``exp``,
    ``log``, ``sqrt``, a real power) is never inlined into a summation, even
    when a single summation reads it: inside the reduction the vector libm call
    forces the accumulators to memory and stops unrolling, while a separate map
    loop is vectorised (reportes/brechas/ronda2-ge-cmr.md, UNIQUAC G0).
    """
    settings = _FuseSettings(dedup, guard, keep_libm)
    exprs = list(exprs)
    removed = set()
    changed = True
    while changed:
        changed = False
        for definition in list(array_defs):
            exprs, name = _try_inline_temporary(
                definition, array_defs, exprs, settings
            )
            if name is not None:
                removed.add(name)
                changed = True
    return array_defs, exprs, removed


def _prefers_column_order(definition):
    """Whether a rank-1 sum or a rank-2 table reads its arrays by rows.

    In ``t(f) = sum_b M(f, b) x(b)`` the natural nest (``f`` outer, ``b``
    inner) walks ``M`` along a row, against Fortran's column-major layout;
    swapping the loops (``b`` outer, ``f`` inner) turns it into a sequence
    of contiguous axpy updates.  Same for a rank-2 table ``t(p, q)`` whose
    body reads ``X(p, q)``: ``q`` should be the outer loop.
    """
    free = definition.free
    limits = definition.limits
    body = definition.body
    if definition.mirror or body.has(sp.Piecewise):
        return False
    if any(isinstance(x, _LoopStart) for x in body.atoms(sp.Symbol)):
        return False
    if len(free) == 1 and len(limits) == 1:
        f, b = free[0], limits[0][0]
        if limits[0][1] != 1 or str(limits[0][2]) != "nc":
            return False
        rows = sum(
            1
            for x in body.atoms(sp.Indexed)
            if len(x.indices) == 2 and x.indices == (f, b)
        )
        cols = sum(
            1
            for x in body.atoms(sp.Indexed)
            if len(x.indices) == 2 and x.indices == (b, f)
        )
        return rows > cols
    if len(free) == 2 and not limits:
        p_, q_ = free
        rows = sum(
            1
            for x in body.atoms(sp.Indexed)
            if len(x.indices) == 2 and x.indices == (q_, p_)
        )
        cols = sum(
            1
            for x in body.atoms(sp.Indexed)
            if len(x.indices) == 2 and x.indices == (p_, q_)
        )
        # the target t(p, q) itself is written with p fastest when q is outer
        return cols + 1 > rows
    return False


def _prefers_column_sum(definition, symmetric_params=()):
    """Whether a rank-2 sum ``t(p, q) = sum_b B`` reads its tables by rows.

    ``sum_b G(p, b) H(q, b) w(b)`` (the subscript order of ``table_layout``)
    walks the non-symmetric arrays with the bound index last, a stride of
    ``nc`` in the natural nest (``b`` innermost).  The ``jki`` order
    (``q`` outer, ``b`` middle, ``p`` inner) reads them contiguously.
    Symmetric arrays are not counted: ``sym_perm`` reorders their subscripts.
    """
    free = definition.free
    limits = definition.limits
    if len(free) != 2 or len(limits) != 1 or definition.mirror:
        return False
    b, low, high = limits[0]
    if low != 1 or str(high) != "nc":
        return False
    body = definition.body
    if body.has(sp.Piecewise) or any(
        isinstance(x, _LoopStart) for x in body.atoms(sp.Symbol)
    ):
        return False
    p_, q_ = free
    names = set(symmetric_params)
    last = first = 0
    for x in body.atoms(sp.Indexed):
        if len(x.indices) != 2 or str(x.base) in names:
            continue
        if tuple(x.indices) in ((p_, b), (q_, b)):
            last += 1
        elif tuple(x.indices) in ((b, p_), (b, q_)):
            first += 1
    return last > first


def _fuse_loop_nests(definitions, namer):
    """Merge consecutive vector sums that share their loop structure.

    ``v1(f) = sum_b B1``, ``v2(f) = sum_b B2``, ... computed one after the
    other become one nest with several accumulators, so the arrays they
    share (the same matrix with different weights) are read once.  Only
    consecutive definitions with one free index, one full-range bound
    index, no special rendering, and no dependence on each other are
    merged.
    """

    def simple(d):
        if len(d.free) != 1 or len(d.limits) != 1:
            return False
        if d.mirror or d.mirror_sum or d.symv:
            return False
        b, low, high = d.limits[0]
        if low != 1 or str(high) != "nc":
            return False
        body = d.body
        return not body.has(sp.Piecewise) and not any(
            isinstance(x, _LoopStart) for x in body.atoms(sp.Symbol)
        )

    out = []
    group = []

    def flush():
        if len(group) <= 1:
            out.extend(group)
        else:
            out.append(
                FusedNest(
                    target=group[0].target,
                    free=group[0].free,
                    limits=group[0].limits,
                    body=group[0].body,
                    fused=list(group),
                )
            )
        group.clear()

    for d in definitions:
        if simple(d) and group:
            first = group[0]
            f0, b0 = first.free[0], first.limits[0][0]
            renamed = d.copy()
            mapping = {d.free[0]: f0, d.limits[0][0]: b0}
            renamed.target = d.target.xreplace(mapping)
            renamed.free = [f0]
            renamed.limits = [(b0, 1, first.limits[0][2])]
            renamed.body = d.body.xreplace(mapping)
            names = {_temp_name(g.target) for g in group}
            depends = any(
                str(x.base) in names for x in renamed.body.atoms(sp.Indexed)
            )
            same_order = bool(d.colmajor) == bool(first.colmajor)
            if not depends and same_order:
                group.append(renamed)
                continue
            flush()
        if simple(d):
            group.append(d)
        else:
            flush()
            out.append(d)
    flush()
    return out


def _loop_order(definition):
    """Loop variables of a rendered definition, outermost first (None if unknown)."""
    free = list(definition.free)
    bound = [lim[0] for lim in definition.limits]
    if (
        definition.symv
        or definition.colmajor
        or isinstance(definition, ComponentFusedMap)
    ):
        return None
    if definition.colsum:
        return [free[1], bound[0], free[0]]
    if isinstance(definition, SiblingNest):
        return [free[1], free[0]] + bound
    if isinstance(definition, FusedNest):
        first = definition.fused[0]
        f, b = first.free[0], first.limits[0][0]
        return [b, f] if first.colmajor else [f, b]
    if definition.mirror_sum and len(free) == 2 and bound:
        return [free[1], free[0]] + bound
    return free + bound


def _permute_symmetric_subscripts(definitions, symmetric_params):
    """Order the subscripts of symmetric parameters by loop depth.

    A parameter declared ``symmetric`` has the same value under any
    permutation of its subscripts, so ``k(a, b, m)`` inside a nest whose
    innermost loop is ``m`` can be read as ``k(m, a, b)``: contiguous in
    Fortran's column-major layout instead of a stride of ``nc**2``.  The
    rewrite is exact; it is applied last, after every pass that relies on
    the canonical (sorted) subscript order.
    """
    names = set(symmetric_params)
    if not names:
        return definitions

    def permute(body, order):
        depth = {idx: k for k, idx in enumerate(order)}
        replacements = {}
        for indexed in body.atoms(sp.Indexed):
            if str(indexed.base) not in names or len(indexed.indices) < 2:
                continue
            if not all(idx in depth for idx in indexed.indices):
                continue
            ordered = sorted(indexed.indices, key=lambda idx: -depth[idx])
            if list(ordered) != list(indexed.indices):
                replacements[indexed] = sp.Indexed(indexed.base, *ordered)
        return body.xreplace(replacements) if replacements else body

    for definition in definitions:
        order = _loop_order(definition)
        if order is None:
            continue
        if isinstance(definition, (FusedNest, SiblingNest)):
            members = (
                definition.fused
                if isinstance(definition, FusedNest)
                else definition.sibling
            )
            for member in members:
                member.body = permute(member.body, order)
        else:
            definition.body = permute(definition.body, order)
    return definitions


def _is_fusable_map(definition) -> bool:
    """Whether ``definition`` is a plain per-component map over ``1..nc``."""
    if len(definition.free) != 1 or definition.limits:
        return False
    if (
        definition.mirror
        or definition.mirror_sum
        or definition.symv
        or isinstance(definition, FusedNest)
        or definition.colmajor
    ):
        return False
    body = definition.body
    return (
        not body.has(sp.Sum)
        and not body.has(sp.Piecewise)
        and not any(isinstance(x, _LoopStart) for x in body.atoms(sp.Symbol))
    )


def _reads_group(definition, group) -> bool:
    """Whether the body of ``definition`` reads an array defined in group."""
    names = {_temp_name(member.target) for member in group}
    return any(str(x.base) in names for x in definition.body.atoms(sp.Indexed))


def _is_independent_scalar(definition, group) -> bool:
    """Whether a non-map definition may move ahead of the open group."""
    scalar = (
        not definition.free
        and not definition.limits
        and not (
            definition.mirror
            or definition.mirror_sum
            or definition.symv
            or isinstance(definition, FusedNest)
        )
    )
    return bool(group) and scalar and not _reads_group(definition, group)


def _rename_map(definition, index):
    """Return a copy of a map with its free index renamed to ``index``."""
    renamed = definition.copy()
    mapping = {definition.free[0]: index}
    renamed.target = definition.target.xreplace(mapping)
    renamed.free = [index]
    renamed.body = definition.body.xreplace(mapping)
    return renamed


def _reads_group_at_element(renamed, group, index) -> bool:
    """Whether every read of a group member in ``renamed`` is at ``index``."""
    names = {_temp_name(member.target) for member in group}
    return all(
        x.indices == (index,)
        for x in renamed.body.atoms(sp.Indexed)
        if str(x.base) in names
    )


def _cse_symbols(namer):
    """Yield the names of the loop-local scalars, drawn lazily."""
    while True:
        yield sp.Symbol(str(namer("t")))


def _order_steps(pending, needed):
    """Order the ``(kind, symbol, value)`` items by their dependencies."""
    known = set()
    steps = []
    while pending:
        for item in pending:
            deps = item[2].free_symbols & needed
            if deps <= known:
                steps.append(item)
                known.add(item[1])
                pending.remove(item)
                break
        else:  # pragma: no cover - cse output is acyclic
            raise RuntimeError("component fusion: cyclic dependencies")
    return steps


def _fuse_map_group(group, namer):
    """Fuse a group of two or more maps into a :class:`ComponentFusedMap`.

    Member values read by later members become placeholders, so the
    elimination never moves a read of ``cmp0(i)`` before its assignment.
    Names for the loop-local scalars are drawn from ``namer`` lazily.
    """
    idx = group[0].free[0]
    holder_of = {
        _temp_name(d.target): sp.Dummy(f"m{k}") for k, d in enumerate(group)
    }
    holders = {d.target: holder_of[_temp_name(d.target)] for d in group}
    bodies = [_to_holders(d.body, holder_of, idx) for d in group]
    replacements, reduced = sp.cse(
        bodies, symbols=_cse_symbols(namer), optimizations=None
    )
    back = {h: t for t, h in holders.items()}
    pending = [("local", sym, value) for sym, value in replacements] + [
        ("member", holders[d.target], b) for d, b in zip(group, reduced)
    ]
    needed = {sym for sym, _ in replacements} | set(back)
    ordered = [
        (kind, back.get(sym, sym), value.xreplace(back))
        for kind, sym, value in _order_steps(pending, needed)
    ]
    return ComponentFusedMap(
        target=group[0].target,
        free=[idx],
        limits=[],
        body=group[0].body,
        steps=ordered,
        members=[(d.target, d.body) for d in group],
    )


def _to_holders(expr, holder_of, index):
    """Replace the reads of group members at ``index`` by placeholders."""
    return expr.xreplace(
        {
            x: holder_of[str(x.base)]
            for x in expr.atoms(sp.Indexed)
            if str(x.base) in holder_of and x.indices == (index,)
        }
    )


def _flush_map_group(out, group, namer) -> None:
    """Append the open group to ``out`` (fused if it has several maps)."""
    if len(group) <= 1:
        out.extend(group)
    else:
        out.append(_fuse_map_group(group, namer))
    group.clear()


def _fuse_component_maps(definitions, namer):
    """Fuse consecutive per-component maps into one loop with per-element CSE.

    ``cmp0(i) = f(i)``, ``cmp0_T(i) = f'(i)``, ``cmp0_TT(i) = f''(i)``, ...
    are emitted as separate loops, each re-evaluating the transcendental
    sub-expressions they share (``sqrt(T/Tc(i))``, ``exp(-3*eps(i)/T)``).
    Consecutive definitions with one free index over ``1..nc``, no
    summation, and that read earlier members only at the same element are
    merged into a single loop; a common sub-expression elimination over the
    bodies of one element computes every shared sub-expression once, into
    loop-local scalars.
    """
    out = []
    group = []
    for definition in definitions:
        if not _is_fusable_map(definition):
            if _is_independent_scalar(definition, group):
                # an independent scalar between two maps moves ahead of
                # the group
                out.append(definition)
                continue
            _flush_map_group(out, group, namer)
            out.append(definition)
            continue
        if group:
            index = group[0].free[0]
            renamed = _rename_map(definition, index)
            if _reads_group_at_element(renamed, group, index):
                group.append(renamed)
                continue
            _flush_map_group(out, group, namer)
        group.append(definition)
    _flush_map_group(out, group, namer)
    return out


_MAP = "map"
_REDUCTION = "red"


def _is_plain_outer_candidate(definition):
    """Whether ``definition`` has no annotation that blocks outer fusion.

    Parameters
    ----------
    definition : TempDef
        Definition to test.

    Returns
    -------
    bool
        ``True`` without ``Piecewise``, folded-loop bounds, mirror, symv,
        fused-nest or column-major annotations.
    """
    body = definition.body
    return (
        not body.has(sp.Piecewise)
        and not any(
            isinstance(symbol, _LoopStart) for symbol in body.atoms(sp.Symbol)
        )
        and not (
            definition.mirror
            or definition.mirror_sum
            or definition.symv
            or isinstance(definition, FusedNest)
            or definition.colmajor
        )
    )


def _outer_kind(definition):
    """Classify a definition for outer-loop fusion.

    Parameters
    ----------
    definition : TempDef
        Definition to classify.

    Returns
    -------
    str or None
        ``"map"`` for a per-component map, ``"red"`` for a scalar reduction
        whose outermost loop runs over ``1..nc``, ``None`` if not fusable.
    """
    if isinstance(definition, ComponentFusedMap):
        return _MAP
    if not _is_plain_outer_candidate(definition):
        return None
    if (
        len(definition.free) == 1
        and not definition.limits
        and not definition.body.has(sp.Sum)
    ):
        return _MAP
    if not definition.free and definition.limits:
        _index, low, high = definition.limits[0]
        if low == 1 and str(high) == "nc":
            return _REDUCTION
    return None


def _outer_index(definition):
    """Return the index of the loop that outer fusion would share."""
    if _outer_kind(definition) == _MAP:
        return definition.free[0]
    return definition.limits[0][0]


def _is_libm_map(definition):
    """Whether ``definition`` is a map that calls a libm routine."""
    if _outer_kind(definition) != _MAP:
        return False
    if isinstance(definition, ComponentFusedMap):
        return any(_calls_libm(value) for _k, _t, value in definition.steps)
    return _calls_libm(definition.body)


def _is_double_reduction(definition):
    """Whether ``definition`` is a reduction with an inner loop."""
    return _outer_kind(definition) == _REDUCTION and len(definition.limits) > 1


def _is_libm_reduction(definition, libm_reductions):
    """Whether ``definition`` is a single reduction calling libm."""
    return (
        libm_reductions
        and _outer_kind(definition) == _REDUCTION
        and len(definition.limits) == 1
        and _calls_libm(definition.body)
    )


def _is_libm_member(definition, libm_reductions):
    """Whether ``definition`` is a libm map or (optionally) reduction."""
    return _is_libm_map(definition) or _is_libm_reduction(
        definition, libm_reductions
    )


def _libm_clash(definition, group, libm_reductions):
    """Whether ``definition`` cannot join ``group`` under ``libm_guard``.

    Parameters
    ----------
    definition : TempDef
        Candidate member.
    group : list
        Current group members.
    libm_reductions : bool
        Also treat single reductions that call libm as libm members.

    Returns
    -------
    bool
        ``True`` when a double reduction would meet a libm member.
    """
    if _is_double_reduction(definition):
        return any(
            _is_libm_member(member, libm_reductions) for member in group
        )
    if _is_libm_member(definition, libm_reductions):
        return any(_is_double_reduction(member) for member in group)
    return False


def _produced_names(definition):
    """Return the base names of the arrays ``definition`` writes."""
    if isinstance(definition, ComponentFusedMap):
        return {_temp_name(target) for target, _ in definition.members}
    return {_temp_name(definition.target)}


def _outer_reads(definition):
    """Return ``(indexed reads, free symbols)`` of a definition's values."""
    if isinstance(definition, ComponentFusedMap):
        values = [value for _k, _t, value in definition.steps]
        return (
            [ref for value in values for ref in value.atoms(sp.Indexed)],
            set().union(*[value.free_symbols for value in values]),
        )
    return (
        list(definition.body.atoms(sp.Indexed)),
        definition.body.free_symbols,
    )


def _loop_start_renames(definition, old, new):
    """Return the substitution of ``old`` by ``new`` for a definition.

    The opaque lower bound ``old + 1`` of a triangular loop follows too.
    """
    mapping = {old: new}
    for limit in definition.limits:
        for bound in (limit[1], limit[2]):
            for symbol in sp.sympify(bound).atoms(sp.Symbol):
                if isinstance(symbol, _LoopStart) and symbol.outer == old:
                    mapping[symbol] = _LoopStart(new)
    return mapping


def _rename_outer_member(definition, old, new):
    """Return ``definition`` with the outer index ``old`` renamed ``new``.

    Parameters
    ----------
    definition : TempDef
        Candidate member.
    old, new : sympy.Idx
        Current and group outer index.

    Returns
    -------
    TempDef
        ``definition`` itself if ``old == new``, else a shallow copy.
    """
    if old == new:
        return definition
    mapping = _loop_start_renames(definition, old, new)
    renamed = definition.copy()
    if isinstance(renamed, ComponentFusedMap):
        renamed.steps = [
            (kind, target.xreplace(mapping), value.xreplace(mapping))
            for kind, target, value in renamed.steps
        ]
        renamed.free = [new]
    elif _outer_kind(renamed) == _MAP:
        renamed.target = renamed.target.xreplace(mapping)
        renamed.body = renamed.body.xreplace(mapping)
        renamed.free = [new]
    else:
        renamed.body = renamed.body.xreplace(mapping)
        renamed.limits = [
            (
                (
                    bound.xreplace(mapping)
                    if hasattr(bound, "xreplace")
                    else bound
                ),
                sp.sympify(low).xreplace(mapping),
                sp.sympify(high).xreplace(mapping),
            )
            for bound, low, high in renamed.limits
        ]
    return renamed


class _OuterGroups:
    """Output list and the group of members being fused.

    Attributes
    ----------
    out : list
        Definitions already emitted, in order.
    group : list
        Members of the open group; cleared (same list object) on flush.
    """

    def __init__(self):
        self.out = []
        self.group = []

    def flush(self):
        """Emit the open group, as an outer nest if it has 2+ members."""
        group = self.group
        if len(group) <= 1:
            self.out.extend(group)
        else:
            limits = []
            for member in group:
                limits += [lim for lim in member.limits if lim not in limits]
            self.out.append(
                OuterFusedNest(
                    target=group[0].target,
                    free=[],
                    limits=limits,
                    body=sp.S.Zero,
                    members=list(group),
                    outer=_outer_index(group[0]),
                )
            )
        group.clear()


def _joined_member(definition, kind, group, libm_guard, libm_reductions):
    """Return the renamed member that joins ``group``, or ``None``.

    Parameters
    ----------
    definition : TempDef
        Fusable definition (``kind`` is not ``None``).
    kind : str
        Result of ``_outer_kind(definition)``.
    group : list
        Non-empty open group.
    libm_guard, libm_reductions : bool
        The libm options of ``_fuse_outer_loops``.

    Returns
    -------
    TempDef or None
        ``None`` if the group must be flushed before ``definition``.
    """
    outer = _outer_index(group[0])
    inner = (
        [limit[0] for limit in definition.limits[1:]]
        if kind == _REDUCTION
        else []
    )
    if outer in inner:
        return None
    if libm_guard and _libm_clash(definition, group, libm_reductions):
        return None
    candidate = _rename_outer_member(
        definition, _outer_index(definition), outer
    )
    arrays = set().union(*[_produced_names(member) for member in group])
    accumulators = {
        _temp_name(member.target)
        for member in group
        if _outer_kind(member) == _REDUCTION
    }
    reads, symbols = _outer_reads(candidate)
    at_outer = all(
        ref.indices == (outer,) for ref in reads if str(ref.base) in arrays
    )
    if at_outer and not any(str(sym) in accumulators for sym in symbols):
        return candidate
    return None


def _moves_ahead_of_group(definition, group):
    """Whether an unfusable definition is an independent scalar.

    Such a scalar is emitted before the open group instead of ending it.
    """
    if not (
        group
        and not definition.free
        and not definition.limits
        and _is_plain_outer_candidate(definition)
    ):
        return False
    names = set().union(*[_produced_names(member) for member in group])
    reads_group = any(
        str(symbol) in names for symbol in definition.body.free_symbols
    ) or any(
        str(ref.base) in names for ref in definition.body.atoms(sp.Indexed)
    )
    return not reads_group


def _fuse_outer_loops(definitions, libm_guard=False, libm_reductions=False):
    """Fuse consecutive loop nests that share the outer loop ``1..nc``.

    Per-component maps (``x(i) = ...``, fused or not) and scalar reductions
    whose outermost loop runs over ``1..nc`` (single sums, triangular double
    sums) are merged into one outer loop, so the data are traversed once
    and the loop overhead is paid once.  A member may read an array
    produced earlier in the group only at the current outer index (it is
    already computed in this iteration), and never an accumulator of the
    group.  Independent scalar definitions in between move ahead.

    With ``libm_guard`` a double (triangular) reduction never shares the outer
    loop with a map that calls a libm routine: the inner loop would stop
    gfortran from vectorising the map, and ``pow``/``sqrt``/``exp`` would lose
    their SIMD versions (reportes/brechas/ronda2-rkpr.md, RKPR W0).  With
    ``libm_reductions`` the same holds for a single reduction whose body calls
    a libm routine (``sum_w (m(w) - 1) log(g(w))``).
    """
    groups = _OuterGroups()
    for definition in definitions:
        kind = _outer_kind(definition)
        if kind is None:
            if not _moves_ahead_of_group(definition, groups.group):
                groups.flush()
            groups.out.append(definition)
        elif not groups.group:
            groups.group.append(definition)
        else:
            member = _joined_member(
                definition, kind, groups.group, libm_guard, libm_reductions
            )
            if member is None:
                groups.flush()
                groups.group.append(definition)
            else:
                groups.group.append(member)
    groups.flush()
    return groups.out


_VECTOR_SLOT = sp.Symbol("__slot")


def _pair_readers(name, definitions, outputs):
    """Return the definitions and outputs that read the array ``name``."""
    readers = [
        d
        for d in definitions
        if any(str(x.base) == name for x in d.body.atoms(sp.Indexed))
    ]
    outs = [
        e
        for e in outputs
        if any(str(x.base) == name for x in e.atoms(sp.Indexed))
    ]
    return readers, outs


def _is_triangular(definition) -> bool:
    """Whether an inner limit starts at the outer loop index."""
    return any(
        isinstance(x, _LoopStart)
        for lim in definition.limits[1:]
        for x in sp.sympify(lim[1]).atoms(sp.Symbol)
    )


def _is_plain_pair_table(table) -> bool:
    """Whether ``table`` is a rank-2 table that may be inlined.

    It has two free indexes and an indexed target, no summation, no layout
    annotation, and a body without ``Sum``, ``Piecewise`` or libm calls.
    """
    if (
        len(table.free) != 2
        or table.limits
        or not isinstance(table.target, sp.Indexed)
    ):
        return False
    if (
        table.mirror_sum
        or table.symv
        or isinstance(table, FusedNest)
        or table.colmajor
        or table.colsum
    ):
        return False
    body = table.body
    return not (
        body.has(sp.Sum) or body.has(sp.Piecewise) or _calls_libm(body)
    )


def _read_only_by_reductions(table, readers, outs) -> bool:
    """Whether the readers of ``table`` are scalar reductions only.

    At most one of them may be a double reduction, and when the table is
    mirrored that one must be triangular.
    """
    if outs or not readers:
        return False
    if any(d.free or not d.limits or len(d.limits) > 2 for d in readers):
        return False
    double = [d for d in readers if len(d.limits) == 2]
    if len(double) > 1:
        return False
    return not (double and table.mirror and not _is_triangular(double[0]))


def _find_droppable_table(definitions, outputs):
    """Return ``(table, name, readers)`` of the last droppable table.

    Returns ``None`` when no table qualifies.
    """
    for table in reversed(definitions):
        if not _is_plain_pair_table(table):
            continue
        name = _temp_name(table.target)
        readers, outs = _pair_readers(name, definitions, outputs)
        if _read_only_by_reductions(table, readers, outs):
            return table, name, readers
    return None


def _inline_pair_table(table, name, readers) -> None:
    """Substitute the body of ``table`` into its readers, in place."""
    free = table.free
    for reader in readers:
        replacements = {
            x: table.body.xreplace(
                {free[0]: x.indices[0], free[1]: x.indices[1]}
            )
            for x in reader.body.atoms(sp.Indexed)
            if str(x.base) == name
        }
        reader.body = reader.body.xreplace(replacements)


def _inline_pair_tables(definitions, outputs, arrays):
    """Inline every droppable table; ``arrays`` loses the dropped ones."""
    while True:
        found = _find_droppable_table(definitions, outputs)
        if found is None:
            return definitions
        table, name, readers = found
        _inline_pair_table(table, name, readers)
        definitions = [d for d in definitions if d is not table]
        arrays.pop(name, None)


def _shared_inner_vector(only_inner, outer, inner, vectors, namer, arrays):
    """Return ``(factors, new definitions)`` for a product over ``inner``.

    The product becomes a vector over ``outer`` (created unless an equal
    one exists in ``vectors``) read at the inner index.
    """
    product = sp.Mul(*only_inner)
    key = product.xreplace({inner: _VECTOR_SLOT})
    created = []
    if key not in vectors:
        target = _new_shared_temp(namer, [outer], arrays)
        vectors[key] = target.base
        created.append(
            TempDef(
                target=target,
                free=[outer],
                limits=[],
                body=product.xreplace({inner: outer}),
            )
        )
    return [sp.Indexed(vectors[key], inner)], created


def _outer_factors_with_vector(outer_factors, outer, vectors):
    """Read an existing vector in place of a product over ``outer``."""
    only_outer = [f for f in outer_factors if f.atoms(sp.Idx) == {outer}]
    if len(only_outer) > 1:
        key = sp.Mul(*only_outer).xreplace({outer: _VECTOR_SLOT})
        if key in vectors:
            return [f for f in outer_factors if f not in only_outer] + [
                sp.Indexed(vectors[key], outer)
            ]
    return outer_factors


def _factor_double_reduction(definition, vectors, namer, arrays):
    """Return the definitions that replace a scalar double reduction.

    The factors independent of the inner index are applied once per outer
    iteration (``factored``); a product over the inner index only becomes
    a vector.  A definition that does not qualify comes back unchanged.
    """
    if (
        definition.free
        or len(definition.limits) != 2
        or not isinstance(definition.body, sp.Mul)
        or definition.body.has(sp.Piecewise)
    ):
        return [definition]
    outer, inner = definition.limits[0][0], definition.limits[1][0]
    factors = sp.Mul.make_args(definition.body)
    outer_factors = [f for f in factors if inner not in f.atoms(sp.Idx)]
    inner_factors = [f for f in factors if inner in f.atoms(sp.Idx)]
    if not outer_factors or not inner_factors:
        return [definition]
    only_inner = [f for f in inner_factors if f.atoms(sp.Idx) == {inner}]
    rest = [f for f in inner_factors if f not in only_inner]
    created = []
    if len(only_inner) > 1:
        only_inner, created = _shared_inner_vector(
            only_inner, outer, inner, vectors, namer, arrays
        )
    outer_factors = _outer_factors_with_vector(outer_factors, outer, vectors)
    factored = definition.copy(
        body=sp.Mul(*outer_factors) * sp.Mul(*(only_inner + rest)),
        factored=True,
        acc=sp.Symbol(str(namer("t"))),
    )
    return created + [factored]


def _place_new_vectors(out, vectors) -> None:
    """Move each new vector right after the definitions it reads.

    The double reductions then stay consecutive and share their inner
    loops.  ``out`` is edited in place.
    """
    new_vectors = {str(b) for b in vectors.values()}
    for vec in [d for d in out if _temp_name(d.target) in new_vectors]:
        out.remove(vec)
        names = {str(x.base) for x in vec.body.atoms(sp.Indexed)} | {
            str(x) for x in vec.body.free_symbols
        }
        position = 0
        for k, other in enumerate(out):
            if other.target is not None and _temp_name(other.target) in names:
                position = k + 1
        out.insert(position, vec)


def _drop_pair_tables(definitions, outputs, namer, arrays):
    """Inline pair tables read only by scalar reductions; factor double sums.

    A rank-2 table ``M(a, b)`` without summation, whose body calls no libm
    routine, read by no output and only by scalar reductions -- at most one
    double reduction (triangular when the table is filled as a triangle) and
    any number of single reductions (its diagonal) -- is not materialised:
    its body is substituted where it is read (reportes/brechas/ronda2-pcsaft.md,
    T1).  Then, in every scalar double reduction ``sum_p sum_x B``:

    * the factors of ``B`` that do not depend on the inner index ``x`` are
      applied once per outer iteration (``acc = sum_x B_x``, then
      ``t = t + acc*B_p``, rendered from the flag ``factored``);
    * a product of several factors over ``x`` only (``sqrt(eps(x))*m(x)*n(x)``)
      becomes a vector computed once before the reduction; the same product
      over the outer index reads the same vector.

    Returns the new definitions; ``arrays`` loses the dropped tables.
    """
    definitions = _inline_pair_tables(definitions, outputs, arrays)
    vectors = {}
    out = []
    for definition in definitions:
        out.extend(
            _factor_double_reduction(definition, vectors, namer, arrays)
        )
    _place_new_vectors(out, vectors)
    return out


_PLAIN_SCALAR_FLAGS = frozenset(
    {"target", "free", "limits", "body", "mirror", "factored", "acc"}
)
_ELEMENT_SLOT = sp.Idx("__elem")


def _is_plain_scalar(definition) -> bool:
    """Whether ``definition`` is a plain scalar without a summation.

    Parameters
    ----------
    definition : TempDef
        Definition to test.

    Returns
    -------
    bool
        ``True`` for a scalar symbol target with no free index, no limits,
        no annotation beyond the plain ones and no ``Sum`` in the body.
    """
    return (
        isinstance(definition.target, sp.Symbol)
        and not definition.free
        and not definition.limits
        and definition.present_keys() <= _PLAIN_SCALAR_FLAGS
        and not definition.body.has(sp.Sum)
    )


def _canonical_scalar(scalar, scalar_def, memo, max_ops):
    """Expand a plain scalar temporary into its recursive definition.

    Parameters
    ----------
    scalar : sympy.Symbol
        Plain scalar temporary (a key of ``scalar_def``).
    scalar_def : dict
        Plain scalar definitions by target.
    memo : dict
        Per-call cache of the expansions already computed; it also guards
        against cycles.
    max_ops : int
        Expansions with more operations are replaced by the symbol itself.

    Returns
    -------
    sympy.Expr
        The expanded value.
    """
    if scalar in memo:
        return memo[scalar]
    memo[scalar] = scalar  # cycle guard
    body = scalar_def[scalar].body
    replacements = {
        symbol: _canonical_scalar(symbol, scalar_def, memo, max_ops)
        for symbol in body.free_symbols
        if symbol in scalar_def
    }
    value = body.xreplace(replacements)
    if sp.count_ops(value) > max_ops:
        value = scalar
    memo[scalar] = value
    return value


def _is_shareable_candidate(sub) -> bool:
    """Whether ``sub`` is a rank-1 libm sub-expression worth sharing."""
    if not isinstance(sub, sp.Expr) or sub.is_Atom:
        return False
    if isinstance(sub, (sp.Indexed, sp.Sum, sp.Tuple)) or sub.has(sp.Sum):
        return False
    if sub.has(sp.Piecewise) or sub.is_Relational or sub.is_Boolean:
        return False
    if any(isinstance(x, _LoopStart) for x in sub.atoms(sp.Symbol)):
        return False
    return len(sub.atoms(sp.Idx)) == 1 and _calls_libm(sub)


def _occurrence_key(sub, scalar_def, memo, max_ops):
    """Return ``sub`` with its index as a slot and scalars expanded."""
    (index,) = sub.atoms(sp.Idx)
    replacements = {
        symbol: _canonical_scalar(symbol, scalar_def, memo, max_ops)
        for symbol in sub.free_symbols
        if symbol in scalar_def
    }
    return sub.xreplace({index: _ELEMENT_SLOT}).xreplace(replacements)


def _is_editable(definition) -> bool:
    """Whether the body of ``definition`` may be rewritten."""
    return definition.present_keys() <= _PLAIN_SCALAR_FLAGS


def _is_component_map(defs, pos) -> bool:
    """Whether site ``pos`` is a definition that is a per-component map."""
    if pos >= len(defs):
        return False
    return len(defs[pos].free) == 1 and not defs[pos].limits


@dataclass
class _ShareState:
    """Mutable state of one :func:`_share_elementwise` call.

    Attributes
    ----------
    defs : list
        Current definitions.
    outs : list
        Current output expressions.
    max_ops : int
        Operation limit of the scalar expansion.
    memo : dict
        Per-call cache of :func:`_canonical_scalar`; it is never reset
        between rounds.
    touched : set
        Scalars read by the replaced occurrences.
    merged : set
        Scalars renamed into an earlier spelling.
    sites : list
        Bodies of ``defs`` followed by ``outs``.
    scalar_def : dict
        Plain scalar definitions of ``defs`` by target.
    position : dict
        Index in ``defs`` of every plain scalar definition.
    """

    defs: list
    outs: list
    max_ops: int
    memo: dict = field(default_factory=dict)
    touched: set = field(default_factory=set)
    merged: set = field(default_factory=set)
    sites: list = field(init=False)
    scalar_def: dict = field(init=False)
    position: dict = field(init=False)

    def __post_init__(self):
        """Index the plain scalars and collect the sites."""
        self.refresh()

    def refresh(self) -> None:
        """Recompute ``sites``, ``scalar_def`` and ``position``."""
        self.sites = [d.body for d in self.defs] + self.outs
        self.scalar_def = {
            d.target: d for d in self.defs if _is_plain_scalar(d)
        }
        self.position = {
            d.target: k for k, d in enumerate(self.defs) if _is_plain_scalar(d)
        }


def _collect_occurrences(state):
    """Group the shareable sub-expressions of every editable site by key."""
    n_defs = len(state.defs)
    found = {}
    for pos, expr in enumerate(state.sites):
        if pos < n_defs and not _is_editable(state.defs[pos]):
            continue
        for sub in sp.preorder_traversal(expr):
            if _is_shareable_candidate(sub):
                key = _occurrence_key(
                    sub, state.scalar_def, state.memo, state.max_ops
                )
                found.setdefault(key, []).append((pos, sub))
    return found


def _best_occurrences(found, defs):
    """Return the occurrences of the key worth sharing, or ``None``.

    Per-component maps share per element anyway (``comp_fuse``): they do
    not count towards the two readers.  The key read in most sites wins,
    ties go to the one with more operations, then to the first in order.
    """
    best, best_rank = None, None
    for key in sorted(found, key=sp.default_sort_key):
        n_sites = len(
            {pos for pos, _ in found[key] if not _is_component_map(defs, pos)}
        )
        if n_sites < 2:
            continue
        rank = (n_sites, sp.count_ops(key))
        if best_rank is None or rank > best_rank:
            best, best_rank = key, rank
    return None if best is None else found[best]


def _scalar_merges(occurrences, state):
    """Map each later spelling of a scalar to the earliest equal one."""
    scalar_def = state.scalar_def
    rep_sub = occurrences[0][1]
    rep_scalars = {
        _canonical_scalar(x, scalar_def, state.memo, state.max_ops): x
        for x in rep_sub.free_symbols
        if x in scalar_def
    }
    merge = {}
    for _pos, sub in occurrences[1:]:
        for symbol in sub.free_symbols:
            if symbol not in scalar_def:
                continue
            canonical = _canonical_scalar(
                symbol, scalar_def, state.memo, state.max_ops
            )
            other = rep_scalars.get(canonical)
            if other is None or other == symbol:
                continue
            position = state.position
            if position[other] < position[symbol]:
                early, late = other, symbol
            else:
                early, late = symbol, other
            merge[late] = early
    return merge


def _resolve_merge_chains(merge):
    """Point every merged scalar at the end of its chain."""

    def final(symbol):
        while symbol in merge:
            symbol = merge[symbol]
        return symbol

    return {symbol: final(symbol) for symbol in merge}


def _share_occurrences(state, occurrences, namer, arrays):
    """Replace ``occurrences`` by a new vector placed before its first reader.

    Parameters
    ----------
    state : _ShareState
        Updated in place.
    occurrences : list
        ``(site position, sub-expression)`` pairs of one key.
    namer : _Namer
        Names the vector.
    arrays : dict
        Records the vector.
    """
    first_pos, rep_sub = occurrences[0]
    (index0,) = rep_sub.atoms(sp.Idx)
    merge = _scalar_merges(occurrences, state)
    target = _new_shared_temp(namer, [index0], arrays)
    vector = TempDef(target=target, free=[index0], limits=[], body=rep_sub)
    replacements = {}
    for pos, sub in occurrences:
        (index,) = sub.atoms(sp.Idx)
        replacements.setdefault(pos, {})[sub] = sp.Indexed(target.base, index)
        state.touched |= {x for x in sub.free_symbols if x in state.scalar_def}
    for pos, mapping in replacements.items():
        state.sites[pos] = state.sites[pos].xreplace(mapping)
    if merge:
        # resolve chains, then rename everywhere (vector body included)
        merge = _resolve_merge_chains(merge)
        state.sites = [e.xreplace(merge) for e in state.sites]
        vector.body = vector.body.xreplace(merge)
        state.merged |= set(merge)
    _insert_vector(state, vector, first_pos, set(merge))


def _insert_vector(state, vector, first_pos, dropped):
    """Write the new sites back and insert ``vector`` before its reader."""
    n_defs = len(state.defs)
    new_sites = list(state.sites)
    for pos in range(n_defs):
        state.defs[pos] = state.defs[pos].copy(body=new_sites[pos])
    insert_at = first_pos if first_pos < n_defs else n_defs
    defs = state.defs[:insert_at] + [vector] + state.defs[insert_at:]
    state.defs = [
        d for d in defs if not (d.target in dropped and _is_plain_scalar(d))
    ]
    state.outs = new_sites[n_defs:]
    state.refresh()


def _drop_dead_scalars(state) -> None:
    """Drop the touched plain scalars that nothing reads any more."""
    while True:
        read = set()
        for expr in [d.body for d in state.defs] + state.outs:
            read |= expr.free_symbols
        for d in state.defs:
            if d.acc is not None:
                read.add(d.acc)
        dead = [
            d
            for d in state.defs
            if _is_plain_scalar(d)
            and d.target in state.touched
            and d.target not in read
        ]
        if not dead:
            break
        state.defs = [d for d in state.defs if not any(d is x for x in dead)]


def _undeclare_dropped(state, namer) -> None:
    """Remove the dropped scalars from the namer's declarations."""
    alive = {d.target for d in state.defs} | {d.acc for d in state.defs}
    for expr in state.outs:
        alive |= expr.free_symbols
    dropped = {s for s in state.touched | set(state.merged) if s not in alive}
    namer.created[:] = [s for s in namer.created if s not in dropped]


def _share_elementwise(definitions, outputs, namer, arrays, max_ops=40):
    """Share costly rank-1 sub-expressions between summations and outputs.

    ``log(t13*r(k))`` in the body of a scalar reduction over ``k`` and
    ``log(t21*r(i))`` in an output over ``i`` are the same vector when
    ``t13 = t6/t8`` and ``t21 = t16*t6`` with ``t16 = 1/t8``
    (reportes/brechas/ronda2-ge-cmr.md, UNIQUAC G1).  The array CSE does not
    see it: the index is bound in one place and free in the other, and the
    scalar prefactors are spelt differently.  Here every rank-1 sub-expression
    that calls a libm routine is keyed by renaming its index to a slot and
    replacing each plain scalar temporary by its definition, expanded
    recursively (up to ``max_ops`` operations), so ``x/y`` and ``(1/y)*x``
    give the same key.  A key read in two or more places (definitions or
    outputs; per-component maps do not count, ``comp_fuse`` already shares
    inside them) becomes a vector ``v(k)``, computed right before its first
    reader, and every occurrence reads it.  Scalar temporaries that only
    differ in spelling inside the shared occurrences are merged into the one
    defined first; the ones left unread are dropped.

    Returns ``(definitions, outputs)``.
    """
    state = _ShareState(list(definitions), list(outputs), max_ops)
    while True:
        occurrences = _best_occurrences(
            _collect_occurrences(state), state.defs
        )
        if occurrences is None:
            break
        _share_occurrences(state, occurrences, namer, arrays)
    _drop_dead_scalars(state)
    _undeclare_dropped(state, namer)
    return state.defs, state.outs


class _DiagonalContext(NamedTuple):
    """Inputs of the canonical comparison of diagonal bodies.

    Attributes
    ----------
    maps : dict
        Inlinable per-component maps by name, as ``(index, body)``.
    symmetric_params : iterable of str
        Parameters declared ``symmetric``.
    sym_rank3 : bool
        Also sort the subscripts of symmetric tensors of rank three.
    """

    maps: dict
    symmetric_params: Sequence[str]
    sym_rank3: bool


def _is_inlinable_map(definition) -> bool:
    """Whether ``definition`` is a plain rank-1 map without a summation."""
    return (
        len(definition.free) == 1
        and not definition.limits
        and not definition.body.has(sp.Sum)
        and not (
            definition.mirror
            or definition.mirror_sum
            or definition.symv
            or isinstance(
                definition, (FusedNest, SiblingNest, ComponentFusedMap)
            )
            or definition.colmajor
        )
    )


def _inline_maps(expr, maps):
    """Replace the reads of the maps in ``maps`` by their bodies."""
    replacements = {}
    for indexed in expr.atoms(sp.Indexed):
        if str(indexed.base) in maps and len(indexed.indices) == 1:
            index, body = maps[str(indexed.base)]
            replacements[indexed] = body.xreplace({index: indexed.indices[0]})
    return expr.xreplace(replacements) if replacements else expr


def _canonical_diagonal_body(expr, context):
    """Expand ``expr`` after inlining the maps, symmetric subscripts sorted."""
    expr = sp.expand(_inline_maps(expr, context.maps))
    if context.symmetric_params:
        return _canonical_symmetric(
            expr, context.symmetric_params, context.sym_rank3
        )
    return expr


def _mirror_tables(definitions):
    """Return the rank-2 summed tables and the definition hosting each.

    Parameters
    ----------
    definitions : list
        Definitions in emission order.

    Returns
    -------
    tuple
        ``(tables, owner)``: the tables, and a dict from ``id(table)`` to
        the definition that holds it (the sibling nest or the table itself).
    """
    tables, owner = [], {}
    for definition in definitions:
        if isinstance(definition, SiblingNest):
            members = definition.sibling
        else:
            members = [definition] if definition.mirror_sum else []
        for table in members:
            if len(table.free) == 2 and len(table.limits) == 1:
                tables.append(table)
                owner[id(table)] = definition
    return tables, owner


def _is_diagonal_candidate(definition) -> bool:
    """Whether ``definition`` is a plain rank-1 reduction over one index."""
    return not (
        len(definition.free) != 1
        or len(definition.limits) != 1
        or definition.mirror
        or definition.mirror_sum
        or definition.symv
        or isinstance(definition, FusedNest)
        or definition.colmajor
    )


def _matching_table(definition, tables, context):
    """Return the first table whose diagonal equals ``definition``.

    Tables with other limits, another diagonal body, or a body that reads
    the definition's own array are skipped.  Returns ``None`` if none is
    left.
    """
    (k,), (m, low, high) = definition.free, definition.limits[0]
    name = _temp_name(definition.target)
    for table in tables:
        (a, b), (m2, low2, high2) = table.free, table.limits[0]
        if (low2, high2) != (low, high):
            continue
        diagonal = _canonical_diagonal_body(
            table.body.xreplace({a: k, b: k, m2: m}), context
        )
        difference = diagonal - _canonical_diagonal_body(
            definition.body, context
        )
        if sp.expand(difference) != 0:
            continue
        if any(str(x.base) == name for x in table.body.atoms(sp.Indexed)):
            continue
        return table
    return None


def _move_diagonal_after_table(out, definition, table, host) -> None:
    """Replace ``definition`` by a read of ``table`` placed after ``host``.

    Nothing changes when a consumer of the definition sits between it and
    the table.  ``out`` is edited in place.
    """
    (k,) = definition.free
    name = _temp_name(definition.target)
    new = TempDef(
        target=definition.target,
        free=[k],
        limits=[],
        body=sp.Indexed(table.target.base, k, k),
    )
    pos_table = out.index(host)
    pos_diagonal = out.index(definition)
    if pos_diagonal < pos_table:
        # every consumer of the diagonal must come after the table
        between = out[pos_diagonal + 1 : pos_table + 1]
        if any(
            str(x.base) == name
            for other in between
            for x in other.body.atoms(sp.Indexed)
        ):
            return
    out.remove(definition)
    out.insert(out.index(host) + 1, new)


def _diagonal_from_tables(definitions, symmetric_params, *, sym_rank3):
    """Read the diagonal of an existing rank-2 table instead of recomputing it.

    ``d(k) = sum_m B(k, k, m)`` and ``t(a, b) = sum_m B(a, b, m)`` (filled in
    a triangle and mirrored) give ``d(k) == t(k, k)``.  The rank-1 sum is
    replaced by the map ``d(k) = t(k, k)``, placed right after the table.
    """
    maps = {
        _temp_name(d.target): (d.free[0], d.body)
        for d in definitions
        if _is_inlinable_map(d)
    }
    context = _DiagonalContext(maps, symmetric_params, sym_rank3)
    tables, owner = _mirror_tables(definitions)
    out = list(definitions)
    for definition in definitions:
        if not _is_diagonal_candidate(definition):
            continue
        table = _matching_table(definition, tables, context)
        if table is not None:
            _move_diagonal_after_table(
                out, definition, table, owner[id(table)]
            )
    return out


def _is_sibling_table(definition) -> bool:
    """Whether ``definition`` is a mirrored rank-2 table with one sum."""
    return bool(
        definition.mirror_sum
        and len(definition.free) == 2
        and len(definition.limits) == 1
        and not definition.body.has(sp.Piecewise)
        and not definition.colsum
    )


def _reads_names(definition, names) -> bool:
    """Whether the body of ``definition`` reads any of the ``names``."""
    return any(
        str(x.base) in names for x in definition.body.atoms(sp.Indexed)
    ) or any(str(sy) in names for sy in definition.body.free_symbols)


def _target_names(definitions) -> set:
    """Return the names of the temporaries the definitions produce."""
    names = set()
    for definition in definitions:
        if definition.target is not None:
            names.add(_temp_name(definition.target))
    return names


def _rename_sibling(definition, first):
    """Return a copy of a table over the indices of the group's first one."""
    (first_a, first_b), (first_m, _, _) = first.free, first.limits[0]
    mapping = {
        definition.free[0]: first_a,
        definition.free[1]: first_b,
        definition.limits[0][0]: first_m,
    }
    renamed = definition.copy()
    renamed.target = definition.target.xreplace(mapping)
    renamed.body = definition.body.xreplace(mapping)
    renamed.free = [first_a, first_b]
    renamed.limits = [(first_m, first.limits[0][1], first.limits[0][2])]
    return renamed


def _flush_sibling_group(out, group, waiting, namer) -> None:
    """Append the open group, then the waiting definitions, to ``out``.

    A group of several tables becomes a :class:`SiblingNest` and draws one
    accumulator name per member from ``namer``.
    """
    if len(group) == 1:
        out.extend(group)
    elif group:
        out.append(
            SiblingNest(
                target=group[0].target,
                free=group[0].free,
                limits=group[0].limits,
                body=group[0].body,
                sibling=list(group),
                accs=[sp.Symbol(str(namer("t"))) for _ in group],
            )
        )
    out.extend(waiting)
    group.clear()
    waiting.clear()


def _fuse_sibling_tables(definitions, namer):
    """One nest for symmetric rank-2 tables with the same loops.

    ``t5(k, l) = sum_m A``, ``t7(k, l) = sum_m B`` (triangle + mirror) become
    one triangular nest whose inner loop accumulates both sums in scalars:
    each slice of the operands is read in a single pass.  Definitions found
    between two such tables that do not read the first one are emitted
    after the fused nest, unless the second table reads them.
    """
    out, group, waiting = [], [], []
    for definition in definitions:
        if _is_sibling_table(definition) and group:
            renamed = _rename_sibling(definition, group[0])
            blocked = _target_names(group + waiting)
            same_range = definition.limits[0][1:] == group[0].limits[0][1:]
            if same_range and not _reads_names(renamed, blocked):
                group.append(renamed)
                continue
            _flush_sibling_group(out, group, waiting, namer)
            group.append(definition)
            continue
        if _is_sibling_table(definition):
            group.append(definition)
            continue
        if group and not _reads_names(definition, _target_names(group)):
            waiting.append(definition)
            continue
        _flush_sibling_group(out, group, waiting, namer)
        out.append(definition)
    _flush_sibling_group(out, group, waiting, namer)
    return out


def _split_accumulators(definitions, namer):
    """Mark the O(N^3) nests (``split_acc=2``).

    A symmetric rank-2 table filled by a sum whose body reads a parameter of
    rank 3 or more (the CMR nest ``t(k, l) = sum_m L(m, k, l) n(m)``) gets two
    scalar accumulators per sum in its inner loop.  A table alone (``mirror_sum``)
    first becomes a one-member nest of :func:`_fuse_sibling_tables`, which
    renders the same loops with a scalar accumulator.
    """

    def rank3(body):
        return any(len(x.indices) >= 3 for x in body.atoms(sp.Indexed))

    def lone(d):
        return (
            d.mirror_sum
            and len(d.free) == 2
            and len(d.limits) == 1
            and not d.body.has(sp.Piecewise)
            and d.present_keys()
            <= {"target", "free", "limits", "body", "mirror_sum"}
        )

    out = []
    for d in definitions:
        if isinstance(d, SiblingNest) and any(
            rank3(g.body) for g in d.sibling
        ):
            d = d.copy(
                accumulators=2,
                accs2=[sp.Symbol(str(namer("t"))) for _ in d.accs],
            )
        elif lone(d) and rank3(d.body):
            d = SiblingNest(
                target=d.target,
                free=d.free,
                limits=d.limits,
                body=d.body,
                sibling=[d],
                accs=[sp.Symbol(str(namer("t")))],
                accumulators=2,
            )
            d.accs2 = [sp.Symbol(str(namer("t")))]
        out.append(d)
    return out
