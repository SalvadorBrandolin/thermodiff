"""Common-subexpression search for the Fortran code generator."""

import functools
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, cast

import sympy as sp

from thermodiff.codegen.common import (
    _AUX_INDEX_NAMES,
    _MAX_ARRAY_CSE_RANK,
    _Namer,
    _free_indices,
    _index_set,
    _is_expensive,
    _memoised_call,
    _new_shared_temp,
    _order_free,
    _sorted_indices,
)
from thermodiff.codegen.extents import slot_mapping
from thermodiff.codegen.ir import TempDef
from thermodiff.codegen.symmetry import (
    _alpha_key,
    _canonical_symmetric,
    _free_alpha_key,
)


def _sum_atoms(
    expr: sp.Basic, memo: dict[sp.Basic, set[sp.Sum]]
) -> set[sp.Sum]:
    """Return ``expr.atoms(sp.Sum)``, memoised in ``memo`` by expression.

    Parameters
    ----------
    expr : sympy.Basic
        Expression to scan.
    memo : dict
        Per-call cache from expression to its set of ``Sum`` atoms.

    Returns
    -------
    set
        The ``Sum`` nodes of ``expr``; callers must not mutate it.
    """
    found = memo.get(expr)
    if found is None:
        found = memo[expr] = expr.atoms(sp.Sum)
    return found


def _sorted_sums(
    expr: sp.Basic,
    memo: dict[sp.Basic, set[sp.Sum]],
    sorted_memo: dict[sp.Basic, tuple[sp.Sum, ...]],
) -> tuple[sp.Sum, ...]:
    """Return the ``Sum`` atoms of ``expr`` sorted by ``default_sort_key``.

    Parameters
    ----------
    expr : sympy.Basic
        Expression to scan.
    memo : dict
        Per-call cache used by :func:`_sum_atoms`.
    sorted_memo : dict
        Per-call cache from expression to its sorted ``Sum`` atoms.

    Returns
    -------
    tuple
        The sorted ``Sum`` nodes of ``expr``.
    """
    ordered = sorted_memo.get(expr)
    if ordered is None:
        ordered = tuple(
            sorted(_sum_atoms(expr, memo), key=sp.default_sort_key)
        )
        sorted_memo[expr] = ordered
    return ordered


@dataclass(frozen=True)
class _SumKeySettings:
    """Options of :func:`_share_sums` that decide when two sums are equal.

    Attributes
    ----------
    indexes : list
        User index order, used to sort the free indices of a temporary.
    alpha : bool
        Share sums equal up to the names of their bound indices.
    free_alpha : bool
        Share sums equal up to the names of all their indices.
    symmetric_params : tuple of str
        Symmetric parameters, canonicalised in the keys.
    transpose : bool
        Reuse the transpose of a rank-2 sum already computed.
    sym_rank3 : bool
        Also canonicalise symmetric parameters of rank three.
    """

    indexes: list[sp.Idx]
    alpha: bool
    free_alpha: bool
    symmetric_params: tuple[str, ...]
    transpose: bool
    sym_rank3: bool


def _innermost_sums(
    exprs: list[sp.Expr],
    sums_memo: dict[sp.Basic, set[sp.Sum]],
    sorted_memo: dict[sp.Basic, tuple[sp.Sum, ...]],
) -> list[sp.Sum]:
    """Return the distinct sums without inner sums, in encounter order.

    Parameters
    ----------
    exprs : list of sympy expressions
        Expressions that may contain summations.
    sums_memo, sorted_memo : dict
        Per-call caches of :func:`_sum_atoms` and :func:`_sorted_sums`.

    Returns
    -------
    list
        The ``Sum`` nodes whose body holds no ``Sum``, each once.
    """
    innermost: list[sp.Sum] = []
    seen: set[sp.Sum] = set()
    for expr in exprs:
        for total in _sorted_sums(expr, sums_memo, sorted_memo):
            if _sum_atoms(total.function, sums_memo) or total in seen:
                continue
            seen.add(total)
            innermost.append(total)
    return innermost


def _transpose_replacement(
    total: sp.Sum,
    free_here: list[sp.Idx],
    by_key: dict[sp.Basic, Any],
    settings: _SumKeySettings,
) -> sp.Indexed | None:
    """Return the transpose of a rank-2 sum already hoisted, or ``None``.

    Parameters
    ----------
    total : sympy.Sum
        The sum to replace.
    free_here : list
        Its two free indices, in user order.
    by_key : dict
        Hoisted temporaries as ``key -> (target, free)``.
    settings : _SumKeySettings
        Options that decide when two sums are equal.

    Returns
    -------
    sympy.Indexed or None
        The earlier temporary indexed by the swapped free indices.
    """
    swapped = list(reversed(free_here))
    mapping = {
        index: sp.Idx(f"_free{pos}") for pos, index in enumerate(swapped)
    }
    transposed_key = _alpha_key(total.xreplace(mapping))
    if settings.symmetric_params:
        transposed_key = _canonical_symmetric(
            transposed_key, settings.symmetric_params, settings.sym_rank3
        )
    if transposed_key not in by_key:
        return None
    target, _ = by_key[transposed_key]
    return sp.Indexed(target.base, *swapped)


def _free_alpha_replacement(
    total: sp.Sum, by_key: dict[sp.Basic, Any], settings: _SumKeySettings
) -> tuple[sp.Basic | None, sp.Basic]:
    """Find an earlier temporary equal to ``total`` up to index names.

    Parameters
    ----------
    total : sympy.Sum
        The sum to replace.
    by_key : dict
        Hoisted temporaries as ``key -> (target, free)``.
    settings : _SumKeySettings
        Options that decide when two sums are equal.

    Returns
    -------
    replacement : sympy expression or None
        The earlier temporary, indexed by the free indices of ``total``.
    key : sympy expression
        The canonical key of ``total``.
    """
    key, free_here = _free_alpha_key(
        total,
        settings.indexes,
        settings.symmetric_params,
        sym_rank3=settings.sym_rank3,
    )
    if settings.transpose and key not in by_key and len(free_here) == 2:
        # the transpose of a rank-2 sum already computed
        found = _transpose_replacement(total, free_here, by_key, settings)
        if found is not None:
            return found, key
    if key in by_key:
        # equal up to the names of its bound and free indices: reuse the
        # temporary, indexed by this sum's free indices
        target, _ = by_key[key]
        found = sp.Indexed(target.base, *free_here) if free_here else target
        return found, key
    return None, key


def _share_one_sum(
    total: sp.Sum,
    by_key: dict[sp.Basic, Any],
    replacements: dict[sp.Sum, sp.Basic],
    definitions: list[TempDef],
    namer: _Namer,
    arrays: dict[str, int],
    settings: _SumKeySettings,
) -> None:
    """Replace ``total`` by an earlier temporary or hoist a new one.

    Parameters
    ----------
    total : sympy.Sum
        The innermost sum to process.
    by_key : dict
        Hoisted temporaries of this round, updated in place.
    replacements : dict
        ``Sum -> temporary`` of this round, updated in place.
    definitions : list
        Temporaries defined so far, appended to in place.
    namer : _Namer
        Generator of temporary variable names.
    arrays : dict
        Records the rank of every array temporary created.
    settings : _SumKeySettings
        Options that decide when two sums are equal.
    """
    key: sp.Basic | None = None
    found: Any = None
    if settings.free_alpha:
        found, key = _free_alpha_replacement(total, by_key, settings)
    elif settings.alpha:
        key = _alpha_key(total)
        # alpha-equivalent to a summation already hoisted
        found = by_key[key] if key in by_key else None
    if found is not None:
        replacements[total] = found
        return
    free = _order_free(_free_indices(total), settings.indexes)
    target = _new_shared_temp(namer, free, arrays)
    definitions.append(
        TempDef(
            target=target,
            free=free,
            limits=list(total.limits),
            body=total.function,
        )
    )
    replacements[total] = target
    if settings.free_alpha:
        by_key[key] = (target, free)
    elif settings.alpha:
        by_key[key] = target


def _share_sums(
    expressions: Iterable[sp.Expr],
    namer: _Namer,
    indexes: list[sp.Idx],
    arrays: dict[str, int],
    alpha: bool = False,
    free_alpha: bool = False,
    symmetric_params: tuple[str, ...] = (),
    transpose: bool = False,
    *,
    sym_rank3: bool,
) -> tuple[list[sp.Expr], list[TempDef]]:
    """Replace every ``Sum`` by a temporary that is computed once.

    Identical summations reused by several outputs are evaluated a single
    time.  Sums that appear in only one output are hoisted too, which keeps
    the output residuals free of loop nests and lets the scalar common
    sub-expression pass below see through them.

    Parameters
    ----------
    expressions : list of sympy expressions
        Expressions that may contain summations.
    namer : _Namer
        Generator of temporary variable names.
    indexes : list
        User index order, used to sort the free indices of a temporary.
    arrays : dict
        Records the rank of every array temporary created.

    Returns
    -------
    residuals : list of sympy expressions
        ``expressions`` with every ``Sum`` replaced by a temporary.
    definitions : list of dict
        How to compute each temporary, in dependency order.
    """
    settings = _SumKeySettings(
        indexes=indexes,
        alpha=alpha,
        free_alpha=free_alpha,
        symmetric_params=symmetric_params,
        transpose=transpose,
        sym_rank3=sym_rank3,
    )
    exprs = list(expressions)
    definitions: list[TempDef] = []
    sums_memo: dict[sp.Basic, set[sp.Sum]] = {}
    sorted_memo: dict[sp.Basic, tuple[sp.Sum, ...]] = {}
    while True:
        innermost = _innermost_sums(exprs, sums_memo, sorted_memo)
        if not innermost:
            break
        replacements: dict[sp.Sum, sp.Basic] = {}
        by_key: dict[sp.Basic, Any] = {}
        for total in innermost:
            _share_one_sum(
                total,
                by_key,
                replacements,
                definitions,
                namer,
                arrays,
                settings,
            )
        exprs = [expr.xreplace(replacements) for expr in exprs]
    return exprs, definitions


def _common_scalar_cse(
    expressions: Iterable[sp.Expr], namer: _Namer, arrays: dict[str, int]
) -> tuple[list[sp.Expr], list[TempDef]]:
    """Extract scalar sub-expressions shared by several outputs.

    Only sub-expressions without free indices are hoisted, so every
    temporary can be computed once, before the output loops, without
    worrying about the index scope of a particular output.

    Parameters
    ----------
    expressions : list of sympy expressions
        Sum free expressions to optimise.
    namer : _Namer
        Generator of temporary variable names.
    arrays : dict
        Records the rank of every array temporary created.

    Returns
    -------
    residuals : list of sympy expressions
        ``expressions`` with the shared sub-expressions replaced.
    definitions : list of dict
        How to compute each shared scalar, in dependency order.
    """
    exprs = list(expressions)
    definitions: list[TempDef] = []
    own_entries = functools.partial(
        _own_scalar_candidate, index_memo={}, eligible_memo={}
    )
    entries_memo: dict[sp.Basic, tuple[Any, ...]] = {}
    rank_memo: dict[sp.Expr, tuple[Any, ...]] = {}
    candidate_sets: dict[sp.Basic, frozenset[Any]] = {}
    while True:
        counts = _count_scalar_candidates(exprs, own_entries, entries_memo)
        repeated = [sub for sub, count in counts.items() if count > 1]
        if not repeated:
            break
        best = _best_scalar_candidate(repeated, counts, rank_memo)
        target = _new_shared_temp(namer, [], arrays)
        definitions.append(
            TempDef(target=target, free=[], limits=[], body=best)
        )
        rule = {best: target}
        for position, expr in enumerate(exprs):
            candidates = _candidate_set(
                expr, own_entries, entries_memo, candidate_sets
            )
            if best in candidates:
                exprs[position] = expr.xreplace(rule)
    return exprs, definitions


def _best_scalar_candidate(
    repeated: list[sp.Expr],
    counts: dict[sp.Expr, int],
    rank_memo: dict[sp.Expr, tuple[Any, ...]],
) -> sp.Expr:
    """Pick the candidate with the highest ``(count, count_ops, sort key)``.

    Only the candidates with the maximum count can win, so the rank is
    evaluated for those alone.  ``repeated`` keeps the order of ``counts``
    and ``max`` returns the first maximum, as it would over all of them.
    """
    top_count = max(counts[sub] for sub in repeated)
    contenders = [sub for sub in repeated if counts[sub] == top_count]
    return max(contenders, key=lambda sub: _candidate_rank(sub, rank_memo))


def _candidate_set(
    expr: sp.Basic,
    own_entries: Callable[[sp.Basic], tuple[Any, ...]],
    entries_memo: dict[sp.Basic, tuple[Any, ...]],
    sets_memo: dict[sp.Basic, frozenset[Any]],
) -> frozenset[Any]:
    """Return the set of candidates occurring in ``expr`` (memoised).

    ``sets_memo`` is the per-call cache ``expression -> frozenset``; it is
    keyed by the expression object, which ``xreplace`` leaves unchanged
    when nothing in it is replaced.
    """
    if expr not in sets_memo:
        sets_memo[expr] = frozenset(
            _preorder_entries(expr, own_entries, entries_memo)
        )
    return sets_memo[expr]


def _is_scalar_candidate(
    sub: sp.Basic,
    index_memo: dict[sp.Basic, Any],
    eligible_memo: dict[sp.Basic, bool],
) -> bool:
    """Tell whether ``sub`` may be hoisted as a shared scalar (memoised)."""
    if sub not in eligible_memo:
        eligible_memo[sub] = (
            isinstance(sub, sp.Expr)
            and not sub.is_Atom
            and not _free_indices(sub, index_memo)
            and not sub.has(sp.Sum)
        )
    return eligible_memo[sub]


def _own_scalar_candidate(
    sub: sp.Basic,
    index_memo: dict[sp.Basic, Any],
    eligible_memo: dict[sp.Basic, bool],
) -> tuple[sp.Basic, ...]:
    """Return ``(sub,)`` when ``sub`` is a scalar candidate, else ``()``."""
    if _is_scalar_candidate(sub, index_memo, eligible_memo):
        return (sub,)
    return ()


def _count_scalar_candidates(
    exprs: list[sp.Expr],
    own_entries: Callable[[sp.Basic], tuple[Any, ...]],
    entries_memo: dict[sp.Basic, tuple[Any, ...]],
) -> dict[Any, int]:
    """Count the occurrences of every hoistable sub-expression.

    The dict keeps the order in which candidates are first met in a
    pre-order walk of ``exprs``, which fixes the tie-break of the caller.
    ``entries_memo`` is the per-call memo of :func:`_preorder_entries`.
    """
    counts: dict[Any, int] = {}
    for expr in exprs:
        for sub in _preorder_entries(expr, own_entries, entries_memo):
            counts[sub] = counts.get(sub, 0) + 1
    return counts


def _candidate_rank(
    sub: sp.Expr, rank_memo: dict[sp.Expr, tuple[Any, ...]]
) -> tuple[Any, ...]:
    """Return ``(count_ops, default_sort_key)`` of ``sub`` (memoised)."""
    if sub not in rank_memo:
        rank_memo[sub] = (sp.count_ops(sub), sp.default_sort_key(sub))
    return rank_memo[sub]


def _node_children(node: sp.Basic) -> Iterable[sp.Basic]:
    """Return the children that ``sp.preorder_traversal`` visits.

    Mirrors ``sp.preorder_traversal(expr)`` with ``keys=None`` for SymPy
    ``Basic`` nodes: ``LatticeOp`` nodes (``Max``, ``Min``, ``And``...) are
    walked through their ``_argset``, everything else through ``args``.
    Non-``Basic`` iterable args (e.g. matrices) are not supported.
    """
    if hasattr(node, "_argset"):
        return cast(Iterable[sp.Basic], node._argset)
    return cast(Iterable[sp.Basic], node.args)


def _preorder_entries(
    expr: sp.Basic,
    own_entries: Callable[[sp.Basic], tuple[Any, ...]],
    memo: dict[sp.Basic, tuple[Any, ...]],
) -> tuple[Any, ...]:
    """Collect per-node entries of ``expr`` in pre-order, memoised.

    The result equals the concatenation of ``own_entries(node)`` over
    ``sp.preorder_traversal(expr)`` with ``keys=None``, for SymPy ``Basic``
    trees.  Non-``Basic`` iterable args (e.g. matrices) are not supported;
    they never occur in the expressions of the generator.  Entries are
    cached per subtree in ``memo``; ``xreplace`` returns the very same
    objects for the subtrees it leaves alone, so a later walk of a
    rewritten expression only visits the nodes that changed.

    Parameters
    ----------
    expr : sympy expression
        Root of the walk.
    own_entries : callable
        Maps a node to the tuple of entries it contributes by itself.
        It must be pure for the lifetime of ``memo``.
    memo : dict
        Per-call cache ``subtree -> tuple of entries``, owned by the
        caller and valid for one fixed ``own_entries``.

    Returns
    -------
    tuple
        The entries of ``expr`` in pre-order.
    """
    stack = [expr]
    while stack:
        node = stack[-1]
        if node in memo:
            stack.pop()
            continue
        children = _node_children(node)
        pending = [child for child in children if child not in memo]
        if pending:
            stack.extend(pending)
            continue
        memo[node] = (
            *own_entries(node),
            *(entry for child in children for entry in memo[child]),
        )
        stack.pop()
    return memo[expr]


def _is_hoistable(
    sub: sp.Basic, index_memo: dict[sp.Basic, frozenset[sp.Idx]] | None = None
) -> bool:
    """Whether ``sub`` may be replaced by an array temporary.

    ``index_memo`` optionally supplies the memo of :func:`_index_set`
    used for the final index test instead of ``sub.atoms(sp.Idx)``.
    """
    if not isinstance(sub, sp.Expr) or sub.is_Atom:
        return False
    if isinstance(sub, (sp.Indexed, sp.Sum, sp.Tuple)):
        return False
    if sub.has(sp.Sum) or sub.has(sp.Piecewise):
        return False
    if sub.is_Relational or sub.is_Boolean:
        return False
    if index_memo is not None:
        return bool(_index_set(sub, index_memo))
    return bool(sub.atoms(sp.Idx))


def _canonical_form(
    sub: sp.Basic, indices: list[sp.Idx], slots: list[sp.Idx]
) -> sp.Basic:
    """Rename ``indices`` of ``sub`` to the leading auxiliary ``slots``.

    A slot keeps the range of the index it replaces, so that two shapes
    that differ only in the dimension an index runs over stay different.
    """
    return sub.xreplace(slot_mapping(indices, slots))


def _cached_hoistable(
    sub: sp.Basic,
    index_memo: dict[sp.Basic, frozenset[sp.Idx]],
    cache: dict[sp.Basic, bool],
) -> bool:
    """:func:`_is_hoistable` memoised in the per-call ``cache``."""
    if sub not in cache:
        cache[sub] = _is_hoistable(sub, index_memo)
    return cache[sub]


def _cached_canonical(
    sub: sp.Basic,
    indices: list[sp.Idx],
    slots: list[sp.Idx],
    cache: dict[sp.Basic, sp.Basic],
) -> sp.Basic:
    """:func:`_canonical_form` memoised in the per-call ``cache``.

    The result depends on ``slots``, so ``cache`` must not outlive the
    call that fixed them.
    """
    if sub not in cache:
        cache[sub] = _canonical_form(sub, indices, slots)
    return cache[sub]


def _own_hoistable_entry(
    sub: sp.Basic,
    slots: list[sp.Idx],
    index_memo: dict[sp.Basic, frozenset[sp.Idx]],
    hoistable_cache: dict[sp.Basic, bool],
    canonical_cache: dict[sp.Basic, sp.Basic],
) -> tuple[Any, ...]:
    """Return the ``(sub, canonical, indices)`` entry of ``sub`` itself.

    The tuple is empty when ``sub`` is not hoistable or its rank exceeds
    :data:`_MAX_ARRAY_CSE_RANK`.
    """
    if not _cached_hoistable(sub, index_memo, hoistable_cache):
        return ()
    indices = _sorted_indices(sub, index_memo)
    if len(indices) > _MAX_ARRAY_CSE_RANK:
        return ()
    canon = _cached_canonical(sub, indices, slots, canonical_cache)
    return ((sub, canon, indices),)


def _best_indexed_candidate(
    counts: dict[sp.Basic, int],
    expensive_twice: bool,
    ops_memo: dict[sp.Basic, Any],
    sort_key_memo: dict[sp.Basic, Any],
) -> sp.Basic | None:
    """Pick the canonical sub-expression to hoist next, or ``None``.

    The winner maximises ``(count, count_ops)`` among the candidates that
    pass the cheap-operation filter, the first one in ``default_sort_key``
    order on ties.  Count groups are visited in descending order, so
    ``count_ops`` and the sort key are only computed for the groups that
    can still win: a lower count is reached only when every candidate of
    the higher groups was filtered out.
    """
    repeated = [canon for canon, count in counts.items() if count >= 2]
    for count in sorted({counts[canon] for canon in repeated}, reverse=True):
        group = sorted(
            (canon for canon in repeated if counts[canon] == count),
            key=lambda item: _memoised_call(
                sp.default_sort_key, item, sort_key_memo
            ),
        )
        best: sp.Basic | None = None
        best_ops: Any = None
        for canon in group:
            ops = _memoised_call(sp.count_ops, canon, ops_memo)
            # A single repeated operation is not worth the extra array
            # unless it is used at least three times.
            if (
                ops < 2
                and count < 3
                and not (expensive_twice and _is_expensive(canon))
            ):
                continue
            if best_ops is None or ops > best_ops:
                best_ops = ops
                best = canon
        if best is not None:
            return best
    return None


def _slots_in_use(canonical: sp.Basic, slots: list[sp.Idx]) -> list[sp.Idx]:
    """Return the indices of ``canonical`` in the order of ``slots``."""
    position = {str(slot): place for place, slot in enumerate(slots)}
    return sorted(
        canonical.atoms(sp.Idx), key=lambda index: position[str(index)]
    )


def _free_aux_indices(exprs: Iterable[sp.Basic]) -> list[sp.Idx]:
    """Return the auxiliary indices that do not clash with ``exprs``.

    Fortran is case insensitive: lower-cased names are compared, so that
    an auxiliary index ``r`` never clashes with e.g. a module constant
    ``R``.
    """
    used = {
        str(symbol).lower()
        for expr in exprs
        for symbol in expr.atoms(sp.Symbol)
    }
    used.update(
        {str(b).lower() for expr in exprs for b in expr.atoms(sp.IndexedBase)}
    )
    used.update({"nc", "res"})
    return [sp.Idx(name) for name in _AUX_INDEX_NAMES if name not in used]


def _count_canonical_entries(
    entries: list[tuple[Any, ...]], counted: list[bool] | None
) -> tuple[dict[sp.Basic, int], dict[sp.Basic, int]]:
    """Count the canonical forms and record the rank of each one.

    Positions flagged ``False`` in ``counted`` do not contribute (all do
    when ``counted`` is ``None``).

    Returns
    -------
    counts : dict
        Number of occurrences of every canonical form.
    sizes : dict
        Number of indices of every canonical form.
    """
    counts: dict[sp.Basic, int] = {}
    sizes: dict[sp.Basic, int] = {}
    for position, position_entries in enumerate(entries):
        if counted is not None and not counted[position]:
            continue
        for _, canon, indices in position_entries:
            counts[canon] = counts.get(canon, 0) + 1
            sizes[canon] = len(indices)
    return counts, sizes


def _replace_with_temp(
    exprs: list[sp.Expr],
    entries: list[tuple[Any, ...]],
    best: sp.Basic,
    target: sp.Indexed,
    rank: int,
) -> list[sp.Expr]:
    """Replace every occurrence of ``best`` by the array ``target``.

    Returns a new list; expressions without an occurrence are the very
    same objects.
    """
    base = target.base
    replacements: dict[sp.Basic, sp.Basic] = {}
    touched: set[int] = set()
    for position, position_entries in enumerate(entries):
        for sub, canon, indices in position_entries:
            if len(indices) == rank and canon == best:
                replacements[sub] = sp.Indexed(base, *indices)
                touched.add(position)
    # ``xreplace`` returns the very same object when no key occurs.
    return [
        expr.xreplace(replacements) if position in touched else expr
        for position, expr in enumerate(exprs)
    ]


def _indexed_cse(
    expressions: Iterable[sp.Expr],
    namer: _Namer,
    arrays: dict[str, int],
    expensive_twice: bool = False,
    counted: list[bool] | None = None,
) -> tuple[list[sp.Expr], list[TempDef]]:
    """Hoist index-dependent sub-expressions shared by several outputs.

    :func:`_common_scalar_cse` only hoists sub-expressions that do not
    depend on an index.  This pass also handles sub-expressions that *do*:
    the indices are first renamed to a canonical set, so that
    ``sqrt(ac(k)*ac(l))`` and ``sqrt(ac(i)*ac(j))`` are recognised as the
    same array.  Every such sub-expression is then materialised into an
    array temporary that is computed a single time.

    For a mixture model this turns the ``sqrt``/``log``/``pow`` calls
    repeated inside the ``nc``-nested summations into one pass, so a
    derivative that needs a shared factor reads it from memory instead of
    recomputing it.

    The rewrite is exact: a sub-expression is only replaced by an indexed
    temporary whose definition is that very sub-expression, evaluated over
    the same indices.

    Only sub-expressions of rank up to :data:`_MAX_ARRAY_CSE_RANK` are
    materialised; see that constant for why rank three and above are left
    alone.

    Parameters
    ----------
    expressions : list of sympy expressions
        Expressions to optimise (they may still contain ``Sum`` objects).
    namer : _Namer
        Generator of temporary variable names.
    arrays : dict
        Records the rank of every array temporary created.

    Returns
    -------
    residuals : list of sympy expressions
        ``expressions`` with the shared index-dependent sub-expressions
        replaced by array temporaries.
    definitions : list of dict
        How to compute each array temporary, in dependency order.
    """
    exprs = list(expressions)
    definitions: list[TempDef] = []
    slots = _free_aux_indices(exprs)
    index_memo: dict[sp.Basic, frozenset[sp.Idx]] = {}
    own_entries = functools.partial(
        _own_hoistable_entry,
        slots=slots,
        index_memo=index_memo,
        hoistable_cache={},
        canonical_cache={},
    )
    entries_memo: dict[sp.Basic, tuple[Any, ...]] = {}
    ops_memo: dict[sp.Basic, Any] = {}
    sort_key_memo: dict[sp.Basic, Any] = {}

    while True:
        entries = [
            _preorder_entries(expr, own_entries, entries_memo)
            for expr in exprs
        ]
        counts, _sizes = _count_canonical_entries(entries, counted)
        best = _best_indexed_candidate(
            counts, expensive_twice, ops_memo, sort_key_memo
        )
        if best is None:
            break
        free = _slots_in_use(best, slots)
        target = _new_shared_temp(namer, free, arrays)
        definitions.append(
            TempDef(target=target, free=free, limits=[], body=best)
        )
        exprs = _replace_with_temp(exprs, entries, best, target, len(free))
    return exprs, definitions


# ---------------------------------------------------------------------------
# SymPy CSE backend
# ---------------------------------------------------------------------------


def _sympy_scalar_cse(
    expressions: Iterable[sp.Expr], namer: _Namer, arrays: dict[str, int]
) -> tuple[list[sp.Expr], list[TempDef]]:
    """Scalar common sub-expression elimination with :func:`sympy.cse`.

    Drop-in alternative to :func:`_common_scalar_cse` used to compare the
    greedy thermodiff pass against SymPy's own algorithm.  ``sympy.cse`` is
    unaware of index scopes, so every replacement that depends on an index
    (or contains a summation) is substituted back; only index-free scalar
    temporaries are kept.
    """
    exprs = list(expressions)
    if not exprs:
        return exprs, []
    replacements, reduced = sp.cse(
        exprs,
        symbols=sp.numbered_symbols("cse_tmp_"),
        order="canonical",
        optimizations=None,
    )
    inlined: dict[sp.Basic, sp.Basic] = {}
    kept: list[tuple[sp.Basic, sp.Basic]] = []
    for symbol, value in replacements:
        value = value.xreplace(inlined)
        if value.atoms(sp.Idx) or value.has(sp.Sum):
            inlined[symbol] = value
            continue
        kept.append((symbol, value))
    renames: dict[sp.Basic, sp.Basic] = {}
    definitions: list[TempDef] = []
    for symbol, value in kept:
        target = _new_shared_temp(namer, [], arrays)
        renames[symbol] = target
        definitions.append(
            TempDef(
                target=target, free=[], limits=[], body=value.xreplace(renames)
            )
        )
    reduced = [expr.xreplace(inlined).xreplace(renames) for expr in reduced]
    return reduced, definitions
