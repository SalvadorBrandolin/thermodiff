"""Invariant and affine hoisting for the Fortran code generator."""

from collections.abc import Callable, Iterable, Sequence
from typing import NamedTuple

import sympy as sp

from thermodiff.codegen.common import (
    _AUX_INDEX_NAMES,
    _LoopStart,
    _MAX_ARRAY_CSE_RANK,
    _Namer,
    _free_indices,
    _index_symbols,
    _is_expensive,
    _new_shared_temp,
    _temp_name,
)
from thermodiff.codegen.extents import slot_mapping
from thermodiff.codegen.ir import Limit, TempDef

#: A separable term ``(scalar, factor over first, factor over second)``.
_SeparableTerm = tuple[sp.Expr, sp.Expr, sp.Expr]

#: A rank-2 output residual and its free indices.
_Output = tuple[sp.Expr, Sequence[sp.Idx]]


class _HoistState(NamedTuple):
    """State of one :func:`_hoist_invariants` call.

    Attributes
    ----------
    slots : list of sympy.Idx
        Auxiliary indices free for the temporaries.
    namer : _Namer
        Generator of temporary names.
    arrays : dict
        Records the rank of every array temporary created.
    definitions : list
        Temporaries created since the last reset (mutated in place).
    temps : dict
        Canonical body -> target of every temporary created so far.
    """

    slots: list[sp.Idx]
    namer: _Namer
    arrays: dict[str, int]
    definitions: list[TempDef]
    temps: dict[sp.Basic, sp.Basic]


def _used_names(
    expressions: Iterable[sp.Basic],
    sum_definitions: Iterable[TempDef],
    arrays: Iterable[str],
    reserved: Iterable[object],
) -> set[str]:
    """Lower-cased names the slot indices of the temporaries must avoid.

    ``reserved`` holds every name of the subroutine (arguments, outputs,
    module symbols); the slot indices must not clash with any of them,
    case-insensitively (Fortran).
    """
    used = {
        str(symbol).lower()
        for expr in expressions
        for symbol in expr.atoms(sp.Symbol)
    }
    used.update(
        {
            str(b).lower()
            for expr in expressions
            for b in expr.atoms(sp.IndexedBase)
        }
    )
    for definition in sum_definitions:
        body = definition.body
        used.update(str(symbol).lower() for symbol in body.atoms(sp.Symbol))
        used.update(str(b).lower() for b in body.atoms(sp.IndexedBase))
    used.update({str(name).lower() for name in arrays})
    used.update(str(name).lower() for name in reserved)
    used.update({"nc", "res"})
    return used


def _new_hoist_state(
    used: set[str], namer: _Namer, arrays: dict[str, int]
) -> _HoistState:
    """Empty state whose slot indices avoid the names in ``used``."""
    return _HoistState(
        slots=[sp.Idx(name) for name in _AUX_INDEX_NAMES if name not in used],
        namer=namer,
        arrays=arrays,
        definitions=[],
        temps={},
    )


def _is_hoist_candidate(sub: sp.Basic, bound: set[sp.Idx]) -> bool:
    """Whether ``sub`` is a costly sub-expression worth hoisting.

    Parameters
    ----------
    sub : sympy.Basic
        Sub-expression inside the loops over ``bound``.
    bound : set
        Indices bound by the enclosing summations.
    """
    if not isinstance(sub, sp.Expr) or sub.is_Atom:
        return False
    if isinstance(sub, (sp.Indexed, sp.Sum, sp.Tuple)):
        return False
    if sub.has(sp.Sum) or sub.has(sp.Piecewise):
        return False
    if sub.is_Relational or sub.is_Boolean:
        return False
    if any(isinstance(s, _LoopStart) for s in sub.atoms(sp.Symbol)):
        return False
    indices = set(_index_symbols(sub))
    if len(indices) > _MAX_ARRAY_CSE_RANK or bound <= indices:
        return False
    return _is_expensive(sub)


def _hoisted_temporary(state: _HoistState, sub: sp.Expr) -> sp.Basic:
    """Replace ``sub`` by its temporary, created on first use.

    The temporary is canonicalised onto the slot indices, so equal
    sub-expressions over different indices share one definition.
    """
    indices = _index_symbols(sub)
    mapping = slot_mapping(indices, state.slots)
    canon = sub.xreplace(mapping)
    if canon not in state.temps:
        free = list(mapping.values())
        target = _new_shared_temp(state.namer, free, state.arrays)
        state.definitions.append(
            TempDef(target=target, free=free, limits=[], body=canon)
        )
        state.temps[canon] = target
    target = state.temps[canon]
    if not indices:
        return target
    return sp.Indexed(target.base, *indices)


def _rewrite_leaf(
    state: _HoistState, expr: sp.Basic, bound: set[sp.Idx]
) -> sp.Basic:
    """Rewrite an expression without arguments, ``Indexed`` or ``Piecewise``.

    Only a ``Piecewise`` that holds a sum is descended.
    """
    if isinstance(expr, sp.Piecewise) and expr.has(sp.Sum):
        return expr.func(
            *[_rewrite_hoisting(state, arg, bound) for arg in expr.args]
        )
    return expr


def _rewrite_terms(
    state: _HoistState, expr: sp.Basic, bound: set[sp.Idx]
) -> sp.Basic:
    """Rewrite a product or sum, hoisting the groups of shared-index factors.

    Groups of factors that share the same indices are hoisted when the group
    is costly enough to be a candidate.
    """
    groups: dict[frozenset[sp.Idx], list[sp.Basic]] = {}
    rest: list[sp.Basic] = []
    for arg in expr.args:
        indices = frozenset(_index_symbols(arg))
        if bound <= indices or arg.has(sp.Sum) or arg.is_Number:
            rest.append(_rewrite_hoisting(state, arg, bound))
        else:
            groups.setdefault(indices, []).append(arg)
    new_args = list(rest)
    for indices in sorted(groups, key=lambda s: sorted(map(str, s))):
        members = groups[indices]
        combined = expr.func(*members)
        if len(members) > 1 and _is_hoist_candidate(combined, bound):
            new_args.append(_hoisted_temporary(state, combined))
        else:
            new_args.extend(
                _rewrite_hoisting(state, arg, bound) for arg in members
            )
    return expr.func(*new_args)


def _rewrite_hoisting(
    state: _HoistState, expr: sp.Basic, bound: set[sp.Idx] | None
) -> sp.Basic:
    """Rewrite ``expr``; ``bound`` is None outside every summation."""
    if isinstance(expr, sp.Sum):
        inner = {limit[0] for limit in expr.limits}
        return expr.func(
            _rewrite_hoisting(state, expr.function, inner), *expr.limits
        )
    if bound is None:
        if not expr.args or not expr.has(sp.Sum):
            return expr
        return expr.func(
            *[_rewrite_hoisting(state, arg, None) for arg in expr.args]
        )
    if _is_hoist_candidate(expr, bound):
        return _hoisted_temporary(state, expr)
    if not expr.args or isinstance(expr, (sp.Indexed, sp.Piecewise)):
        return _rewrite_leaf(state, expr, bound)
    if isinstance(expr, (sp.Mul, sp.Add)):
        return _rewrite_terms(state, expr, bound)
    if isinstance(expr, sp.Basic) and not isinstance(expr, sp.Expr):
        return expr
    return expr.func(
        *[_rewrite_hoisting(state, arg, bound) for arg in expr.args]
    )


def _rewrite_sum_definitions(
    state: _HoistState, sum_definitions: list[TempDef]
) -> list[TempDef]:
    """Hoist from the body of every looped definition.

    The new temporaries are inserted right before the definition.
    """
    rewritten: list[TempDef] = []
    for definition in sum_definitions:
        if definition.limits:
            bound = {limit[0] for limit in definition.limits}
            definition = definition.copy(
                body=_rewrite_hoisting(state, definition.body, bound)
            )
            rewritten.extend(state.definitions)
            state.definitions.clear()
        rewritten.append(definition)
    return rewritten


def _hoist_invariants(
    expressions: Iterable[sp.Expr],
    namer: _Namer,
    arrays: dict[str, int],
    sum_definitions: Iterable[TempDef] = (),
    reserved: Iterable[object] = (),
) -> tuple[list[sp.Expr], list[TempDef], list[TempDef]]:
    """Move loop-invariant costly sub-expressions out of summation loops.

    Inside ``Sum(body, (a, ...), (b, ...))`` a sub-expression of ``body``
    whose indices do not include every bound index ``{a, b}`` is evaluated
    more times than needed: ``alpha(a)**k(a)`` inside a double loop costs
    ``nc**2`` powers instead of ``nc``.  This pass replaces every such
    sub-expression that is costly (see :func:`_is_expensive`) by a temporary
    indexed only by its own indices (a scalar when it has none), computed
    once before the loop -- even when it appears a single time, which is
    what :func:`_indexed_cse` would require it to be repeated for.

    Products are split by index dependence first, so that in
    ``sqrt(ac(q)*ac(s)*t(q)**k(q)*t(s)**k(s))`` the factor
    ``ac(q)*t(q)**k(q)`` becomes one rank-1 temporary, shared (after index
    canonicalisation) with ``ac(s)*t(s)**k(s)``.

    The pass runs twice: on the expressions before :func:`_share_sums`
    (``expressions``), and on the bodies of the summation temporaries it
    produced (``sum_definitions``), where sub-expressions built on other
    summations -- such as ``1/sum_l n(l)`` -- have become plain scalars.

    The rewrite is exact: every replaced sub-expression is the body of its
    temporary, evaluated over the same indices.

    Returns
    -------
    residuals : list
        ``expressions`` rewritten.
    definitions : list of dict
        Temporaries hoisted from ``expressions``.
    sum_definitions : list of dict
        ``sum_definitions`` rewritten, with the temporaries hoisted from
        each body inserted right before it.
    """
    exprs = list(expressions)
    sum_definitions = list(sum_definitions)
    used = _used_names(exprs, sum_definitions, arrays, reserved)
    state = _new_hoist_state(used, namer, arrays)
    exprs = [_rewrite_hoisting(state, expr, None) for expr in exprs]
    hoisted = state.definitions
    state = state._replace(definitions=[])
    return exprs, hoisted, _rewrite_sum_definitions(state, sum_definitions)


def _collect_terms(expr: sp.Expr) -> sp.Expr:
    """Group the terms of a sum by their index-dependent factors.

    ``a*x(i) + b*x(i)*y(j) + c*x(i) + d`` becomes
    ``(a + c)*x(i) + b*x(i)*y(j) + d``: the coefficients, free of indices,
    can then be computed once outside the loops.
    """
    if not isinstance(expr, sp.Add):
        return expr
    groups: dict[sp.Expr, list[sp.Expr]] = {}
    for term in expr.args:
        factors = term.args if isinstance(term, sp.Mul) else (term,)
        coef = [f for f in factors if not f.atoms(sp.Idx)]
        mono = [f for f in factors if f.atoms(sp.Idx)]
        key = sp.Mul(*mono)
        groups.setdefault(key, []).append(sp.Mul(*coef))
    if len(groups) == len(expr.args):
        return expr
    return sp.Add(
        *[
            sp.Add(*groups[key]) * key
            for key in sorted(groups, key=sp.default_sort_key)
        ]
    )


def _pair_term(term: sp.Expr, free: frozenset[sp.Idx]) -> bool:
    """Whether a factor of ``term`` depends on every index of ``free``.

    A ``Piecewise`` factor, as the Kronecker deltas, also counts.
    """
    return any(
        f.has(sp.Piecewise) or free <= f.atoms(sp.Idx)
        for f in sp.Mul.make_args(term)
    )


def _distribute(
    expr: sp.Expr, free: Iterable[sp.Idx], limit: int = 2000
) -> list[sp.Expr] | None:
    """Terms of ``expr`` with products distributed over the sums.

    Sums that have a term over some but not all of the indices ``free`` (a
    rank-2 residual) are distributed.  Scalar sums, pair sums such as
    ``1 - k(i, j)`` (every term scalar or over both indices) and function
    arguments stay as they are.  Returns None past ``limit`` terms.
    """
    free = frozenset(free)
    if isinstance(expr, sp.Add):
        terms: list[sp.Expr] = []
        for arg in expr.args:
            sub = _distribute(arg, free, limit)
            if sub is None:
                return None
            terms.extend(sub)
            if len(terms) > limit:
                return None
        return terms
    if isinstance(expr, sp.Mul):
        terms = [sp.S.One]
        for arg in expr.args:
            if isinstance(arg, sp.Add) and any(
                t.atoms(sp.Idx) & free and not _pair_term(t, free)
                for t in arg.args
            ):
                sub = _distribute(arg, free, limit)
            else:
                sub = [arg]
            if sub is None or len(terms) * len(sub) > limit:
                return None
            terms = [a * b for a in terms for b in sub]
        return terms
    return [expr]


def _split_separable(
    terms: Iterable[sp.Expr], free: Sequence[sp.Idx]
) -> tuple[list[_SeparableTerm], list[sp.Expr]]:
    """Split ``terms`` into separable ``(S, F, G)`` triples and the rest.

    ``S`` is the product of the scalar factors, ``F`` of the factors over
    ``free[0]`` only and ``G`` of those over ``free[1]`` only.  A pair term,
    or one lacking ``F`` or ``G``, goes to the rest unchanged.
    """
    first, second = free
    separable: list[_SeparableTerm] = []
    rest: list[sp.Expr] = []
    for term in terms:
        scalar: list[sp.Expr] = []
        over_first: list[sp.Expr] = []
        over_second: list[sp.Expr] = []
        pair = _pair_term(term, frozenset(free))
        for factor in sp.Mul.make_args(term):
            indices = factor.atoms(sp.Idx)
            if pair:
                break
            if first in indices:
                over_first.append(factor)
            elif second in indices:
                over_second.append(factor)
            else:
                scalar.append(factor)
        if pair or not over_first or not over_second:
            rest.append(term)
        else:
            separable.append(
                (
                    sp.Mul(*scalar),
                    sp.Mul(*over_first),
                    sp.Mul(*over_second),
                )
            )
    return separable, rest


def _separable_cost(separable: Iterable[_SeparableTerm]) -> int:
    """Cost per element of the separable terms.

    Computed after the scalar coefficients are hoisted: the products inside
    F and G, the scalar factor, F*G and the sum.
    """
    return sum(
        sp.count_ops(over_first, visual=False)
        + sp.count_ops(over_second, visual=False)
        + (scalar != 1)
        + 2
        for scalar, over_first, over_second in separable
    )


def _group_by_side(
    separable: Iterable[_SeparableTerm], side: sp.Idx, second: sp.Idx
) -> tuple[int, dict[sp.Expr, list[sp.Expr]]]:
    """Group the separable terms by the factor over the other index.

    Returns ``(cost, groups)``: the cost per element is one product and one
    sum per group (a compound key is computed once as a vector).
    """
    groups: dict[sp.Expr, list[sp.Expr]] = {}
    for scalar, over_first, over_second in separable:
        if side == second:
            key, coef = over_second, scalar * over_first
        else:
            key, coef = over_first, scalar * over_second
        groups.setdefault(key, []).append(coef)
    return 2 * len(groups), groups


def _grouped_definitions(
    groups: dict[sp.Expr, list[sp.Expr]],
    side: sp.Idx,
    other: sp.Idx,
    namer: _Namer,
    arrays: dict[str, int],
) -> tuple[list[sp.Expr], list[TempDef]]:
    """Terms ``coef * key`` of the groups, with the vectors they need.

    Returns ``(terms, definitions)``; a coefficient that is already a
    vector over ``other`` is used as it is.
    """
    terms: list[sp.Expr] = []
    definitions: list[TempDef] = []
    for key in sorted(groups, key=sp.default_sort_key):
        body = sp.Add(*groups[key])
        if isinstance(body, sp.Indexed) and body.indices == (other,):
            coef = body
        else:
            coef = _new_shared_temp(namer, [other], arrays)
            definitions.append(
                TempDef(target=coef, free=[other], limits=[], body=body)
            )
        if not isinstance(key, sp.Indexed):
            vec = _new_shared_temp(namer, [side], arrays)
            definitions.append(
                TempDef(target=vec, free=[side], limits=[], body=key)
            )
            key = vec
        terms.append(coef * key)
    return terms, definitions


def _lowrank_bilinear(
    expr: sp.Expr,
    free: Sequence[sp.Idx],
    namer: _Namer,
    arrays: dict[str, int],
) -> tuple[sp.Expr, list[TempDef]] | None:
    """Rank-k form of a rank-2 output ``H(p, q)``.

    Every term of the (distributed) residual is split into its scalar
    factors, its factors over ``p`` only, over ``q`` only, and its pair
    factors (both indices, or a ``Piecewise``).  The separable terms
    ``S F(p) G(q)`` are grouped by the factor over one index (the side with
    the cheaper result), ``sum_l r_l(p) G_l(q)``, and each coefficient
    vector ``r_l(p) = sum S F(p)`` becomes a rank-1 definition, O(nc).  The
    terms over one index only and the pair terms are kept as they are.

    Returns ``(new_expr, definitions)``, or None when there is nothing to
    group or the grouped form is not cheaper per element (``count_ops`` of the
    separable terms against the products and sums of the new form).
    """
    first, second = free
    terms = _distribute(expr, (first, second))
    if terms is None:
        return None
    separable, rest = _split_separable(terms, free)
    if not separable:
        return None
    cost_second, groups_second = _group_by_side(separable, second, second)
    cost_first, groups_first = _group_by_side(separable, first, second)
    side, cost, groups = (
        (second, cost_second, groups_second)
        if cost_second <= cost_first
        else (first, cost_first, groups_first)
    )
    if cost >= _separable_cost(separable):
        return None
    other = first if side == second else second
    new_terms, definitions = _grouped_definitions(
        groups, side, other, namer, arrays
    )
    return sp.Add(*rest, *new_terms), definitions


def _loop_names_in_use(
    definitions: Iterable[TempDef],
    outputs: Iterable[_Output],
    arrays: Iterable[str],
    reserved: Iterable[object],
) -> set[str]:
    """Lower-cased names already used by the definitions and the outputs.

    Parameters
    ----------
    definitions : list of TempDef
        Temporaries in dependency order.
    outputs : list of (expr, free_order)
        Output residuals.
    arrays : dict
        Array temporaries created so far (their names are read here).
    reserved : iterable
        Extra names that must stay free.

    Returns
    -------
    set of str
    """
    used = {"nc", "res"} | {str(x).lower() for x in reserved}
    for definition in definitions:
        used |= {str(x).lower() for x in definition.body.atoms(sp.Symbol)}
        used |= {str(x).lower() for x in definition.body.atoms(sp.IndexedBase)}
        used |= {str(x).lower() for x in definition.free}
    for expr, _ in outputs:
        used |= {str(x).lower() for x in expr.atoms(sp.Symbol)}
    used |= {str(name).lower() for name in arrays}
    return used


def _loop_aux_slots(
    definitions: Iterable[TempDef],
    outputs: Iterable[_Output],
    arrays: Iterable[str],
    reserved: Iterable[object],
) -> list[sp.Idx]:
    """Auxiliary index slots whose names are not in use.

    Returns
    -------
    list of sympy.Idx
    """
    used = _loop_names_in_use(definitions, outputs, arrays, reserved)
    return [sp.Idx(name) for name in _AUX_INDEX_NAMES if name not in used]


def _has_unhoistable_shape(sub: sp.Basic) -> bool:
    """Whether ``sub`` can never become a loop-invariant temporary."""
    if not isinstance(sub, sp.Expr) or sub.is_Atom or sub.is_Number:
        return True
    if isinstance(sub, (sp.Indexed, sp.Sum)) or sub.has(sp.Sum):
        return True
    if sub.has(sp.Piecewise) or sub.is_Relational or sub.is_Boolean:
        return True
    return any(isinstance(x, _LoopStart) for x in sub.atoms(sp.Symbol))


def _is_loop_candidate(sub: sp.Basic, bound: set[sp.Idx]) -> bool:
    """Whether ``sub`` is worth a temporary hoisted out of the loops."""
    if _has_unhoistable_shape(sub):
        return False
    indices = set(_index_symbols(sub))
    if indices >= bound or len(indices) > _MAX_ARRAY_CSE_RANK:
        return False
    if not indices:
        return True
    return _is_expensive(sub) or sp.count_ops(sub) >= 3


class _LoopHoister:
    """Per-call state of :func:`_hoist_loop_invariants`.

    Parameters
    ----------
    namer : _Namer
        Generator of temporary names; called by :meth:`make` only, in the
        depth-first order of :meth:`rewrite`.
    arrays : dict
        Records the rank of every array temporary created.
    slots : list of sympy.Idx
        Auxiliary indices used to canonicalise the temporaries.

    Attributes
    ----------
    temps : dict
        Canonical body -> target; a per-call cache.
    pending : list of TempDef
        Temporaries created and not yet placed before a definition.
    """

    def __init__(
        self, namer: _Namer, arrays: dict[str, int], slots: list[sp.Idx]
    ) -> None:
        self.namer = namer
        self.arrays = arrays
        self.slots = slots
        self.temps: dict[sp.Basic, sp.Basic] = {}
        self.pending: list[TempDef] = []

    def flush(self) -> list[TempDef]:
        """Return the pending temporaries and clear the list."""
        flushed = list(self.pending)
        self.pending.clear()
        return flushed

    def make(self, sub: sp.Expr) -> sp.Basic:
        """Return the reference to the temporary holding ``sub``."""
        indices = _index_symbols(sub)
        mapping = slot_mapping(indices, self.slots)
        canon = sub.xreplace(mapping)
        if canon not in self.temps:
            free = list(mapping.values())
            target = _new_shared_temp(self.namer, free, self.arrays)
            self.pending.append(
                TempDef(target=target, free=free, limits=[], body=canon)
            )
            self.temps[canon] = target
        target = self.temps[canon]
        return target if not indices else sp.Indexed(target.base, *indices)

    def rewrite(self, sub: sp.Basic, bound: set[sp.Idx]) -> sp.Basic:
        """Replace the hoistable parts of ``sub`` by temporaries."""
        if _is_loop_candidate(sub, bound):
            return self.make(sub)
        if not getattr(sub, "args", ()) or isinstance(
            sub, (sp.Indexed, sp.Idx)
        ):
            return sub
        if isinstance(sub, (sp.Sum, sp.Piecewise)):
            return sub
        if isinstance(sub, (sp.Mul, sp.Add)):
            return self._rewrite_terms(sub, bound)
        return sub.func(*[self.rewrite(arg, bound) for arg in sub.args])

    def _rewrite_terms(self, sub: sp.Basic, bound: set[sp.Idx]) -> sp.Basic:
        """Rewrite a product or sum, grouping the arguments by indices."""
        groups: dict[frozenset[sp.Idx], list[sp.Basic]] = {}
        rest: list[sp.Basic] = []
        for arg in sub.args:
            key = frozenset(_index_symbols(arg))
            if (
                key >= bound
                or arg.has(sp.Sum)
                or arg.is_Number
                or arg.has(sp.Piecewise)
            ):
                rest.append(self.rewrite(arg, bound))
            else:
                groups.setdefault(key, []).append(arg)
        new = list(rest)
        for key in sorted(groups, key=lambda x: sorted(map(str, x))):
            new.extend(self._rewrite_group(sub.func, groups[key], bound))
        return sub.func(*new)

    def _rewrite_group(
        self,
        func: Callable[..., sp.Basic],
        members: list[sp.Basic],
        bound: set[sp.Idx],
    ) -> list[sp.Basic]:
        """Rewrite the arguments of one index group; combine if possible."""
        combined = func(*members)
        if len(members) > 1 and _is_loop_candidate(combined, bound):
            return [self.make(combined)]
        return [self.rewrite(arg, bound) for arg in members]


def _hoist_loop_invariants(
    definitions: Iterable[TempDef],
    outputs: Iterable[_Output],
    namer: _Namer,
    arrays: dict[str, int],
    reserved: Iterable[object] = (),
) -> tuple[list[TempDef], list[sp.Expr]]:
    """Move loop invariants out of the loops over free indices.

    :func:`_hoist_invariants` handles the summation loops.  The loops over
    the *free* indices of a temporary or of an output -- the double loop of
    the composition Hessian, the loop of a rank-2 temporary -- also contain
    invariants: in ``dfdndn(i, j) = c1*x(i)*y(j) + c2*m(i, j) + ...`` the
    scalar coefficients ``c1``, ``c2`` do not depend on ``i`` or ``j``, and
    a factor ``g(i)`` does not depend on the inner index ``j``.

    The terms of every loop body are first grouped by their index-dependent
    factors (:func:`_collect_terms`).  Then every index-free compound
    sub-expression becomes a scalar temporary, and every costly
    sub-expression that depends on a strict subset of the loop indices
    becomes a temporary of that rank, computed once before the loop.

    Parameters
    ----------
    definitions : list of dict
        Temporaries in dependency order; the ones with free indices are
        rewritten and the new temporaries are inserted right before them.
    outputs : list of (expr, free_order)
        Output residuals; their new temporaries go after all definitions.

    Returns
    -------
    definitions : list of dict
    outputs : list of sympy expressions
    """
    slots = _loop_aux_slots(definitions, outputs, arrays, reserved)
    hoister = _LoopHoister(namer, arrays, slots)
    new_definitions: list[TempDef] = []
    for definition in definitions:
        bound = set(definition.free) | {lim[0] for lim in definition.limits}
        if bound:
            body = _collect_terms(definition.body)
            definition = definition.copy(body=hoister.rewrite(body, bound))
            new_definitions.extend(hoister.flush())
        new_definitions.append(definition)
    new_outputs: list[sp.Expr] = []
    for expr, free in outputs:
        if free:
            expr = hoister.rewrite(_collect_terms(expr), set(free))
        new_outputs.append(expr)
    new_definitions.extend(hoister.flush())
    return new_definitions, new_outputs


#: Keys a record may carry and still be split (``present_keys`` subset).
_AFFINE_FLAGS = frozenset({"target", "free", "limits", "body", "mirror"})
_AFFINE_SLOT = sp.Idx("__affine")


class _AffineMatch(NamedTuple):
    """Parts of a sum ``(const + var) * others`` over one ``nc`` loop."""

    index: sp.Idx
    low: sp.Basic
    high: sp.Basic
    add: sp.Add
    others: list[sp.Expr]


def _alpha(body: sp.Basic, index: sp.Idx) -> sp.Basic:
    """``body`` with the loop ``index`` renamed to a common slot."""
    return body.xreplace({index: _AFFINE_SLOT})


def _is_affine_shape(definition: TempDef) -> bool:
    """Whether ``definition`` is a one-loop ``nc`` sum with a Mul body."""
    free = set(definition.free)
    if (
        not free
        or len(definition.limits) != 1
        or not isinstance(definition.target, sp.Indexed)
        or not definition.present_keys() <= _AFFINE_FLAGS
    ):
        return False
    _, low, high = definition.limits[0]
    body = definition.body
    return not (
        low != 1
        or str(high) != "nc"
        or not isinstance(body, sp.Mul)
        or body.has(sp.Piecewise)
        or any(isinstance(x, _LoopStart) for x in body.atoms(sp.Symbol))
    )


def _affine_match(definition: TempDef) -> _AffineMatch | None:
    """Split the body of ``definition`` as ``(const + var) * others``.

    Returns
    -------
    _AffineMatch or None
        ``None`` unless exactly one ``Add`` factor depends on the free
        indices and no other factor does.
    """
    if not _is_affine_shape(definition):
        return None
    free = set(definition.free)
    body = definition.body
    adds = [
        f
        for f in body.args
        if isinstance(f, sp.Add) and f.atoms(sp.Idx) & free
    ]
    others = [
        f for f in body.args if not (isinstance(f, sp.Add) and f in adds)
    ]
    if len(adds) != 1 or any(f.atoms(sp.Idx) & free for f in others):
        return None
    index, low, high = definition.limits[0]
    return _AffineMatch(index, low, high, adds[0], others)


def _split_terms(
    add: sp.Add, free: set[sp.Idx]
) -> tuple[list[sp.Expr], list[sp.Expr]]:
    """``(const, var)`` terms of ``add`` by dependence on ``free``."""
    const = [a for a in add.args if not (a.atoms(sp.Idx) & free)]
    var = [a for a in add.args if a.atoms(sp.Idx) & free]
    return const, var


def _would_split(definition: TempDef) -> bool:
    """Whether ``definition`` would be split (no side effects)."""
    match = _affine_match(definition)
    if match is None:
        return False
    const, var = _split_terms(match.add, set(definition.free))
    if not const or not var:
        return False
    inner = sp.Add(*var)
    if inner.could_extract_minus_sign():
        inner = -inner
    factors = sp.Mul.make_args(inner * sp.Mul(*match.others))
    only = [
        f
        for f in factors
        if f.atoms(sp.Idx) == {match.index} and not f.is_Number
    ]
    if len(only) >= 2:
        factors = [f for f in factors if f not in only] + [
            sp.Indexed(sp.IndexedBase("__vec"), match.index)
        ]
    return bool(sp.count_ops(sp.Mul(*factors)) < sp.count_ops(definition.body))


def _matrix_bases(expr: sp.Basic) -> set[str]:
    """Names of the bases of the rank-2 or higher arrays read by ``expr``."""
    return {str(x.base) for x in expr.atoms(sp.Indexed) if len(x.indices) >= 2}


def _loop_shape(
    definition: TempDef,
) -> tuple[int, tuple[tuple[sp.Basic, sp.Basic], ...]] | None:
    """Free count and limits of a sum, or ``None`` for other records."""
    if not definition.free or not definition.limits:
        return None
    return (
        len(definition.free),
        tuple((lo, hi) for _b, lo, hi in definition.limits),
    )


def _blocked_by_pairs(
    defs: Iterable[TempDef], outs: Iterable[sp.Expr]
) -> set[str]:
    """Names of the sums whose matrix a rank-2 output reads too."""
    pair_bases: set[str] = set()
    for expr in outs:
        if len(_free_indices(expr)) >= 2:
            pair_bases |= _matrix_bases(expr)
    return {
        _temp_name(definition.target)
        for definition in defs
        if definition.free
        and definition.limits
        and _matrix_bases(definition.body) & pair_bases
    }


def _blocked_by_runs(defs: list[TempDef]) -> set[str]:
    """Names of the sums of a run of equal loops that cannot all split."""
    blocked: set[str] = set()
    run: list[TempDef] = []
    for definition in defs + [None]:
        if (
            definition is not None
            and run
            and _loop_shape(definition) is not None
            and _loop_shape(definition) == _loop_shape(run[0])
        ):
            run.append(definition)
            continue
        if len(run) > 1 and not all(_would_split(member) for member in run):
            blocked |= {_temp_name(member.target) for member in run}
        run = (
            [definition]
            if definition is not None and _loop_shape(definition) is not None
            else []
        )
    return blocked


def _close_blocked_matrices(sums: list[TempDef], blocked: set[str]) -> None:
    """Block the sums that read a matrix read by a blocked sum (fixpoint)."""
    changed = True
    while changed:
        changed = False
        held = set().union(
            *[
                _matrix_bases(definition.body)
                for definition in sums
                if _temp_name(definition.target) in blocked
            ]
        )
        for definition in sums:
            name = _temp_name(definition.target)
            if name not in blocked and _matrix_bases(definition.body) & held:
                blocked.add(name)
                changed = True


def _blocked_sums(defs: list[TempDef], outs: list[sp.Expr]) -> set[str]:
    """Names of the sums that must not be split.

    Sums over the same loops next to each other share one loop nest
    (fuse_loops, sib_fuse, sym_matvec): split all of them or none.  Neither
    split a sum whose matrix a rank-2 output reads too (the Hessian of the
    cubics reads kij next to the matrix-vector products that share it).
    Sums that read the same matrix go together: all split or none.
    """
    blocked = _blocked_by_pairs(defs, outs)
    blocked |= _blocked_by_runs(defs)
    sums = [
        definition
        for definition in defs
        if definition.free and definition.limits
    ]
    for definition in sums:
        if _temp_name(definition.target) not in blocked and not _would_split(
            definition
        ):
            blocked.add(_temp_name(definition.target))
    _close_blocked_matrices(sums, blocked)
    return blocked


class _AffineState:
    """Per-call state of :func:`_split_affine`.

    Parameters
    ----------
    definitions : list of TempDef
        The definitions; copied, and edited in place afterwards.
    namer : _Namer
        Generator of temporary names.
    arrays : dict
        Records the rank of every array temporary created.

    Attributes
    ----------
    defs : list of TempDef
        The working definitions.
    """

    def __init__(
        self,
        definitions: Iterable[TempDef],
        namer: _Namer,
        arrays: dict[str, int],
    ) -> None:
        self.defs = list(definitions)
        self.namer = namer
        self.arrays = arrays

    def vector_of(
        self, product: sp.Expr, index: sp.Idx, position: int
    ) -> tuple[sp.Indexed, int]:
        """Indexed vector holding ``product`` (over ``index`` only).

        Returns
        -------
        (sympy.Indexed, int)
            The reference and the next insertion position.  A new vector
            consumes a namer number and is inserted at ``position`` even if
            the caller later drops it (see :func:`_split_at`).
        """
        key = _alpha(product, index)
        for definition in self.defs:
            if (
                len(definition.free) == 1
                and not definition.limits
                and definition.present_keys() <= _AFFINE_FLAGS
                and isinstance(definition.target, sp.Indexed)
                and _alpha(definition.body, definition.free[0]) == key
            ):
                return sp.Indexed(definition.target.base, index), position
        target = _new_shared_temp(self.namer, [index], self.arrays)
        self.defs.insert(
            position,
            TempDef(target=target, free=[index], limits=[], body=product),
        )
        return sp.Indexed(target.base, index), position + 1

    def vectorise(
        self, body: sp.Expr, index: sp.Idx, position: int
    ) -> tuple[sp.Expr, int]:
        """Replace the factors over ``index`` only by one vector."""
        factors = sp.Mul.make_args(body)
        only = [
            f
            for f in factors
            if f.atoms(sp.Idx) == {index} and not f.is_Number
        ]
        if len(only) < 2:
            return body, position
        vec, position = self.vector_of(sp.Mul(*only), index, position)
        rest = [f for f in factors if f not in only]
        return sp.Mul(*rest) * vec, position

    def scalar_for(
        self, position: int, body: sp.Expr, limit: Limit
    ) -> tuple[sp.Symbol, int]:
        """Scalar sum of ``body`` over ``limit``: reuse or create it.

        Returns
        -------
        (sympy.Symbol, int)
            The scalar and the next insertion position.
        """
        index, low, high = limit
        key = _alpha(body, index)
        for other in self.defs[:position]:
            if (
                not other.free
                and len(other.limits) == 1
                and other.present_keys() <= _AFFINE_FLAGS
                and isinstance(other.target, sp.Symbol)
                and tuple(other.limits[0][1:]) == (low, high)
                and _alpha(other.body, other.limits[0][0]) == key
            ):
                return other.target, position
        scalar = _new_shared_temp(self.namer, [], self.arrays)
        self.defs.insert(
            position,
            TempDef(
                target=scalar, free=[], limits=[(index, low, high)], body=body
            ),
        )
        return scalar, position + 1


def _reread(
    expr: sp.Expr, name: str, scalar: sp.Expr, coeff: sp.Expr, sign: int
) -> sp.Expr:
    """Rewrite the reads of the split table ``name`` in ``expr``."""
    base = scalar if float(coeff) == 1.0 else coeff * scalar
    rep = {
        x: base + sign * x
        for x in expr.atoms(sp.Indexed)
        if str(x.base) == name
    }
    return expr.xreplace(rep) if rep else expr


def _split_at(
    state: _AffineState, k: int, outs: list[sp.Expr]
) -> tuple[int, list[sp.Expr]]:
    """Try to split the definition at ``k``.

    Returns
    -------
    (int, list)
        The index to continue from and the (possibly rewritten) outputs.
    """
    defs = state.defs
    definition = defs[k]
    match = _affine_match(definition)
    if match is None:
        return k + 1, outs
    const, var = _split_terms(match.add, set(definition.free))
    if not const or not var:
        return k + 1, outs
    m = match.index
    weight = sp.Mul(*match.others)
    inner = sp.Add(*var)
    sign = 1
    if inner.could_extract_minus_sign():
        inner, sign = -inner, -1
    position = k
    new_body, position = state.vectorise(inner * weight, m, position)
    if sp.count_ops(new_body) >= sp.count_ops(definition.body):
        # The vectors made for nothing leave the definitions, but their
        # namer number and their entry in ``arrays`` stay (and shift every
        # later temporary name).  Candidate cleanup; it changes the output.
        del defs[k:position]
        return k + 1, outs
    coeff = sp.Add(*const)
    if not coeff.is_Number:
        weight, coeff = coeff * weight, sp.S.One
    s_body, position = state.vectorise(weight, m, position)
    scalar, position = state.scalar_for(position, s_body, definition.limits[0])
    table = definition.copy(body=new_body)
    defs[position] = table
    name = _temp_name(definition.target)
    for q, other in enumerate(defs):
        if other is not table:
            defs[q] = other.copy(
                body=_reread(other.body, name, scalar, coeff, sign)
            )
    outs = [_reread(expr, name, scalar, coeff, sign) for expr in outs]
    return position + 1, outs


def _split_affine(
    definitions: Iterable[TempDef],
    outputs: Iterable[sp.Expr],
    namer: _Namer,
    arrays: dict[str, int],
) -> tuple[list[TempDef], list[sp.Expr]]:
    """Take the free-index-invariant part of a sum out of the loop nest.

    In ``t(k, l) = sum_m (1 - L(m, k, l)) n(m)`` (the O(N^3) nest of CMR,
    reportes/brechas/ronda2-ge-cmr.md) the term ``n(m)`` does not depend on
    ``k, l``: ``t(k, l) = S - sum_m L(m, k, l) n(m)`` with the scalar
    ``S = sum_m n(m)`` computed once, so the innermost body is one
    multiply-add.
    A definition with free indices and one summation, whose body is a product
    ``A * w`` with an ``A = a + b`` such that ``a`` and ``w`` do not depend on
    the free indices, keeps ``sum_m b w`` (sign folded in) and its readers read
    ``S +- t``; a product of several factors over the bound index only
    (``cmp0(m)*n(m)``) becomes a vector, an existing one when a map with the
    same body exists.  Applied only when the body gets cheaper, and to a run of
    consecutive sums over the same loops (fused into one nest later) only when
    every member of the run can be split; the sums that read one matrix are
    split together or not at all; never when a rank-2 output reads the matrix.

    Returns ``(definitions, outputs)``.
    """
    state = _AffineState(definitions, namer, arrays)
    outs = list(outputs)
    blocked = _blocked_sums(state.defs, outs)
    k = 0
    while k < len(state.defs):
        if _temp_name(state.defs[k].target) in blocked:
            k += 1
            continue
        k, outs = _split_at(state, k, outs)
    return state.defs, outs


def _split_factored(
    definition: TempDef, limits: Sequence[Limit]
) -> tuple[sp.Expr, sp.Expr]:
    """``(outer factor, inner body)`` of a ``factored`` double reduction."""
    inner = {lim[0] for lim in limits[1:]}
    factors = sp.Mul.make_args(definition.body)
    o_f = [f for f in factors if not (f.atoms(sp.Idx) & inner)]
    i_f = [f for f in factors if f.atoms(sp.Idx) & inner]
    return sp.Mul(*o_f), sp.Mul(*i_f)


# ---------------------------------------------------------------------------
# Symmetry handling
# ---------------------------------------------------------------------------


def _separate_sums(
    expr: sp.Expr, max_terms: int = 16, selective: bool = False
) -> sp.Expr:
    """Distribute products over sums inside summation bodies and split.

    ``sum_k (r_i r_j r_k + s_i s_k) n_k`` becomes
    ``r_i r_j sum_k r_k n_k + s_i sum_k s_k n_k``: after the split, the
    factors that do not depend on the bound index leave each sum
    (:func:`thermodiff.chain.factor_sums`), and a rank-2 sum over a third
    index may become an outer product of a scalar sum.  Bodies that would
    expand into more than ``max_terms`` terms are left alone.
    """
    from thermodiff.chain import factor_sums

    def visit(sub: sp.Basic) -> sp.Basic:
        if isinstance(sub, sp.Sum):
            body = visit(sub.function)
            if body.has(sp.Piecewise):
                return sub.func(body, *sub.limits)
            # inner sums are atoms for the expansion
            original = sub.func(body, *sub.limits)
            expanded = sp.expand_mul(body)
            if (
                isinstance(expanded, sp.Add)
                and len(expanded.args) <= max_terms
            ):
                # thermodiff.chain is not typed yet
                split = factor_sums(  # type: ignore[no-untyped-call]
                    sp.Add(*[sp.Sum(t, *sub.limits) for t in expanded.args])
                )
                if not selective or cost(split) < cost(original):
                    return split
            return original
        if not getattr(sub, "args", ()) or isinstance(
            sub, (sp.Idx, sp.Indexed)
        ):
            return sub
        return sub.func(*[visit(arg) for arg in sub.args])

    def cost(expr: sp.Basic) -> int:
        """Estimated work: sum over loop nests of nc**depth (nc = 10)."""
        total: int = 0
        for total_sum in expr.atoms(sp.Sum):
            depth = len(_free_indices(total_sum)) + len(total_sum.limits)
            total += 10**depth
        return total

    return visit(expr)
