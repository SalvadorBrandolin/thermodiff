"""Invariant and affine hoisting for the Fortran code generator."""

import sympy as sp

from thermodiff.codegen.common import (
    _AUX_INDEX_NAMES,
    _LoopStart,
    _MAX_ARRAY_CSE_RANK,
    _free_indices,
    _index_symbols,
    _is_expensive,
    _new_shared_temp,
    _temp_name,
)
from thermodiff.codegen.ir import TempDef


def _hoist_invariants(
    expressions, namer, arrays, sum_definitions=(), reserved=()
):
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
    # ``reserved``: every name of the subroutine (arguments, outputs, module
    # symbols); the slot indices of the temporaries must not clash with any
    # of them, case-insensitively (Fortran).
    used = {
        str(symbol).lower()
        for expr in exprs
        for symbol in expr.atoms(sp.Symbol)
    }
    used.update(
        {str(b).lower() for expr in exprs for b in expr.atoms(sp.IndexedBase)}
    )
    for definition in sum_definitions:
        body = definition.body
        used.update(str(symbol).lower() for symbol in body.atoms(sp.Symbol))
        used.update(str(b).lower() for b in body.atoms(sp.IndexedBase))
    used.update({str(name).lower() for name in arrays})
    used.update(str(name).lower() for name in reserved)
    used.update({"nc", "res"})
    slots = [sp.Idx(name) for name in _AUX_INDEX_NAMES if name not in used]
    definitions = []
    temps = {}

    def candidate(sub, bound):
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

    def temporary(sub):
        indices = _index_symbols(sub)
        mapping = {index: slots[pos] for pos, index in enumerate(indices)}
        canon = sub.xreplace(mapping)
        if canon not in temps:
            free = slots[: len(indices)]
            target = _new_shared_temp(namer, free, arrays)
            definitions.append(
                TempDef(target=target, free=free, limits=[], body=canon)
            )
            temps[canon] = target
        target = temps[canon]
        if not indices:
            return target
        return sp.Indexed(target.base, *indices)

    def rewrite(expr, bound):
        if isinstance(expr, sp.Sum):
            inner = {limit[0] for limit in expr.limits}
            return expr.func(rewrite(expr.function, inner), *expr.limits)
        if bound is None:
            if not expr.args or not expr.has(sp.Sum):
                return expr
            return expr.func(*[rewrite(arg, None) for arg in expr.args])
        if candidate(expr, bound):
            return temporary(expr)
        if not expr.args or isinstance(expr, (sp.Indexed, sp.Piecewise)):
            if isinstance(expr, sp.Piecewise) and expr.has(sp.Sum):
                return expr.func(*[rewrite(arg, bound) for arg in expr.args])
            return expr
        if isinstance(expr, (sp.Mul, sp.Add)):
            groups = {}
            rest = []
            for arg in expr.args:
                indices = frozenset(_index_symbols(arg))
                if bound <= indices or arg.has(sp.Sum) or arg.is_Number:
                    rest.append(rewrite(arg, bound))
                else:
                    groups.setdefault(indices, []).append(arg)
            new_args = list(rest)
            for indices in sorted(groups, key=lambda s: sorted(map(str, s))):
                members = groups[indices]
                combined = expr.func(*members)
                if len(members) > 1 and candidate(combined, bound):
                    new_args.append(temporary(combined))
                else:
                    new_args.extend(rewrite(arg, bound) for arg in members)
            return expr.func(*new_args)
        if isinstance(expr, sp.Basic) and not isinstance(expr, sp.Expr):
            return expr
        return expr.func(*[rewrite(arg, bound) for arg in expr.args])

    exprs = [rewrite(expr, None) for expr in exprs]
    hoisted = definitions
    definitions = []
    rewritten = []
    for definition in sum_definitions:
        if definition.limits:
            bound = {limit[0] for limit in definition.limits}
            definition = definition.copy(body=rewrite(definition.body, bound))
            rewritten.extend(definitions)
            definitions.clear()
        rewritten.append(definition)
    return exprs, hoisted, rewritten


def _collect_terms(expr):
    """Group the terms of a sum by their index-dependent factors.

    ``a*x(i) + b*x(i)*y(j) + c*x(i) + d`` becomes
    ``(a + c)*x(i) + b*x(i)*y(j) + d``: the coefficients, free of indices,
    can then be computed once outside the loops.
    """
    if not isinstance(expr, sp.Add):
        return expr
    groups = {}
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


def _pair_term(term, free):
    """Whether a factor of ``term`` depends on every index of ``free`` (or is
    a ``Piecewise``, as the Kronecker deltas)."""
    return any(
        f.has(sp.Piecewise) or free <= f.atoms(sp.Idx)
        for f in sp.Mul.make_args(term)
    )


def _distribute(expr, free, limit=2000):
    """Terms of ``expr`` with products distributed over the sums that have a
    term over some but not all of the indices ``free`` (a rank-2 residual).
    Scalar sums, pair sums such as ``1 - k(i, j)`` (every term scalar or over
    both indices) and function arguments stay as they are.  Returns None past
    ``limit`` terms."""
    free = frozenset(free)
    if isinstance(expr, sp.Add):
        terms = []
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


def _lowrank_bilinear(expr, free, namer, arrays):
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
    p_, q_ = free
    terms = _distribute(expr, (p_, q_))
    if terms is None:
        return None
    sep, rest = [], []
    for term in terms:
        S, F, G = [], [], []
        pair = _pair_term(term, frozenset(free))
        for factor in sp.Mul.make_args(term):
            idx = factor.atoms(sp.Idx)
            if pair:
                break
            if p_ in idx:
                F.append(factor)
            elif q_ in idx:
                G.append(factor)
            else:
                S.append(factor)
        if pair or not F or not G:
            rest.append(term)
        else:
            sep.append((sp.Mul(*S), sp.Mul(*F), sp.Mul(*G)))
    if not sep:
        return None
    # per element, after the scalar coefficients are hoisted: the products
    # inside F and G, the scalar factor, F*G and the sum
    old_cost = sum(
        sp.count_ops(F, visual=False)
        + sp.count_ops(G, visual=False)
        + (S != 1)
        + 2
        for S, F, G in sep
    )

    def grouped(side):
        # cost per element: one product and one sum per group (a compound
        # key is computed once as a vector)
        groups = {}
        for S, F, G in sep:
            key, coef = (G, S * F) if side == q_ else (F, S * G)
            groups.setdefault(key, []).append(coef)
        return 2 * len(groups), groups

    cost_q, groups_q = grouped(q_)
    cost_p, groups_p = grouped(p_)
    side, cost, groups = (
        (q_, cost_q, groups_q) if cost_q <= cost_p else (p_, cost_p, groups_p)
    )
    if cost >= old_cost:
        return None
    other = p_ if side == q_ else q_
    definitions = []
    new_terms = list(rest)
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
        new_terms.append(coef * key)
    return sp.Add(*new_terms), definitions


def _hoist_loop_invariants(definitions, outputs, namer, arrays, reserved=()):
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
    used = {"nc", "res"} | {str(x).lower() for x in reserved}
    for definition in definitions:
        used |= {str(x).lower() for x in definition.body.atoms(sp.Symbol)}
        used |= {str(x).lower() for x in definition.body.atoms(sp.IndexedBase)}
        used |= {str(x).lower() for x in definition.free}
    for expr, _ in outputs:
        used |= {str(x).lower() for x in expr.atoms(sp.Symbol)}
    used |= {str(name).lower() for name in arrays}
    slots = [sp.Idx(name) for name in _AUX_INDEX_NAMES if name not in used]
    temps = {}
    pending = []

    def make(sub):
        indices = _index_symbols(sub)
        mapping = {index: slots[pos] for pos, index in enumerate(indices)}
        canon = sub.xreplace(mapping)
        if canon not in temps:
            free = slots[: len(indices)]
            target = _new_shared_temp(namer, free, arrays)
            pending.append(
                TempDef(target=target, free=free, limits=[], body=canon)
            )
            temps[canon] = target
        target = temps[canon]
        return target if not indices else sp.Indexed(target.base, *indices)

    def candidate(sub, bound):
        if not isinstance(sub, sp.Expr) or sub.is_Atom or sub.is_Number:
            return False
        if isinstance(sub, (sp.Indexed, sp.Sum)) or sub.has(sp.Sum):
            return False
        if sub.has(sp.Piecewise) or sub.is_Relational or sub.is_Boolean:
            return False
        if any(isinstance(x, _LoopStart) for x in sub.atoms(sp.Symbol)):
            return False
        indices = set(_index_symbols(sub))
        if indices >= bound or len(indices) > _MAX_ARRAY_CSE_RANK:
            return False
        if not indices:
            return True
        return _is_expensive(sub) or sp.count_ops(sub) >= 3

    def rewrite(sub, bound):
        if candidate(sub, bound):
            return make(sub)
        if not getattr(sub, "args", ()) or isinstance(
            sub, (sp.Indexed, sp.Idx)
        ):
            return sub
        if isinstance(sub, sp.Sum) or isinstance(sub, sp.Piecewise):
            return sub
        if isinstance(sub, (sp.Mul, sp.Add)):
            groups = {}
            rest = []
            for arg in sub.args:
                key = frozenset(_index_symbols(arg))
                if (
                    key >= bound
                    or arg.has(sp.Sum)
                    or arg.is_Number
                    or arg.has(sp.Piecewise)
                ):
                    rest.append(rewrite(arg, bound))
                else:
                    groups.setdefault(key, []).append(arg)
            new = list(rest)
            for key in sorted(groups, key=lambda x: sorted(map(str, x))):
                members = groups[key]
                combined = sub.func(*members)
                if len(members) > 1 and candidate(combined, bound):
                    new.append(make(combined))
                else:
                    new.extend(rewrite(arg, bound) for arg in members)
            return sub.func(*new)
        return sub.func(*[rewrite(arg, bound) for arg in sub.args])

    new_definitions = []
    for definition in definitions:
        bound = set(definition.free) | {lim[0] for lim in definition.limits}
        if bound:
            body = _collect_terms(definition.body)
            definition = definition.copy(body=rewrite(body, bound))
            new_definitions.extend(pending)
            pending.clear()
        new_definitions.append(definition)
    new_outputs = []
    for expr, free in outputs:
        if free:
            expr = rewrite(_collect_terms(expr), set(free))
        new_outputs.append(expr)
    new_definitions.extend(pending)
    pending.clear()
    return new_definitions, new_outputs


def _split_affine(definitions, outputs, namer, arrays):
    """Take the part of a summation body that does not depend on the free
    indices out of the loop nest.

    In ``t(k, l) = sum_m (1 - L(m, k, l)) n(m)`` (the O(N^3) nest of CMR,
    reportes/brechas/ronda2-ge-cmr.md) the term ``n(m)`` does not depend on
    ``k, l``: ``t(k, l) = S - sum_m L(m, k, l) n(m)`` with the scalar
    ``S = sum_m n(m)`` computed once, so the innermost body is one multiply-add.
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
    slot = sp.Idx("__affine")
    flags_ok = {"target", "free", "limits", "body", "mirror"}

    def alpha(body, index):
        return body.xreplace({index: slot})

    defs = list(definitions)
    outs = list(outputs)

    def vector_of(product, index, position):
        """Indexed vector holding ``product`` (over ``index`` only)."""
        key = alpha(product, index)
        for d in defs:
            if (
                len(d.free) == 1
                and not d.limits
                and d.present_keys() <= flags_ok
                and isinstance(d.target, sp.Indexed)
                and alpha(d.body, d.free[0]) == key
            ):
                return sp.Indexed(d.target.base, index), position
        target = _new_shared_temp(namer, [index], arrays)
        defs.insert(
            position,
            TempDef(target=target, free=[index], limits=[], body=product),
        )
        return sp.Indexed(target.base, index), position + 1

    def vectorise(body, index, position):
        factors = sp.Mul.make_args(body)
        only = [
            f
            for f in factors
            if f.atoms(sp.Idx) == {index} and not f.is_Number
        ]
        if len(only) < 2:
            return body, position
        vec, position = vector_of(sp.Mul(*only), index, position)
        rest = [f for f in factors if f not in only]
        return sp.Mul(*rest) * vec, position

    def eligible(d):
        """Whether ``d`` would be split (same tests as below, no side effects)."""
        free = set(d.free)
        if (
            not free
            or len(d.limits) != 1
            or not isinstance(d.target, sp.Indexed)
            or not d.present_keys() <= flags_ok
        ):
            return False
        m, low, high = d.limits[0]
        body = d.body
        if (
            low != 1
            or str(high) != "nc"
            or not isinstance(body, sp.Mul)
            or body.has(sp.Piecewise)
            or any(isinstance(x, _LoopStart) for x in body.atoms(sp.Symbol))
        ):
            return False
        adds = [
            f
            for f in body.args
            if isinstance(f, sp.Add) and f.atoms(sp.Idx) & free
        ]
        others = [
            f for f in body.args if not (isinstance(f, sp.Add) and f in adds)
        ]
        if len(adds) != 1 or any(f.atoms(sp.Idx) & free for f in others):
            return False
        var = [a for a in adds[0].args if a.atoms(sp.Idx) & free]
        if len(var) == len(adds[0].args) or not var:
            return False
        inner = sp.Add(*var)
        if inner.could_extract_minus_sign():
            inner = -inner
        factors = sp.Mul.make_args(inner * sp.Mul(*others))
        only = [
            f for f in factors if f.atoms(sp.Idx) == {m} and not f.is_Number
        ]
        if len(only) >= 2:
            factors = [f for f in factors if f not in only] + [
                sp.Indexed(sp.IndexedBase("__vec"), m)
            ]
        return sp.count_ops(sp.Mul(*factors)) < sp.count_ops(body)

    # sums over the same loops next to each other share one loop nest
    # (fuse_loops, sib_fuse, sym_matvec): split all of them or none.  Neither
    # split a sum whose matrix a rank-2 output reads too (the Hessian of the
    # cubics reads kij next to the matrix-vector products that share it).
    blocked = set()
    pair_bases = {
        str(x.base)
        for e in outs
        if len(_free_indices(e)) >= 2
        for x in e.atoms(sp.Indexed)
        if len(x.indices) >= 2
    }
    for d in defs:
        if (
            d.free
            and d.limits
            and any(
                str(x.base) in pair_bases
                for x in d.body.atoms(sp.Indexed)
                if len(x.indices) >= 2
            )
        ):
            blocked.add(_temp_name(d.target))
    run = []

    def shape(d):
        if not d.free or not d.limits:
            return None
        return (len(d.free), tuple((lo, hi) for _b, lo, hi in d.limits))

    for d in defs + [None]:
        if (
            d is not None
            and run
            and shape(d) is not None
            and shape(d) == shape(run[0])
        ):
            run.append(d)
            continue
        if len(run) > 1 and not all(eligible(e) for e in run):
            blocked |= {_temp_name(e.target) for e in run}
        run = [d] if d is not None and shape(d) is not None else []

    # sums that read the same matrix go together: all split or none
    def matrices(d):
        return {
            str(x.base)
            for x in d.body.atoms(sp.Indexed)
            if len(x.indices) >= 2
        }

    sums = [d for d in defs if d.free and d.limits]
    for d in sums:
        if _temp_name(d.target) not in blocked and not eligible(d):
            blocked.add(_temp_name(d.target))
    changed = True
    while changed:
        changed = False
        held = set().union(
            *[matrices(d) for d in sums if _temp_name(d.target) in blocked]
        )
        for d in sums:
            if _temp_name(d.target) not in blocked and matrices(d) & held:
                blocked.add(_temp_name(d.target))
                changed = True

    k = 0
    while k < len(defs):
        d = defs[k]
        if _temp_name(d.target) in blocked:
            k += 1
            continue
        free = set(d.free)
        if (
            not free
            or len(d.limits) != 1
            or not isinstance(d.target, sp.Indexed)
            or not d.present_keys() <= flags_ok
        ):
            k += 1
            continue
        m, low, high = d.limits[0]
        body = d.body
        if (
            low != 1
            or str(high) != "nc"
            or not isinstance(body, sp.Mul)
            or body.has(sp.Piecewise)
            or any(isinstance(x, _LoopStart) for x in body.atoms(sp.Symbol))
        ):
            k += 1
            continue
        adds = [
            f
            for f in body.args
            if isinstance(f, sp.Add) and f.atoms(sp.Idx) & free
        ]
        others = [
            f for f in body.args if not (isinstance(f, sp.Add) and f in adds)
        ]
        if len(adds) != 1 or any(f.atoms(sp.Idx) & free for f in others):
            k += 1
            continue
        const = [a for a in adds[0].args if not (a.atoms(sp.Idx) & free)]
        var = [a for a in adds[0].args if a.atoms(sp.Idx) & free]
        if not const or not var:
            k += 1
            continue
        weight = sp.Mul(*others)
        inner = sp.Add(*var)
        sign = 1
        if inner.could_extract_minus_sign():
            inner, sign = -inner, -1
        position = k
        new_body, position = vectorise(inner * weight, m, position)
        if sp.count_ops(new_body) >= sp.count_ops(body):
            del defs[k:position]  # vectors made for nothing
            k += 1
            continue
        coeff = sp.Add(*const)
        if not coeff.is_Number:
            weight, coeff = coeff * weight, sp.S.One
        s_body, position = vectorise(weight, m, position)
        key = alpha(s_body, m)
        scalar = None
        for e in defs[:position]:
            if (
                not e.free
                and len(e.limits) == 1
                and e.present_keys() <= flags_ok
                and isinstance(e.target, sp.Symbol)
                and tuple(e.limits[0][1:]) == (low, high)
                and alpha(e.body, e.limits[0][0]) == key
            ):
                scalar = e.target
                break
        if scalar is None:
            scalar = _new_shared_temp(namer, [], arrays)
            defs.insert(
                position,
                TempDef(
                    target=scalar,
                    free=[],
                    limits=[(m, low, high)],
                    body=s_body,
                ),
            )
            position += 1
        table = d.copy(body=new_body)
        defs[position] = table
        name = _temp_name(d.target)

        def reread(expr):
            base_ = scalar if float(coeff) == 1.0 else coeff * scalar
            rep = {
                x: base_ + sign * x
                for x in expr.atoms(sp.Indexed)
                if str(x.base) == name
            }
            return expr.xreplace(rep) if rep else expr

        for q, e in enumerate(defs):
            if e is not table:
                defs[q] = e.copy(body=reread(e.body))
        outs = [reread(e) for e in outs]
        k = position + 1
    return defs, outs


def _split_factored(definition, limits):
    """``(outer factor, inner body)`` of a ``factored`` double reduction."""
    inner = {lim[0] for lim in limits[1:]}
    factors = sp.Mul.make_args(definition.body)
    o_f = [f for f in factors if not (f.atoms(sp.Idx) & inner)]
    i_f = [f for f in factors if f.atoms(sp.Idx) & inner]
    return sp.Mul(*o_f), sp.Mul(*i_f)


# ---------------------------------------------------------------------------
# Symmetry handling
# ---------------------------------------------------------------------------


def _separate_sums(expr, max_terms=16, selective=False):
    """Distribute products over sums inside summation bodies and split.

    ``sum_k (r_i r_j r_k + s_i s_k) n_k`` becomes
    ``r_i r_j sum_k r_k n_k + s_i sum_k s_k n_k``: after the split, the
    factors that do not depend on the bound index leave each sum
    (:func:`thermodiff.chain.factor_sums`), and a rank-2 sum over a third
    index may become an outer product of a scalar sum.  Bodies that would
    expand into more than ``max_terms`` terms are left alone.
    """
    from thermodiff.chain import factor_sums

    def visit(sub):
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
                split = factor_sums(
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

    def cost(expr):
        """Estimated work: sum over loop nests of nc**depth (nc = 10)."""
        total = 0
        for total_sum in expr.atoms(sp.Sum):
            depth = len(_free_indices(total_sum)) + len(total_sum.limits)
            total += 10**depth
        return total

    return visit(expr)
