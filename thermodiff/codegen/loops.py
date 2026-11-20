"""Loop fusion and table rewrites for the Fortran code generator."""

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

    def key_of(total):
        free = _order_free(_free_indices(total), indexes)
        mapping = {
            index: sp.Idx(f"_free{pos}") for pos, index in enumerate(free)
        }
        renamed = _alpha_key(total.xreplace(mapping))
        if symmetric_params:
            renamed = _canonical_symmetric(
                renamed, symmetric_params, sym_rank3
            )
        return renamed

    from thermodiff.chain import factor_sums

    # factors that do not depend on the bound indices leave the sums first
    staged = [
        (attr, out_name, factor_sums(expr)) for attr, out_name, expr in staged
    ]
    vectors = set()
    for _, _, expr in staged:
        for total in expr.atoms(sp.Sum):
            if (
                len(total.limits) == 1
                and _free_indices(total)
                and (general or len(_free_indices(total)) >= 1)
            ):
                vectors.add(key_of(total))

    def orientations_general(total):
        """Nestings of a sum over two or more indices: one bound index goes
        to an inner sum (rank 1 or 2 in its free indices), the rest stay
        outside.  Returns (outer limits, outer factors, inner sum, None)."""
        if len(total.limits) < 2 or not isinstance(total.function, sp.Mul):
            return []
        ranges = {(lim[1], lim[2]) for lim in total.limits}
        if len(ranges) != 1:
            return []
        result = []
        for pos, (b, lb, hb) in enumerate(total.limits):
            outer_f = [
                f
                for f in total.function.args
                if b not in f.atoms(sp.Idx) and f.atoms(sp.Idx)
            ]
            numbers = [f for f in total.function.args if f.is_Number]
            rest = [
                f
                for f in total.function.args
                if f not in outer_f and not f.is_Number
            ]
            if not outer_f or not rest:
                continue
            inner = sp.Sum(sp.Mul(*rest), (b, lb, hb))
            if len(_free_indices(inner)) > 2:
                continue
            outer_limits = [
                lim for k, lim in enumerate(total.limits) if k != pos
            ]
            result.append((outer_limits, numbers + outer_f, inner, None))
        return result

    def orientations(total):
        """(outer index, outer factors, inner sum) for both nestings."""
        if general:
            return orientations_general(total)
        if len(total.limits) != 2 or not isinstance(total.function, sp.Mul):
            return []
        (a, la, ha), (b, lb, hb) = total.limits
        if (la, ha) != (lb, hb):
            return []
        result = []
        for outer, inner_idx, lims in ((a, b, (la, ha)), (b, a, (lb, hb))):
            outer_f = [
                f
                for f in total.function.args
                if f.atoms(sp.Idx) <= {outer} and f.atoms(sp.Idx)
            ]
            numbers = [f for f in total.function.args if f.is_Number]
            rest = [
                f
                for f in total.function.args
                if f not in outer_f and not f.is_Number
            ]
            if not outer_f or not rest:
                continue
            inner = sp.Sum(sp.Mul(*rest), (inner_idx, *lims))
            result.append((outer, numbers + outer_f, inner, lims))
        return result

    # inner sums shared by several double sums are worth a vector too
    # (e.g. the temperature derivatives of sum_kl n_k n_l s_k s_l K_kl)
    counts = {}
    for _, _, expr in staged:
        for total in expr.atoms(sp.Sum):
            for key in {key_of(o[2]) for o in orientations(total)}:
                counts[key] = counts.get(key, 0) + 1
    shared = vectors | (
        {k for k, c in counts.items() if c >= 2} if shared_inner else set()
    )
    if not shared and not (cost and general):
        return staged

    def nest(total):
        options = [o for o in orientations(total) if key_of(o[2]) in shared]
        if not options and cost and general and len(total.limits) >= 3:
            # by cost: the inner sum with the fewest free indices, then the
            # one that leaves the most factors outside
            options = sorted(
                orientations(total),
                key=lambda o: (len(_free_indices(o[2])), -len(o[1])),
            )
            if options:
                outer_limits, outer_f, inner, _ = options[0]
                return sp.Sum(sp.Mul(*outer_f) * inner, *outer_limits)
        if not options:
            return None
        # prefer an inner sum that some output needs anyway
        options.sort(
            key=lambda o: (
                key_of(o[2]) not in vectors,
                -counts.get(key_of(o[2]), 0),
            )
        )
        outer, outer_f, inner, lims = options[0]
        if general:
            return sp.Sum(sp.Mul(*outer_f) * inner, *outer)
        return sp.Sum(sp.Mul(*outer_f) * inner, (outer, *lims))

    result = []
    for attr, out_name, expr in staged:
        for _ in range(3 if general else 1):
            replacements = {}
            for total in expr.atoms(sp.Sum):
                nested = nest(total)
                if nested is not None:
                    replacements[total] = nested
            if not replacements:
                break
            expr = expr.xreplace(replacements)
        result.append((attr, out_name, expr))
    return result


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

    def refs(expr, name):
        return [x for x in expr.atoms(sp.Indexed) if str(x.base) == name]

    def outside_sums(expr, name):
        if isinstance(expr, sp.Sum):
            return False
        if isinstance(expr, sp.Indexed):
            return str(expr.base) == name
        return any(
            outside_sums(arg, name) for arg in getattr(expr, "args", ())
        )

    def read_weight(exprs_, name):
        """How many times each element is read: diagonal reads count 0,
        a triangular (folded) summation 1/2, a full summation 1."""
        weight = 0.0
        seen_keys = set()
        for e in exprs_:
            for total in e.atoms(sp.Sum):
                free = sorted(_free_indices(total), key=str)
                key = _alpha_key(
                    total.xreplace(
                        {
                            x: sp.Idx(f"_free{pos}")
                            for pos, x in enumerate(free)
                        }
                    )
                )
                if dedup and key in seen_keys:
                    continue
                seen_keys.add(key)
                inner_refs = [x for x in refs(total.function, name)]
                if not inner_refs or total.function.atoms(sp.Sum):
                    continue
                if all(
                    len(set(x.indices)) < len(x.indices) for x in inner_refs
                ):
                    continue
                folded = any(
                    isinstance(lim[1], _LoopStart) for lim in total.limits
                )
                weight += 0.5 if folded else 1.0
        return weight

    def sum_count(exprs_, name):
        """Number of distinct summations (by alpha key) that read ``name``
        directly in their body, nested summations included."""
        keys = set()
        for e in exprs_:
            for total in e.atoms(sp.Sum):
                body_ = total.function
                direct = [
                    x
                    for x in refs(body_, name)
                    if not any(
                        x in inner.atoms(sp.Indexed)
                        for inner in body_.atoms(sp.Sum)
                    )
                ]
                if not direct:
                    continue
                free = sorted(_free_indices(total), key=str)
                keys.add(
                    _alpha_key(
                        total.xreplace(
                            {
                                x: sp.Idx(f"_free{pos}")
                                for pos, x in enumerate(free)
                            }
                        )
                    )
                )
        return len(keys)

    def pair_libm(body_, free_):
        """A libm call whose argument depends on both indices of the pair (a
        separable sqrt(e_i)*sqrt(e_j) is hoisted into vectors later anyway)."""
        both = set(free_)
        for sub in sp.preorder_traversal(body_):
            call = (
                isinstance(sub, sp.Function)
                and not isinstance(sub, sp.Piecewise)
            ) or (isinstance(sub, sp.Pow) and not sub.exp.is_Integer)
            if call and both <= {
                i for x in sub.atoms(sp.Indexed) for i in x.indices
            }:
                return True
        return False

    def vector_reads_only(exprs_, name):
        """Every summation that reads ``name`` directly keeps a free index
        (a matrix-vector product): a table read by scalar double sums is
        better inlined (reportes/brechas/ronda2-pcsaft.md, T1)."""
        found = False
        for e in exprs_:
            for total in e.atoms(sp.Sum):
                body_ = total.function
                direct = [
                    x
                    for x in refs(body_, name)
                    if not any(
                        x in inner.atoms(sp.Indexed)
                        for inner in body_.atoms(sp.Sum)
                    )
                ]
                if not direct:
                    continue
                found = True
                if not _free_indices(total):
                    return False
        return found

    exprs = list(exprs)
    removed = set()
    changed = True
    while changed:
        changed = False
        for definition in list(array_defs):
            free = definition["free"]
            if len(free) != 2 or definition["limits"]:
                continue
            body = definition["body"]
            name = _temp_name(definition["target"])
            trivial = not _is_expensive(body) and sp.count_ops(body) <= 3
            if not trivial and any(outside_sums(e, name) for e in exprs):
                continue
            cheap = not _is_expensive(body) and sp.count_ops(body) <= 8
            if (
                keep_libm
                and pair_libm(body, free)
                and sum_count(exprs, name) >= 1
                and vector_reads_only(exprs, name)
            ):
                continue
            if guard and _is_expensive(body) and sum_count(exprs, name) >= 2:
                # an expensive body (exp, log, root, ...) read by two or more
                # different summations, at any nesting depth, is computed once
                continue
            if not cheap and read_weight(exprs, name) > 1:
                # costly body read more than once per element: keep it
                continue
            other = [
                d
                for d in array_defs
                if d is not definition and refs(d["body"], name)
            ]
            if any(
                len(set(x.indices)) == len(x.indices)
                for d in other
                for x in refs(d["body"], name)
            ):
                # another temporary reads it off the diagonal: keep it
                continue

            def inline(expr):
                mapping = {
                    x: body.xreplace(dict(zip(free, x.indices)))
                    for x in refs(expr, name)
                }
                return expr.xreplace(mapping) if mapping else expr

            exprs = [inline(e) for e in exprs]
            for d in other:
                d["body"] = inline(d["body"])
            array_defs.remove(definition)
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
    free = definition["free"]
    limits = definition["limits"]
    body = definition["body"]
    if definition.get("mirror") or body.has(sp.Piecewise):
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
    free = definition["free"]
    limits = definition["limits"]
    if len(free) != 2 or len(limits) != 1 or definition.get("mirror"):
        return False
    b, low, high = limits[0]
    if low != 1 or str(high) != "nc":
        return False
    body = definition["body"]
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
        if len(d["free"]) != 1 or len(d["limits"]) != 1:
            return False
        if d.get("mirror") or d.get("mirror_sum") or d.get("symv"):
            return False
        b, low, high = d["limits"][0]
        if low != 1 or str(high) != "nc":
            return False
        body = d["body"]
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
                    target=group[0]["target"],
                    free=group[0]["free"],
                    limits=group[0]["limits"],
                    body=group[0]["body"],
                    fused=list(group),
                )
            )
        group.clear()

    for d in definitions:
        if simple(d) and group:
            first = group[0]
            f0, b0 = first["free"][0], first["limits"][0][0]
            renamed = d.copy()
            mapping = {d["free"][0]: f0, d["limits"][0][0]: b0}
            renamed["target"] = d["target"].xreplace(mapping)
            renamed["free"] = [f0]
            renamed["limits"] = [(b0, 1, first["limits"][0][2])]
            renamed["body"] = d["body"].xreplace(mapping)
            names = {_temp_name(g["target"]) for g in group}
            depends = any(
                str(x.base) in names for x in renamed["body"].atoms(sp.Indexed)
            )
            same_order = bool(d.get("colmajor")) == bool(first.get("colmajor"))
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
    free = list(definition["free"])
    bound = [lim[0] for lim in definition["limits"]]
    if (
        definition.get("symv")
        or definition.get("colmajor")
        or definition.get("compfused")
    ):
        return None
    if definition.get("colsum"):
        return [free[1], bound[0], free[0]]
    if definition.get("sibling"):
        return [free[1], free[0]] + bound
    if definition.get("fused"):
        first = definition["fused"][0]
        f, b = first["free"][0], first["limits"][0][0]
        return [b, f] if first.get("colmajor") else [f, b]
    if definition.get("mirror_sum") and len(free) == 2 and bound:
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
        if definition.get("fused") or definition.get("sibling"):
            for member in definition.get("fused") or definition["sibling"]:
                member["body"] = permute(member["body"], order)
        else:
            definition["body"] = permute(definition["body"], order)
    return definitions


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

    def is_map(d):
        if len(d["free"]) != 1 or d["limits"]:
            return False
        if (
            d.get("mirror")
            or d.get("mirror_sum")
            or d.get("symv")
            or d.get("fused")
            or d.get("colmajor")
        ):
            return False
        body = d["body"]
        return (
            not body.has(sp.Sum)
            and not body.has(sp.Piecewise)
            and not any(
                isinstance(x, _LoopStart) for x in body.atoms(sp.Symbol)
            )
        )

    out = []
    group = []

    def flush():
        if len(group) <= 1:
            out.extend(group)
        else:
            idx = group[0]["free"][0]
            # member values read by later members become placeholders, so the
            # elimination never moves a read of cmp0(i) before its assignment
            holder_of = {
                _temp_name(d["target"]): sp.Dummy(f"m{k}")
                for k, d in enumerate(group)
            }
            holders = {
                d["target"]: holder_of[_temp_name(d["target"])] for d in group
            }

            def to_holders(expr):
                return expr.xreplace(
                    {
                        x: holder_of[str(x.base)]
                        for x in expr.atoms(sp.Indexed)
                        if str(x.base) in holder_of and x.indices == (idx,)
                    }
                )

            bodies = [to_holders(d["body"]) for d in group]
            replacements, reduced = sp.cse(
                bodies,
                symbols=iter(lambda: sp.Symbol(str(namer("t"))), None),
                optimizations=None,
            )
            back = {h: t for t, h in holders.items()}
            pending = [
                ("local", sym, value) for sym, value in replacements
            ] + [
                ("member", holders[d["target"]], b)
                for d, b in zip(group, reduced)
            ]
            known = set()
            local_syms = {sym for sym, _ in replacements}
            needed = local_syms | set(back)
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
            ordered = [
                (kind, back.get(sym, sym), value.xreplace(back))
                for kind, sym, value in steps
            ]
            out.append(
                ComponentFusedMap(
                    target=group[0]["target"],
                    free=[idx],
                    limits=[],
                    body=group[0]["body"],
                    steps=ordered,
                    members=[(d["target"], d["body"]) for d in group],
                )
            )
        group.clear()

    for d in definitions:
        if not is_map(d):
            names = {_temp_name(g["target"]) for g in group}
            scalar = (
                not d["free"]
                and not d["limits"]
                and not any(
                    d.get(k) for k in ("mirror", "mirror_sum", "symv", "fused")
                )
            )
            reads_group = any(
                str(x.base) in names for x in d["body"].atoms(sp.Indexed)
            )
            if group and scalar and not reads_group:
                # an independent scalar between two maps moves ahead of the group
                out.append(d)
                continue
            flush()
            out.append(d)
            continue
        if group:
            idx = group[0]["free"][0]
            renamed = d.copy()
            mapping = {d["free"][0]: idx}
            renamed["target"] = d["target"].xreplace(mapping)
            renamed["free"] = [idx]
            renamed["body"] = d["body"].xreplace(mapping)
            names = {_temp_name(g["target"]) for g in group}
            ok = all(
                x.indices == (idx,)
                for x in renamed["body"].atoms(sp.Indexed)
                if str(x.base) in names
            )
            if ok:
                group.append(renamed)
                continue
            flush()
        group.append(d)
    flush()
    return out


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

    def plain(d):
        body = d["body"]
        return (
            not body.has(sp.Piecewise)
            and not any(
                isinstance(x, _LoopStart) for x in body.atoms(sp.Symbol)
            )
            and not any(
                d.get(k)
                for k in ("mirror", "mirror_sum", "symv", "fused", "colmajor")
            )
        )

    def kind(d):
        if d.get("compfused"):
            return "map"
        if not plain(d):
            return None
        if (
            len(d["free"]) == 1
            and not d["limits"]
            and not d["body"].has(sp.Sum)
        ):
            return "map"
        if not d["free"] and d["limits"]:
            b, low, high = d["limits"][0]
            if low == 1 and str(high) == "nc":
                return "red"
        return None

    def outer_of(d):
        return d["free"][0] if kind(d) == "map" else d["limits"][0][0]

    def libm_map(d):
        if kind(d) != "map":
            return False
        if d.get("compfused"):
            return any(_calls_libm(v) for _k, _t, v in d["steps"])
        return _calls_libm(d["body"])

    def double_red(d):
        return kind(d) == "red" and len(d["limits"]) > 1

    def libm_red(d):
        return (
            libm_reductions
            and kind(d) == "red"
            and len(d["limits"]) == 1
            and _calls_libm(d["body"])
        )

    def clash(d):
        """libm_guard: d cannot join the current group."""
        if not libm_guard:
            return False
        return (
            double_red(d) and any(libm_map(g) or libm_red(g) for g in group)
        ) or (
            (libm_map(d) or libm_red(d)) and any(double_red(g) for g in group)
        )

    def produced(d):
        if d.get("compfused"):
            return {_temp_name(t) for t, _ in d["members"]}
        return {_temp_name(d["target"])}

    def reads(d):
        if d.get("compfused"):
            return [
                x for _k, _t, v in d["steps"] for x in v.atoms(sp.Indexed)
            ], set().union(*[v.free_symbols for _k, _t, v in d["steps"]])
        return list(d["body"].atoms(sp.Indexed)), d["body"].free_symbols

    def rename(d, old, new):
        if old == new:
            return d
        m = {old: new}
        # the opaque lower bound ``old + 1`` of a triangular loop follows too
        for lim in d["limits"]:
            for bound in (lim[1], lim[2]):
                for sym in sp.sympify(bound).atoms(sp.Symbol):
                    if isinstance(sym, _LoopStart) and sym.outer == old:
                        m[sym] = _LoopStart(new)
        d = d.copy()
        if d.get("compfused"):
            d["steps"] = [
                (k, t.xreplace(m), v.xreplace(m)) for k, t, v in d["steps"]
            ]
            d["free"] = [new]
        elif kind(d) == "map":
            d["target"] = d["target"].xreplace(m)
            d["body"] = d["body"].xreplace(m)
            d["free"] = [new]
        else:
            d["body"] = d["body"].xreplace(m)
            d["limits"] = [
                (
                    b.xreplace(m) if hasattr(b, "xreplace") else b,
                    sp.sympify(lo).xreplace(m),
                    sp.sympify(hi).xreplace(m),
                )
                for b, lo, hi in d["limits"]
            ]
        return d

    out, group = [], []

    def flush():
        if len(group) <= 1:
            out.extend(group)
        else:
            limits = []
            for g in group:
                limits += [lim for lim in g["limits"] if lim not in limits]
            out.append(
                OuterFusedNest(
                    target=group[0].get("target"),
                    free=[],
                    limits=limits,
                    body=sp.S.Zero,
                    members=list(group),
                    outer=outer_of(group[0]),
                )
            )
        group.clear()

    for d in definitions:
        k = kind(d)
        if k is not None and group:
            x = outer_of(group[0])
            inner = [lim[0] for lim in d["limits"][1:]] if k == "red" else []
            if x in inner or clash(d):
                flush()
                group.append(d)
                continue
            cand = rename(d, outer_of(d), x)
            arrays_ = set().union(*[produced(g) for g in group])
            accs = {_temp_name(g["target"]) for g in group if kind(g) == "red"}
            idx_reads, syms = reads(cand)
            ok = all(
                x_.indices == (x,)
                for x_ in idx_reads
                if str(x_.base) in arrays_
            )
            ok = ok and not any(str(sy) in accs for sy in syms)
            if ok:
                group.append(cand)
                continue
            flush()
            group.append(d)
            continue
        if k is not None:
            group.append(d)
            continue
        # not fusable: independent scalars move ahead, anything else ends the group
        if group and not d["free"] and not d["limits"] and plain(d):
            names = set().union(*[produced(g) for g in group])
            if not any(
                str(sy) in names for sy in d["body"].free_symbols
            ) and not any(
                str(x_.base) in names for x_ in d["body"].atoms(sp.Indexed)
            ):
                out.append(d)
                continue
        flush()
        out.append(d)
    flush()
    return out


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

    def readers(name):
        defs = [
            d
            for d in definitions
            if any(str(x.base) == name for x in d["body"].atoms(sp.Indexed))
        ]
        outs = [
            e
            for e in outputs
            if any(str(x.base) == name for x in e.atoms(sp.Indexed))
        ]
        return defs, outs

    def triangular(d):
        return any(
            isinstance(x, _LoopStart)
            for lim in d["limits"][1:]
            for x in sp.sympify(lim[1]).atoms(sp.Symbol)
        )

    changed = True
    while changed:
        changed = False
        for table in reversed(definitions):
            free = table["free"]
            if (
                len(free) != 2
                or table["limits"]
                or not isinstance(table["target"], sp.Indexed)
            ):
                continue
            if any(
                table.get(k)
                for k in ("mirror_sum", "symv", "fused", "colmajor", "colsum")
            ):
                continue
            body = table["body"]
            if body.has(sp.Sum) or body.has(sp.Piecewise) or _calls_libm(body):
                continue
            name = _temp_name(table["target"])
            defs, outs = readers(name)
            if outs or not defs:
                continue
            if any(
                d["free"] or not d["limits"] or len(d["limits"]) > 2
                for d in defs
            ):
                continue
            double = [d for d in defs if len(d["limits"]) == 2]
            if len(double) > 1 or (
                double and table.get("mirror") and not triangular(double[0])
            ):
                continue
            for d in defs:
                rep = {
                    x: body.xreplace(
                        {free[0]: x.indices[0], free[1]: x.indices[1]}
                    )
                    for x in d["body"].atoms(sp.Indexed)
                    if str(x.base) == name
                }
                d["body"] = d["body"].xreplace(rep)
            definitions = [d for d in definitions if d is not table]
            arrays.pop(name, None)
            changed = True
            break

    vectors = {}
    canon_slot = sp.Symbol("__slot")
    out = []
    for d in definitions:
        if (
            d["free"]
            or len(d["limits"]) != 2
            or not isinstance(d["body"], sp.Mul)
        ):
            out.append(d)
            continue
        if d["body"].has(sp.Piecewise):
            out.append(d)
            continue
        outer, inner = d["limits"][0][0], d["limits"][1][0]
        factors = sp.Mul.make_args(d["body"])
        o_f = [f for f in factors if inner not in f.atoms(sp.Idx)]
        i_f = [f for f in factors if inner in f.atoms(sp.Idx)]
        if not o_f or not i_f:
            out.append(d)
            continue
        only_inner = [f for f in i_f if f.atoms(sp.Idx) == {inner}]
        rest = [f for f in i_f if f not in only_inner]
        new_defs = []
        if len(only_inner) > 1:
            prod = sp.Mul(*only_inner)
            key = prod.xreplace({inner: canon_slot})
            if key not in vectors:
                target = _new_shared_temp(namer, [outer], arrays)
                vectors[key] = target.base
                new_defs.append(
                    TempDef(
                        target=target,
                        free=[outer],
                        limits=[],
                        body=prod.xreplace({inner: outer}),
                    )
                )
            only_inner = [sp.Indexed(vectors[key], inner)]
        only_outer = [f for f in o_f if f.atoms(sp.Idx) == {outer}]
        if len(only_outer) > 1:
            key = sp.Mul(*only_outer).xreplace({outer: canon_slot})
            if key in vectors:
                o_f = [f for f in o_f if f not in only_outer] + [
                    sp.Indexed(vectors[key], outer)
                ]
        out.extend(new_defs)
        out.append(
            d.copy(
                body=sp.Mul(*o_f) * sp.Mul(*(only_inner + rest)),
                factored=True,
                acc=sp.Symbol(str(namer("t"))),
            )
        )
    # each new vector goes right after the definitions it reads, so that the
    # double reductions stay consecutive (and share their inner loops)
    new_vectors = {str(b) for b in vectors.values()}
    for vec in [d for d in out if _temp_name(d["target"]) in new_vectors]:
        out.remove(vec)
        names = {str(x.base) for x in vec["body"].atoms(sp.Indexed)} | {
            str(x) for x in vec["body"].free_symbols
        }
        position = 0
        for k, d in enumerate(out):
            if (
                d.get("target") is not None
                and _temp_name(d["target"]) in names
            ):
                position = k + 1
        out.insert(position, vec)
    return out


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
    flags_ok = {
        "target",
        "free",
        "limits",
        "body",
        "mirror",
        "factored",
        "acc",
    }

    def plain_scalar(d):
        return (
            isinstance(d.get("target"), sp.Symbol)
            and not d["free"]
            and not d["limits"]
            and set(d) <= flags_ok
            and not d["body"].has(sp.Sum)
        )

    scalar_def = {d["target"]: d for d in definitions if plain_scalar(d)}
    position = {
        d["target"]: k for k, d in enumerate(definitions) if plain_scalar(d)
    }
    canon_memo = {}

    def canon_scalar(s):
        if s in canon_memo:
            return canon_memo[s]
        canon_memo[s] = s  # cycle guard
        body = scalar_def[s]["body"]
        rep = {
            x: canon_scalar(x) for x in body.free_symbols if x in scalar_def
        }
        value = body.xreplace(rep)
        if sp.count_ops(value) > max_ops:
            value = s
        canon_memo[s] = value
        return value

    slot = sp.Idx("__elem")

    def candidate(sub):
        if not isinstance(sub, sp.Expr) or sub.is_Atom:
            return False
        if isinstance(sub, (sp.Indexed, sp.Sum, sp.Tuple)) or sub.has(sp.Sum):
            return False
        if sub.has(sp.Piecewise) or sub.is_Relational or sub.is_Boolean:
            return False
        if any(isinstance(x, _LoopStart) for x in sub.atoms(sp.Symbol)):
            return False
        return len(sub.atoms(sp.Idx)) == 1 and _calls_libm(sub)

    def key_of(sub):
        (idx,) = sub.atoms(sp.Idx)
        rep = {x: canon_scalar(x) for x in sub.free_symbols if x in scalar_def}
        return sub.xreplace({idx: slot}).xreplace(rep), idx

    def editable(d):
        return set(d) <= flags_ok

    def is_map(pos):
        if pos >= n_defs:
            return False
        d = defs[pos]
        return len(d["free"]) == 1 and not d["limits"]

    sites = [d["body"] for d in definitions] + list(outputs)
    n_defs = len(definitions)
    defs = list(definitions)
    outs = list(outputs)
    touched = set()
    merged_all = set()
    while True:
        found = {}
        for pos, expr in enumerate(sites):
            if pos < n_defs and not editable(defs[pos]):
                continue
            for sub in sp.preorder_traversal(expr):
                if candidate(sub):
                    key, _ = key_of(sub)
                    found.setdefault(key, []).append((pos, sub))
        best, best_rank = None, None
        for key in sorted(found, key=sp.default_sort_key):
            occ = found[key]
            # per-component maps share per element anyway (comp_fuse): they do
            # not count towards the two readers
            n_sites = len({pos for pos, _ in occ if not is_map(pos)})
            if n_sites < 2:
                continue
            rank = (n_sites, sp.count_ops(key))
            if best_rank is None or rank > best_rank:
                best, best_rank = key, rank
        if best is None:
            break
        occ = found[best]
        first_pos, rep_sub = occ[0]
        (idx0,) = rep_sub.atoms(sp.Idx)
        # merge differently spelt scalars of the occurrences into the earliest
        merge = {}
        rep_scalars = {
            canon_scalar(x): x for x in rep_sub.free_symbols if x in scalar_def
        }
        for _pos, sub in occ[1:]:
            for x in sub.free_symbols:
                if x not in scalar_def:
                    continue
                y = rep_scalars.get(canon_scalar(x))
                if y is None or y == x:
                    continue
                early, late = (y, x) if position[y] < position[x] else (x, y)
                merge[late] = early
        target = _new_shared_temp(namer, [idx0], arrays)
        base = target.base
        vector = TempDef(target=target, free=[idx0], limits=[], body=rep_sub)
        rep = {}
        for pos, sub in occ:
            (idx,) = sub.atoms(sp.Idx)
            rep.setdefault(pos, {})[sub] = sp.Indexed(base, idx)
            touched |= {x for x in sub.free_symbols if x in scalar_def}
        for pos, mapping in rep.items():
            sites[pos] = sites[pos].xreplace(mapping)
        if merge:
            # resolve chains, then rename everywhere (vector body included)
            def final(s):
                while s in merge:
                    s = merge[s]
                return s

            merge = {s: final(s) for s in merge}
            sites = [e.xreplace(merge) for e in sites]
            vector["body"] = vector["body"].xreplace(merge)
            merged_all |= set(merge)
        # insert the vector right before its first reader
        new_sites = list(sites)
        for pos in range(n_defs):
            defs[pos] = defs[pos].copy(body=new_sites[pos])
        drop = set(merge)
        insert_at = first_pos if first_pos < n_defs else n_defs
        defs = defs[:insert_at] + [vector] + defs[insert_at:]
        kept = []
        for d in defs:
            if d.get("target") in drop and plain_scalar(d):
                continue
            kept.append(d)
        defs = kept
        outs = new_sites[n_defs:]
        sites = [d["body"] for d in defs] + outs
        n_defs = len(defs)
        position = {
            d["target"]: k for k, d in enumerate(defs) if plain_scalar(d)
        }
        scalar_def = {d["target"]: d for d in defs if plain_scalar(d)}
    # drop the plain scalars of replaced occurrences that nothing reads any more
    while True:
        read = set()
        for e in [d["body"] for d in defs] + outs:
            read |= e.free_symbols
        for d in defs:
            if d.get("acc") is not None:
                read.add(d["acc"])
        dead = [
            d
            for d in defs
            if plain_scalar(d)
            and d["target"] in touched
            and d["target"] not in read
        ]
        if not dead:
            break
        defs = [d for d in defs if not any(d is x for x in dead)]
    # undeclare the dropped scalars
    alive = {d.get("target") for d in defs} | {d.get("acc") for d in defs}
    for e in outs:
        alive |= e.free_symbols
    dropped = {s for s in touched | set(merged_all) if s not in alive}
    namer.created[:] = [s for s in namer.created if s not in dropped]
    return defs, outs


def _diagonal_from_tables(definitions, symmetric_params, *, sym_rank3):
    """Read the diagonal of an existing rank-2 table instead of recomputing it.

    ``d(k) = sum_m B(k, k, m)`` and ``t(a, b) = sum_m B(a, b, m)`` (filled in
    a triangle and mirrored) give ``d(k) == t(k, k)``.  The rank-1 sum is
    replaced by the map ``d(k) = t(k, k)``, placed right after the table.
    """
    maps = {}
    for d in definitions:
        if (
            len(d["free"]) == 1
            and not d["limits"]
            and not d["body"].has(sp.Sum)
            and not any(
                d.get(k)
                for k in (
                    "mirror",
                    "mirror_sum",
                    "symv",
                    "fused",
                    "colmajor",
                    "sibling",
                    "compfused",
                )
            )
        ):
            maps[_temp_name(d["target"])] = (d["free"][0], d["body"])

    def inline_maps(e):
        rep = {}
        for x in e.atoms(sp.Indexed):
            if str(x.base) in maps and len(x.indices) == 1:
                f, body = maps[str(x.base)]
                rep[x] = body.xreplace({f: x.indices[0]})
        return e.xreplace(rep) if rep else e

    def canon(e):
        e = sp.expand(inline_maps(e))
        return (
            _canonical_symmetric(e, symmetric_params, sym_rank3)
            if symmetric_params
            else e
        )

    tables, owner = [], {}
    for d in definitions:
        members = d.get("sibling") or ([d] if d.get("mirror_sum") else [])
        for t in members:
            if len(t["free"]) == 2 and len(t["limits"]) == 1:
                tables.append(t)
                owner[id(t)] = d
    out = list(definitions)
    for d in definitions:
        if (
            len(d["free"]) != 1
            or len(d["limits"]) != 1
            or any(
                d.get(k)
                for k in ("mirror", "mirror_sum", "symv", "fused", "colmajor")
            )
        ):
            continue
        (k,), (m, lo, hi) = d["free"], d["limits"][0]
        for t in tables:
            (a, b), (m2, lo2, hi2) = t["free"], t["limits"][0]
            if (lo2, hi2) != (lo, hi):
                continue
            diag = canon(t["body"].xreplace({a: k, b: k, m2: m}))
            if sp.expand(diag - canon(d["body"])) != 0:
                continue
            name = _temp_name(d["target"])
            if any(str(x.base) == name for x in t["body"].atoms(sp.Indexed)):
                continue
            host = owner[id(t)]
            pos_t = out.index(host)
            new = TempDef(
                target=d["target"],
                free=[k],
                limits=[],
                body=sp.Indexed(t["target"].base, k, k),
            )
            pos_d = out.index(d)
            if pos_d < pos_t:
                # every consumer of d must come after the table
                users = [
                    e
                    for e in out[pos_d + 1 : pos_t + 1]
                    if any(
                        str(x.base) == name
                        for x in e["body"].atoms(sp.Indexed)
                    )
                ]
                if users:
                    break
            out.remove(d)
            out.insert(out.index(host) + 1, new)
            break
    return out


def _fuse_sibling_tables(definitions, namer):
    """One nest for symmetric rank-2 tables with the same loops.

    ``t5(k, l) = sum_m A``, ``t7(k, l) = sum_m B`` (triangle + mirror) become
    one triangular nest whose inner loop accumulates both sums in scalars:
    each slice of the operands is read in a single pass.  Definitions found
    between two such tables that do not read the first one are emitted
    after the fused nest, unless the second table reads them.
    """

    def table(d):
        return (
            d.get("mirror_sum")
            and len(d["free"]) == 2
            and len(d["limits"]) == 1
            and not d["body"].has(sp.Piecewise)
            and not d.get("colsum")
        )

    def reads(d, names):
        return any(
            str(x.base) in names for x in d["body"].atoms(sp.Indexed)
        ) or any(str(sy) in names for sy in d["body"].free_symbols)

    def produced(d):
        return (
            {_temp_name(d["target"])} if d.get("target") is not None else set()
        )

    out, group, waiting = [], [], []

    def flush():
        if len(group) == 1:
            out.extend(group)
        elif group:
            out.append(
                SiblingNest(
                    target=group[0]["target"],
                    free=group[0]["free"],
                    limits=group[0]["limits"],
                    body=group[0]["body"],
                    sibling=list(group),
                    accs=[sp.Symbol(str(namer("t"))) for _ in group],
                )
            )
        out.extend(waiting)
        group.clear()
        waiting.clear()

    for d in definitions:
        if table(d) and group:
            (a0, b0), (m0, lo0, hi0) = group[0]["free"], group[0]["limits"][0]
            (a, b), (m, lo, hi) = d["free"], d["limits"][0]
            renamed = d.copy()
            mp = {a: a0, b: b0, m: m0}
            renamed["target"] = d["target"].xreplace(mp)
            renamed["body"] = d["body"].xreplace(mp)
            renamed["free"], renamed["limits"] = [a0, b0], [(m0, lo0, hi0)]
            blocked = set().union(*[produced(g) for g in group + waiting])
            if (lo, hi) == (lo0, hi0) and not reads(renamed, blocked):
                group.append(renamed)
                continue
            flush()
            group.append(d)
            continue
        if table(d):
            group.append(d)
            continue
        if group and not reads(d, set().union(*[produced(g) for g in group])):
            waiting.append(d)
            continue
        flush()
        out.append(d)
    flush()
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
            d.get("mirror_sum")
            and len(d["free"]) == 2
            and len(d["limits"]) == 1
            and not d["body"].has(sp.Piecewise)
            and set(d) <= {"target", "free", "limits", "body", "mirror_sum"}
        )

    out = []
    for d in definitions:
        if d.get("sibling") and any(rank3(g["body"]) for g in d["sibling"]):
            d = d.copy(
                split_acc=2,
                accs2=[sp.Symbol(str(namer("t"))) for _ in d["accs"]],
            )
        elif lone(d) and rank3(d["body"]):
            d = SiblingNest(
                target=d["target"],
                free=d["free"],
                limits=d["limits"],
                body=d["body"],
                sibling=[d],
                accs=[sp.Symbol(str(namer("t")))],
                accumulators=2,
            )
            d["accs2"] = [sp.Symbol(str(namer("t")))]
        out.append(d)
    return out
