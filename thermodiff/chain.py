"""Structured differentiation by the chain rule over summation nodes.

The default path of thermodiff differentiates the fully expanded function:
every definition of a ``.model`` file is substituted before differentiating,
so the derivatives lose the structure of the mixing rules.  A hand-written
implementation does the opposite: it writes

    Ar = F(S_1, ..., S_m, T, V)

in terms of a few scalar "mixing" quantities ``S_k`` (``sum_i n_i``,
``B``, ``D``, ...), computes the gradient and the Hessian of each ``S_k``
with respect to the mole numbers, and assembles the derivatives of ``Ar``
by the chain rule.  The composition Hessian is then a combination of a few
outer products of vectors and a few matrices, instead of one large
expression per element.

This module reproduces that structure automatically:

1. The summations of the expanded function are first *factored*: factors of
   a summation body that do not depend on its bound indices are moved out
   of it (``sum_ij x_ij / ntot`` becomes ``(sum_ij x_ij) / ntot``).
2. Every outermost summation without free indices becomes a node ``S_k``
   (summations equal up to the names of their bound indices share one
   node).  What remains, ``F``, has no summation and no mole number.
3. The partial derivatives of ``F`` are scalar derivatives; the derivatives
   of every node are obtained with the existing machinery of
   :class:`thermodiff.diffplz.DiffPlz` (Kronecker reduction included).
4. The derivatives of ``Ar`` are assembled with the closed chain-rule
   formulas and the node quantities substituted back.  The usual passes of
   :func:`thermodiff.fortran.to_fortran` then share the node summations.

The rewrite is exact.  When the function does not fit the pattern (a mole
number outside every summation, pressure as a variable) the caller falls
back to the plain derivatives.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import sympy as sp

from thermodiff.diffplz import DiffPlz
from thermodiff.thermovars import P, T, V, i, j, n, nc

__all__ = [
    "chain_rule_derivatives",
    "factor_sums",
    "summation_nodes",
    "abstract_component_functions",
]

#: derivative attribute -> (variables) ; "n" means a mole-number index
_ATTR_VARS = {
    "dt": ("T",),
    "dv": ("V",),
    "dni": ("n",),
    "dt2": ("T", "T"),
    "dv2": ("V", "V"),
    "dtdv": ("T", "V"),
    "dnidnj": ("n", "n"),
    "dtdni": ("T", "n"),
    "dvdni": ("V", "n"),
}
_SYM = {"T": T, "V": V}


def _bound(total):
    return {limit[0] for limit in total.limits}


def factor_sums(expr):
    """Move factors that do not depend on the bound indices out of sums.

    ``Sum(a*b(k), k)`` becomes ``a*Sum(b(k), k)`` when ``a`` does not
    contain ``k``.  Applied bottom-up, so nested summations are factored
    first.  Exact for finite sums.
    """
    if not isinstance(expr, sp.Basic) or not expr.args:
        return expr
    args = [factor_sums(arg) for arg in expr.args]
    if isinstance(expr, sp.Sum):
        body = args[0]
        bound = _bound(expr)
        if isinstance(body, sp.Mul):
            inner = [f for f in body.args if f.atoms(sp.Idx) & bound]
            outer = [f for f in body.args if not (f.atoms(sp.Idx) & bound)]
            if outer and inner:
                return sp.Mul(*outer) * sp.Sum(sp.Mul(*inner), *expr.limits)
            if outer and not inner:
                # body independent of the bound indices: sum_k c = c * count
                return expr.func(body, *expr.limits).doit()
        return expr.func(body, *args[1:])
    if isinstance(expr, sp.Idx) or isinstance(expr, sp.Indexed):
        return expr
    try:
        return expr.func(*args)
    except TypeError:
        return expr


def _alpha_key(total):
    mapping = {}
    for position, limit in enumerate(total.limits):
        mapping[limit[0]] = sp.Idx(f"_chainbound{position}")
    return total.xreplace(mapping)


def summation_nodes(expr):
    """Replace every outermost index-free summation by a node symbol.

    Returns
    -------
    outer : sympy expression
        ``expr`` with the summations replaced by ``S0``, ``S1``, ...
    nodes : list of (Symbol, Sum)
        The node symbols and the summations they stand for, in a
        deterministic order.
    """
    found = {}

    def visit(sub):
        if isinstance(sub, sp.Sum):
            if not (sub.atoms(sp.Idx) - _bound_all(sub)):
                key = _alpha_key(sub)
                found.setdefault(key, sub)
                return
        for arg in getattr(sub, "args", ()):
            visit(arg)

    visit(expr)
    ordered = sorted(found.items(), key=lambda kv: sp.default_sort_key(kv[0]))
    nodes = []
    replacements = {}
    for position, (key, total) in enumerate(ordered):
        # a Dummy cannot clash with a model symbol called S0, S1, ...
        symbol = sp.Dummy(f"S{position}")
        nodes.append((symbol, total))
        replacements[key] = symbol

    def substitute(sub):
        if isinstance(sub, sp.Sum):
            key = _alpha_key(sub)
            if key in replacements:
                return replacements[key]
        if not getattr(sub, "args", ()) or isinstance(
            sub, (sp.Idx, sp.Indexed)
        ):
            return sub
        return sub.func(*[substitute(arg) for arg in sub.args])

    return substitute(expr), nodes


def _bound_all(total):
    """Every index bound by ``total`` or by a summation nested inside it."""
    bound = set()
    for sub in sp.preorder_traversal(total):
        if isinstance(sub, sp.Sum):
            bound |= _bound(sub)
    return bound


#: keys of :attr:`NameScheme.prefixes`
CMP_KIND = "cmp"
MAT_KIND = "mat"

_SLOT_CANDIDATES = [
    "ic",
    "ia",
    "ib",
    "jc",
    "ja",
    "jb",
    "kc",
    "ka",
    "kb",
    "ua",
    "ub",
    "uc",
    "va",
    "vb",
    "vc",
]
_PREFIX_CANDIDATES = {
    CMP_KIND: ["cmp", "cpf", "cfn", "cvec"],
    MAT_KIND: ["mat", "pmt", "pfn", "pmat"],
}


#: structure key under which the scheme of a chain-rule run is returned
NAME_SCHEME_KEY = "_name_scheme"


@dataclass(frozen=True)
class NameScheme:
    """Slot index names and array prefixes of one chain-rule run.

    Attributes
    ----------
    component_slot : sympy.Idx
        Index of the definitions of the component-function vectors.
    pair_slots : tuple of sympy.Idx
        Indices of the definitions of the pair-function matrices.
    prefixes : mapping of str to str
        Prefix of the generated arrays, keyed by :data:`CMP_KIND` and
        :data:`MAT_KIND`; read-only.
    """

    component_slot: sp.Idx
    pair_slots: tuple[sp.Idx, sp.Idx]
    prefixes: Mapping[str, str]


#: the names used when the model does not clash with them
DEFAULT_NAME_SCHEME = NameScheme(
    component_slot=sp.Idx("ic"),
    pair_slots=(sp.Idx("ia"), sp.Idx("ib")),
    prefixes=MappingProxyType({CMP_KIND: "cmp", MAT_KIND: "mat"}),
)


def _name_scheme(expression: sp.Expr) -> NameScheme:
    """Choose slot indices and array prefixes that the model does not use.

    Fortran is case insensitive, so the comparison is done in lower case.
    When every prefix candidate is used by the model, the default prefix
    is kept.

    Parameters
    ----------
    expression : sympy.Expr
        The function whose symbols the names must not clash with.

    Returns
    -------
    NameScheme
        The slots and prefixes for this expression.
    """
    used = {"nc", "res"}
    used |= {str(x).lower() for x in expression.atoms(sp.Symbol)}
    used |= {str(x).lower() for x in expression.atoms(sp.IndexedBase)}
    used |= {str(x).lower() for x in expression.atoms(sp.Idx)}
    slots = [name for name in _SLOT_CANDIDATES if name not in used]
    prefixes = dict(DEFAULT_NAME_SCHEME.prefixes)
    for kind, candidates in _PREFIX_CANDIDATES.items():
        for prefix in candidates:
            if not any(name.startswith(prefix) for name in used):
                prefixes[kind] = prefix
                break
    return NameScheme(
        component_slot=sp.Idx(slots[0]),
        pair_slots=(sp.Idx(slots[1]), sp.Idx(slots[2])),
        prefixes=MappingProxyType(prefixes),
    )


class _Param:
    """A function of temperature over one or two component indices."""

    def __init__(self, name, body, rank, symmetric=False):
        self.name = name
        self.body = body  # in terms of the name scheme slots
        self.rank = rank
        self.symmetric = symmetric
        self.func = sp.Function(name)


def _mole_free(sub):
    return not any(base == n for base in sub.atoms(sp.IndexedBase))


def _subscript_order(body, a, b):
    """Order ``(a, b)`` or ``(b, a)`` in which ``body`` reads its parameters.

    Counts the rank-2 parameters subscripted by exactly the two indices in
    each order; the majority wins and a tie keeps ``(a, b)``.
    """
    ab = ba = 0
    for x in body.atoms(sp.Indexed):
        if len(x.indices) != 2:
            continue
        if tuple(x.indices) == (a, b):
            ab += 1
        elif tuple(x.indices) == (b, a):
            ba += 1
    return (b, a) if ba > ab else (a, b)


def abstract_component_functions(
    expr,
    registry,
    symmetric_params=(),
    pairs=True,
    split_sqrt=False,
    table_layout=False,
    scheme=DEFAULT_NAME_SCHEME,
):
    """Abstract per-component and per-pair functions of temperature.

    Inside summations, the factors of a product that do not contain mole
    numbers are grouped by the indices they use:

    * a group over two indices that depends on ``T`` (``sqrt(a_k a_l)
      (1 - k_kl)``) becomes a pair function ``M(k, l, T)``;
    * a group over one index that depends on ``T`` (``alpha_k``) becomes a
      component function ``A(k, T)``.

    Their values and temperature derivatives are later computed once, as
    vectors (O(nc)) or matrices (O(nc^2), a single triangle when symmetric),
    instead of being expanded inside every summation that needs them.  The
    bodies of pair functions are themselves abstracted into component
    functions, so that ``dM/dT`` costs a few operations per element.

    ``registry`` maps the canonical body to a :class:`_Param` and is shared
    between the nodes.

    With ``table_layout`` the slots ``(ia, ib)`` of a non-symmetric pair
    function follow the subscript order of the parameters in its body
    (``G(l, k) = exp(-c(l, k) tau(l, k))`` for ``G_lk``) instead of the
    alphabetical order of the summation indices, so the table is written and
    read in the parameters' own (column-major) layout.

    ``scheme`` gives the slot indices and array prefixes of the new
    functions (:class:`NameScheme`).
    """
    from thermodiff.fortran import _canonical_symmetric, _swap

    def plain(arg):
        # only functions of T (and parameters): V and P dependence would be
        # lost, because the abstract functions are differentiated in T only
        return (
            _mole_free(arg)
            and not arg.has(sp.Sum)
            and not arg.has(sp.Piecewise)
            and not arg.has(V)
            and not arg.has(P)
        )

    def make(body, indices):
        indices = sorted(indices, key=str)
        if len(indices) == 1:
            canon = body.xreplace({indices[0]: scheme.component_slot})
            key = ("c", canon)
            if key not in registry:
                registry[key] = _Param(
                    f"{scheme.prefixes[CMP_KIND]}{len(registry)}", canon, 1
                )
            return registry[key].func(indices[0], T)
        a, b = indices
        slot_a, slot_b = scheme.pair_slots
        canon = body.xreplace({a: slot_a, b: slot_b})
        canon = _canonical_symmetric(canon, symmetric_params)
        if (
            table_layout
            and _subscript_order(body, a, b) == (b, a)
            and _swap(canon, slot_a, slot_b, symmetric_params) != canon
        ):
            a, b = b, a
            canon = body.xreplace({a: slot_a, b: slot_b})
            canon = _canonical_symmetric(canon, symmetric_params)
        inner = abstract_component_functions(
            canon, registry, symmetric_params, pairs=False, scheme=scheme
        )
        key = ("p", canon)
        if key not in registry:
            sym = _swap(canon, slot_a, slot_b, symmetric_params) == canon
            registry[key] = _Param(
                f"{scheme.prefixes[MAT_KIND]}{len(registry)}", inner, 2, sym
            )
        if canon.has(T):
            return registry[key].func(a, b, T)
        return registry[key].func(a, b)

    def _split_pair_roots(group, idx):
        """sqrt(c*A_a*A_b)*K_ab -> sqrt(c)*sqrt(A_a)*sqrt(A_b)*K_ab.

        The per-component factors are first abstracted, then the roots of
        their products are split (valid for non-negative component
        functions, as the attractive parameters of cubic equations).
        Returns the list of factors, or None when the group does not split
        into per-component functions and a temperature-free pair factor.
        """
        inner = abstract_component_functions(
            group, registry, symmetric_params, pairs=False, scheme=scheme
        )
        expanded = sp.expand_power_base(inner, force=True)
        factors = (
            expanded.args if isinstance(expanded, sp.Mul) else (expanded,)
        )
        out = []
        for f in factors:
            fidx = f.atoms(sp.Idx)
            if len(fidx) == 2 and f.has(T):
                return None  # still a genuine function of the pair and T
            if len(fidx) == 1 and f.has(T):
                if isinstance(f, sp.core.function.AppliedUndef):
                    out.append(f)
                else:
                    # e.g. sqrt(cmp0(k, T)): a new component function
                    (index,) = fidx
                    canon = f.xreplace({index: scheme.component_slot})
                    key = ("c", canon)
                    if key not in registry:
                        registry[key] = _Param(
                            f"{scheme.prefixes[CMP_KIND]}{len(registry)}",
                            canon,
                            1,
                        )
                    out.append(registry[key].func(index, T))
            else:
                out.append(f)
        return out

    def rewrite(sub, inside):
        if not getattr(sub, "args", ()) or isinstance(
            sub, (sp.Idx, sp.Indexed)
        ):
            return sub
        if isinstance(sub, sp.Sum):
            return sub.func(rewrite(sub.function, True), *sub.limits)
        if inside and isinstance(sub, sp.Mul):
            free = [a for a in sub.args if plain(a) and a.atoms(sp.Idx)]
            rest = [a for a in sub.args if a not in free]
            pair_groups = {}
            for a in free:
                idx = frozenset(a.atoms(sp.Idx))
                if len(idx) == 2 and pairs:
                    pair_groups.setdefault(idx, []).append(a)
            new = [rewrite(a, inside) for a in rest]
            used = set()
            for idx in sorted(pair_groups, key=lambda x: sorted(map(str, x))):
                members = [
                    a
                    for a in free
                    if a.atoms(sp.Idx) <= idx and id(a) not in used
                ]
                if len(members) < 2 and not any(m.has(T) for m in members):
                    continue
                for m in members:
                    used.add(id(m))
                group = sp.Mul(*members)
                if (
                    split_sqrt
                    and group.has(T)
                    and any(
                        isinstance(x, sp.Pow) and not x.exp.is_integer
                        for x in sp.preorder_traversal(group)
                    )
                ):
                    split = _split_pair_roots(group, idx)
                    if split is not None:
                        new.extend(split)
                        continue
                new.append(make(group, idx))
            singles = {}
            for a in free:
                if id(a) in used:
                    continue
                idx = a.atoms(sp.Idx)
                if len(idx) == 1:
                    singles.setdefault(next(iter(idx)), []).append(a)
                else:
                    new.append(rewrite(a, inside))
            for index in sorted(singles, key=str):
                members = singles[index]
                if any(m.has(T) for m in members):
                    new.append(make(sp.Mul(*members), {index}))
                else:
                    new.extend(members)
            return sp.Mul(*new)
        if inside and plain(sub) and sub.has(T):
            idx = sub.atoms(sp.Idx)
            if len(idx) == 1 or (len(idx) == 2 and pairs):
                return make(sub, idx)
        return sub.func(*[rewrite(arg, inside) for arg in sub.args])

    # called on a pair body (no Sum): treat it as "inside"
    return rewrite(expr, not expr.has(sp.Sum))


def component_definitions(
    registry,
    order,
    inline_rates=False,
    pair_cse=False,
    pair_rates=False,
    scheme=DEFAULT_NAME_SCHEME,
):
    """Values and temperature derivatives of the abstracted functions.

    Returns
    -------
    defs : list of (name, body, rank, symmetric)
        ``body`` is in terms of ``scheme.component_slot`` (rank 1) or
        ``scheme.pair_slots`` (rank 2), with the component functions it uses
        already replaced by their arrays.  Rank-1 definitions come first.
    mapping : callable
        Replaces the abstract functions and their derivatives by arrays.
    """
    names = {}
    params = {p.name: p for p in registry.values()}
    for p in params.values():
        for d in range(order + 1):
            names[(p.name, d)] = sp.IndexedBase(
                p.name + ("" if d == 0 else "_" + "T" * d),
                shape=(nc,) * p.rank,
            )

    def mapping(expr):
        def visit(sub):
            if isinstance(sub, sp.Derivative):
                f = sub.expr
                if (
                    isinstance(f, sp.core.function.AppliedUndef)
                    and str(f.func) in params
                ):
                    d = sum(c for var, c in sub.variable_count if var == T)
                    idx = [a for a in f.args if a != T]
                    if (
                        inline_rates
                        and str(f.func) in pair_dep_rates
                        and d in (1, 2)
                    ):
                        G = pair_dep_rates[str(f.func)].xreplace(
                            dict(zip(scheme.pair_slots, idx))
                        )
                        m = sp.Indexed(names[(str(f.func), 0)], *idx)
                        if d == 1:
                            return visit(m * G)
                        return visit(m * (G**2 + sp.diff(G, T)))
                    if inline_rates and str(f.func) in rates and d in (1, 2):
                        # M' = M*rate, M'' = M*(rate**2 + rate'), written in
                        # place: the sums then split into dot products
                        g0, vecs = rates[str(f.func)]
                        a_, b_ = idx
                        m = sp.Indexed(names[(str(f.func), 0)], a_, b_)
                        rate = g0 + vecs[0][a_] + vecs[0][b_]
                        if d == 1:
                            return m * rate
                        rate_T = sp.diff(g0, T) + vecs[1][a_] + vecs[1][b_]
                        return m * (rate**2 + rate_T)
                    return sp.Indexed(names[(str(f.func), d)], *idx)
            if (
                isinstance(sub, sp.core.function.AppliedUndef)
                and str(sub.func) in params
            ):
                return sp.Indexed(
                    names[(str(sub.func), 0)], *[a for a in sub.args if a != T]
                )
            if not getattr(sub, "args", ()) or isinstance(
                sub, (sp.Idx, sp.Indexed)
            ):
                return sub
            return sub.func(*[visit(arg) for arg in sub.args])

        return visit(expr)

    if pair_cse:
        # a pair body that contains an earlier one (tau*exp(-c tau) after
        # exp(-c tau)) is written in terms of it, through its function so
        # that its temperature derivatives propagate
        earlier = []
        for p in params.values():
            if p.rank != 2:
                continue
            for q in earlier:
                if q.body.has(T) and q.body != p.body:
                    new_body = p.body.subs(
                        q.body, q.func(*scheme.pair_slots, T)
                    )
                    if new_body != p.body:
                        p.body = new_body
            earlier.append(p)

    # logarithmic rates of the pair functions: M' = M*(c + g(a) + g(b))
    rates = {}
    pair_dep_rates = {}
    extra_vectors = []
    for p in params.values():
        if p.rank != 2 or not p.body.has(T) or order == 0:
            continue
        # the logarithmic form divides by the T-dependent factors; it is only
        # used when every such factor sits under a non-integer power (as in
        # sqrt(a_a*a_b)), whose plain derivative divides by it too
        factors = p.body.args if isinstance(p.body, sp.Mul) else (p.body,)
        if not all(
            (not f.has(T))
            or (isinstance(f, sp.Pow) and not f.exp.is_integer)
            or (pair_rates and isinstance(f, sp.exp))
            for f in factors
        ):
            continue
        G = sp.expand(sp.diff(sp.expand_log(sp.log(p.body), force=True), T))
        terms = G.args if isinstance(G, sp.Add) else (G,)
        if (
            pair_rates
            and not G.has(sp.log)
            and not G.has(sp.exp)
            and any(len(t.atoms(sp.Idx)) > 1 for t in terms)
        ):
            # a cheap rate that depends on the pair (c_ab b_ab / T**2 for
            # exp(-c_ab tau_ab)): M' = M*G_ab written in place
            pair_dep_rates[p.name] = G
            continue
        if G.has(sp.log) or any(len(t.atoms(sp.Idx)) > 1 for t in terms):
            continue
        slot_a, slot_b = scheme.pair_slots
        ga = sp.Add(*[t for t in terms if slot_a in t.atoms(sp.Idx)])
        gb = sp.Add(*[t for t in terms if slot_b in t.atoms(sp.Idx)])
        g0 = sp.Add(*[t for t in terms if not t.atoms(sp.Idx)])
        ca = ga.xreplace({slot_a: scheme.component_slot})
        cb = gb.xreplace({slot_b: scheme.component_slot})
        if sp.simplify(ca - cb) != 0:
            continue
        rate_vecs = []
        for d in range(order):  # rate and its T derivatives up to order-1
            vname = f"{p.name}_g" + "T" * d
            body = ca if d == 0 else sp.diff(ca, T, d)
            extra_vectors.append((vname, body))
            rate_vecs.append(sp.IndexedBase(vname, shape=(nc,)))
        rates[p.name] = (g0, rate_vecs)

    # logarithmic rates of the component functions with costly powers:
    # A' = A*h, A'' = A'*h + A*h' with h = d(log A)/dT, so that a power such
    # as (3/(2 + T/Tc))**k is evaluated once instead of once per derivative
    comp_rates = {}
    for p in params.values():
        if p.rank != 1 or not p.body.has(T) or order == 0:
            continue
        factors = p.body.args if isinstance(p.body, sp.Mul) else (p.body,)
        costly = [f for f in factors if f.has(T)]
        if not costly or not all(
            (isinstance(f, sp.Pow) and not f.exp.is_integer)
            or isinstance(f, sp.exp)
            for f in costly
        ):
            continue
        h = sp.diff(sp.expand_log(sp.log(p.body), force=True), T)
        if h.has(sp.log) and any(
            isinstance(f, sp.Pow) and f.exp.has(T) for f in costly
        ):
            continue
        comp_rates[p.name] = h

    defs = []
    for rank in (1, 2):
        for p in params.values():
            if p.rank != rank:
                continue
            slots = (
                (scheme.component_slot,) if rank == 1 else scheme.pair_slots
            )
            arrays_ = [
                sp.Indexed(names[(p.name, d)], *slots)
                for d in range(order + 1)
            ]
            opaque = sp.Function("_opaque_" + p.name)(T)
            current = p.body
            for d in range(order + 1 if p.body.has(T) else 1):
                if d == 0:
                    value = p.body
                elif p.name in comp_rates:
                    h = comp_rates[p.name]
                    value = (
                        arrays_[0] * h
                        if d == 1
                        else arrays_[1] * h + arrays_[0] * sp.diff(h, T)
                    )
                elif p.name in rates:
                    g0, vecs = rates[p.name]
                    a_, b_ = scheme.pair_slots
                    rate = g0 + vecs[0][a_] + vecs[0][b_]
                    if d == 1:
                        value = arrays_[0] * rate
                    else:
                        rate_T = sp.diff(g0, T) + vecs[1][a_] + vecs[1][b_]
                        value = arrays_[1] * rate + arrays_[0] * rate_T
                else:
                    current = sp.diff(current, T)
                    if d == 1 and rank == 2:
                        current = current.subs(p.body, opaque)
                    value = current
                    for dd in range(d, 0, -1):
                        value = value.subs(
                            sp.Derivative(opaque, (T, dd)), arrays_[dd]
                        )
                    value = value.subs(opaque, arrays_[0])
                if (
                    inline_rates
                    and d > 0
                    and (p.name in rates or p.name in pair_dep_rates)
                ):
                    continue  # written in place by mapping()
                defs.append(
                    (
                        str(names[(p.name, d)]),
                        mapping(value),
                        rank,
                        p.symmetric,
                    )
                )
        if rank == 1:
            for vname, body in extra_vectors:
                defs.append((vname, mapping(body), 1, False))
    return defs, mapping


def split_sum_terms(expr, scheme=DEFAULT_NAME_SCHEME):
    """Expand summation bodies and split sums of terms into sums of sums.

    ``sum_kl n_k n_l (g_k + g_l) M_kl`` becomes
    ``sum_kl n_k g_k n_l M_kl + sum_kl n_k n_l g_l M_kl``; then the factors
    that depend only on the outer index can be moved out of the inner sum
    and every term becomes a dot product with a vector sum
    (:func:`thermodiff.fortran._nest_double_sums`).  Exact.  ``scheme``
    gives the prefixes of the abstracted arrays, which count as functions
    of ``T``.
    """

    def depends_on_t(e):
        return e.has(T) or any(
            str(b).startswith(
                (scheme.prefixes[CMP_KIND], scheme.prefixes[MAT_KIND])
            )
            for b in e.atoms(sp.IndexedBase)
        )

    def expand_rates(body):
        # factors that do not depend on T (1 - k_ij, b_i + b_j) stay atomic
        masks = {}
        for sub in sorted(body.atoms(sp.Add), key=sp.count_ops, reverse=True):
            if not depends_on_t(sub) and not any(
                sub.has(m) for m in masks.values()
            ):
                masks[sub] = sp.Dummy("mask")
        expanded = sp.expand(body.xreplace(masks))
        return expanded.xreplace({v: k for k, v in masks.items()})

    def visit(sub):
        if isinstance(sub, sp.Sum):
            body = visit(sub.function)
            if not body.has(sp.Sum) and not body.has(sp.Piecewise):
                body = expand_rates(body)
            if isinstance(body, sp.Add):
                return sp.Add(
                    *[sp.Sum(term, *sub.limits) for term in body.args]
                )
            return sub.func(body, *sub.limits)
        if not getattr(sub, "args", ()) or isinstance(
            sub, (sp.Idx, sp.Indexed)
        ):
            return sub
        return sub.func(*[visit(arg) for arg in sub.args])

    return visit(expr)


def _nest_inner_nodes(nodes):
    """Turn index-free summations nested inside node bodies into nodes.

    ``S = sum_i n_i f_i(sum_j n_j m_j d_j**3)`` becomes
    ``S = sum_i n_i f_i(Z)`` with a new node ``Z = sum_j n_j m_j d_j**3``,
    so that the derivatives of ``S`` are composed through ``Z`` by the
    chain rule instead of differentiating the nested sum inside every term.
    Returns the extended node list and {symbol: set of inner symbols}.
    """
    by_key = {_alpha_key(total): symbol for symbol, total in nodes}
    result = list(nodes)
    inner_map = {}
    queue = list(nodes)
    counter = [len(nodes)]

    def inner_sums(body):
        found = []

        def visit(sub, bound):
            if isinstance(sub, sp.Sum):
                if not (sub.atoms(sp.Idx) - _bound_all(sub)):
                    found.append(sub)
                    return
                bound = bound | _bound(sub)
                visit(sub.function, bound)
                return
            for arg in getattr(sub, "args", ()):
                visit(arg, bound)

        visit(body, set())
        return found

    while queue:
        symbol, total = queue.pop(0)
        replacements = {}
        kids = set()
        for inner in inner_sums(total.function):
            key = _alpha_key(inner)
            if key not in by_key:
                new_symbol = sp.Dummy(f"S{counter[0]}")
                counter[0] += 1
                by_key[key] = new_symbol
                result.append((new_symbol, inner))
                queue.append((new_symbol, inner))
            replacements[inner] = by_key[key]
            kids.add(by_key[key])
        if replacements:
            new_total = total.func(
                total.function.xreplace(replacements), *total.limits
            )
            result = [
                (s_, new_total if s_ == symbol else t_) for s_, t_ in result
            ]
            for pos, (s_, t_) in enumerate(queue):
                if s_ == symbol:
                    queue[pos] = (s_, new_total)
        inner_map[symbol] = kids
    return result, inner_map


def _nested_node_derivatives(nodes, inner_map, needed, indexes):
    """Total derivatives of nodes whose bodies contain inner node symbols.

    With P the partial derivatives of the body (inner nodes held fixed),
    B_k = dS/dZ_k and D the total derivatives of the inner nodes:
      D_x S  = P_x + sum_k B_k D_x Z_k
      D_xy S = P_xy + sum_k (P_{x,k} D_y Z_k + P_{y,k} D_x Z_k)
               + sum_kl B_kl D_x Z_k D_y Z_l + sum_k B_k D_xy Z_k
    Nodes without inner nodes are left to the caller (plain DiffPlz).
    """
    bodies = dict(nodes)
    done = {}
    first = {"dni": "n", "dt": "T", "dv": "V"}
    second = {
        "dnidnj": ("n", "n"),
        "dt2": ("T", "T"),
        "dv2": ("V", "V"),
        "dtdv": ("T", "V"),
        "dtdni": ("n", "T"),
        "dvdni": ("n", "V"),
    }
    attr_of = {"n": "dni", "T": "dt", "V": "dv"}

    def plain(symbol):
        dz = (
            DiffPlz(
                bodies[symbol],
                internal_functions=[],
                indexes=indexes,
                name="s",
                derivatives=needed,
            )
            if needed
            else None
        )
        entry = {"value": bodies[symbol]}
        for attr in needed:
            entry[attr] = _node_derivative(dz, attr)
        return entry

    def first_of(entry, x, index):
        value = entry.get(attr_of[x], sp.S.Zero)
        if x == "n" and index != i:
            value = value.xreplace({i: index})
        return value

    def second_of(entry, x, y):
        key = {
            ("n", "n"): "dnidnj",
            ("T", "T"): "dt2",
            ("V", "V"): "dv2",
            ("T", "V"): "dtdv",
            ("V", "T"): "dtdv",
            ("n", "T"): "dtdni",
            ("T", "n"): "dtdni",
            ("n", "V"): "dvdni",
            ("V", "n"): "dvdni",
        }[(x, y)]
        return entry.get(key, sp.S.Zero)

    def compute(symbol):
        if symbol in done:
            return done[symbol]
        kids = sorted(inner_map.get(symbol, ()), key=str)
        if not kids:
            done[symbol] = plain(symbol)
            return done[symbol]
        kid_d = {k: compute(k) for k in kids}
        body = bodies[symbol]
        partial = plain(symbol)
        firsts = sorted({a for a in needed if a in first})
        B = {k: sp.diff(body, k) for k in kids}
        PB = {}
        for k in kids:
            dz = (
                DiffPlz(
                    B[k],
                    internal_functions=[],
                    indexes=indexes,
                    name="s",
                    derivatives=firsts,
                )
                if firsts
                else None
            )
            PB[k] = {a: _node_derivative(dz, a) for a in firsts}
        entry = {"value": body}
        for attr in needed:
            if attr in first:
                x = first[attr]
                value = partial[attr]
                for k in kids:
                    value += B[k] * first_of(kid_d[k], x, i)
            else:
                x, y = second[attr]
                ix = i
                iy = j if (x == "n" and y == "n") else i
                value = partial[attr]
                for k in kids:
                    pbx = first_of(PB[k], x, ix)
                    pby = first_of(PB[k], y, iy)
                    value += pbx * first_of(kid_d[k], y, iy)
                    value += pby * first_of(kid_d[k], x, ix)
                    for l_ in kids:
                        value += (
                            sp.diff(B[k], l_)
                            * first_of(kid_d[k], x, ix)
                            * first_of(kid_d[l_], y, iy)
                        )
                    value += B[k] * second_of(kid_d[k], x, y)
            entry[attr] = value
        done[symbol] = entry
        return entry

    for symbol, _ in nodes:
        if inner_map.get(symbol):
            compute(symbol)
    return {s_: e for s_, e in done.items() if inner_map.get(s_)} | {
        s_: done[s_] for s_ in done
    }


#: functions without a usable symbolic derivative: the outer function falls back
#: to the plain ``sp.diff`` path when it contains one of them
_NONSMOOTH = (
    sp.Piecewise,
    sp.Max,
    sp.Min,
    sp.Abs,
    sp.sign,
    sp.Heaviside,
    sp.floor,
    sp.ceiling,
    sp.KroneckerDelta,
    sp.DiracDelta,
)


def _scalar_dag(F, mode):
    """Nodes ``[(u, phi)]`` of the scalar function ``F`` in dependency order.

    ``"cse"``: the common sub-expressions of ``F`` (``sympy.cse``) and ``F``
    itself, so each node is a coarse block (``dm1 = S0/S1``, the rest of F).
    ``"fine"``: in addition every compound sub-expression is a node of its own
    (one operation per node, n-ary sums and products kept whole).
    """
    rep, red = sp.cse(
        F, symbols=sp.numbered_symbols("_scu"), optimizations="basic"
    )
    nodes = list(rep) + [(sp.Symbol("_sctop"), red[0])]
    if mode == "cse":
        return nodes
    names = sp.numbered_symbols("_scv")
    out, memo = [], {}

    def lift(e, top=False):
        if e.is_Atom:
            return e
        e2 = e.func(*[lift(a) for a in e.args])
        if top or e2.is_Atom:
            return e2
        if e2 not in memo:
            memo[e2] = next(names)
            out.append((memo[e2], e2))
        return memo[e2]

    for u, phi in nodes:
        out.append((u, lift(phi, top=True)))
    return out


def _propagate(F, leaves, first, pairs, mode):
    """Forward second-order propagation of ``F`` over the DAG of ``mode``.

    Every node ``u = phi(p, ...)`` gets its gradient and the needed Hessian
    entries with respect to ``leaves`` from the local partials ``phi_p``,
    ``phi_pq`` (``sympy.diff`` of the small node body):
    ``u_l = sum_p phi_p p_l`` and ``u_lm = sum_p phi_p p_lm + sum_pq phi_pq
    p_l q_m``; structural zeros are never formed.  Returns the flat
    expressions of ``[F] + [F_l for l in first] + [F_lm for (l, m) in
    pairs]``, or ``None`` when a local partial is not explicit.
    """
    pos = {leaf: k for k, leaf in enumerate(leaves)}
    active = set(first) | {leaf for pair in pairs for leaf in pair}
    pairs_set = set(pairs)
    grad = {leaf: {leaf: sp.S.One} for leaf in leaves if leaf in active}
    hess = {leaf: {} for leaf in leaves if leaf in active}
    value = {}

    def ordered(a, b):
        return (a, b) if pos[a] <= pos[b] else (b, a)

    for u, phi in _scalar_dag(F, mode):
        phi_v = phi.xreplace(value)
        value[u] = phi_v
        parents = sorted(
            (p for p in phi.free_symbols if grad.get(p)),
            key=sp.default_sort_key,
        )
        g_u, h_u = {}, {}
        for a, p in enumerate(parents):
            d1 = sp.diff(phi, p)
            if d1.has(sp.Derivative) or d1.has(sp.Subs):
                return None
            d1 = d1.xreplace(value)
            if d1 != 0:
                for leaf, x in grad[p].items():
                    g_u[leaf] = g_u.get(leaf, 0) + d1 * x
                for key, x in hess[p].items():
                    h_u[key] = h_u.get(key, 0) + d1 * x
            for q in parents[a:]:
                d2 = sp.diff(phi, p, q)
                if d2 == 0:
                    continue
                if d2.has(sp.Derivative) or d2.has(sp.Subs):
                    return None
                d2 = d2.xreplace(value)
                for l_, x in grad[p].items():
                    for m_, y in grad[q].items():
                        key = ordered(l_, m_)
                        if key not in pairs_set:
                            continue
                        if p == q:
                            if pos[l_] > pos[m_]:
                                continue
                            term = d2 * x * y
                        else:
                            term = d2 * x * y if l_ != m_ else 2 * d2 * x * y
                        h_u[key] = h_u.get(key, 0) + term
        grad[u] = {leaf: x for leaf, x in g_u.items() if x != 0}
        hess[u] = {key: x for key, x in h_u.items() if x != 0}
    top = u
    return (
        [value[top]]
        + [grad[top].get(leaf, sp.S.Zero) for leaf in first]
        + [hess[top].get(pair, sp.S.Zero) for pair in pairs]
    )


def _scalar_chain(F, leaves, first, pairs):
    """Value, gradient and Hessian entries of ``F`` as named scalars.

    The derivatives are propagated forward over a DAG of ``F``
    (:func:`_propagate`) instead of differentiating the expanded ``F``, and
    the result goes through one common sub-expression elimination.  Both
    DAG granularities are tried and the one with fewer operations is kept.

    Returns ``(definitions, value, d1, d2)``: ``definitions`` is a list of
    ``(symbol, expr)`` in dependency order (placeholder symbols ``_sc*``),
    ``value`` the expression of ``F``, ``d1[l]`` and ``d2[(l, m)]`` the
    expressions of the requested partial derivatives (a placeholder symbol,
    a number or a leaf), or ``None`` when ``F`` is not smooth enough.
    """
    if any(F.has(func) for func in _NONSMOOTH):
        return None
    best = None
    for mode in ("cse", "fine"):
        flat = _propagate(F, leaves, first, pairs, mode)
        if flat is None:
            return None
        rep, red = sp.cse(
            flat, symbols=sp.numbered_symbols("_scc"), optimizations="basic"
        )
        ops = sum(sp.count_ops(e) for _, e in rep) + sum(
            sp.count_ops(e) for e in red
        )
        if best is None or ops < best[0]:
            best = (ops, rep, red)
    _, rep, red = best
    definitions = list(rep)
    names = sp.numbered_symbols("_scd")
    results = []
    for expr in red:
        if expr.is_Atom:
            results.append(expr)
        else:
            symbol = next(names)
            definitions.append((symbol, expr))
            results.append(symbol)
    value = results[0]
    d1 = dict(zip(first, results[1 : 1 + len(first)]))
    d2 = dict(zip(pairs, results[1 + len(first) :]))
    return definitions, value, d1, d2


def _node_derivative(diff, attr):
    value = getattr(diff, attr)
    return sp.S.Zero if value is None else value


def chain_rule_derivatives(
    expression,
    attributes,
    indexes,
    symmetric_params=(),
    split_sqrt=False,
    split_sums=False,
    pair_cse=False,
    pair_rates=False,
    nested_nodes=False,
    table_layout=False,
    scalar_chain=False,
):
    """Derivatives of ``expression`` assembled by the chain rule.

    Parameters
    ----------
    expression : sympy expression
        The fully expanded function (``DiffPlz.expression``).
    attributes : list of str
        Derivative attributes to compute (``dt``, ``dni``, ``dnidnj`` ...).
    indexes : list
        Index order, as given to :class:`~thermodiff.diffplz.DiffPlz`.

    scalar_chain : bool, optional
        Compute the partial derivatives of the outer function ``F`` by forward
        second-order propagation over its DAG (:func:`_scalar_chain`) instead
        of ``sp.diff`` of the expanded ``F``.  They become named scalars,
        returned in ``"_scalar_defs"`` (``[(symbol, expr)]`` in dependency
        order, in terms of the node symbols), and the derivatives read them;
        ``"_scalar_nodes"`` gives the summation of every node they read.  Falls back to ``sp.diff``
        when ``F`` contains a non-smooth function.

    Returns
    -------
    dict or None
        ``{"value": expr, attr: expr, ...}``, or ``None`` when the function
        does not fit the pattern (the caller then keeps the plain
        derivatives).
    """
    if any(attr not in _ATTR_VARS for attr in attributes):
        return None
    if expression.has(P):
        return None
    scheme = _name_scheme(expression)
    factored = factor_sums(expression)
    outer, nodes = summation_nodes(factored)
    if not nodes:
        return None
    # the outer function must not depend on the mole numbers any more
    if any(base == n for base in outer.atoms(sp.IndexedBase)) or outer.has(
        sp.Sum
    ):
        return None

    # node derivatives needed: any attribute that involves n needs the
    # node gradient/Hessian; T/V attributes need the T/V node derivatives
    needed = set()
    for attr in attributes:
        vars_ = _ATTR_VARS[attr]
        if vars_ == ("n",):
            needed.add("dni")
        elif vars_ == ("n", "n"):
            needed |= {"dni", "dnidnj"}
        elif "n" in vars_:
            x = vars_[0]
            needed |= {
                "dni",
                {"T": "dtdni", "V": "dvdni"}[x],
                {"T": "dt", "V": "dv"}[x],
            }
        else:
            for x in vars_:
                needed.add({"T": "dt", "V": "dv"}[x])
            if len(vars_) == 2:
                needed.add(
                    {("T", "T"): "dt2", ("V", "V"): "dv2", ("T", "V"): "dtdv"}[
                        vars_
                    ]
                )
    needed = sorted(needed)

    registry = {}
    nodes = [
        (
            symbol,
            abstract_component_functions(
                total,
                registry,
                symmetric_params,
                split_sqrt=split_sqrt,
                table_layout=table_layout,
                scheme=scheme,
            ),
        )
        for symbol, total in nodes
    ]
    node_symbols = [symbol for symbol, _ in nodes]
    node_d = {}
    if nested_nodes:
        nodes, inner_map = _nest_inner_nodes(nodes)
        node_d = _nested_node_derivatives(nodes, inner_map, needed, indexes)
    else:
        inner_map = {}
    for symbol, total in nodes:
        if symbol in node_d:
            continue
        dz = (
            DiffPlz(
                total,
                internal_functions=[],
                indexes=indexes,
                name="s",
                derivatives=needed,
            )
            if needed
            else None
        )
        entry = {"value": total}
        for attr in needed:
            entry[attr] = _node_derivative(dz, attr)
        node_d[symbol] = entry

    def node(symbol, key):
        return node_d[symbol].get(key, sp.S.Zero)

    # node derivative with respect to one variable ("T", "V", "n")
    def dS(symbol, x, index=None):
        if x == "n":
            value = node(symbol, "dni")
            return (
                value
                if index is None or index == i
                else value.xreplace({i: index})
            )
        return node(symbol, {"T": "dt", "V": "dv"}[x])

    def d2S(symbol, x, y):
        key = tuple(sorted((x, y)))
        if key == ("n", "n"):
            return node(symbol, "dnidnj")
        if "n" in key:
            other = key[0] if key[1] == "n" else key[1]
            return node(symbol, {"T": "dtdni", "V": "dvdni"}[other])
        return node(
            symbol,
            {("T", "T"): "dt2", ("V", "V"): "dv2", ("T", "V"): "dtdv"}[key],
        )

    # partial derivatives of the outer function (scalar)
    F = outer
    Fk = {s: sp.diff(F, s) for s in node_symbols}

    def Fx(x):
        return sp.S.Zero if x == "n" else sp.diff(F, _SYM[x])

    back = {symbol: total for symbol, total in nodes}

    def resolve(expr):
        # inner node symbols may appear inside node bodies
        for _ in range(len(back) + 1):
            new_expr = expr.xreplace(back)
            if new_expr == expr:
                break
            expr = new_expr
        return expr

    sc = None
    scalar_estimate = None
    if scalar_chain:
        leaves = list(node_symbols) + [_SYM[x] for x in ("V", "T")]
        first, pairs = [], []

        def want(lst, item):
            if item not in lst:
                lst.append(item)

        def pair(a, b):
            want(
                pairs, (a, b) if leaves.index(a) <= leaves.index(b) else (b, a)
            )

        # only the partials multiplied by a non-zero node derivative
        for attr in attributes:
            vars_ = _ATTR_VARS[attr]
            if len(vars_) == 1:
                x = vars_[0]
                if x != "n":
                    want(first, _SYM[x])
                for s_ in node_symbols:
                    if dS(s_, x, i) != 0:
                        want(first, s_)
                continue
            x, y = vars_
            if y == "n" and x != "n":
                x, y = "n", x
            if x != "n" and y != "n":
                pair(_SYM[x], _SYM[y])
            for s_ in node_symbols:
                if d2S(s_, x, y) != 0:
                    want(first, s_)
                for a, b in ((x, y), (y, x)):
                    if b != "n" and dS(s_, a, i) != 0:
                        pair(s_, _SYM[b])
                for t_ in node_symbols:
                    if dS(s_, x, i) != 0 and dS(t_, y, j) != 0:
                        pair(s_, t_)
        first = [leaf for leaf in leaves if leaf in first]
        pairs = sorted(
            pairs, key=lambda ab: (leaves.index(ab[0]), leaves.index(ab[1]))
        )
        # nothing to propagate (only the value): keep the plain path
        if first or pairs:
            sc = _scalar_chain(F, leaves, first, pairs)
            if sc is not None:
                # operation estimates of the two scalar blocks (scalar_chain=
                # "estimate"): the plain partials of F after one CSE, against
                # the propagated definitions and results
                plain_exprs = (
                    [F]
                    + [sp.diff(F, a) for a in first]
                    + [sp.diff(F, a, b) for a, b in pairs]
                )
                rep_, red_ = sp.cse(
                    plain_exprs,
                    symbols=sp.numbered_symbols("_sce"),
                    optimizations="basic",
                )
                est_plain = sum(sp.count_ops(e) for _, e in rep_) + sum(
                    sp.count_ops(e) for e in red_
                )
                est_sc = sum(sp.count_ops(e) for _, e in sc[0])
                scalar_estimate = (est_plain, est_sc)

    def F2_plain(a, b):
        if a in Fk_plain:
            return sp.diff(Fk_plain[a], b)
        return sp.diff(F, a, b)

    Fk_plain, Fx_plain = Fk, Fx
    if sc is not None:
        sc_defs, sc_value, sc_d1, sc_d2 = sc
        Fk = {s: sc_d1.get(s, sp.S.Zero) for s in node_symbols}

        def Fx(x):
            return sp.S.Zero if x == "n" else sc_d1.get(_SYM[x], sp.S.Zero)

        def F2(a, b):
            if leaves.index(a) > leaves.index(b):
                a, b = b, a
            return sc_d2.get((a, b), sp.S.Zero)

    else:
        F2 = F2_plain

    def total_first(x):
        # d/dx (x in T, V, n with index i)
        expr = Fx(x)
        for s in node_symbols:
            expr += Fk[s] * dS(s, x, i)
        return expr

    def total_second(x, y):
        # d2/dx dy ; for n the first n uses index i, the second j
        ix = i
        iy = j if (x == "n" and y == "n") else i
        expr = sp.S.Zero
        if x != "n" and y != "n":
            expr += F2(_SYM[x], _SYM[y])
        for s in node_symbols:
            if y != "n":
                expr += F2(s, _SYM[y]) * dS(s, x, ix)
            if x != "n":
                expr += F2(s, _SYM[x]) * dS(s, y, iy)
            for t in node_symbols:
                expr += F2(s, t) * dS(s, x, ix) * dS(t, y, iy)
            expr += Fk[s] * d2S(s, x, y)
        return expr

    order = max(
        (sum(1 for v in _ATTR_VARS[a] if v == "T") for a in attributes),
        default=0,
    )
    defs, mapping = component_definitions(
        registry,
        order,
        inline_rates=split_sums,
        pair_cse=pair_cse,
        pair_rates=pair_rates,
        scheme=scheme,
    )
    value0 = mapping(resolve(outer)) if sc is None else sc_value
    if split_sums:
        value0 = split_sum_terms(value0, scheme)
    out = {
        "value": value0,
        "_component_defs": defs,
        NAME_SCHEME_KEY: scheme,
    }
    if sc is not None:
        # the scalars keep the node symbols; the nodes they read go apart
        out["_scalar_defs"] = [
            (symbol, mapping(body)) for symbol, body in sc_defs
        ]
        read = set().union(*[body.free_symbols for _, body in sc_defs]) | (
            sc_value.free_symbols if isinstance(sc_value, sp.Expr) else set()
        )
        nodes_read = []
        for symbol in node_symbols:
            if symbol in read:
                body = mapping(resolve(symbol))
                if split_sums:
                    body = split_sum_terms(body, scheme)
                nodes_read.append((symbol, body))
        out["_scalar_nodes"] = nodes_read
        out["_scalar_estimate"] = scalar_estimate

    def assemble(target):
        for attr in attributes:
            vars_ = _ATTR_VARS[attr]
            if len(vars_) == 1:
                value = total_first(vars_[0])
            else:
                x, y = vars_
                if y == "n" and x != "n":
                    x, y = "n", x
                value = total_second(x, y)
            value = mapping(resolve(value))
            if split_sums:
                value = split_sum_terms(value, scheme)
            target[attr] = value

    assemble(out)
    if sc is not None:
        # the plain outputs as well: to_fortran runs its summation passes on
        # them too, so the loops come out as without scalar_chain
        Fk, Fx, F2 = Fk_plain, Fx_plain, F2_plain
        plain = {"value": mapping(resolve(outer))}
        if split_sums:
            plain["value"] = split_sum_terms(plain["value"], scheme)
        assemble(plain)
        out["_plain"] = plain
    return out
