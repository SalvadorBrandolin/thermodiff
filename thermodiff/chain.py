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

import sympy as sp

from thermodiff.diffplz import DiffPlz
from thermodiff.thermovars import P, T, V, i, j, n, nc

__all__ = [
    "chain_rule_derivatives",
    "factor_sums",
    "summation_nodes",
    "abstract_component_functions",
]

#: index used by the definitions of the component-function vectors
COMPONENT_SLOT = sp.Idx("ic")

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
        if not getattr(sub, "args", ()) or isinstance(sub, (sp.Idx, sp.Indexed)):
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


#: indices used by the definitions of the pair-function matrices
PAIR_SLOTS = (sp.Idx("ia"), sp.Idx("ib"))

#: prefixes of the generated arrays (changed if the model uses them)
_PREFIX = {"cmp": "cmp", "mat": "mat"}

_SLOT_CANDIDATES = ["ic", "ia", "ib", "jc", "ja", "jb", "kc", "ka", "kb",
                    "ua", "ub", "uc", "va", "vb", "vc"]
_PREFIX_CANDIDATES = {"cmp": ["cmp", "cpf", "cfn", "cvec"],
                      "mat": ["mat", "pmt", "pfn", "pmat"]}


def _configure_names(expression):
    """Choose slot indices and array prefixes that the model does not use.

    Fortran is case insensitive, so the comparison is done in lower case.
    """
    global COMPONENT_SLOT, PAIR_SLOTS
    used = {"nc", "res"}
    used |= {str(x).lower() for x in expression.atoms(sp.Symbol)}
    used |= {str(x).lower() for x in expression.atoms(sp.IndexedBase)}
    used |= {str(x).lower() for x in expression.atoms(sp.Idx)}
    slots = [name for name in _SLOT_CANDIDATES if name not in used]
    COMPONENT_SLOT = sp.Idx(slots[0])
    PAIR_SLOTS = (sp.Idx(slots[1]), sp.Idx(slots[2]))
    for kind, candidates in _PREFIX_CANDIDATES.items():
        for prefix in candidates:
            if not any(name.startswith(prefix) for name in used):
                _PREFIX[kind] = prefix
                break


class _Param:
    """A function of temperature over one or two component indices."""

    def __init__(self, name, body, rank, symmetric=False):
        self.name = name
        self.body = body        # in terms of COMPONENT_SLOT / PAIR_SLOTS
        self.rank = rank
        self.symmetric = symmetric
        self.func = sp.Function(name)


def _mole_free(sub):
    return not any(base == n for base in sub.atoms(sp.IndexedBase))


def abstract_component_functions(expr, registry, symmetric_params=(), pairs=True):
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
    """
    from thermodiff.fortran import _canonical_symmetric, _swap

    def plain(arg):
        # only functions of T (and parameters): V and P dependence would be
        # lost, because the abstract functions are differentiated in T only
        return (_mole_free(arg) and not arg.has(sp.Sum)
                and not arg.has(sp.Piecewise) and not arg.has(V)
                and not arg.has(P))

    def make(body, indices):
        indices = sorted(indices, key=str)
        if len(indices) == 1:
            canon = body.xreplace({indices[0]: COMPONENT_SLOT})
            key = ("c", canon)
            if key not in registry:
                registry[key] = _Param(f"{_PREFIX['cmp']}{len(registry)}", canon, 1)
            return registry[key].func(indices[0], T)
        a, b = indices
        canon = body.xreplace({a: PAIR_SLOTS[0], b: PAIR_SLOTS[1]})
        canon = _canonical_symmetric(canon, symmetric_params)
        inner = abstract_component_functions(canon, registry, symmetric_params, pairs=False)
        key = ("p", canon)
        if key not in registry:
            sym = _swap(canon, PAIR_SLOTS[0], PAIR_SLOTS[1], symmetric_params) == canon
            registry[key] = _Param(f"{_PREFIX['mat']}{len(registry)}", inner, 2, sym)
        if canon.has(T):
            return registry[key].func(a, b, T)
        return registry[key].func(a, b)

    def rewrite(sub, inside):
        if not getattr(sub, "args", ()) or isinstance(sub, (sp.Idx, sp.Indexed)):
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
                members = [a for a in free if a.atoms(sp.Idx) <= idx and id(a) not in used]
                if len(members) < 2 and not any(m.has(T) for m in members):
                    continue
                for m in members:
                    used.add(id(m))
                new.append(make(sp.Mul(*members), idx))
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


def component_definitions(registry, order):
    """Values and temperature derivatives of the abstracted functions.

    Returns
    -------
    defs : list of (name, body, rank, symmetric)
        ``body`` is in terms of :data:`COMPONENT_SLOT` (rank 1) or
        :data:`PAIR_SLOTS` (rank 2), with the component functions it uses
        already replaced by their arrays.  Rank-1 definitions come first.
    mapping : callable
        Replaces the abstract functions and their derivatives by arrays.
    """
    names = {}
    params = {p.name: p for p in registry.values()}
    for p in params.values():
        for d in range(order + 1):
            names[(p.name, d)] = sp.IndexedBase(
                p.name + ("" if d == 0 else "_" + "T" * d), shape=(nc,) * p.rank
            )

    def mapping(expr):
        def visit(sub):
            if isinstance(sub, sp.Derivative):
                f = sub.expr
                if isinstance(f, sp.core.function.AppliedUndef) and str(f.func) in params:
                    d = sum(c for var, c in sub.variable_count if var == T)
                    return sp.Indexed(names[(str(f.func), d)], *[a for a in f.args if a != T])
            if isinstance(sub, sp.core.function.AppliedUndef) and str(sub.func) in params:
                return sp.Indexed(names[(str(sub.func), 0)], *[a for a in sub.args if a != T])
            if not getattr(sub, "args", ()) or isinstance(sub, (sp.Idx, sp.Indexed)):
                return sub
            return sub.func(*[visit(arg) for arg in sub.args])
        return visit(expr)

    # logarithmic rates of the pair functions: M' = M*(c + g(a) + g(b))
    rates = {}
    extra_vectors = []
    for p in params.values():
        if p.rank != 2 or not p.body.has(T) or order == 0:
            continue
        # the logarithmic form divides by the T-dependent factors; it is only
        # used when every such factor sits under a non-integer power (as in
        # sqrt(a_a*a_b)), whose plain derivative divides by it too
        factors = p.body.args if isinstance(p.body, sp.Mul) else (p.body,)
        if not all(
            (not f.has(T)) or (isinstance(f, sp.Pow) and not f.exp.is_integer)
            for f in factors
        ):
            continue
        G = sp.expand(sp.diff(sp.expand_log(sp.log(p.body), force=True), T))
        terms = G.args if isinstance(G, sp.Add) else (G,)
        if G.has(sp.log) or any(len(t.atoms(sp.Idx)) > 1 for t in terms):
            continue
        ga = sp.Add(*[t for t in terms if PAIR_SLOTS[0] in t.atoms(sp.Idx)])
        gb = sp.Add(*[t for t in terms if PAIR_SLOTS[1] in t.atoms(sp.Idx)])
        g0 = sp.Add(*[t for t in terms if not t.atoms(sp.Idx)])
        ca = ga.xreplace({PAIR_SLOTS[0]: COMPONENT_SLOT})
        cb = gb.xreplace({PAIR_SLOTS[1]: COMPONENT_SLOT})
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
            (isinstance(f, sp.Pow) and not f.exp.is_integer) or isinstance(f, sp.exp)
            for f in costly
        ):
            continue
        h = sp.diff(sp.expand_log(sp.log(p.body), force=True), T)
        if h.has(sp.log) and any(isinstance(f, sp.Pow) and f.exp.has(T) for f in costly):
            continue
        comp_rates[p.name] = h

    defs = []
    for rank in (1, 2):
        for p in params.values():
            if p.rank != rank:
                continue
            slots = (COMPONENT_SLOT,) if rank == 1 else PAIR_SLOTS
            arrays_ = [sp.Indexed(names[(p.name, d)], *slots)
                       for d in range(order + 1)]
            opaque = sp.Function("_opaque_" + p.name)(T)
            current = p.body
            for d in range(order + 1 if p.body.has(T) else 1):
                if d == 0:
                    value = p.body
                elif p.name in comp_rates:
                    h = comp_rates[p.name]
                    value = (arrays_[0] * h if d == 1
                             else arrays_[1] * h + arrays_[0] * sp.diff(h, T))
                elif p.name in rates:
                    g0, vecs = rates[p.name]
                    a_, b_ = PAIR_SLOTS
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
                        value = value.subs(sp.Derivative(opaque, (T, dd)), arrays_[dd])
                    value = value.subs(opaque, arrays_[0])
                defs.append((str(names[(p.name, d)]), mapping(value), rank,
                             p.symmetric))
        if rank == 1:
            for vname, body in extra_vectors:
                defs.append((vname, mapping(body), 1, False))
    return defs, mapping


def _node_derivative(diff, attr):
    value = getattr(diff, attr)
    return sp.S.Zero if value is None else value


def chain_rule_derivatives(expression, attributes, indexes, symmetric_params=()):
    """Derivatives of ``expression`` assembled by the chain rule.

    Parameters
    ----------
    expression : sympy expression
        The fully expanded function (``DiffPlz.expression``).
    attributes : list of str
        Derivative attributes to compute (``dt``, ``dni``, ``dnidnj`` ...).
    indexes : list
        Index order, as given to :class:`~thermodiff.diffplz.DiffPlz`.

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
    _configure_names(expression)
    factored = factor_sums(expression)
    outer, nodes = summation_nodes(factored)
    if not nodes:
        return None
    # the outer function must not depend on the mole numbers any more
    if any(base == n for base in outer.atoms(sp.IndexedBase)) or outer.has(sp.Sum):
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
            needed |= {"dni", {"T": "dtdni", "V": "dvdni"}[x], {"T": "dt", "V": "dv"}[x]}
        else:
            for x in vars_:
                needed.add({"T": "dt", "V": "dv"}[x])
            if len(vars_) == 2:
                needed.add({("T", "T"): "dt2", ("V", "V"): "dv2",
                            ("T", "V"): "dtdv"}[vars_])
    needed = sorted(needed)

    registry = {}
    nodes = [(symbol, abstract_component_functions(total, registry, symmetric_params))
             for symbol, total in nodes]
    node_symbols = [symbol for symbol, _ in nodes]
    node_d = {}
    for symbol, total in nodes:
        dz = DiffPlz(total, internal_functions=[], indexes=indexes, name="s",
                     derivatives=needed) if needed else None
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
            return value if index is None or index == i else value.xreplace({i: index})
        return node(symbol, {"T": "dt", "V": "dv"}[x])

    def d2S(symbol, x, y):
        key = tuple(sorted((x, y)))
        if key == ("n", "n"):
            return node(symbol, "dnidnj")
        if "n" in key:
            other = key[0] if key[1] == "n" else key[1]
            return node(symbol, {"T": "dtdni", "V": "dvdni"}[other])
        return node(symbol, {("T", "T"): "dt2", ("V", "V"): "dv2",
                             ("T", "V"): "dtdv"}[key])

    # partial derivatives of the outer function (scalar)
    F = outer
    Fk = {s: sp.diff(F, s) for s in node_symbols}
    def Fx(x):
        return sp.S.Zero if x == "n" else sp.diff(F, _SYM[x])

    back = {symbol: total for symbol, total in nodes}

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
            expr += sp.diff(F, _SYM[x], _SYM[y])
        for s in node_symbols:
            if y != "n":
                expr += sp.diff(Fk[s], _SYM[y]) * dS(s, x, ix)
            if x != "n":
                expr += sp.diff(Fk[s], _SYM[x]) * dS(s, y, iy)
            for t in node_symbols:
                expr += sp.diff(Fk[s], t) * dS(s, x, ix) * dS(t, y, iy)
            expr += Fk[s] * d2S(s, x, y)
        return expr

    order = max((sum(1 for v in _ATTR_VARS[a] if v == "T") for a in attributes), default=0)
    defs, mapping = component_definitions(registry, order)
    out = {"value": mapping(outer.xreplace(back)), "_component_defs": defs}
    for attr in attributes:
        vars_ = _ATTR_VARS[attr]
        if len(vars_) == 1:
            value = total_first(vars_[0])
        else:
            x, y = vars_
            if y == "n" and x != "n":
                x, y = "n", x
            value = total_second(x, y)
        out[attr] = mapping(value.xreplace(back))
    return out
