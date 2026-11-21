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

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
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
    context = _AbstractionContext(
        registry, symmetric_params, pairs, split_sqrt, table_layout, scheme
    )
    # called on a pair body (no Sum): treat it as "inside"
    return _rewrite_factors(expr, not expr.has(sp.Sum), context)


@dataclass(frozen=True)
class _AbstractionContext:
    """Options and per-call state of :func:`abstract_component_functions`.

    Attributes
    ----------
    registry : dict
        Canonical body -> :class:`_Param`; filled while rewriting.
    symmetric_params : tuple
        Names of the symmetric parameters.
    pairs : bool
        Abstract the pair functions.
    split_sqrt : bool
        Split the roots of pair groups into per-component functions.
    table_layout : bool
        Follow the subscript order of the parameters in pair slots.
    scheme : NameScheme
        Slot indices and array prefixes of the new functions.
    """

    registry: dict
    symmetric_params: tuple
    pairs: bool
    split_sqrt: bool
    table_layout: bool
    scheme: NameScheme


def _is_plain_factor(arg) -> bool:
    """Tell whether ``arg`` is a function of ``T`` and parameters only.

    ``V`` and ``P`` dependence would be lost, because the abstract
    functions are differentiated in ``T`` only.
    """
    return (
        _mole_free(arg)
        and not arg.has(sp.Sum)
        and not arg.has(sp.Piecewise)
        and not arg.has(V)
        and not arg.has(P)
    )


def _register_component(canon, index, context):
    """Return the component function of ``canon`` applied to ``index``.

    The function is added to the registry when the body is new.
    """
    registry, scheme = context.registry, context.scheme
    key = ("c", canon)
    if key not in registry:
        registry[key] = _Param(
            f"{scheme.prefixes[CMP_KIND]}{len(registry)}", canon, 1
        )
    return registry[key].func(index, T)


def _canonical_pair(body, first, second, context):
    """Return ``body`` in the pair slots, with canonical symmetric terms."""
    from thermodiff.fortran import _canonical_symmetric

    slot_a, slot_b = context.scheme.pair_slots
    canon = body.xreplace({first: slot_a, second: slot_b})
    return _canonical_symmetric(canon, context.symmetric_params)


def _make_pair_function(body, first, second, context):
    """Return the pair function of ``body`` over two indices.

    With ``table_layout`` the slots follow the subscript order of the
    parameters when that differs from the alphabetical order and swapping
    the slots changes the canonical body.
    """
    from thermodiff.fortran import _swap

    registry, scheme = context.registry, context.scheme
    slot_a, slot_b = scheme.pair_slots
    canon = _canonical_pair(body, first, second, context)
    if (
        context.table_layout
        and _subscript_order(body, first, second) == (second, first)
        and _swap(canon, slot_a, slot_b, context.symmetric_params) != canon
    ):
        first, second = second, first
        canon = _canonical_pair(body, first, second, context)
    inner = abstract_component_functions(
        canon,
        registry,
        context.symmetric_params,
        pairs=False,
        scheme=scheme,
    )
    key = ("p", canon)
    if key not in registry:
        symmetric = (
            _swap(canon, slot_a, slot_b, context.symmetric_params) == canon
        )
        registry[key] = _Param(
            f"{scheme.prefixes[MAT_KIND]}{len(registry)}",
            inner,
            2,
            symmetric,
        )
    if canon.has(T):
        return registry[key].func(first, second, T)
    return registry[key].func(first, second)


def _make_function(body, indices, context):
    """Abstract ``body`` as a component or a pair function of ``indices``."""
    indices = sorted(indices, key=str)
    if len(indices) == 1:
        slot = context.scheme.component_slot
        canon = body.xreplace({indices[0]: slot})
        return _register_component(canon, indices[0], context)
    first, second = indices
    return _make_pair_function(body, first, second, context)


def _as_component_function(factor, index, context):
    """Return ``factor`` as a component function of ``index``.

    A factor that is already an applied function is kept; anything else
    (e.g. ``sqrt(cmp0(k, T))``) becomes a new component function.
    """
    if isinstance(factor, sp.core.function.AppliedUndef):
        return factor
    canon = factor.xreplace({index: context.scheme.component_slot})
    return _register_component(canon, index, context)


def _split_pair_roots(group, context):
    """sqrt(c*A_a*A_b)*K_ab -> sqrt(c)*sqrt(A_a)*sqrt(A_b)*K_ab.

    The per-component factors are first abstracted, then the roots of
    their products are split (valid for non-negative component
    functions, as the attractive parameters of cubic equations).
    Returns the list of factors, or None when the group does not split
    into per-component functions and a temperature-free pair factor.
    """
    inner = abstract_component_functions(
        group,
        context.registry,
        context.symmetric_params,
        pairs=False,
        scheme=context.scheme,
    )
    expanded = sp.expand_power_base(inner, force=True)
    factors = expanded.args if isinstance(expanded, sp.Mul) else (expanded,)
    out = []
    for factor in factors:
        factor_indices = factor.atoms(sp.Idx)
        if len(factor_indices) == 2 and factor.has(T):
            return None  # still a genuine function of the pair and T
        if len(factor_indices) == 1 and factor.has(T):
            (index,) = factor_indices
            out.append(_as_component_function(factor, index, context))
        else:
            out.append(factor)
    return out


def _has_non_integer_power(group) -> bool:
    """Tell whether ``group`` contains a power with a non-integer exponent."""
    return any(
        isinstance(node, sp.Pow) and not node.exp.is_integer
        for node in sp.preorder_traversal(group)
    )


def _abstract_pair_groups(free, pair_groups, context):
    """Abstract the groups of factors over the same two indices.

    Returns ``(items, used)``: the new factors, and the ``id`` of every
    member of ``free`` that went into a group.
    """
    items, used = [], set()
    for indices in sorted(pair_groups, key=lambda x: sorted(map(str, x))):
        members = [
            factor
            for factor in free
            if factor.atoms(sp.Idx) <= indices and id(factor) not in used
        ]
        if len(members) < 2 and not any(m.has(T) for m in members):
            continue
        used.update(id(member) for member in members)
        group = sp.Mul(*members)
        if (
            context.split_sqrt
            and group.has(T)
            and _has_non_integer_power(group)
        ):
            split = _split_pair_roots(group, context)
            if split is not None:
                items.extend(split)
                continue
        items.append(_make_function(group, indices, context))
    return items, used


def _abstract_single_indices(free, used, context):
    """Abstract the factors of ``free`` that depend on a single index.

    Factors over other index sets are rewritten on their own, and come
    first; the groups of single-index factors follow in index order.
    """
    items, singles = [], {}
    for factor in free:
        if id(factor) in used:
            continue
        indices = factor.atoms(sp.Idx)
        if len(indices) == 1:
            singles.setdefault(next(iter(indices)), []).append(factor)
        else:
            items.append(_rewrite_factors(factor, True, context))
    for index in sorted(singles, key=str):
        members = singles[index]
        if any(member.has(T) for member in members):
            items.append(_make_function(sp.Mul(*members), {index}, context))
        else:
            items.extend(members)
    return items


def _rewrite_product(product, context):
    """Abstract the plain factors of a product inside a summation."""
    free = [
        arg
        for arg in product.args
        if _is_plain_factor(arg) and arg.atoms(sp.Idx)
    ]
    rest = [arg for arg in product.args if arg not in free]
    pair_groups = {}
    for factor in free:
        indices = frozenset(factor.atoms(sp.Idx))
        if len(indices) == 2 and context.pairs:
            pair_groups.setdefault(indices, []).append(factor)
    new = [_rewrite_factors(arg, True, context) for arg in rest]
    pair_items, used = _abstract_pair_groups(free, pair_groups, context)
    new.extend(pair_items)
    new.extend(_abstract_single_indices(free, used, context))
    return sp.Mul(*new)


def _rewrite_factors(sub, inside, context):
    """Rewrite ``sub``, abstracting what lies inside summations.

    ``inside`` tells whether ``sub`` is the body of a summation.
    """
    if not getattr(sub, "args", ()) or isinstance(sub, (sp.Idx, sp.Indexed)):
        return sub
    if isinstance(sub, sp.Sum):
        body = _rewrite_factors(sub.function, True, context)
        return sub.func(body, *sub.limits)
    if inside and isinstance(sub, sp.Mul):
        return _rewrite_product(sub, context)
    if inside and _is_plain_factor(sub) and sub.has(T):
        indices = sub.atoms(sp.Idx)
        if len(indices) == 1 or (len(indices) == 2 and context.pairs):
            return _make_function(sub, indices, context)
    return sub.func(
        *[_rewrite_factors(arg, inside, context) for arg in sub.args]
    )


@dataclass(frozen=True)
class _MappingContext:
    """State read by :func:`_map_to_arrays`.

    Attributes
    ----------
    names : dict
        ``(function name, derivative order)`` -> ``IndexedBase``.
    params : dict
        Name -> abstracted function (``registry`` values).
    rates : dict
        Name -> ``(g0, rate vectors)`` of the separable logarithmic rates.
    pair_dep_rates : dict
        Name -> logarithmic rate that depends on the pair.
    inline_rates : bool
        Write the first two derivatives of the rate functions in place.
    scheme : NameScheme
        Prefixes and slots of the abstracted arrays.
    """

    names: dict
    params: dict
    rates: dict
    pair_dep_rates: dict
    inline_rates: bool
    scheme: NameScheme


def _array_names(params: dict, order: int) -> dict:
    """Return the ``IndexedBase`` of every function and derivative order.

    Parameters
    ----------
    params : dict
        Name -> abstracted function.
    order : int
        Highest temperature derivative.

    Returns
    -------
    dict
        ``(name, derivative order)`` -> ``IndexedBase``.
    """
    names = {}
    for p in params.values():
        for deriv in range(order + 1):
            suffix = "" if deriv == 0 else "_" + "T" * deriv
            names[(p.name, deriv)] = sp.IndexedBase(
                p.name + suffix, shape=(nc,) * p.rank
            )
    return names


def _inline_derivative(func, deriv, idx, context):
    """Return ``M'`` or ``M''`` written in place, or ``None``.

    Parameters
    ----------
    func : AppliedUndef
        The differentiated abstracted function.
    deriv : int
        Order of the temperature derivative.
    idx : list
        Its arguments without ``T``.
    context : _MappingContext
        State of the mapping.

    Returns
    -------
    sympy expression or None
        ``None`` when the derivative is not written in place.
    """
    name = str(func.func)
    if not context.inline_rates or deriv not in (1, 2):
        return None
    scheme = context.scheme
    if name in context.pair_dep_rates:
        rate = context.pair_dep_rates[name].xreplace(
            dict(zip(scheme.pair_slots, idx))
        )
        base = sp.Indexed(context.names[(name, 0)], *idx)
        if deriv == 1:
            return _map_to_arrays(base * rate, context)
        return _map_to_arrays(base * (rate**2 + sp.diff(rate, T)), context)
    if name in context.rates:
        # M' = M*rate, M'' = M*(rate**2 + rate'), written in place: the sums
        # then split into dot products
        g0, vecs = context.rates[name]
        slot_a, slot_b = idx
        base = sp.Indexed(context.names[(name, 0)], slot_a, slot_b)
        rate = g0 + vecs[0][slot_a] + vecs[0][slot_b]
        if deriv == 1:
            return base * rate
        rate_t = sp.diff(g0, T) + vecs[1][slot_a] + vecs[1][slot_b]
        return base * (rate**2 + rate_t)
    return None


def _map_derivative(deriv_expr, context):
    """Return the array that replaces a derivative, or ``None``.

    Parameters
    ----------
    deriv_expr : sympy.Derivative
        A derivative node.
    context : _MappingContext
        State of the mapping.

    Returns
    -------
    sympy expression or None
        ``None`` when it is not a derivative of an abstracted function.
    """
    func = deriv_expr.expr
    if not (
        isinstance(func, sp.core.function.AppliedUndef)
        and str(func.func) in context.params
    ):
        return None
    deriv = sum(c for var, c in deriv_expr.variable_count if var == T)
    idx = [arg for arg in func.args if arg != T]
    inlined = _inline_derivative(func, deriv, idx, context)
    if inlined is not None:
        return inlined
    return sp.Indexed(context.names[(str(func.func), deriv)], *idx)


def _map_to_arrays(sub, context):
    """Replace the abstract functions and their derivatives by arrays.

    Parameters
    ----------
    sub : sympy expression
        Expression to rewrite.
    context : _MappingContext
        State of the mapping.

    Returns
    -------
    sympy expression
        ``sub`` in terms of the arrays of ``context.names``.
    """
    if isinstance(sub, sp.Derivative):
        replaced = _map_derivative(sub, context)
        if replaced is not None:
            return replaced
    if (
        isinstance(sub, sp.core.function.AppliedUndef)
        and str(sub.func) in context.params
    ):
        return sp.Indexed(
            context.names[(str(sub.func), 0)],
            *[arg for arg in sub.args if arg != T],
        )
    if not getattr(sub, "args", ()) or isinstance(sub, (sp.Idx, sp.Indexed)):
        return sub
    return sub.func(*[_map_to_arrays(arg, context) for arg in sub.args])


def _share_pair_bodies(params: dict, scheme: NameScheme) -> None:
    """Write a pair body in terms of the earlier pair body it contains.

    ``tau*exp(-c tau)`` after ``exp(-c tau)`` is written through the function
    of the earlier one, so that its temperature derivatives propagate.
    Mutates ``body`` of the registry objects, in registry order.

    Parameters
    ----------
    params : dict
        Name -> abstracted function.
    scheme : NameScheme
        Slots of the pair functions.
    """
    earlier = []
    for p in params.values():
        if p.rank != 2:
            continue
        for q in earlier:
            if q.body.has(T) and q.body != p.body:
                new_body = p.body.subs(q.body, q.func(*scheme.pair_slots, T))
                if new_body != p.body:
                    p.body = new_body
        earlier.append(p)


def _product_factors(body):
    """Return the factors of a product (or the body itself)."""
    return body.args if isinstance(body, sp.Mul) else (body,)


def _log_rate_applies(body, pair_rates):
    """Tell whether the logarithmic rate form is valid for a pair body.

    The form divides by the T-dependent factors; it is only used when every
    such factor sits under a non-integer power (as in ``sqrt(a_a*a_b)``),
    whose plain derivative divides by it too.
    """
    return all(
        (not factor.has(T))
        or (isinstance(factor, sp.Pow) and not factor.exp.is_integer)
        or (pair_rates and isinstance(factor, sp.exp))
        for factor in _product_factors(body)
    )


def _pair_dependent_terms(terms):
    """Tell whether some term of a rate depends on both pair indexes."""
    return any(len(term.atoms(sp.Idx)) > 1 for term in terms)


def _separable_rate(rate, terms, scheme):
    """Split ``g(a) + g(b)`` rates into ``(g0, g)``, or return ``None``.

    Parameters
    ----------
    rate : sympy expression
        ``d(log M)/dT``.
    terms : tuple
        Terms of ``rate``.
    scheme : NameScheme
        Pair slots and component slot.

    Returns
    -------
    tuple or None
        ``(g0, g_component)``; ``None`` when the terms do not separate.
    """
    if rate.has(sp.log) or _pair_dependent_terms(terms):
        return None
    slot_a, slot_b = scheme.pair_slots
    rate_a = sp.Add(*[t for t in terms if slot_a in t.atoms(sp.Idx)])
    rate_b = sp.Add(*[t for t in terms if slot_b in t.atoms(sp.Idx)])
    constant = sp.Add(*[t for t in terms if not t.atoms(sp.Idx)])
    comp_a = rate_a.xreplace({slot_a: scheme.component_slot})
    comp_b = rate_b.xreplace({slot_b: scheme.component_slot})
    if sp.simplify(comp_a - comp_b) != 0:
        return None
    return constant, comp_a


def _pair_rate_terms(body):
    """Return ``d(log body)/dT`` expanded, and its terms."""
    rate = sp.expand(sp.diff(sp.expand_log(sp.log(body), force=True), T))
    return rate, rate.args if isinstance(rate, sp.Add) else (rate,)


def _rate_vector_definitions(name, component_rate, order):
    """Return the vectors ``g`` and its T derivatives up to ``order - 1``.

    Parameters
    ----------
    name : str
        Function name.
    component_rate : sympy expression
        The rate ``g`` of one component.
    order : int
        Highest temperature derivative.

    Returns
    -------
    list of (str, sympy expression)
        Vector name and body, from the rate to its highest derivative.
    """
    vectors = []
    for deriv in range(order):
        body = (
            component_rate if deriv == 0 else sp.diff(component_rate, T, deriv)
        )
        vectors.append((f"{name}_g" + "T" * deriv, body))
    return vectors


def _pair_log_rates(params, order, pair_rates, scheme):
    """Logarithmic rates of the pair functions: ``M' = M*(c + g(a) + g(b))``.

    Parameters
    ----------
    params : dict
        Name -> abstracted function.
    order : int
        Highest temperature derivative.
    pair_rates : bool
        Accept cheap rates that depend on the pair.
    scheme : NameScheme
        Slots of the arrays.

    Returns
    -------
    rates : dict
        Name -> ``(g0, rate vectors)``.
    pair_dep_rates : dict
        Name -> rate that depends on the pair.
    extra_vectors : list of (str, sympy expression)
        The rate vectors to define, in creation order.
    """
    rates, pair_dep_rates, extra_vectors = {}, {}, []
    for p in params.values():
        if p.rank != 2 or not p.body.has(T) or order == 0:
            continue
        if not _log_rate_applies(p.body, pair_rates):
            continue
        rate, terms = _pair_rate_terms(p.body)
        if (
            pair_rates
            and not rate.has(sp.log)
            and not rate.has(sp.exp)
            and _pair_dependent_terms(terms)
        ):
            # a cheap rate that depends on the pair (c_ab b_ab / T**2 for
            # exp(-c_ab tau_ab)): M' = M*G_ab written in place
            pair_dep_rates[p.name] = rate
            continue
        split = _separable_rate(rate, terms, scheme)
        if split is None:
            continue
        vectors = _rate_vector_definitions(p.name, split[1], order)
        extra_vectors.extend(vectors)
        rates[p.name] = (
            split[0],
            [sp.IndexedBase(vname, shape=(nc,)) for vname, _ in vectors],
        )
    return rates, pair_dep_rates, extra_vectors


def _component_log_rates(params, order):
    """Logarithmic rates of the component functions with costly powers.

    ``A' = A*h``, ``A'' = A'*h + A*h'`` with ``h = d(log A)/dT``, so that a
    power such as ``(3/(2 + T/Tc))**k`` is evaluated once instead of once per
    derivative.

    Parameters
    ----------
    params : dict
        Name -> abstracted function.
    order : int
        Highest temperature derivative.

    Returns
    -------
    dict
        Name -> ``h``.
    """
    comp_rates = {}
    for p in params.values():
        if p.rank != 1 or not p.body.has(T) or order == 0:
            continue
        costly = [f for f in _product_factors(p.body) if f.has(T)]
        if not costly or not all(
            (isinstance(f, sp.Pow) and not f.exp.is_integer)
            or isinstance(f, sp.exp)
            for f in costly
        ):
            continue
        rate = sp.diff(sp.expand_log(sp.log(p.body), force=True), T)
        if rate.has(sp.log) and any(
            isinstance(f, sp.Pow) and f.exp.has(T) for f in costly
        ):
            continue
        comp_rates[p.name] = rate
    return comp_rates


def _rate_based_value(name, deriv, arrays, comp_rates, rates, scheme):
    """Return the ``deriv``-th derivative from a logarithmic rate, or None.

    Parameters
    ----------
    name : str
        Function name.
    deriv : int
        Derivative order (1 or more).
    arrays : list
        ``Indexed`` arrays of the derivatives of the function.
    comp_rates : dict
        Logarithmic rates of the component functions.
    rates : dict
        Logarithmic rates of the pair functions.
    scheme : NameScheme
        Pair slots.

    Returns
    -------
    sympy expression or None
        ``None`` when ``name`` has no rate.
    """
    if name in comp_rates:
        rate = comp_rates[name]
        if deriv == 1:
            return arrays[0] * rate
        return arrays[1] * rate + arrays[0] * sp.diff(rate, T)
    if name in rates:
        g0, vecs = rates[name]
        slot_a, slot_b = scheme.pair_slots
        rate = g0 + vecs[0][slot_a] + vecs[0][slot_b]
        if deriv == 1:
            return arrays[0] * rate
        rate_t = sp.diff(g0, T) + vecs[1][slot_a] + vecs[1][slot_b]
        return arrays[1] * rate + arrays[0] * rate_t
    return None


def _plain_derivative(param, deriv, arrays, opaque, current):
    """Differentiate once more and put the arrays in place of the functions.

    Parameters
    ----------
    param : object
        Abstracted function.
    deriv : int
        Derivative order (1 or more) being built.
    arrays : list
        ``Indexed`` arrays of the derivatives of the function.
    opaque : sympy expression
        Placeholder of the pair body.
    current : sympy expression
        The ``deriv - 1`` derivative of the body.

    Returns
    -------
    current : sympy expression
        The ``deriv`` derivative of the body, for the next order.
    value : sympy expression
        It, in terms of ``arrays``.
    """
    current = sp.diff(current, T)
    if deriv == 1 and param.rank == 2:
        current = current.subs(param.body, opaque)
    value = current
    for lower in range(deriv, 0, -1):
        value = value.subs(sp.Derivative(opaque, (T, lower)), arrays[lower])
    return current, value.subs(opaque, arrays[0])


def _derivative_value(param, deriv, state, context, comp_rates):
    """Return the ``deriv``-th derivative of a function and the new state.

    Parameters
    ----------
    param : object
        Abstracted function.
    deriv : int
        Derivative order (1 or more).
    state : tuple
        ``(arrays, opaque, current)``: the arrays of the derivatives, the
        placeholder of the pair body and the ``deriv - 1`` derivative of the
        body, which only plain differentiation advances.
    context : _MappingContext
        State of the mapping (rates, scheme).
    comp_rates : dict
        Logarithmic rates of the component functions.

    Returns
    -------
    current : sympy expression
        The ``current`` of the next order.
    value : sympy expression
        The derivative, in terms of the arrays.
    """
    arrays, opaque, current = state
    value = _rate_based_value(
        param.name, deriv, arrays, comp_rates, context.rates, context.scheme
    )
    if value is None:
        return _plain_derivative(param, deriv, arrays, opaque, current)
    return current, value


def _derivative_arrays(param, order, context):
    """Return the ``Indexed`` arrays of a function and its derivatives."""
    scheme = context.scheme
    slots = (scheme.component_slot,) if param.rank == 1 else scheme.pair_slots
    return [
        sp.Indexed(context.names[(param.name, deriv)], *slots)
        for deriv in range(order + 1)
    ]


def _parameter_definitions(param, order, context, comp_rates):
    """Definitions of one function and its temperature derivatives.

    Parameters
    ----------
    param : object
        Abstracted function.
    order : int
        Highest temperature derivative.
    context : _MappingContext
        State of the mapping (names, rates, scheme).
    comp_rates : dict
        Logarithmic rates of the component functions.

    Returns
    -------
    list of (name, body, rank, symmetric)
        One entry per derivative that is not written in place.
    """
    arrays = _derivative_arrays(param, order, context)
    opaque = sp.Function("_opaque_" + param.name)(T)
    in_place = context.inline_rates and (
        param.name in context.rates or param.name in context.pair_dep_rates
    )
    current = param.body
    entries = []
    for deriv in range(order + 1 if param.body.has(T) else 1):
        if deriv == 0:
            value = param.body
        else:
            current, value = _derivative_value(
                param, deriv, (arrays, opaque, current), context, comp_rates
            )
            if in_place:
                continue  # written in place by the mapping
        entries.append(
            (
                str(context.names[(param.name, deriv)]),
                _map_to_arrays(value, context),
                param.rank,
                param.symmetric,
            )
        )
    return entries


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
    params = {p.name: p for p in registry.values()}
    if pair_cse:
        _share_pair_bodies(params, scheme)
    rates, pair_dep_rates, extra_vectors = _pair_log_rates(
        params, order, pair_rates, scheme
    )
    comp_rates = _component_log_rates(params, order)
    context = _MappingContext(
        names=_array_names(params, order),
        params=params,
        rates=rates,
        pair_dep_rates=pair_dep_rates,
        inline_rates=inline_rates,
        scheme=scheme,
    )
    defs = []
    for rank in (1, 2):
        for p in params.values():
            if p.rank == rank:
                defs.extend(
                    _parameter_definitions(p, order, context, comp_rates)
                )
        if rank == 1:
            for vname, body in extra_vectors:
                defs.append((vname, _map_to_arrays(body, context), 1, False))
    return defs, partial(_map_to_arrays, context=context)


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


def _collect_inner_sums(sub, found):
    """Append to ``found`` the index-free summations nested in ``sub``.

    The search does not enter a summation that is index-free: it is taken
    whole.
    """
    if isinstance(sub, sp.Sum):
        if not (sub.atoms(sp.Idx) - _bound_all(sub)):
            found.append(sub)
            return
        _collect_inner_sums(sub.function, found)
        return
    for arg in getattr(sub, "args", ()):
        _collect_inner_sums(arg, found)


def _inner_sums(body):
    """Return the index-free summations nested in ``body``."""
    found = []
    _collect_inner_sums(body, found)
    return found


def _extract_inner_sums(total, by_key, result, queue):
    """Give every inner sum of ``total`` a node symbol.

    New nodes are appended to ``result`` and ``queue`` (named after the
    number of nodes so far) and registered in ``by_key``, all in place.

    Returns
    -------
    replacements : dict
        Inner sum -> node symbol.
    kids : set
        The node symbols of the inner sums.
    """
    replacements, kids = {}, set()
    for inner in _inner_sums(total.function):
        key = _alpha_key(inner)
        if key not in by_key:
            new_symbol = sp.Dummy(f"S{len(result)}")
            by_key[key] = new_symbol
            result.append((new_symbol, inner))
            queue.append((new_symbol, inner))
        replacements[inner] = by_key[key]
        kids.add(by_key[key])
    return replacements, kids


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
    while queue:
        symbol, total = queue.pop(0)
        replacements, kids = _extract_inner_sums(total, by_key, result, queue)
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


#: derivative attribute -> variable, for the first derivatives of a node
_FIRST_VARIABLE = MappingProxyType({"dni": "n", "dt": "T", "dv": "V"})
#: derivative attribute -> pair of variables, for the second derivatives
_SECOND_VARIABLES = MappingProxyType(
    {
        "dnidnj": ("n", "n"),
        "dt2": ("T", "T"),
        "dv2": ("V", "V"),
        "dtdv": ("T", "V"),
        "dtdni": ("n", "T"),
        "dvdni": ("n", "V"),
    }
)
#: variable -> attribute of its first derivative
_FIRST_ATTRIBUTE = MappingProxyType({"n": "dni", "T": "dt", "V": "dv"})
#: pair of variables (in any order) -> attribute of the second derivative
_SECOND_ATTRIBUTE = MappingProxyType(
    {
        ("n", "n"): "dnidnj",
        ("T", "T"): "dt2",
        ("V", "V"): "dv2",
        ("T", "V"): "dtdv",
        ("V", "T"): "dtdv",
        ("n", "T"): "dtdni",
        ("T", "n"): "dtdni",
        ("n", "V"): "dvdni",
        ("V", "n"): "dvdni",
    }
)


@dataclass(frozen=True)
class _NestedContext:
    """Inputs shared by the recursion of :func:`_nested_node_derivatives`.

    Attributes
    ----------
    bodies : dict
        Node symbol -> body, in terms of the inner node symbols.
    inner_map : dict
        Node symbol -> set of the inner node symbols of its body.
    needed : list of str
        Node derivative attributes to compute.
    indexes : list
        Index order of the :class:`~thermodiff.diffplz.DiffPlz`.
    """

    bodies: dict
    inner_map: dict
    needed: list
    indexes: list


def _plain_node_entry(body, needed, indexes):
    """Return ``{"value": body, attribute: derivative}`` of one body.

    The inner node symbols of ``body`` are held fixed (partial derivatives).
    """
    diff = (
        DiffPlz(
            body,
            internal_functions=[],
            indexes=indexes,
            name="s",
            derivatives=needed,
        )
        if needed
        else None
    )
    entry = {"value": body}
    for attr in needed:
        entry[attr] = _node_derivative(diff, attr)
    return entry


def _first_of(entry, variable, index):
    """Return the first derivative of ``entry`` in ``variable``.

    For the mole numbers the derivative is renamed to ``index``.
    """
    value = entry.get(_FIRST_ATTRIBUTE[variable], sp.S.Zero)
    if variable == "n" and index != i:
        value = value.xreplace({i: index})
    return value


def _second_of(entry, variable_x, variable_y):
    """Return the second derivative of ``entry`` in two variables."""
    key = _SECOND_ATTRIBUTE[(variable_x, variable_y)]
    return entry.get(key, sp.S.Zero)


def _body_gradient_partials(gradient, kids, firsts, indexes):
    """Return ``{kid: {attribute: P_attribute}}`` of ``B_k = dS/dZ_k``."""
    partials = {}
    for kid in kids:
        diff = (
            DiffPlz(
                gradient[kid],
                internal_functions=[],
                indexes=indexes,
                name="s",
                derivatives=firsts,
            )
            if firsts
            else None
        )
        partials[kid] = {attr: _node_derivative(diff, attr) for attr in firsts}
    return partials


def _total_first_derivative(attr, partial, gradient, kid_d, kids):
    """``D_x S = P_x + sum_k B_k D_x Z_k``."""
    variable = _FIRST_VARIABLE[attr]
    value = partial[attr]
    for kid in kids:
        value += gradient[kid] * _first_of(kid_d[kid], variable, i)
    return value


def _total_second_derivative(attr, partial, gradient, partials, kid_d, kids):
    """``D_xy S`` of the module docstring, for the attribute ``attr``."""
    variable_x, variable_y = _SECOND_VARIABLES[attr]
    index_x = i
    index_y = j if (variable_x == "n" and variable_y == "n") else i
    value = partial[attr]
    for kid in kids:
        pbx = _first_of(partials[kid], variable_x, index_x)
        pby = _first_of(partials[kid], variable_y, index_y)
        value += pbx * _first_of(kid_d[kid], variable_y, index_y)
        value += pby * _first_of(kid_d[kid], variable_x, index_x)
        for other in kids:
            value += (
                sp.diff(gradient[kid], other)
                * _first_of(kid_d[kid], variable_x, index_x)
                * _first_of(kid_d[other], variable_y, index_y)
            )
        value += gradient[kid] * _second_of(kid_d[kid], variable_x, variable_y)
    return value


def _compute_nested_entry(symbol, context, done):
    """Return the total derivatives of one node, filling ``done``.

    The inner nodes are computed first (recursively); a node without inner
    nodes keeps its plain partial derivatives.
    """
    if symbol in done:
        return done[symbol]
    needed, indexes = context.needed, context.indexes
    kids = sorted(context.inner_map.get(symbol, ()), key=str)
    if not kids:
        done[symbol] = _plain_node_entry(
            context.bodies[symbol], needed, indexes
        )
        return done[symbol]
    kid_d = {kid: _compute_nested_entry(kid, context, done) for kid in kids}
    body = context.bodies[symbol]
    partial = _plain_node_entry(body, needed, indexes)
    firsts = sorted({attr for attr in needed if attr in _FIRST_VARIABLE})
    gradient = {kid: sp.diff(body, kid) for kid in kids}
    partials = _body_gradient_partials(gradient, kids, firsts, indexes)
    entry = {"value": body}
    for attr in needed:
        if attr in _FIRST_VARIABLE:
            entry[attr] = _total_first_derivative(
                attr, partial, gradient, kid_d, kids
            )
        else:
            entry[attr] = _total_second_derivative(
                attr, partial, gradient, partials, kid_d, kids
            )
    done[symbol] = entry
    return entry


def _nested_node_derivatives(nodes, inner_map, needed, indexes):
    """Total derivatives of nodes whose bodies contain inner node symbols.

    With P the partial derivatives of the body (inner nodes held fixed),
    B_k = dS/dZ_k and D the total derivatives of the inner nodes:
      D_x S  = P_x + sum_k B_k D_x Z_k
      D_xy S = P_xy + sum_k (P_{x,k} D_y Z_k + P_{y,k} D_x Z_k)
               + sum_kl B_kl D_x Z_k D_y Z_l + sum_k B_k D_xy Z_k
    Nodes without inner nodes are left to the caller (plain DiffPlz).
    """
    context = _NestedContext(dict(nodes), inner_map, needed, indexes)
    done = {}
    for symbol, _ in nodes:
        if inner_map.get(symbol):
            _compute_nested_entry(symbol, context, done)
    return {
        symbol: entry
        for symbol, entry in done.items()
        if inner_map.get(symbol)
    } | {symbol: done[symbol] for symbol in done}


#: functions without a usable symbolic derivative: the outer function
#: falls back to the plain ``sp.diff`` path when it contains one of
#: them
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


def _scalar_dag(scalar_expr, mode):
    """Nodes ``[(u, phi)]`` of the scalar function in dependency order.

    The function is ``scalar_expr``.  ``"cse"``: the common sub-expressions
    of ``scalar_expr`` (``sympy.cse``) and ``scalar_expr`` itself, so each
    node is a coarse block (``dm1 = S0/S1``, the rest of F).
    ``"fine"``: in addition every compound sub-expression is a node of its own
    (one operation per node, n-ary sums and products kept whole).
    """
    rep, red = sp.cse(
        scalar_expr, symbols=sp.numbered_symbols("_scu"), optimizations="basic"
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


@dataclass(frozen=True)
class _PropagationState:
    """Forward-propagation state of :func:`_propagate` (mutated in place).

    Attributes
    ----------
    position : dict
        Leaf -> its position in the leaf list.
    pairs_set : set
        The requested Hessian entries, as ``(leaf, leaf)`` in leaf order.
    grad : dict
        Node or leaf -> ``{leaf: derivative}``; filled node by node.
    hess : dict
        Node or leaf -> ``{(leaf, leaf): derivative}``; filled node by node.
    value : dict
        Node symbol -> its expression in terms of the leaves.
    """

    position: dict
    pairs_set: set
    grad: dict
    hess: dict
    value: dict


def _is_explicit(derivative) -> bool:
    """Tell whether a derivative has no unevaluated ``Derivative``/``Subs``."""
    return not (derivative.has(sp.Derivative) or derivative.has(sp.Subs))


def _add_first_partial(phi, parent, state, grad_u, hess_u):
    """Add ``phi_p`` times the gradient and Hessian of ``parent``.

    Returns ``False`` when the local partial is not explicit.
    """
    first = sp.diff(phi, parent)
    if not _is_explicit(first):
        return False
    first = first.xreplace(state.value)
    if first != 0:
        for leaf, term in state.grad[parent].items():
            grad_u[leaf] = grad_u.get(leaf, 0) + first * term
        for key, term in state.hess[parent].items():
            hess_u[key] = hess_u.get(key, 0) + first * term
    return True


def _add_second_partial(phi, parent_p, parent_q, state, hess_u):
    """Add ``phi_pq p_l q_m`` to the Hessian entries ``hess_u``.

    Returns ``False`` when the local partial is not explicit.
    """
    second = sp.diff(phi, parent_p, parent_q)
    if second == 0:
        return True
    if not _is_explicit(second):
        return False
    second = second.xreplace(state.value)
    position = state.position
    for leaf_l, term_p in state.grad[parent_p].items():
        for leaf_m, term_q in state.grad[parent_q].items():
            if position[leaf_l] <= position[leaf_m]:
                key = (leaf_l, leaf_m)
            else:
                key = (leaf_m, leaf_l)
            if key not in state.pairs_set:
                continue
            if parent_p == parent_q:
                if position[leaf_l] > position[leaf_m]:
                    continue
                term = second * term_p * term_q
            elif leaf_l != leaf_m:
                term = second * term_p * term_q
            else:
                term = 2 * second * term_p * term_q
            hess_u[key] = hess_u.get(key, 0) + term
    return True


def _propagate_node(phi, state):
    """Return ``(gradient, Hessian)`` of a node, or ``None`` if not explicit.

    The result holds the structural non-zeros only.
    """
    parents = sorted(
        (p for p in phi.free_symbols if state.grad.get(p)),
        key=sp.default_sort_key,
    )
    grad_u, hess_u = {}, {}
    for position, parent in enumerate(parents):
        if not _add_first_partial(phi, parent, state, grad_u, hess_u):
            return None
        for other in parents[position:]:
            if not _add_second_partial(phi, parent, other, state, hess_u):
                return None
    return (
        {leaf: term for leaf, term in grad_u.items() if term != 0},
        {key: term for key, term in hess_u.items() if term != 0},
    )


def _propagate(scalar_expr, leaves, first, pairs, mode):
    """Forward second-order propagation of ``scalar_expr`` over a DAG.

    The DAG is the one of ``mode``.

    Every node ``u = phi(p, ...)`` gets its gradient and the needed Hessian
    entries with respect to ``leaves`` from the local partials ``phi_p``,
    ``phi_pq`` (``sympy.diff`` of the small node body):
    ``u_l = sum_p phi_p p_l`` and ``u_lm = sum_p phi_p p_lm + sum_pq phi_pq
    p_l q_m``; structural zeros are never formed.  Returns the flat
    expressions of ``[F] + [F_l for l in first] + [F_lm for (l, m) in
    pairs]``, or ``None`` when a local partial is not explicit.
    """
    active = set(first) | {leaf for pair in pairs for leaf in pair}
    state = _PropagationState(
        position={leaf: pos for pos, leaf in enumerate(leaves)},
        pairs_set=set(pairs),
        grad={leaf: {leaf: sp.S.One} for leaf in leaves if leaf in active},
        hess={leaf: {} for leaf in leaves if leaf in active},
        value={},
    )
    for symbol, phi in _scalar_dag(scalar_expr, mode):
        state.value[symbol] = phi.xreplace(state.value)
        propagated = _propagate_node(phi, state)
        if propagated is None:
            return None
        state.grad[symbol], state.hess[symbol] = propagated
    top = symbol
    return (
        [state.value[top]]
        + [state.grad[top].get(leaf, sp.S.Zero) for leaf in first]
        + [state.hess[top].get(pair, sp.S.Zero) for pair in pairs]
    )


def _scalar_chain(scalar_expr, leaves, first, pairs):
    """Value, gradient and Hessian entries of ``scalar_expr`` as named scalars.

    The derivatives are propagated forward over a DAG of ``scalar_expr``
    (:func:`_propagate`) instead of differentiating the expanded
    ``scalar_expr``, and the result goes through one common sub-expression
    elimination.  Both
    DAG granularities are tried and the one with fewer operations is kept.

    Returns ``(definitions, value, d1, d2)``: ``definitions`` is a list of
    ``(symbol, expr)`` in dependency order (placeholder symbols ``_sc*``),
    ``value`` the expression of ``scalar_expr``, ``d1[l]`` and ``d2[(l, m)]``
    the expressions of the requested partial derivatives (a placeholder
    symbol, a number or a leaf), or ``None`` when ``scalar_expr`` is not
    smooth enough.
    """
    if any(scalar_expr.has(func) for func in _NONSMOOTH):
        return None
    best = None
    for mode in ("cse", "fine"):
        flat = _propagate(scalar_expr, leaves, first, pairs, mode)
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


#: first-order node derivative attribute of a variable
_FIRST_ATTR = {"T": "dt", "V": "dv"}
#: mixed (variable, n) node derivative attribute of a variable
_MIXED_ATTR = {"T": "dtdni", "V": "dvdni"}
#: second-order node derivative attribute of two T/V variables
_SECOND_ATTR = {("T", "T"): "dt2", ("V", "V"): "dv2", ("T", "V"): "dtdv"}


def _needed_for_attribute(variables):
    """Return the node derivatives that one attribute needs.

    Parameters
    ----------
    variables : tuple of str
        The variables of the attribute (``_ATTR_VARS`` value).

    Returns
    -------
    set of str
        Attributes of the node derivatives.
    """
    if variables == ("n",):
        return {"dni"}
    if variables == ("n", "n"):
        return {"dni", "dnidnj"}
    if "n" in variables:
        first = variables[0]
        return {"dni", _MIXED_ATTR[first], _FIRST_ATTR[first]}
    needed = {_FIRST_ATTR[var] for var in variables}
    if len(variables) == 2:
        needed.add(_SECOND_ATTR[variables])
    return needed


def _needed_node_derivatives(attributes):
    """Return the sorted node derivatives that ``attributes`` involve.

    Any attribute that involves ``n`` needs the node gradient or Hessian;
    the T and V attributes need the T and V node derivatives.

    Parameters
    ----------
    attributes : list of str
        Derivative attributes of the outer function.

    Returns
    -------
    list of str
        Sorted node derivative attributes.
    """
    needed = set()
    for attr in attributes:
        needed |= _needed_for_attribute(_ATTR_VARS[attr])
    return sorted(needed)


def _outer_and_nodes(expression):
    """Split ``expression`` into the outer function and its summation nodes.

    Parameters
    ----------
    expression : sympy expression
        The fully expanded function.

    Returns
    -------
    tuple or None
        ``(outer, nodes)``; ``None`` when there are no nodes or the outer
        function still depends on the mole numbers.
    """
    outer, nodes = summation_nodes(factor_sums(expression))
    if not nodes:
        return None
    if any(base == n for base in outer.atoms(sp.IndexedBase)) or outer.has(
        sp.Sum
    ):
        return None
    return outer, nodes


def _abstract_nodes(
    nodes, registry, symmetric_params, split_sqrt, table_layout, scheme
):
    """Abstract the component functions of every node, filling ``registry``.

    Parameters
    ----------
    nodes : list of (symbol, sympy expression)
        Summation nodes.
    registry : dict
        Filled with the abstracted functions.
    symmetric_params : tuple
        Symmetric parameters.
    split_sqrt, table_layout : bool
        Options of :func:`abstract_component_functions`.
    scheme : NameScheme
        Names of the abstracted arrays.

    Returns
    -------
    list of (symbol, sympy expression)
        The nodes in terms of the abstracted functions.
    """
    return [
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


def _node_derivative_table(nodes, indexes, needed, nested_nodes):
    """Compute the derivatives of every node.

    Parameters
    ----------
    nodes : list of (symbol, sympy expression)
        Summation nodes.
    indexes : list
        Index order of the :class:`~thermodiff.diffplz.DiffPlz`.
    needed : list of str
        Node derivative attributes.
    nested_nodes : bool
        Nest the inner sums of the nodes as nodes of their own.

    Returns
    -------
    nodes : list of (symbol, sympy expression)
        The nodes, with the inner nodes when ``nested_nodes``.
    node_d : dict
        Symbol -> ``{"value": ..., attribute: derivative}``.
    """
    node_d = {}
    if nested_nodes:
        nodes, inner_map = _nest_inner_nodes(nodes)
        node_d = _nested_node_derivatives(nodes, inner_map, needed, indexes)
    for symbol, total in nodes:
        if symbol in node_d:
            continue
        diff = (
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
            entry[attr] = _node_derivative(diff, attr)
        node_d[symbol] = entry
    return nodes, node_d


def _node_partial(node_d, symbol, key):
    """Return one derivative of a node (zero when it was not computed)."""
    return node_d[symbol].get(key, sp.S.Zero)


def _node_first(node_d, symbol, var, index=None):
    """Return the derivative of a node with respect to ``T``, ``V`` or ``n``.

    Parameters
    ----------
    node_d : dict
        Derivatives of the nodes.
    symbol : sympy.Symbol
        The node.
    var : str
        ``"T"``, ``"V"`` or ``"n"``.
    index : sympy.Idx, optional
        Index that replaces ``i`` in the derivative with respect to ``n``.

    Returns
    -------
    sympy expression
        The derivative.
    """
    if var == "n":
        value = _node_partial(node_d, symbol, "dni")
        if index is None or index == i:
            return value
        return value.xreplace({i: index})
    return _node_partial(node_d, symbol, _FIRST_ATTR[var])


def _node_second(node_d, symbol, var_x, var_y):
    """Return the second derivative of a node with respect to two variables."""
    key = tuple(sorted((var_x, var_y)))
    if key == ("n", "n"):
        return _node_partial(node_d, symbol, "dnidnj")
    if "n" in key:
        other = key[0] if key[1] == "n" else key[1]
        return _node_partial(node_d, symbol, _MIXED_ATTR[other])
    return _node_partial(node_d, symbol, _SECOND_ATTR[key])


@dataclass(frozen=True)
class _OuterPartials:
    """Partial derivatives of the outer function ``F``.

    Attributes
    ----------
    outer : sympy expression
        ``F``.
    node_first : dict
        Node symbol -> ``dF/dS``.
    leaves : tuple
        Node symbols, ``V`` and ``T``, in propagation order (scalar chain).
    scalar_first : dict or None
        Leaf -> propagated first partial; ``None`` for plain ``sp.diff``.
    scalar_second : dict or None
        Leaf pair -> propagated second partial; ``None`` for ``sp.diff``.
    """

    outer: sp.Expr
    node_first: dict
    leaves: tuple = ()
    scalar_first: dict | None = None
    scalar_second: dict | None = None

    def wrt_variable(self, var):
        """Return ``dF/dT`` or ``dF/dV`` (zero for ``n``)."""
        if var == "n":
            return sp.S.Zero
        if self.scalar_first is None:
            return sp.diff(self.outer, _SYM[var])
        return self.scalar_first.get(_SYM[var], sp.S.Zero)

    def second(self, first, second):
        """Return the second partial with respect to two leaves."""
        if self.scalar_second is None:
            if first in self.node_first:
                return sp.diff(self.node_first[first], second)
            return sp.diff(self.outer, first, second)
        if self.leaves.index(first) > self.leaves.index(second):
            first, second = second, first
        return self.scalar_second.get((first, second), sp.S.Zero)


def _plain_partials(outer, node_symbols):
    """Return the partials of ``outer`` by ``sp.diff``."""
    return _OuterPartials(
        outer, {symbol: sp.diff(outer, symbol) for symbol in node_symbols}
    )


def _scalar_partials(outer, node_symbols, leaves, propagated):
    """Return the partials of ``outer`` read from the scalar chain.

    Parameters
    ----------
    outer : sympy expression
        ``F``.
    node_symbols : list
        Node symbols.
    leaves : list
        Node symbols, ``V`` and ``T``.
    propagated : tuple
        Result of :func:`_scalar_chain`.

    Returns
    -------
    _OuterPartials
        Partials backed by the propagated first and second derivatives.
    """
    _, _, first, second = propagated
    return _OuterPartials(
        outer,
        {symbol: first.get(symbol, sp.S.Zero) for symbol in node_symbols},
        tuple(leaves),
        first,
        second,
    )


def _total_first(var, partials, node_d, node_symbols):
    """Return ``d/dvar`` of the function (``n`` with index ``i``)."""
    expr = partials.wrt_variable(var)
    for symbol in node_symbols:
        expr += partials.node_first[symbol] * _node_first(
            node_d, symbol, var, i
        )
    return expr


def _total_second(var_x, var_y, partials, node_d, node_symbols):
    """Return ``d2/dvar_x dvar_y`` (indexes ``i`` and ``j`` for two ``n``)."""
    index_x = i
    index_y = j if (var_x == "n" and var_y == "n") else i
    expr = sp.S.Zero
    if var_x != "n" and var_y != "n":
        expr += partials.second(_SYM[var_x], _SYM[var_y])
    for symbol in node_symbols:
        grad_x = _node_first(node_d, symbol, var_x, index_x)
        grad_y = _node_first(node_d, symbol, var_y, index_y)
        if var_y != "n":
            expr += partials.second(symbol, _SYM[var_y]) * grad_x
        if var_x != "n":
            expr += partials.second(symbol, _SYM[var_x]) * grad_y
        for other in node_symbols:
            other_y = _node_first(node_d, other, var_y, index_y)
            expr += partials.second(symbol, other) * grad_x * other_y
        expr += partials.node_first[symbol] * _node_second(
            node_d, symbol, var_x, var_y
        )
    return expr


def _want(items, item):
    """Append ``item`` to ``items`` unless it is already there."""
    if item not in items:
        items.append(item)


def _want_pair(pairs, leaves, first, second):
    """Add the pair in leaf order to ``pairs``."""
    if leaves.index(first) <= leaves.index(second):
        _want(pairs, (first, second))
    else:
        _want(pairs, (second, first))


def _plan_first_order(var, node_symbols, node_d, first):
    """Add the first partials that a first-order attribute reads."""
    if var != "n":
        _want(first, _SYM[var])
    for symbol in node_symbols:
        if _node_first(node_d, symbol, var, i) != 0:
            _want(first, symbol)


def _plan_second_node(symbol, variables, node_symbols, node_d, plan):
    """Add the partials that a second-order attribute reads for one node.

    Parameters
    ----------
    symbol : sympy.Symbol
        The node.
    variables : tuple of str
        The two variables, ``n`` first when present.
    node_symbols : list
        All the node symbols.
    node_d : dict
        Derivatives of the nodes.
    plan : tuple
        ``(leaves, first, pairs)``; the last two are extended in place.
    """
    var_x, var_y = variables
    leaves, first, pairs = plan
    if _node_second(node_d, symbol, var_x, var_y) != 0:
        _want(first, symbol)
    for var_a, var_b in ((var_x, var_y), (var_y, var_x)):
        if var_b != "n" and _node_first(node_d, symbol, var_a, i) != 0:
            _want_pair(pairs, leaves, symbol, _SYM[var_b])
    for other in node_symbols:
        if (
            _node_first(node_d, symbol, var_x, i) != 0
            and _node_first(node_d, other, var_y, j) != 0
        ):
            _want_pair(pairs, leaves, symbol, other)


def _scalar_plan(attributes, node_symbols, node_d, leaves):
    """Choose the partials of ``F`` that are multiplied by a node derivative.

    Parameters
    ----------
    attributes : list of str
        Derivative attributes.
    node_symbols : list
        Node symbols.
    node_d : dict
        Derivatives of the nodes.
    leaves : list
        Node symbols, ``V`` and ``T``.

    Returns
    -------
    first : list
        Leaves with a first partial, in leaf order.
    pairs : list of tuple
        Leaf pairs with a second partial, in leaf order.
    """
    first, pairs = [], []
    for attr in attributes:
        variables = _ATTR_VARS[attr]
        if len(variables) == 1:
            _plan_first_order(variables[0], node_symbols, node_d, first)
            continue
        var_x, var_y = variables
        if var_y == "n" and var_x != "n":
            var_x, var_y = "n", var_x
        if var_x != "n" and var_y != "n":
            _want_pair(pairs, leaves, _SYM[var_x], _SYM[var_y])
        for symbol in node_symbols:
            _plan_second_node(
                symbol,
                (var_x, var_y),
                node_symbols,
                node_d,
                (leaves, first, pairs),
            )
    first = [leaf for leaf in leaves if leaf in first]
    pairs = sorted(
        pairs, key=lambda ab: (leaves.index(ab[0]), leaves.index(ab[1]))
    )
    return first, pairs


def _scalar_estimate(outer, first, pairs, propagated):
    """Estimate the operations of the plain and the propagated partials.

    The plain partials of ``outer`` after one CSE, against the propagated
    definitions (``scalar_chain="estimate"``).

    Parameters
    ----------
    outer : sympy expression
        ``F``.
    first, pairs : list
        The partials that were propagated.
    propagated : tuple
        Result of :func:`_scalar_chain`.

    Returns
    -------
    tuple of int
        ``(plain operations, propagated operations)``.
    """
    plain_exprs = (
        [outer]
        + [sp.diff(outer, a) for a in first]
        + [sp.diff(outer, a, b) for a, b in pairs]
    )
    replacements, reduced = sp.cse(
        plain_exprs,
        symbols=sp.numbered_symbols("_sce"),
        optimizations="basic",
    )
    est_plain = sum(sp.count_ops(e) for _, e in replacements) + sum(
        sp.count_ops(e) for e in reduced
    )
    est_scalar = sum(sp.count_ops(e) for _, e in propagated[0])
    return est_plain, est_scalar


def _scalar_chain_block(outer, attributes, node_symbols, node_d, leaves):
    """Propagate the partials of ``outer`` and estimate their cost.

    Parameters
    ----------
    outer : sympy expression
        ``F``.
    attributes : list of str
        Derivative attributes.
    node_symbols : list
        Node symbols.
    node_d : dict
        Derivatives of the nodes.
    leaves : list
        Node symbols, ``V`` and ``T``.

    Returns
    -------
    propagated : tuple or None
        Result of :func:`_scalar_chain`; ``None`` when there is nothing to
        propagate (only the value) or ``F`` is not smooth.
    estimate : tuple or None
        Result of :func:`_scalar_estimate`.
    """
    first, pairs = _scalar_plan(attributes, node_symbols, node_d, leaves)
    if not (first or pairs):
        return None, None
    propagated = _scalar_chain(outer, leaves, first, pairs)
    if propagated is None:
        return None, None
    return propagated, _scalar_estimate(outer, first, pairs, propagated)


@dataclass(frozen=True)
class _Finisher:
    """Last rewrites of a derivative before it enters the structure.

    Attributes
    ----------
    back : dict
        Node symbol -> body, to put the nodes back.
    mapping : callable
        Replaces the abstract functions by arrays.
    split_sums : bool
        Split the sums of terms into sums of sums.
    scheme : NameScheme
        Names of the abstracted arrays.
    """

    back: dict
    mapping: Callable
    split_sums: bool
    scheme: NameScheme

    def resolve(self, expr):
        """Put the inner node bodies back (they may be nested)."""
        for _ in range(len(self.back) + 1):
            new_expr = expr.xreplace(self.back)
            if new_expr == expr:
                break
            expr = new_expr
        return expr

    def split(self, expr):
        """Split the sums of terms when ``split_sums`` is on."""
        if self.split_sums:
            return split_sum_terms(expr, self.scheme)
        return expr

    def finish(self, expr):
        """Resolve, map to arrays and split ``expr``."""
        return self.split(self.mapping(self.resolve(expr)))


def _scalar_outputs(propagated, estimate, node_symbols, finisher):
    """Return the scalar-chain entries of the structure.

    The scalars keep the node symbols; the nodes they read go apart.

    Parameters
    ----------
    propagated : tuple
        Result of :func:`_scalar_chain`.
    estimate : tuple
        Result of :func:`_scalar_estimate`.
    node_symbols : list
        Node symbols.
    finisher : _Finisher
        Rewrites of the bodies.

    Returns
    -------
    dict
        ``_scalar_defs``, ``_scalar_nodes`` and ``_scalar_estimate``.
    """
    scalar_defs, scalar_value = propagated[0], propagated[1]
    mapped = [(symbol, finisher.mapping(body)) for symbol, body in scalar_defs]
    read = set().union(*[body.free_symbols for _, body in scalar_defs]) | (
        scalar_value.free_symbols
        if isinstance(scalar_value, sp.Expr)
        else set()
    )
    nodes_read = [
        (symbol, finisher.finish(symbol))
        for symbol in node_symbols
        if symbol in read
    ]
    return {
        "_scalar_defs": mapped,
        "_scalar_nodes": nodes_read,
        "_scalar_estimate": estimate,
    }


def _assemble_attributes(
    target, attributes, partials, node_d, node_symbols, finisher
):
    """Fill ``target`` with the chain-rule derivative of every attribute.

    Parameters
    ----------
    target : dict
        Receives ``attribute -> expression``.
    attributes : list of str
        Derivative attributes.
    partials : _OuterPartials
        Partials of the outer function to read.
    node_d : dict
        Derivatives of the nodes.
    node_symbols : list
        Node symbols.
    finisher : _Finisher
        Rewrites of the derivatives.
    """
    for attr in attributes:
        variables = _ATTR_VARS[attr]
        if len(variables) == 1:
            value = _total_first(variables[0], partials, node_d, node_symbols)
        else:
            var_x, var_y = variables
            if var_y == "n" and var_x != "n":
                var_x, var_y = "n", var_x
            value = _total_second(var_x, var_y, partials, node_d, node_symbols)
        target[attr] = finisher.finish(value)


def _temperature_order(attributes):
    """Return the highest temperature derivative of ``attributes``."""
    return max(
        (sum(1 for var in _ATTR_VARS[a] if var == "T") for a in attributes),
        default=0,
    )


def _head_structure(outer, defs, scheme, propagated, finisher):
    """Return the ``value``, component definitions and scheme entries.

    Parameters
    ----------
    outer : sympy expression
        ``F``.
    defs : list
        Result of :func:`component_definitions`.
    scheme : NameScheme
        Names of the abstracted arrays.
    propagated : tuple or None
        Result of :func:`_scalar_chain`, if it is used.
    finisher : _Finisher
        Rewrites of the value.

    Returns
    -------
    dict
        The first three entries of the structure.
    """
    if propagated is None:
        value = finisher.finish(outer)
    else:
        value = finisher.split(propagated[1])
    return {
        "value": value,
        "_component_defs": defs,
        NAME_SCHEME_KEY: scheme,
    }


def _plain_structure(outer, attributes, plain, node_d, node_symbols, finisher):
    """Return the plain outputs that accompany the scalar chain.

    ``to_fortran`` runs its summation passes on them too, so the loops come
    out as without ``scalar_chain``.

    Parameters
    ----------
    outer : sympy expression
        ``F``.
    attributes : list of str
        Derivative attributes.
    plain : _OuterPartials
        Partials of ``F`` by ``sp.diff``.
    node_d : dict
        Derivatives of the nodes.
    node_symbols : list
        Node symbols.
    finisher : _Finisher
        Rewrites of the derivatives.

    Returns
    -------
    dict
        ``value`` and the attributes.
    """
    plain_out = {"value": finisher.finish(outer)}
    _assemble_attributes(
        plain_out, attributes, plain, node_d, node_symbols, finisher
    )
    return plain_out


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
        ``"_scalar_nodes"`` gives the summation of every node they read.
        Falls back to ``sp.diff`` when ``F`` contains a non-smooth
        function.

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
    split = _outer_and_nodes(expression)
    if split is None:
        return None
    outer, nodes = split
    needed = _needed_node_derivatives(attributes)
    registry = {}
    nodes = _abstract_nodes(
        nodes, registry, symmetric_params, split_sqrt, table_layout, scheme
    )
    node_symbols = [symbol for symbol, _ in nodes]
    nodes, node_d = _node_derivative_table(
        nodes, indexes, needed, nested_nodes
    )
    plain = _plain_partials(outer, node_symbols)
    leaves = list(node_symbols) + [_SYM[var] for var in ("V", "T")]
    propagated, estimate = None, None
    if scalar_chain:
        propagated, estimate = _scalar_chain_block(
            outer, attributes, node_symbols, node_d, leaves
        )
    partials = plain
    if propagated is not None:
        partials = _scalar_partials(outer, node_symbols, leaves, propagated)
    defs, mapping = component_definitions(
        registry,
        _temperature_order(attributes),
        inline_rates=split_sums,
        pair_cse=pair_cse,
        pair_rates=pair_rates,
        scheme=scheme,
    )
    finisher = _Finisher(dict(nodes), mapping, split_sums, scheme)
    out = _head_structure(outer, defs, scheme, propagated, finisher)
    if propagated is not None:
        out.update(
            _scalar_outputs(propagated, estimate, node_symbols, finisher)
        )
    _assemble_attributes(
        out, attributes, partials, node_d, node_symbols, finisher
    )
    if propagated is not None:
        out["_plain"] = _plain_structure(
            outer, attributes, plain, node_d, node_symbols, finisher
        )
    return out
