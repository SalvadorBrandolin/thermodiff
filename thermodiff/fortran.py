"""Fortran code generation for thermodynamic models.

This module turns a :class:`~thermodiff.diffplz.DiffPlz` instance into a
free-form Fortran subroutine that evaluates the original function together
with all its available derivatives.

The generated subroutine takes the independent variables (mole numbers
``n``, volume ``V``, pressure ``P``, temperature ``T``), the model
parameters, and returns the function value and the derivatives as ``intent
(out)`` arguments, for example::

    subroutine f(n, V, T, f, dfdT, dfdV, dfdn)
      implicit none
      integer, intent(in) :: nc
      real(8), intent(in) :: n(:)
      real(8), intent(in) :: V, T
      real(8), intent(out) :: f, dfdT, dfdV
      real(8), intent(out) :: dfdn(:)
      ...
    end subroutine

Summations produced with :func:`~thermodiff.core.easy_sums.sum_components`
are expanded into explicit ``do`` loops and the Kronecker deltas handled by
``DiffPlz`` become ``if``/``else`` blocks.  Internal functions created with
:func:`~thermodiff.core.idxfunction.idx_function` (and their derivatives)
are exposed as extra input arguments by default; pass ``strict=True`` to
raise an error instead.
"""

import re

import sympy as sp
from sympy.core.function import DefinedFunction
from sympy.printing.fortran import fcode

from thermodiff.derivatives import DERIVATIVE_ATTRIBUTES, resolve_derivatives
from thermodiff.thermovars import P, T, V, i, j, n, nc

__all__ = ["to_fortran"]


#: Mapping of ``DiffPlz`` derivative attributes to the suffix used in the
#: generated Fortran variable names.
_DERIV_SUFFIX = {
    "dt": "dT",
    "dv": "dV",
    "dp": "dP",
    "dni": "dn",
    "dt2": "dT2",
    "dv2": "dV2",
    "dp2": "dP2",
    "dnidnj": "dndn",
    "dtdv": "dTdV",
    "dtdp": "dTdP",
    "dtdni": "dndT",
    "dvdni": "dndV",
    "dvdp": "dVdP",
    "dpdni": "dndP",
}

#: Order in which the derivatives are emitted.
_DERIV_ORDER = list(DERIVATIVE_ATTRIBUTES)

#: Settings passed to :func:`sympy.printing.fortran.fcode`.
_FCODE_SETTINGS = {
    "source_format": "free",
    "standard": 95,
    "contract": False,
}

_INDENT = "   "


def _sanitize(name: str) -> str:
    r"""Turn an arbitrary symbol name into a valid Fortran identifier.

    Parameters
    ----------
    name : str
        Symbol name, potentially containing LaTeX pieces such as
        ``\\tau_{lk}`` or ``G^E``.

    Returns
    -------
    str
        A valid Fortran identifier.
    """
    clean = str(name).replace("\\", "")
    clean = clean.replace("{", "_").replace("}", "")
    clean = clean.replace("^", "_")
    clean = re.sub(r"[^0-9A-Za-z_]", "_", clean)
    clean = re.sub(r"_+", "_", clean).strip("_")
    if not clean:
        clean = "var"
    if clean[0].isdigit():
        clean = "v_" + clean
    return clean


#: Memo entry of :func:`_free_indices`: the ``Idx`` atoms of a subtree and
#: the summation variables of every ``Sum`` inside it.
_IndexInfo = tuple[frozenset, frozenset]


def _node_index_info(node: sp.Basic, memo: dict) -> _IndexInfo:
    """Combine the memoised index info of the children of ``node``."""
    indices = set()
    bound = set()
    for child in node.args:
        child_indices, child_bound = memo[child]
        indices |= child_indices
        bound |= child_bound
    if isinstance(node, sp.Idx):
        indices.add(node)
    if isinstance(node, sp.Sum):
        bound.update(limit[0] for limit in node.limits)
    return frozenset(indices), frozenset(bound)


def _fill_index_memo(expr: sp.Basic, memo: dict) -> None:
    """Memoise the index info of every subtree of ``expr``, bottom-up."""
    stack = [expr]
    while stack:
        node = stack[-1]
        if node in memo:
            stack.pop()
            continue
        pending = [child for child in node.args if child not in memo]
        if pending:
            stack.extend(pending)
            continue
        memo[node] = _node_index_info(node, memo)
        stack.pop()


def _free_indices(expr: sp.Expr, memo: dict | None = None) -> set:
    """Return the indices that are free in ``expr``.

    An index is free when it is not the summation index of any
    :class:`sympy.Sum` inside ``expr``.

    Parameters
    ----------
    expr : sympy expression
        Expression to inspect.
    memo : dict, optional
        Per-call cache shared by the queries of one pass.  It maps every
        visited subtree to its ``Idx`` atoms and summation variables, so
        shared subtrees are walked once.  Without it every call walks
        ``expr`` from scratch.

    Returns
    -------
    set
        A new set with the free indices.
    """
    if memo is None:
        bound = {
            limit[0] for total in expr.atoms(sp.Sum) for limit in total.limits
        }
        return {idx for idx in expr.atoms(sp.Idx) if idx not in bound}
    _fill_index_memo(expr, memo)
    indices, bound = memo[expr]
    return set(indices - bound)


def _order_free(free: set, indexes) -> list:
    """Order free indices deterministically.

    Derivative indices (``i`` and ``j``) come first, then the user supplied
    ``indexes`` in their given order, then any remaining index sorted by
    name.
    """
    ordered = []
    for idx in (i, j):
        if idx in free and idx not in ordered:
            ordered.append(idx)
    for idx in indexes:
        if idx in free and idx not in ordered:
            ordered.append(idx)
    for idx in sorted(free, key=str):
        if idx not in ordered:
            ordered.append(idx)
    return ordered


def _kron_to_piecewise(expr: sp.Expr) -> sp.Expr:
    """Replace ``KroneckerDelta`` by a two branch ``Piecewise``."""
    replacements = {}
    for delta in expr.atoms(sp.KroneckerDelta):
        first, second = delta.args
        replacements[delta] = sp.Piecewise(
            (sp.Float(1), sp.Eq(first, second)), (sp.Float(0), True)
        )
    return expr.xreplace(replacements)


def _is_placeholder(func) -> bool:
    """Return whether ``func`` is a user defined placeholder function.

    Placeholder functions are the ones created with
    :func:`~thermodiff.core.idxfunction.idx_function` or plain undefined
    :class:`sympy.Function` objects.  SymPy provided functions (``exp``,
    ``KroneckerDelta``, ``Piecewise`` ...) are defined functions and are
    handled by the Fortran printer directly.
    """
    return isinstance(func, sp.Function) and not isinstance(
        func, DefinedFunction
    )


class _PlaceholderResolver:
    """Replace undefined functions and derivatives with input arguments.

    Parameters
    ----------
    strict : bool
        When ``True``, raise a ``ValueError`` if any placeholder is found
        instead of exposing it as an argument.
    """

    def __init__(self, strict: bool = False):
        self.strict = strict
        #: name -> rank (0 for scalars)
        self.ranks: dict = {}

    def resolve(self, expr: sp.Expr) -> sp.Expr:
        """Return ``expr`` with placeholders replaced by plain variables."""
        derivatives = list(expr.atoms(sp.Derivative))
        functions = [f for f in expr.atoms(sp.Function) if _is_placeholder(f)]

        if self.strict and (derivatives or functions):
            names = sorted(
                {str(d) for d in derivatives} | {str(f) for f in functions}
            )
            raise ValueError(
                "Expression contains unresolved functions or derivatives "
                "(strict mode): " + ", ".join(names)
            )

        replacements = {
            deriv: self._derivative_var(deriv) for deriv in derivatives
        }
        resolved = expr.xreplace(replacements)

        bare = {
            func
            for func in resolved.atoms(sp.Function)
            if _is_placeholder(func)
        }
        replacements = {func: self._function_var(func) for func in bare}
        return resolved.xreplace(replacements)

    def _index_dims(self, func: sp.Function) -> list:
        """Return the index labels used as array dimensions by ``func``."""
        dims = []
        for arg in func.args:
            if isinstance(arg, sp.Idx):
                dims.append(str(arg))
            elif isinstance(arg, sp.Indexed):
                dims.extend(str(idx) for idx in arg.indices)
        return dims

    def _register(self, name: str, rank: int) -> None:
        previous = self.ranks.get(name)
        if previous is not None and previous != rank:
            raise ValueError(
                f"Function '{name}' is used with inconsistent ranks: "
                f"{previous} and {rank}."
            )
        self.ranks[name] = rank

    def _build(self, name: str, labels: list) -> sp.Expr:
        rank = len(labels)
        self._register(name, rank)
        if rank == 0:
            return sp.Symbol(name)
        base = sp.IndexedBase(name, shape=(nc,) * rank)
        indices = tuple(sp.Idx(label) for label in labels)
        return sp.Indexed(base, *indices)

    def _derivative_var(self, deriv: sp.Derivative) -> sp.Expr:
        inner = deriv.expr
        if _is_placeholder(inner):
            base = _sanitize(inner.func.__name__)
            dims = self._index_dims(inner)
        else:
            base = _sanitize(str(inner))
            dims = []

        suffix = ""
        for var in deriv.variables:
            if isinstance(var, sp.Indexed):
                label = str(var.indices[0])
                dims.append(label)
                suffix += "d" + label
            else:
                suffix += "d" + _sanitize(str(var))

        return self._build("d" + base + suffix, dims)

    def _function_var(self, func: sp.Function) -> sp.Expr:
        base = _sanitize(func.func.__name__)
        dims = self._index_dims(func)
        return self._build(base, dims)


class _Namer:
    """Generate sequential temporary variable names."""

    def __init__(self):
        self._count = 0
        self.created = []

    def __call__(self, prefix: str = "t") -> sp.Symbol:
        self._count += 1
        symbol = sp.Symbol(f"{prefix}{self._count}")
        self.created.append(symbol)
        return symbol


def _expand_sums(expr: sp.Expr, namer: _Namer):
    """Expand ``Sum`` sub-expressions into loop statements.

    Parameters
    ----------
    expr : sympy expression
        Expression that may contain summations.
    namer : _Namer
        Generator of temporary variable names.

    Returns
    -------
    statements : list of str
        The generated assignment and loop statements, innermost first.
    residual : sympy expression
        ``expr`` with every ``Sum`` replaced by a temporary symbol.
    """
    statements = []
    residual = expr

    def enclosing_bound(root):
        """Map every Sum in ``root`` to the indices bound by its ancestors."""
        found = {}

        def walk(sub, bound):
            if isinstance(sub, sp.Sum):
                found.setdefault(sub, set()).update(bound)
                bound = bound | {limit[0] for limit in sub.limits}
                walk(sub.function, bound)
                return
            for arg in getattr(sub, "args", ()):
                walk(arg, bound)

        walk(root, frozenset())
        return found

    while True:
        outer = enclosing_bound(residual)
        # an innermost sum can be computed before everything else only if
        # it does not depend on an index bound by an enclosing sum
        innermost = [
            total
            for total in sorted(
                residual.atoms(sp.Sum), key=sp.default_sort_key
            )
            if not total.function.atoms(sp.Sum)
            and not (_free_indices(total) & outer.get(total, set()))
        ]
        if not innermost:
            break
        for total in innermost:
            tmp = namer("t")
            statements.append(f"{tmp} = 0.0d0")
            depth = 0
            for idx, low, high in total.limits:
                low_s = _print_expr(low)
                high_s = _print_expr(high)
                statements.append(
                    f"{_INDENT * depth}do {idx} = {low_s}, {high_s}"
                )
                depth += 1
            body_s = _print_expr(total.function)
            statements.append(f"{_INDENT * depth}{tmp} = {tmp} + {body_s}")
            for _ in total.limits:
                depth -= 1
                statements.append(f"{_INDENT * depth}end do")
            residual = residual.xreplace({total: tmp})

    # remaining sums contain inner sums that depend on their bound indices
    # (e.g. sum_i n_i/(sum_j n_j G_ji)): the inner sums are expanded inside
    # the loops of the enclosing one
    remaining = sorted(
        (
            total
            for total in residual.atoms(sp.Sum)
            if not enclosing_bound(residual).get(total)
        ),
        key=sp.default_sort_key,
    )
    for total in remaining:
        tmp = namer("t")
        statements.append(f"{tmp} = 0.0d0")
        depth = 0
        for idx, low, high in total.limits:
            statements.append(
                f"{_INDENT * depth}do {idx} = {_print_expr(low)}, {_print_expr(high)}"
            )
            depth += 1
        inner_statements, body = _expand_sums(total.function, namer)
        for statement in inner_statements:
            statements.append(_INDENT * depth + statement)
        statements.append(
            f"{_INDENT * depth}{tmp} = {tmp} + {_print_expr(body)}"
        )
        for _ in total.limits:
            depth -= 1
            statements.append(f"{_INDENT * depth}end do")
        residual = residual.xreplace({total: tmp})

    return statements, residual


def _print_expr(expr: sp.Expr) -> str:
    """Print ``expr`` as a single Fortran expression."""
    return fcode(expr, **_FCODE_SETTINGS).strip()


def _indent_lines(text: str, depth: int) -> list:
    """Indent every non-empty line of ``text`` by ``depth`` levels."""
    prefix = _INDENT * depth
    return [prefix + line for line in text.splitlines() if line.strip()]


def _declare_output(
    name: str, rank: int, optional: bool = False, contiguous: bool = False
) -> str:
    """Return the declaration line for an output argument.

    With ``contiguous=True`` assumed-shape outputs carry the ``contiguous``
    attribute (option ``contiguous_args``)."""
    if optional:
        attributes = "real(8), optional, intent(out)"
    else:
        attributes = "real(8), intent(out)"
    if rank == 0:
        return f"{attributes} :: {name}"
    if contiguous:
        attributes = attributes.replace("real(8), ", "real(8), contiguous, ")
    dims = ", ".join([":"] * rank)
    return f"{attributes} :: {name}({dims})"


def _render_output(
    name: str,
    expr: sp.Expr,
    free_order: list,
    namer,
    mirror=False,
    column=False,
) -> list:
    """Render the statements that compute a single output.

    With ``mirror=True`` (rank-2 outputs known to be symmetric) only the
    upper triangle is computed and every value is stored twice.  With
    ``column=True`` the loops are opened in reverse order (the first
    subscript innermost, column-major); the subscripts are unchanged.
    """
    statements, residual = _expand_sums(expr, namer)
    residual = residual.xreplace(
        {idx: sp.Symbol(str(idx)) for idx in free_order}
    )

    mirror = mirror and len(free_order) == 2
    lines = []
    depth = 0
    loop_order = (
        list(reversed(free_order)) if column and not mirror else free_order
    )
    for position, idx in enumerate(loop_order):
        low = str(free_order[0]) if (mirror and position == 1) else "1"
        lines.append(_INDENT * depth + f"do {idx} = {low}, nc")
        depth += 1

    for statement in statements:
        lines.append(_INDENT * depth + statement)

    body = fcode(residual, assign_to="res", **_FCODE_SETTINGS)
    lines.extend(_indent_lines(body, depth))

    if free_order:
        subscript = ", ".join(str(idx) for idx in free_order)
        lines.append(_INDENT * depth + f"{name}({subscript}) = res")
        if mirror:
            transposed = ", ".join(str(idx) for idx in reversed(free_order))
            lines.append(_INDENT * depth + f"{name}({transposed}) = res")
    else:
        lines.append(_INDENT * depth + f"{name} = res")

    for _ in free_order:
        depth -= 1
        lines.append(_INDENT * depth + "end do")

    return lines


def _new_shared_temp(namer, free_order, arrays):
    """Create a temporary for a shared sub-expression.

    Parameters
    ----------
    namer : _Namer
        Generator of temporary variable names.
    free_order : list
        Free indices the temporary depends on; empty for a scalar.
    arrays : dict
        Records the rank of every array temporary created.

    Returns
    -------
    sympy expression
        A ``Symbol`` for a scalar temporary, or an ``Indexed`` (indexed by
        ``free_order``) for an array temporary.
    """
    name = str(namer("t"))
    if not free_order:
        return sp.Symbol(name)
    arrays[name] = len(free_order)
    base = sp.IndexedBase(name, shape=(nc,) * len(free_order))
    return sp.Indexed(base, *free_order)


def _alpha_key(total):
    """Canonical form of a summation modulo renaming of its bound indices.

    Two summations that differ only in the names of their dummy indices
    (``sum_l x(i,l)`` and ``sum_m x(i,m)``) get the same key.
    """
    mapping = {}
    for position, limit in enumerate(total.limits):
        mapping[limit[0]] = sp.Idx(f"_bound{position}")
    return total.xreplace(mapping)


def _free_alpha_key(total, indexes, symmetric_params=()):
    """Canonical form of a summation modulo renaming of all its indices.

    Bound indices are renamed by position, free indices by their order, so
    that ``sum_k x(i,k)`` and ``sum_l x(j,l)`` get the same key.
    """
    free = _order_free(_free_indices(total), indexes)
    mapping = {index: sp.Idx(f"_free{pos}") for pos, index in enumerate(free)}
    key = _alpha_key(total.xreplace(mapping))
    if symmetric_params:
        key = _canonical_symmetric(key, symmetric_params)
    return key, free


def _share_sums(
    expressions,
    namer,
    indexes,
    arrays,
    alpha=False,
    free_alpha=False,
    symmetric_params=(),
    transpose=False,
):
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
    exprs = list(expressions)
    definitions = []
    while True:
        innermost = []
        seen = set()
        for expr in exprs:
            for total in sorted(expr.atoms(sp.Sum), key=sp.default_sort_key):
                if total.function.atoms(sp.Sum) or total in seen:
                    continue
                seen.add(total)
                innermost.append(total)
        if not innermost:
            break
        replacements = {}
        by_key = {}
        for total in innermost:
            if free_alpha:
                key, free_here = _free_alpha_key(
                    total, indexes, symmetric_params
                )
                if transpose and key not in by_key and len(free_here) == 2:
                    # the transpose of a rank-2 sum already computed
                    swapped = list(reversed(free_here))
                    mapping = {
                        index: sp.Idx(f"_free{pos}")
                        for pos, index in enumerate(swapped)
                    }
                    tkey = _alpha_key(total.xreplace(mapping))
                    if symmetric_params:
                        tkey = _canonical_symmetric(tkey, symmetric_params)
                    if tkey in by_key:
                        target_, _ = by_key[tkey]
                        replacements[total] = sp.Indexed(
                            target_.base, *swapped
                        )
                        continue
                if key in by_key:
                    # equal up to the names of its bound and free indices:
                    # reuse the temporary, indexed by this sum's free indices
                    target_, _ = by_key[key]
                    replacements[total] = (
                        sp.Indexed(target_.base, *free_here)
                        if free_here
                        else target_
                    )
                    continue
            elif alpha:
                key = _alpha_key(total)
                if key in by_key:
                    # alpha-equivalent to a summation already hoisted
                    replacements[total] = by_key[key]
                    continue
            free = _order_free(_free_indices(total), indexes)
            target = _new_shared_temp(namer, free, arrays)
            definitions.append(
                {
                    "target": target,
                    "free": free,
                    "limits": list(total.limits),
                    "body": total.function,
                }
            )
            replacements[total] = target
            if free_alpha:
                by_key[
                    _free_alpha_key(total, indexes, symmetric_params)[0]
                ] = (
                    target,
                    free,
                )
            elif alpha:
                by_key[_alpha_key(total)] = target
        exprs = [expr.xreplace(replacements) for expr in exprs]
    return exprs, definitions


def _common_scalar_cse(expressions, namer, arrays):
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
    definitions = []
    index_memo = {}
    eligible_memo = {}
    rank_memo = {}
    while True:
        counts = _count_scalar_candidates(exprs, index_memo, eligible_memo)
        repeated = [sub for sub, count in counts.items() if count > 1]
        if not repeated:
            break
        best = max(
            repeated,
            key=lambda sub: (counts[sub], *_candidate_rank(sub, rank_memo)),
        )
        target = _new_shared_temp(namer, [], arrays)
        definitions.append(
            {
                "target": target,
                "free": [],
                "limits": [],
                "body": best,
            }
        )
        exprs = [expr.xreplace({best: target}) for expr in exprs]
    return exprs, definitions


def _is_scalar_candidate(
    sub: sp.Basic, index_memo: dict, eligible_memo: dict
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


def _count_scalar_candidates(
    exprs: list, index_memo: dict, eligible_memo: dict
) -> dict:
    """Count the occurrences of every hoistable sub-expression.

    The dict keeps the order in which candidates are first met in a
    pre-order walk of ``exprs``, which fixes the tie-break of the caller.
    """
    counts = {}
    for expr in exprs:
        for sub in sp.preorder_traversal(expr):
            if _is_scalar_candidate(sub, index_memo, eligible_memo):
                counts[sub] = counts.get(sub, 0) + 1
    return counts


def _candidate_rank(sub: sp.Expr, rank_memo: dict) -> tuple:
    """Return ``(count_ops, default_sort_key)`` of ``sub`` (memoised)."""
    if sub not in rank_memo:
        rank_memo[sub] = (sp.count_ops(sub), sp.default_sort_key(sub))
    return rank_memo[sub]


def _node_children(node: sp.Basic):
    """Return the children that ``sp.preorder_traversal`` visits.

    Mirrors SymPy: ``LatticeOp`` nodes (``Max``, ``Min``, ``And``...) are
    walked through their ``_argset``, everything else through ``args``.
    """
    if hasattr(node, "_argset"):
        return node._argset
    return node.args


def _preorder_entries(expr: sp.Basic, own_entries, memo: dict) -> tuple:
    """Collect per-node entries of ``expr`` in pre-order, memoised.

    The result equals the concatenation of ``own_entries(node)`` over
    ``sp.preorder_traversal(expr)``.  Entries are cached per subtree in
    ``memo``; ``xreplace`` returns the very same objects for the subtrees
    it leaves alone, so a later walk of a rewritten expression only
    visits the nodes that changed.

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


#: Candidate names for the auxiliary loop variables used by the array
#: temporaries of :func:`_indexed_cse`.  Names already present in the model
#: are skipped, so the generated code stays clash free.
_AUX_INDEX_NAMES = ["p", "q", "r", "u", "w", "x", "y", "z"] + [
    # large models (PC-SAFT) can use every single letter; the extra names
    # only appear when the first eight are taken, so existing output is
    # unchanged
    f"{letter}{number}"
    for number in range(2, 10)
    for letter in "pqruwxyz"
]

#: Largest rank that :func:`_indexed_cse` materialises.  The pass trades
#: memory for speed, and that only pays off up to rank two.  A rank three
#: temporary has ``nc**3`` entries, which is large enough to fall out of
#: cache: benchmarks show it makes the generated code slower than simply
#: recomputing the factor, while still allocating ``O(nc**3)`` memory.
_MAX_ARRAY_CSE_RANK = 2


def _index_symbols(expr) -> list:
    """Return the model indices appearing in ``expr``, sorted by name."""
    return sorted(expr.atoms(sp.Idx), key=str)


def _index_set(expr, memo: dict) -> frozenset:
    """Return the ``sp.Idx`` atoms of ``expr``, memoised bottom-up.

    Parameters
    ----------
    expr : sympy expression
        Expression to inspect.
    memo : dict
        Maps every visited sub-tree to its index set.  It is owned by the
        caller (one per pass), so it can be reused while the expressions
        stay valid keys; each node is visited once.

    Returns
    -------
    frozenset
        The same set as ``expr.atoms(sp.Idx)``.
    """
    stack = [expr]
    while stack:
        node = stack[-1]
        if node in memo:
            stack.pop()
            continue
        if isinstance(node, sp.Idx):
            memo[node] = frozenset((node,))
            stack.pop()
            continue
        pending = [arg for arg in node.args if arg not in memo]
        if pending:
            stack.extend(pending)
            continue
        memo[node] = frozenset().union(*(memo[arg] for arg in node.args))
        stack.pop()
    return memo[expr]


def _sorted_indices(expr, memo: dict) -> list:
    """Like :func:`_index_symbols`, but reusing the ``memo`` index sets."""
    return sorted(_index_set(expr, memo), key=str)


def _is_hoistable(sub, index_memo: dict | None = None) -> bool:
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


def _canonical_form(sub, indices: list, slots: list):
    """Rename ``indices`` of ``sub`` to the leading auxiliary ``slots``."""
    mapping = {
        index: slots[position] for position, index in enumerate(indices)
    }
    return sub.xreplace(mapping)


def _cached_hoistable(sub, index_memo: dict, cache: dict) -> bool:
    """:func:`_is_hoistable` memoised in the per-call ``cache``."""
    if sub not in cache:
        cache[sub] = _is_hoistable(sub, index_memo)
    return cache[sub]


def _cached_canonical(sub, indices: list, slots: list, cache: dict):
    """:func:`_canonical_form` memoised in the per-call ``cache``.

    The result depends on ``slots``, so ``cache`` must not outlive the
    call that fixed them.
    """
    if sub not in cache:
        cache[sub] = _canonical_form(sub, indices, slots)
    return cache[sub]


def _hoistable_entries(
    expr,
    slots: list,
    index_memo: dict,
    hoistable_cache: dict,
    canonical_cache: dict,
) -> list:
    """List ``(sub, canonical, indices)`` of the hoistable sub-trees.

    Sub-trees come in pre-order, and only those whose rank does not exceed
    :data:`_MAX_ARRAY_CSE_RANK` are kept.
    """
    entries = []
    for sub in sp.preorder_traversal(expr):
        if not _cached_hoistable(sub, index_memo, hoistable_cache):
            continue
        indices = _sorted_indices(sub, index_memo)
        if len(indices) > _MAX_ARRAY_CSE_RANK:
            continue
        canon = _cached_canonical(sub, indices, slots, canonical_cache)
        entries.append((sub, canon, indices))
    return entries


def _indexed_cse(
    expressions, namer, arrays, expensive_twice=False, counted=None
):
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
    definitions = []
    # Fortran is case insensitive: compare lower-cased names, so that an
    # auxiliary index ``r`` never clashes with e.g. a module constant ``R``.
    used = {
        str(symbol).lower()
        for expr in exprs
        for symbol in expr.atoms(sp.Symbol)
    }
    used.update(
        {str(b).lower() for expr in exprs for b in expr.atoms(sp.IndexedBase)}
    )
    used.update({"nc", "res"})
    slots = [sp.Idx(name) for name in _AUX_INDEX_NAMES if name not in used]

    index_memo = {}
    hoistable_cache = {}
    canonical_cache = {}

    while True:
        entries = [
            _hoistable_entries(
                expr, slots, index_memo, hoistable_cache, canonical_cache
            )
            for expr in exprs
        ]
        counts = {}
        sizes = {}
        for position, position_entries in enumerate(entries):
            if counted is not None and not counted[position]:
                continue
            for _, canon, indices in position_entries:
                counts[canon] = counts.get(canon, 0) + 1
                sizes[canon] = len(indices)
        best = None
        best_key = None
        for canon in sorted(counts, key=sp.default_sort_key):
            count = counts[canon]
            if count < 2:
                continue
            ops = sp.count_ops(canon)
            # A single repeated operation is not worth the extra array
            # unless it is used at least three times.
            if (
                ops < 2
                and count < 3
                and not (expensive_twice and _is_expensive(canon))
            ):
                continue
            key = (count, ops)
            if best_key is None or key > best_key:
                best_key = key
                best = canon
        if best is None:
            break
        free = slots[: sizes[best]]
        target = _new_shared_temp(namer, free, arrays)
        definitions.append(
            {
                "target": target,
                "free": free,
                "limits": [],
                "body": best,
            }
        )
        base = target.base
        rank = len(free)
        replacements = {}
        for position_entries in entries:
            for sub, canon, indices in position_entries:
                if len(indices) == rank and canon == best:
                    replacements[sub] = sp.Indexed(base, *indices)
        exprs = [expr.xreplace(replacements) for expr in exprs]
    return exprs, definitions


def _is_expensive(expr) -> bool:
    """Whether evaluating ``expr`` involves a costly operation.

    A function call (``log``, ``exp``, ...), a power whose exponent is not a
    positive integer (``sqrt``, a real exponent, a division) is costly; plain
    sums and products of loads are not.
    """
    for sub in sp.preorder_traversal(expr):
        if isinstance(sub, sp.Function) and not isinstance(sub, sp.Piecewise):
            return True
        if isinstance(sub, sp.Pow):
            exponent = sub.exp
            if not (exponent.is_Integer and exponent > 0):
                return True
    return False


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
        body = definition["body"]
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
                {"target": target, "free": free, "limits": [], "body": canon}
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
        if definition["limits"]:
            bound = {limit[0] for limit in definition["limits"]}
            definition = dict(
                definition, body=rewrite(definition["body"], bound)
            )
            rewritten.extend(definitions)
            definitions.clear()
        rewritten.append(definition)
    return exprs, hoisted, rewritten


def _canonical_floats(expr):
    """Round every floating-point constant to its double value, with one
    representation per value.  Constants that SymPy carried with different
    precisions (``5.96554738597035e-7`` and ``5.96554738597034e-7`` from the
    same product of literals) become the same object, so the common
    sub-expression passes can merge the expressions that contain them."""
    floats = expr.atoms(sp.Float)
    if not floats:
        return expr
    return expr.xreplace({f: sp.Float(repr(float(f)), 17) for f in floats})


def _merge_dummy_images(expr, symmetric_params=(), shared_bases=frozenset()):
    """Merge the terms of a summation that are images under a renaming of
    its bound indices.

    In ``sum_{k,l,m} F`` with all indices over the same range, renaming the
    dummies never changes the value, so ``sum (b_k + b_l + b_m) L_klm n_k n_l n_m``
    with ``L`` symmetric equals ``3 sum b_k L_klm n_k n_l n_m``.  Each term of
    the body is mapped to the smallest of its images under every permutation
    of the bound indices (symmetric parameters canonicalised), equal images
    are merged with their multiplicity.  Exact; applied innermost first.

    A summation that reads an array of ``shared_bases`` (arrays that other
    outputs of the same kernel read inside sums with free indices) is left
    as is: its unmerged form shares the inner tables with those outputs, and
    merging would add a second pass over the array.
    """
    from itertools import permutations

    def canon(term, bound):
        best = None
        for perm in permutations(bound):
            image = term.xreplace(dict(zip(bound, perm)))
            if symmetric_params:
                image = _canonical_symmetric(image, symmetric_params)
            key = sp.default_sort_key(image)
            if best is None or key < best[0]:
                best = (key, image)
        return best[1]

    def visit(sub):
        if not getattr(sub, "args", ()) or isinstance(
            sub, (sp.Idx, sp.Indexed)
        ):
            return sub
        if isinstance(sub, sp.Sum):
            body = visit(sub.function)
            bound = [lim[0] for lim in sub.limits]
            ranges = {(lim[1], lim[2]) for lim in sub.limits}
            reads_shared = any(
                str(x.base) in shared_bases for x in body.atoms(sp.Indexed)
            )
            if 2 <= len(bound) <= 3 and len(ranges) == 1 and not reads_shared:
                expanded = sp.expand(body)
                terms = (
                    expanded.args
                    if isinstance(expanded, sp.Add)
                    else (expanded,)
                )
                if len(terms) > 1:
                    merged = {}
                    for term in terms:
                        image = canon(term, bound)
                        coeff, rest = image.as_coeff_Mul()
                        merged[rest] = merged.get(rest, 0) + coeff
                    if len(merged) < len(terms):
                        body = sp.factor_terms(
                            sp.Add(*[c * r for r, c in merged.items()])
                        )
            return sub.func(body, *sub.limits)
        return sub.func(*[visit(a) for a in sub.args])

    return visit(expr)


def _nest_double_sums(
    staged,
    indexes,
    symmetric_params=(),
    shared_inner=False,
    general=False,
    cost=False,
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
            renamed = _canonical_symmetric(renamed, symmetric_params)
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
                {"target": coef, "free": [other], "limits": [], "body": body}
            )
        if not isinstance(key, sp.Indexed):
            vec = _new_shared_temp(namer, [side], arrays)
            definitions.append(
                {"target": vec, "free": [side], "limits": [], "body": key}
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
        used |= {str(x).lower() for x in definition["body"].atoms(sp.Symbol)}
        used |= {
            str(x).lower() for x in definition["body"].atoms(sp.IndexedBase)
        }
        used |= {str(x).lower() for x in definition["free"]}
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
                {"target": target, "free": free, "limits": [], "body": canon}
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
        bound = set(definition["free"]) | {
            lim[0] for lim in definition["limits"]
        }
        if bound and not definition.get("mirror_of"):
            body = _collect_terms(definition["body"])
            definition = dict(definition, body=rewrite(body, bound))
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


def _symv_parts(definition, symmetric_params):
    """Split ``v(f) = sum_b P(b,f)*R(b)`` with ``P`` symmetric.

    Returns ``(pair, rest)`` when the body is a product of exactly one
    factor that depends on both indices and is symmetric under their swap,
    and factors that depend only on the bound index (or on none); else None.
    """
    free = definition["free"]
    limits = definition["limits"]
    if len(free) != 1 or len(limits) != 1:
        return None
    b, low, high = limits[0]
    if low != 1 or str(high) != "nc":
        return None
    f = free[0]
    body = definition["body"]
    factors = body.args if isinstance(body, sp.Mul) else (body,)
    pair = [x for x in factors if {f, b} <= x.atoms(sp.Idx)]
    rest = [x for x in factors if x not in pair]
    if len(pair) != 1 or any(f in x.atoms(sp.Idx) for x in rest):
        return None
    if body.has(sp.Piecewise) or any(
        isinstance(x, _LoopStart) for x in body.atoms(sp.Symbol)
    ):
        return None
    canon = _canonical_symmetric(pair[0], symmetric_params)
    if _swap(canon, f, b, symmetric_params) != canon:
        return None
    return pair[0], sp.Mul(*rest)


def _render_symv(definition) -> list:
    """Symmetric matrix-vector product reading the upper triangle only.

    For every column ``f``: an axpy into ``v(1:f-1)`` and a dot product for
    ``v(f)``, both vectorisable.
    """
    target = definition["target"]
    f = definition["free"][0]
    b = definition["limits"][0][0]
    pair, rest = definition["symv"]
    acc = definition["symv_acc"]
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


def _render_column_order(definition) -> list:
    """Loop nest with the first subscript innermost (column-major)."""
    target = definition["target"]
    free = definition["free"]
    limits = definition["limits"]
    target_s = _print_expr(target)
    body_s = _print_expr(definition["body"])
    if limits:
        (f,) = free
        b, low, high = limits[0]
        return [
            f"do {f} = 1, nc",
            f"{_INDENT}{target_s} = 0.0d0",
            "end do",
            f"do {b} = {_print_expr(low)}, {_print_expr(high)}",
            f"{_INDENT}do {f} = 1, nc",
            f"{_INDENT * 2}{target_s} = {target_s} + {body_s}",
            f"{_INDENT}end do",
            "end do",
        ]
    p_, q_ = free
    return [
        f"do {q_} = 1, nc",
        f"{_INDENT}do {p_} = 1, nc",
        f"{_INDENT * 2}{target_s} = {body_s}",
        f"{_INDENT}end do",
        "end do",
    ]


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


def _render_column_sum(definition) -> list:
    """Rank-2 sum in ``jki`` order: for every column ``q``, axpy updates over
    the bound index; a symmetric (``mirror_sum``) sum fills ``p <= q`` and
    mirrors the row."""
    target = definition["target"]
    p_, q_ = definition["free"]
    b, low, high = definition["limits"][0]
    target_s = _print_expr(target)
    upper = q_ if definition.get("mirror_sum") else "nc"
    lines = [
        f"do {q_} = 1, nc",
        f"{_INDENT}do {p_} = 1, {upper}",
        f"{_INDENT * 2}{target_s} = 0.0d0",
        f"{_INDENT}end do",
        f"{_INDENT}do {b} = {_print_expr(low)}, {_print_expr(high)}",
        f"{_INDENT * 2}do {p_} = 1, {upper}",
        f"{_INDENT * 3}{target_s} = {target_s} + {_print_expr(definition['body'])}",
        f"{_INDENT * 2}end do",
        f"{_INDENT}end do",
    ]
    if definition.get("mirror_sum"):
        transposed = _print_expr(sp.Indexed(target.base, q_, p_))
        lines += [
            f"{_INDENT}do {p_} = 1, {q_} - 1",
            f"{_INDENT * 2}{transposed} = {target_s}",
            f"{_INDENT}end do",
        ]
    lines.append("end do")
    return lines


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
                {
                    "target": group[0]["target"],
                    "free": group[0]["free"],
                    "limits": group[0]["limits"],
                    "body": group[0]["body"],
                    "fused": list(group),
                }
            )
        group.clear()

    for d in definitions:
        if simple(d) and group:
            first = group[0]
            f0, b0 = first["free"][0], first["limits"][0][0]
            renamed = dict(d)
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


def _render_fused(definition) -> list:
    group = definition["fused"]
    f = group[0]["free"][0]
    b, low, high = group[0]["limits"][0]
    colmajor = bool(group[0].get("colmajor"))
    lines = [f"do {f} = 1, nc"]
    for d in group:
        lines.append(f"{_INDENT}{_print_expr(d['target'])} = 0.0d0")
    lines.append("end do")
    outer, inner = (b, f) if colmajor else (f, b)
    lo_o, hi_o = (low, high) if colmajor else (1, sp.Symbol("nc"))
    lo_i, hi_i = (1, sp.Symbol("nc")) if colmajor else (low, high)
    lines.append(
        f"do {outer} = {_print_expr(sp.sympify(lo_o))}, {_print_expr(sp.sympify(hi_o))}"
    )
    lines.append(
        f"{_INDENT}do {inner} = {_print_expr(sp.sympify(lo_i))}, {_print_expr(sp.sympify(hi_i))}"
    )
    for d in group:
        t = _print_expr(d["target"])
        lines.append(f"{_INDENT * 2}{t} = {t} + {_print_expr(d['body'])}")
    lines.append(f"{_INDENT}end do")
    lines.append("end do")
    return lines


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
                {
                    "target": group[0]["target"],
                    "free": [idx],
                    "limits": [],
                    "body": group[0]["body"],
                    "compfused": True,
                    "steps": ordered,
                    "members": [(d["target"], d["body"]) for d in group],
                }
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
            renamed = dict(d)
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


def _render_component_fused(definition) -> list:
    idx = definition["free"][0]
    lines = [f"do {idx} = 1, nc"]
    for _kind, target, value in definition["steps"]:
        lines.append(f"{_INDENT}{_print_expr(target)} = {_print_expr(value)}")
    lines.append("end do")
    return lines


def _calls_libm(expr) -> bool:
    """Whether ``expr`` calls a vectorisable libm routine (``exp``, ``log``,
    ``sqrt``, a real power): a function other than ``Piecewise`` or a power whose
    exponent is not an integer."""
    for sub in sp.preorder_traversal(expr):
        if isinstance(sub, sp.Function) and not isinstance(sub, sp.Piecewise):
            return True
        if isinstance(sub, sp.Pow) and not sub.exp.is_Integer:
            return True
    return False


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
        d = dict(d)
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
                {
                    "target": group[0].get("target"),
                    "free": [],
                    "limits": limits,
                    "body": sp.S.Zero,
                    "outerfused": list(group),
                    "outer": outer_of(group[0]),
                }
            )
        group.clear()

    for d in definitions:
        k = kind(d)
        if k is None or (k == "map" and group and False):
            pass
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
                    {
                        "target": target,
                        "free": [outer],
                        "limits": [],
                        "body": prod.xreplace({inner: outer}),
                    }
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
            dict(
                d,
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
        vector = {
            "target": target,
            "free": [idx0],
            "limits": [],
            "body": rep_sub,
        }
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
            defs[pos] = dict(defs[pos], body=new_sites[pos])
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
                len(d["free"]) == 1
                and not d["limits"]
                and set(d) <= flags_ok
                and isinstance(d["target"], sp.Indexed)
                and alpha(d["body"], d["free"][0]) == key
            ):
                return sp.Indexed(d["target"].base, index), position
        target = _new_shared_temp(namer, [index], arrays)
        defs.insert(
            position,
            {"target": target, "free": [index], "limits": [], "body": product},
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
        free = set(d["free"])
        if (
            not free
            or len(d["limits"]) != 1
            or not isinstance(d["target"], sp.Indexed)
            or not set(d) <= flags_ok
        ):
            return False
        m, low, high = d["limits"][0]
        body = d["body"]
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
            d["free"]
            and d["limits"]
            and any(
                str(x.base) in pair_bases
                for x in d["body"].atoms(sp.Indexed)
                if len(x.indices) >= 2
            )
        ):
            blocked.add(_temp_name(d["target"]))
    run = []

    def shape(d):
        if not d["free"] or not d["limits"]:
            return None
        return (len(d["free"]), tuple((lo, hi) for _b, lo, hi in d["limits"]))

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
            blocked |= {_temp_name(e["target"]) for e in run}
        run = [d] if d is not None and shape(d) is not None else []

    # sums that read the same matrix go together: all split or none
    def matrices(d):
        return {
            str(x.base)
            for x in d["body"].atoms(sp.Indexed)
            if len(x.indices) >= 2
        }

    sums = [d for d in defs if d["free"] and d["limits"]]
    for d in sums:
        if _temp_name(d["target"]) not in blocked and not eligible(d):
            blocked.add(_temp_name(d["target"]))
    changed = True
    while changed:
        changed = False
        held = set().union(
            *[matrices(d) for d in sums if _temp_name(d["target"]) in blocked]
        )
        for d in sums:
            if _temp_name(d["target"]) not in blocked and matrices(d) & held:
                blocked.add(_temp_name(d["target"]))
                changed = True

    k = 0
    while k < len(defs):
        d = defs[k]
        if _temp_name(d["target"]) in blocked:
            k += 1
            continue
        free = set(d["free"])
        if (
            not free
            or len(d["limits"]) != 1
            or not isinstance(d["target"], sp.Indexed)
            or not set(d) <= flags_ok
        ):
            k += 1
            continue
        m, low, high = d["limits"][0]
        body = d["body"]
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
                not e["free"]
                and len(e["limits"]) == 1
                and set(e) <= flags_ok
                and isinstance(e["target"], sp.Symbol)
                and tuple(e["limits"][0][1:]) == (low, high)
                and alpha(e["body"], e["limits"][0][0]) == key
            ):
                scalar = e["target"]
                break
        if scalar is None:
            scalar = _new_shared_temp(namer, [], arrays)
            defs.insert(
                position,
                {
                    "target": scalar,
                    "free": [],
                    "limits": [(m, low, high)],
                    "body": s_body,
                },
            )
            position += 1
        table = dict(d, body=new_body)
        defs[position] = table
        name = _temp_name(d["target"])

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
                defs[q] = dict(e, body=reread(e["body"]))
        outs = [reread(e) for e in outs]
        k = position + 1
    return defs, outs


def _split_factored(definition, limits):
    """``(outer factor, inner body)`` of a ``factored`` double reduction."""
    inner = {lim[0] for lim in limits[1:]}
    factors = sp.Mul.make_args(definition["body"])
    o_f = [f for f in factors if not (f.atoms(sp.Idx) & inner)]
    i_f = [f for f in factors if f.atoms(sp.Idx) & inner]
    return sp.Mul(*o_f), sp.Mul(*i_f)


def _render_outer_fused(definition) -> list:
    x = definition["outer"]
    lines = []
    for g in definition["outerfused"]:
        if not g.get("compfused") and not g["free"]:
            lines.append(f"{_print_expr(g['target'])} = 0.0d0")
    lines.append(f"do {x} = 1, nc")
    members = definition["outerfused"]
    k = 0
    while k < len(members):
        g = members[k]
        if g.get("compfused"):
            for _k, t, v in g["steps"]:
                lines.append(f"{_INDENT}{_print_expr(t)} = {_print_expr(v)}")
            k += 1
        elif g["free"]:
            lines.append(
                f"{_INDENT}{_print_expr(g['target'])} = {_print_expr(g['body'])}"
            )
            k += 1
        else:
            # consecutive reductions with the same inner loops share them
            run = [g]
            while (
                k + len(run) < len(members)
                and not members[k + len(run)].get("compfused")
                and not members[k + len(run)]["free"]
                and members[k + len(run)]["limits"][1:] == g["limits"][1:]
            ):
                run.append(members[k + len(run)])
            depth = 1
            for r in run:
                if r.get("factored"):
                    lines.append(_INDENT + f"{r['acc']} = 0.0d0")
            for idx, low, high in g["limits"][1:]:
                lines.append(
                    _INDENT * depth
                    + f"do {idx} = {_print_expr(low)}, {_print_expr(high)}"
                )
                depth += 1
            for r in run:
                if r.get("factored"):
                    _o, inner_body = _split_factored(r, r["limits"])
                    lines.append(
                        _INDENT * depth
                        + f"{r['acc']} = {r['acc']} + {_print_expr(inner_body)}"
                    )
                    continue
                t = _print_expr(r["target"])
                lines.append(
                    _INDENT * depth + f"{t} = {t} + {_print_expr(r['body'])}"
                )
            for _ in g["limits"][1:]:
                depth -= 1
                lines.append(_INDENT * depth + "end do")
            for r in run:
                if r.get("factored"):
                    t = _print_expr(r["target"])
                    outer_f, _i = _split_factored(r, r["limits"])
                    lines.append(
                        _INDENT
                        + f"{t} = {t} + {_print_expr(r['acc'] * outer_f)}"
                    )
            k += len(run)
    lines.append("end do")
    return lines


def _diagonal_from_tables(definitions, symmetric_params):
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
            _canonical_symmetric(e, symmetric_params)
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
            new = {
                "target": d["target"],
                "free": [k],
                "limits": [],
                "body": sp.Indexed(t["target"].base, k, k),
            }
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
                {
                    "target": group[0]["target"],
                    "free": group[0]["free"],
                    "limits": group[0]["limits"],
                    "body": group[0]["body"],
                    "sibling": list(group),
                    "accs": [sp.Symbol(str(namer("t"))) for _ in group],
                }
            )
        out.extend(waiting)
        group.clear()
        waiting.clear()

    for d in definitions:
        if table(d) and group:
            (a0, b0), (m0, lo0, hi0) = group[0]["free"], group[0]["limits"][0]
            (a, b), (m, lo, hi) = d["free"], d["limits"][0]
            renamed = dict(d)
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
            d = dict(
                d,
                split_acc=2,
                accs2=[sp.Symbol(str(namer("t"))) for _ in d["accs"]],
            )
        elif lone(d) and rank3(d["body"]):
            d = {
                "target": d["target"],
                "free": d["free"],
                "limits": d["limits"],
                "body": d["body"],
                "sibling": [d],
                "accs": [sp.Symbol(str(namer("t")))],
                "split_acc": 2,
            }
            d["accs2"] = [sp.Symbol(str(namer("t")))]
        out.append(d)
    return out


def _render_sibling(definition) -> list:
    a, b = definition["free"]
    lines = [f"do {b} = 1, nc", f"{_INDENT}do {a} = 1, {b}"]
    for acc in definition["accs"]:
        lines.append(f"{_INDENT * 2}{acc} = 0.0d0")
    m, lo, hi = definition["limits"][0]
    if definition.get("split_acc"):
        # two partial accumulators (even/odd m): a reduction gcc vectorises
        # without reassociation (reportes/brechas/ronda3-cmr-o3.md)
        pairs = list(
            zip(definition["accs"], definition["accs2"], definition["sibling"])
        )
        for _acc, acc2, _g in pairs:
            lines.append(f"{_INDENT * 2}{acc2} = 0.0d0")
        lines.append(
            f"{_INDENT * 2}do {m} = {_print_expr(lo)}, {_print_expr(hi - 1)}, 2"
        )
        for acc, acc2, g in pairs:
            lines.append(
                f"{_INDENT * 3}{acc} = {acc} + {_print_expr(g['body'])}"
            )
            lines.append(
                f"{_INDENT * 3}{acc2} = {acc2} + "
                f"{_print_expr(g['body'].xreplace({m: m + 1}))}"
            )
        lines.append(f"{_INDENT * 2}end do")
        # remainder: at most one iteration, from the first m not done
        lines.append(f"{_INDENT * 2}do {m} = {m}, {_print_expr(hi)}")
        for _acc, acc2, g in pairs:
            lines.append(
                f"{_INDENT * 3}{acc2} = {acc2} + {_print_expr(g['body'])}"
            )
        lines.append(f"{_INDENT * 2}end do")
        for acc, acc2, _g in pairs:
            lines.append(f"{_INDENT * 2}{acc} = {acc} + {acc2}")
    else:
        lines.append(
            f"{_INDENT * 2}do {m} = {_print_expr(lo)}, {_print_expr(hi)}"
        )
        for acc, g in zip(definition["accs"], definition["sibling"]):
            lines.append(
                f"{_INDENT * 3}{acc} = {acc} + {_print_expr(g['body'])}"
            )
        lines.append(f"{_INDENT * 2}end do")
    for acc, g in zip(definition["accs"], definition["sibling"]):
        t = g["target"]
        lines.append(f"{_INDENT * 2}{_print_expr(t)} = {acc}")
        lines.append(
            f"{_INDENT * 2}{_print_expr(sp.Indexed(t.base, b, a))} = {acc}"
        )
    lines += [f"{_INDENT}end do", "end do"]
    return lines


def _render_definition(definition) -> list:
    """Render the statements that fill one shared temporary."""
    if definition.get("sibling"):
        return _render_sibling(definition)
    if definition.get("outerfused"):
        return _render_outer_fused(definition)
    if definition.get("compfused"):
        return _render_component_fused(definition)
    if definition.get("fused"):
        return _render_fused(definition)
    if definition.get("symv"):
        return _render_symv(definition)
    if definition.get("colmajor"):
        return _render_column_order(definition)
    if definition.get("colsum"):
        return _render_column_sum(definition)
    target = definition["target"]
    free = definition["free"]
    lines = []
    depth = 0
    for idx in free:
        lines.append(_INDENT * depth + f"do {idx} = 1, nc")
        depth += 1

    if (
        definition.get("mirror_sum")
        and len(free) == 2
        and definition["limits"]
    ):
        # symmetric rank-2 summation: upper triangle, then mirror
        a, b = free
        target_s = _print_expr(target)
        transposed = _print_expr(sp.Indexed(target.base, b, a))
        out = [
            f"do {b} = 1, nc",
            f"{_INDENT}do {a} = 1, {b}",
            f"{_INDENT * 2}{target_s} = 0.0d0",
        ]
        depth = 2
        for idx, low, high in definition["limits"]:
            out.append(
                _INDENT * depth
                + f"do {idx} = {_print_expr(low)}, {_print_expr(high)}"
            )
            depth += 1
        out.append(
            _INDENT * depth
            + f"{target_s} = {target_s} + {_print_expr(definition['body'])}"
        )
        for _ in definition["limits"]:
            depth -= 1
            out.append(_INDENT * depth + "end do")
        out.append(_INDENT * 2 + f"{transposed} = {target_s}")
        out += [f"{_INDENT}end do", "end do"]
        return out
    mirror = (
        definition.get("mirror", False)
        and len(free) == 2
        and not (definition["limits"])
    )
    if mirror:
        # Symmetric rank-2 temporary: fill the upper triangle and mirror.
        lines = [
            f"do {free[0]} = 1, nc",
            f"{_INDENT}do {free[1]} = {free[0]}, nc",
        ]
        body_s = _print_expr(definition["body"])
        target_s = _print_expr(target)
        transposed = _print_expr(sp.Indexed(target.base, free[1], free[0]))
        lines.append(_INDENT * 2 + f"{target_s} = {body_s}")
        lines.append(_INDENT * 2 + f"{transposed} = {target_s}")
        lines.append(_INDENT + "end do")
        lines.append("end do")
        return lines

    target_s = _print_expr(target)
    limits = definition["limits"]
    if definition.get("factored") and not free:
        outer_f, inner_body = _split_factored(definition, limits)
        acc = definition["acc"]
        (o, o_lo, o_hi), (x, x_lo, x_hi) = limits
        return [
            f"{target_s} = 0.0d0",
            f"do {o} = {_print_expr(o_lo)}, {_print_expr(o_hi)}",
            f"{_INDENT}{acc} = 0.0d0",
            f"{_INDENT}do {x} = {_print_expr(x_lo)}, {_print_expr(x_hi)}",
            f"{_INDENT * 2}{acc} = {acc} + {_print_expr(inner_body)}",
            f"{_INDENT}end do",
            f"{_INDENT}{target_s} = {target_s} + {_print_expr(acc * outer_f)}",
            "end do",
        ]
    if limits:
        lines.append(_INDENT * depth + f"{target_s} = 0.0d0")
        for idx, low, high in limits:
            low_s = _print_expr(low)
            high_s = _print_expr(high)
            lines.append(_INDENT * depth + f"do {idx} = {low_s}, {high_s}")
            depth += 1
        body_s = _print_expr(definition["body"])
        lines.append(_INDENT * depth + f"{target_s} = {target_s} + {body_s}")
        for _ in limits:
            depth -= 1
            lines.append(_INDENT * depth + "end do")
    else:
        body_s = _print_expr(definition["body"])
        lines.append(_INDENT * depth + f"{target_s} = {body_s}")

    for _ in free:
        depth -= 1
        lines.append(_INDENT * depth + "end do")
    return lines


#: ``attr`` of the pseudo-outputs that carry the scalars of ``scalar_chain``
_SCALAR_CHAIN = "_scalar_chain"
#: ``attr`` of the pseudo-outputs that carry the plain outputs (dropped)
_SCALAR_PLAIN = "_scalar_plain"


def _dependency_order(definitions):
    """Stable topological order: a definition goes after every one it reads,
    otherwise in its original position (lowest index first)."""
    import heapq

    index = {_temp_name(d["target"]): k for k, d in enumerate(definitions)}
    deps = []
    for k, d in enumerate(definitions):
        body = d["body"]
        names = {str(x) for x in body.free_symbols} | {
            str(x.base) for x in body.atoms(sp.Indexed)
        }
        deps.append(
            {index[nm] for nm in names if nm in index and index[nm] != k}
        )
    users = {k: [] for k in range(len(definitions))}
    for k, ds in enumerate(deps):
        for dep in ds:
            users[dep].append(k)
    pending = [len(ds) for ds in deps]
    ready = [k for k, count in enumerate(pending) if count == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        k = heapq.heappop(ready)
        order.append(definitions[k])
        for user in users[k]:
            pending[user] -= 1
            if pending[user] == 0:
                heapq.heappush(ready, user)
    if len(order) != len(definitions):
        raise ValueError("cyclic temporaries")
    return order


def _temp_name(target) -> str:
    """Return the base name of a shared temporary."""
    if isinstance(target, sp.Indexed):
        return str(target.base)
    return str(target)


def _referenced_temp_names(expr, temp_names) -> set:
    """Return the names of the temporaries referenced by ``expr``."""
    names = set()
    for symbol in expr.atoms(sp.Symbol):
        if str(symbol) in temp_names:
            names.add(str(symbol))
    for indexed in expr.atoms(sp.Indexed):
        base = indexed.base
        if isinstance(base, sp.IndexedBase) and str(base) in temp_names:
            names.add(str(base))
    return names


def _guard(attributes, out_names) -> str:
    """Return the ``present(...)`` condition for a group of outputs.

    Returns an empty string when nothing depends on the group.
    """
    if not attributes:
        return ""
    names = []
    if "value" in attributes and "value" in out_names:
        names.append(out_names["value"])
    names.extend(
        out_names[attr]
        for attr in _DERIV_ORDER
        if attr in attributes and attr in out_names
    )
    return " .or. ".join(f"present({name})" for name in names)


def _wrap_if(condition, limit=120) -> list:
    """Wrap an ``if (condition) then`` line with continuations."""
    pieces = condition.split(" .or. ")
    lines = []
    current = "if (" + pieces[0]
    for piece in pieces[1:]:
        token = " .or. " + piece
        if len(current) + len(token) + len(") then") > limit:
            lines.append(current + " .or. &")
            current = "     " + piece
        else:
            current += token
    lines.append(current + ") then")
    return lines


def _guarded(lines, condition) -> list:
    """Wrap ``lines`` in an ``if (condition)`` block when needed."""
    if not condition:
        return list(lines)
    block = _wrap_if(condition)
    block.extend((_INDENT + line) if line else line for line in lines)
    block.append("end if")
    return block


def _wrap_name_list(keyword, names, limit=120):
    """Split ``keyword`` plus names into continuation-free lines.

    Parameters
    ----------
    keyword : str
        Declaration prefix including the ``::``, for example
        ``"real(8) :: "``.
    names : list of str
        Names to declare.
    limit : int, optional
        Maximum line length.

    Returns
    -------
    list of str
        One or more declaration lines.
    """
    lines = []
    current = None
    for name in names:
        if current is None:
            current = name
        elif len(keyword) + len(current) + len(name) + 2 > limit:
            lines.append(keyword + current)
            current = name
        else:
            current += ", " + name
    if current is not None:
        lines.append(keyword + current)
    return lines


def _wrap_signature(sub_name, arg_names, limit=120):
    """Return the subroutine statement, wrapped if necessary.

    Parameters
    ----------
    sub_name : str
        Subroutine name.
    arg_names : list of str
        Argument names.
    limit : int, optional
        Maximum line length.

    Returns
    -------
    list of str
        The (possibly continued) subroutine statement.
    """
    head = f"subroutine {sub_name}("
    if len(head) + len(", ".join(arg_names)) <= limit:
        return [head + ", ".join(arg_names) + ")"]

    continuation = _INDENT + "  "
    lines = []
    current = ""
    for arg in arg_names:
        candidate = arg if not current else current + ", " + arg
        if current and len(head) + len(candidate) > limit:
            lines.append(head + current + ", &")
            head = continuation
            current = arg
        else:
            current = candidate
    lines.append(head + current + ")")
    return lines


# ---------------------------------------------------------------------------
# Symmetry handling
# ---------------------------------------------------------------------------


#: canonicalisation of symmetric parameters of rank >= 3 (option sym_rank3);
#: a one-element list so that to_fortran can switch it for one call
_SYM_RANK3 = [False]


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


def _canonical_symmetric(expr, symmetric_params):
    """Sort the subscripts of every symmetric parameter occurrence.

    For a parameter declared ``symmetric`` (``k(i,j) == k(j,i)``) the
    occurrence ``k(q,p)`` is rewritten as ``k(p,q)``.  The rewrite is exact
    and makes transposed occurrences syntactically identical, which lets
    the common sub-expression passes merge them and lets
    :func:`_fold_symmetric_sums` recognise symmetric summation bodies.
    """
    if not symmetric_params:
        return expr
    names = set(symmetric_params)
    replacements = {}
    for indexed in expr.atoms(sp.Indexed):
        if str(indexed.base) not in names:
            continue
        if len(indexed.indices) >= 3 and _SYM_RANK3[0]:
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


def _swap(expr, a, b, symmetric_params):
    """Swap the indices ``a`` and ``b`` and re-canonicalise."""
    swapped = expr.xreplace({a: b, b: a})
    return _canonical_symmetric(swapped, symmetric_params)


class _LoopStart(sp.Symbol):
    """Lower bound ``outer + 1`` of a folded triangular loop.

    It is an opaque symbol so that no common sub-expression pass can hoist
    the bound out of the loop header; it prints as ``outer + 1``.
    """

    def __new__(cls, outer):
        obj = super().__new__(cls, f"{outer} + 1")
        obj.outer = outer
        return obj


def _fold_symmetric_sums(expr, symmetric_params):
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
    replacements = {}
    for total in expr.atoms(sp.Sum):
        if len(total.limits) != 2:
            continue
        (a, low_a, high_a), (b, low_b, high_b) = total.limits
        if low_a != 1 or low_b != 1 or high_a != high_b:
            continue
        body = total.function
        canon = _canonical_symmetric(body, symmetric_params)
        if _swap(canon, a, b, symmetric_params) != canon:
            continue
        outer, inner = sorted((a, b), key=sp.default_sort_key)
        diagonal = canon.xreplace({inner: outer})
        replacements[total] = sp.Sum(
            diagonal, (outer, 1, high_a)
        ) + sp.Integer(2) * sp.Sum(
            canon, (outer, 1, high_a), (inner, _LoopStart(outer), high_a)
        )
    return expr.xreplace(replacements) if replacements else expr


# ---------------------------------------------------------------------------
# SymPy CSE backend
# ---------------------------------------------------------------------------


def _sympy_scalar_cse(expressions, namer, arrays):
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
    inlined = {}
    kept = []
    for symbol, value in replacements:
        value = value.xreplace(inlined)
        if value.atoms(sp.Idx) or value.has(sp.Sum):
            inlined[symbol] = value
            continue
        kept.append((symbol, value))
    renames = {}
    definitions = []
    for symbol, value in kept:
        target = _new_shared_temp(namer, [], arrays)
        renames[symbol] = target
        definitions.append(
            {
                "target": target,
                "free": [],
                "limits": [],
                "body": value.xreplace(renames),
            }
        )
    reduced = [expr.xreplace(inlined).xreplace(renames) for expr in reduced]
    return reduced, definitions


def to_fortran(
    diff,
    derivatives=None,
    name=None,
    subroutine_name=None,
    strict=False,
    integer_args=None,
    module_symbols=None,
    optional_derivatives=False,
    array_cse=True,
    share=True,
    stats=None,
    share_sums=None,
    scalar_cse=None,
    symmetric_params=(),
    param_symmetry=False,
    hessian_symmetry=False,
    cse_backend="thermodiff",
    sum_alpha=False,
    array_pad=0,
    hoist_invariants=False,
    chain_rule=False,
    hoist_loops=False,
    sum_free_alpha=False,
    fuse_temps=False,
    nest_sums=False,
    split_sqrt=False,
    split_sums=False,
    nest_shared=False,
    sym_matvec=False,
    col_major=False,
    pair_cse=False,
    pair_rates=False,
    sum_transpose=False,
    sep_sums=False,
    sym_rank3=False,
    nest_general=False,
    sym_sum2=False,
    cse_expensive=False,
    fuse_loops=False,
    nested_nodes=False,
    sym_perm=False,
    fuse_guard=False,
    nest_cost=False,
    comp_fuse=False,
    fuse_outer=False,
    fuse_libm_guard=False,
    keep_libm_tables=False,
    diag_table=False,
    sib_fuse=False,
    dummy_canon=False,
    float_canon=False,
    contiguous_args=False,
    table_layout=False,
    lowrank_hess=False,
    drop_pair_tables=False,
    share_elementwise=False,
    scalar_chain=False,
    split_affine=False,
    split_acc=False,
    _structured=None,
    _structure_only=False,
):
    """Generate a Fortran subroutine for a :class:`DiffPlz` instance.

    Parameters
    ----------
    diff : DiffPlz
        The differentiated thermodynamic expression.
    derivatives : str or list of str, optional
        Derivatives to emit, for example ``["dT", "dV", "dn"]``.  Both the
        friendly names and the derivative attributes (``["dt", "dv",
        "dni"]``) are accepted, as well as the groups ``all``, ``none``,
        ``first`` and ``second`` (see
        :func:`thermodiff.derivatives.resolve_derivatives`).  By default
        every non-zero derivative computed by ``diff`` is emitted.
    name : str, optional
        Base name used for the generated variables.  Defaults to the
        ``name`` attribute of ``diff``.
    subroutine_name : str, optional
        Name of the generated subroutine.  Defaults to the base name.
    strict : bool, optional
        When ``True``, raise a ``ValueError`` if the expression contains
        internal functions or unresolved derivatives instead of exposing
        them as input arguments.  By default ``False``.
    integer_args : list of str, optional
        Names of extra arguments that must be declared as
        ``integer, intent(in)`` (for example, extra dimensions).
    module_symbols : list of str, optional
        Names that are provided by a ``use`` statement of the enclosing
        module.  They are neither declared as arguments nor included in
        the subroutine signature.
    optional_derivatives : bool, optional
        When ``True``, every output - the function value included - is
        declared as a Fortran ``optional`` argument and computed only
        when the caller passes it.  A shared temporary is skipped
        whenever none of the outputs that need it is present, so calling
        the subroutine with a subset of the outputs also skips the work.
    array_cse : bool, optional
        When ``True`` (the default), index-dependent sub-expressions that
        several outputs share are materialised into array temporaries
        computed a single time (see :func:`_indexed_cse`).  Only
        sub-expressions of rank up to two are materialised, because a
        larger temporary costs more than it saves.  Disable the pass to
        obtain the plain, fully inlined expressions.
    share : bool, optional
        When ``True`` (the default), summations and common scalar
        sub-expressions shared by several outputs are computed a single
        time as internal temporaries (:func:`_share_sums` and
        :func:`_common_scalar_cse`).  Set it to ``False`` to disable both
        sharing passes: every output is then emitted from its own fully
        inlined expressions and repeated summations are re-evaluated per
        output.  The benchmark variant "V1 none" is
        ``array_cse=False, share=False``; "V2 partial" is
        ``array_cse=False``; "V3 full" is ``array_cse=True``.  The flag
        only changes the generated body: the signature, the declaration
        block and the output order are identical for every setting.
    stats : dict, optional
        When given, it is filled in place with generation statistics:
        ``operation_count`` (``sympy.count_ops`` summed over the final
        output expressions and over the body of every shared temporary),
        ``operation_count_tool`` (the exact tool/options used) and
        ``n_temporaries`` (number of shared temporaries emitted).  The
        count is a symbolic-operation count of the final expressions, not
        a runtime estimate.
    share_sums, scalar_cse : bool, optional
        Fine-grained control of the two passes enabled by ``share``:
        :func:`_share_sums` and the scalar common sub-expression pass.
        ``None`` (the default) inherits the value of ``share``.  Used by the
        ablation study.
    symmetric_params : iterable of str, optional
        Names of rank-2 parameters known to be symmetric (declared with the
        ``symmetric`` attribute in a ``.model`` file).
    param_symmetry : bool, optional
        Exploit ``symmetric_params``: canonicalise their subscripts and
        fold symmetric double summations into a triangular loop (see
        :func:`_fold_symmetric_sums`).  Off by default.
    hessian_symmetry : bool, optional
        Compute only the upper triangle of the composition Hessian
        ``dfdndn`` (``j >= i``) and mirror it.  Exact by Schwarz' theorem.
        Off by default.
    cse_backend : {"thermodiff", "sympy"}, optional
        Algorithm of the scalar common sub-expression pass.
    sum_alpha : bool, optional
        Share summations that are equal up to the names of their bound
        indices (alpha equivalence), not only syntactically identical ones.
        Off by default.
    array_pad : int, optional
        Extra rows added to the leading dimension of rank-2 array
        temporaries (``t(nc + pad, nc)``).  Zero (the default) keeps
        ``t(nc, nc)``.
    hoist_invariants : bool, optional
        Move costly sub-expressions that do not depend on every bound index
        of their summation out of the loop, into temporaries of lower rank
        (see :func:`_hoist_invariants`).  Off by default.
    chain_rule : bool, optional
        Differentiate by the chain rule over the summation nodes of the
        function, as a hand-written implementation does (see
        :mod:`thermodiff.chain`).  Falls back to the plain derivatives when
        the function does not fit the pattern.  Off by default.
    hoist_loops : bool, optional
        Group the terms of every loop body over free indices by their
        index-dependent factors and compute the loop invariants once (see
        :func:`_hoist_loop_invariants`).  Off by default.
    sum_free_alpha : bool, optional
        Share summations equal up to the names of their bound *and free*
        indices (``sum_k x(i,k)`` and ``sum_l x(j,l)``).  Off by default.
    fuse_temps : bool, optional
        Do not materialise cheap rank-2 array temporaries that are only
        read inside summations: their body is inserted into the summation
        (see :func:`_fuse_temporaries`).  Off by default.
    nest_sums : bool, optional
        Rewrite double sums as dot products with vector sums that other
        outputs already need (see :func:`_nest_double_sums`).  Off by
        default.
    split_sqrt : bool, optional
        With ``chain_rule``: split ``sqrt(c*A_a*A_b)`` into
        ``sqrt(c)*sqrt(A_a)*sqrt(A_b)`` for per-component temperature
        functions.  Assumes they are non-negative (true for the attractive
        parameters of cubic equations).  Off by default.
    split_sums : bool, optional
        With ``chain_rule``: expand summation bodies and split sums of terms
        into sums, so that each term can become a dot product with a vector
        sum.  Off by default.
    nest_shared : bool, optional
        With ``nest_sums``: also nest double sums whose inner sum is shared
        by several double sums (not only by an output), and count equal
        summations once when deciding whether to fuse a temporary.  Off by
        default.
    sym_matvec : bool, optional
        With ``param_symmetry``: emit the vector sums ``v(f) = sum_b
        P(b,f) R(b)`` with a symmetric ``P`` as a symmetric matrix-vector
        product that reads only the upper triangle.  Off by default.
    col_major : bool, optional
        Order the loops of vector sums and rank-2 tables so that the arrays
        they read are traversed along columns (see
        :func:`_prefers_column_order`).  Off by default.
    pair_cse, pair_rates : bool, optional
        With ``chain_rule``: write pair functions in terms of earlier ones
        (``tau*exp(-c tau)`` as ``tau*M``), and write the temperature
        derivatives of pair functions with a cheap pair-dependent
        logarithmic rate in place (``M' = M*r_ab``).  Off by default.
    sum_transpose : bool, optional
        With ``sum_free_alpha``: reuse a rank-2 summation that is the
        transpose of another one.  Off by default.
    sep_sums : bool, optional
        Distribute products over sums in summation bodies, split the sums
        and move out the factors that do not depend on the bound index (see
        :func:`_separate_sums`).  Off by default.
    sym_rank3 : bool, optional
        With ``param_symmetry``: also canonicalise symmetric parameters of
        rank three or more (fully symmetric tensors).  Off by default.
    nest_general : bool, optional
        With ``nest_sums``: also nest sums over three or more indices, and
        use rank-2 summations as inner sums.  Off by default.
    sym_sum2 : bool, optional
        With ``param_symmetry``: a rank-2 summation whose body is symmetric
        in its two free indices is computed on the upper triangle and
        mirrored.  Off by default.
    cse_expensive : bool, optional
        The indexed CSE extracts a costly single-operation sub-expression
        (``log``, ``exp``, a root) already when it is used twice.  Off by
        default.
    fuse_loops : bool, optional
        Merge consecutive vector sums with the same loop structure into one
        loop nest (see :func:`_fuse_loop_nests`); only without optional
        outputs.  Off by default.
    nested_nodes : bool, optional
        With ``chain_rule``: index-free summations nested inside a node
        become nodes too, and derivatives are composed through them (see
        :func:`thermodiff.chain._nest_inner_nodes`).  Off by default.
    sym_perm : bool, optional
        Rewrite the subscripts of ``symmetric`` parameters inside loop nests
        so that the innermost loop index comes first (contiguous access).
        Exact by symmetry; applied after every other pass.  Off by default.
    fuse_guard : bool, optional
        With ``fuse_temps``: never inline an expensive rank-2 temporary
        (transcendental body) that two or more different summations read,
        nested ones included.  Off by default.
    nest_cost : bool, optional
        With ``nest_sums`` and ``nest_general``: nest every sum over three or
        more indices into an inner rank-2 table, even when no other output
        shares it.  Off by default.
    comp_fuse : bool, optional
        Fuse consecutive per-component maps (``cmp0``, ``cmp0_T``, ...) into
        one loop with a per-element common sub-expression elimination.
        Only without ``optional_derivatives``.  Off by default.
    dummy_canon : bool, optional
        Merge the terms of a summation over two or three indices that are
        images of each other under a renaming of the bound indices (e.g.
        ``(b_k + b_l + b_m)`` with a symmetric tensor).  Exact.  Off by
        default.
    float_canon : bool, optional
        Round every floating-point constant to its double value with a single
        representation before the common sub-expression passes.  Off by
        default.
    contiguous_args : bool, optional
        Declare every assumed-shape array argument (parameters, mole numbers
        and outputs) ``contiguous``, so the compiler can drop the stride
        arithmetic and the multi-versioned loops.  A caller that passes a
        non-contiguous section gets a copy-in/copy-out (still correct).
        Off by default.
    fuse_outer : bool, optional
        Fuse consecutive per-component maps and scalar reductions that share
        the outer loop ``1..nc`` into one loop.  Only without
        ``optional_derivatives``.  Off by default.
    fuse_libm_guard : bool, optional
        With ``fuse_outer``, never fuse a double reduction into the outer loop
        of a map that calls ``exp``, ``log``, ``sqrt`` or a real power, so the
        map stays vectorisable.  Off by default.
    keep_libm_tables : bool, optional
        With ``fuse_temps``, never inline a rank-2 temporary that calls a libm
        routine into a summation; it stays a table computed by its own map
        loop.  Off by default.
    table_layout : bool, optional
        With ``chain_rule``, store a non-symmetric pair function in the
        subscript order of the parameters it reads (``G(l, k)`` for
        ``exp(-c_lk tau_lk)``) instead of the alphabetical order of the
        summation indices, so the table is written contiguously and the sums
        that read it take the contiguous inner index; with ``col_major`` a
        rank-2 sum that then reads its tables with the bound index last
        (``sum_k G(i, k) H(j, k)``) is emitted in ``jki`` order.  Off by
        default.
    lowrank_hess : bool or "sym", optional
        Assemble a rank-2 output (the composition Hessian) whose residual has
        separable terms ``c F(i) G(j)`` in rank-k form: the terms are grouped
        by the factor over one index, ``sum_l r_l(i) G_l(j)``, with the
        coefficient vectors ``r_l`` computed once (O(nc)); terms over one
        index and pair terms are kept.  Applied only when it lowers the
        per-element operation count.  ``True`` emits the full square in
        column order without mirror; ``"sym"`` keeps the triangle and mirror
        of ``hessian_symmetry``.  Only without ``optional_derivatives``.
        Off by default.
    drop_pair_tables : bool, optional
        Do not materialise a rank-2 table read only by scalar reductions (one
        double, triangular reduction and its diagonal sums) whose body calls
        no libm routine: its body is inlined there.  In the scalar double
        reductions the factors over the outer index are applied once per
        outer iteration and the products over the inner index only become
        vectors.  With ``fuse_libm_guard`` a single reduction that calls a
        libm routine does not share the outer loop of a double reduction
        either.  Only without ``optional_derivatives``.  Off by default.
    share_elementwise : bool, optional
        A costly rank-1 sub-expression (``log``, ``exp``, a root of a
        per-component expression) read in two or more places -- the body of a
        scalar reduction over ``k`` and an output over ``i`` -- is computed
        once as a vector that every place reads.  Occurrences are compared up
        to the index name and to the spelling of their scalar prefactors
        (``x/y`` and ``(1/y)*x``); differently spelt scalar temporaries of the
        shared occurrences are merged.  Only without
        ``optional_derivatives``.  Off by default.
    scalar_chain : bool, optional
        With ``chain_rule``, compute the partial derivatives of the outer
        function ``F(S_1, ..., S_m, V, T)`` by forward second-order
        propagation over its expression DAG, followed by one common
        sub-expression elimination, instead of differentiating the expanded
        ``F`` (see :func:`thermodiff.chain._scalar_chain`).  They become
        scalar temporaries computed after the reductions they read, and the
        outputs read them.  Falls back to the plain path when ``F`` is not
        smooth (``Piecewise``, ``Max`` ...).  ``"cost"`` generates the kernel
        both ways and keeps the one with the lower ``operation_count``
        (``stats["scalar_chain_used"]`` tells which).  With ``"estimate"``
        the choice is made before the array pipeline, from the operation
        counts of the plain partials of ``F`` (after one CSE) and of the
        propagated definitions, so the pipeline runs once.  Only without
        ``optional_derivatives``.  Off by default.
    split_affine : bool, optional
        In a summation with free indices, ``sum_m (c - P(m, k, l)) w(m)``
        with ``c`` and ``w`` independent of the free indices becomes
        ``c*S - sum_m P w``: the scalar ``S = sum_m w(m)`` is computed once
        and the readers of the table read ``S - t``, so the innermost body is
        one multiply-add; products of factors over the bound index only
        become vectors (see :func:`_split_affine`).  Applied only when the
        body gets cheaper; only without ``optional_derivatives``.  Off by
        default.
    split_acc : int or bool, optional
        ``2``: in the inner loop of a symmetric rank-2 table filled by a sum
        that reads a parameter of rank three (the O(N^3) nest of CMR), use two
        scalar partial accumulators per sum (even and odd ``m``, one remainder
        iteration, then their sum), so that gcc vectorises the reduction
        without ``-ffast-math`` (reportes/brechas/ronda3-cmr-o3.md).  Changes
        the summation order.  Only without ``optional_derivatives``.  Off by
        default.

    Returns
    -------
    str
        The free-form Fortran source code.
    """
    if scalar_chain in ("cost", "estimate") and not (
        chain_rule and not optional_derivatives
    ):
        scalar_chain = False
    if scalar_chain == "estimate":
        # one pass of the array pipeline: the path is chosen from the operation
        # estimates of the two scalar blocks made by the chain-rule step
        settings = dict(locals())
        structured_ = to_fortran(
            **dict(settings, scalar_chain=True, _structure_only=True)
        )
        estimate = (structured_ or {}).get("_scalar_estimate")
        # ties and near ties go to propagation: after the scalar CSE of the
        # pipeline it is the cheaper path in every case checked (7 models)
        use = estimate is not None and estimate[1] <= 1.1 * estimate[0]
        if use:
            code = to_fortran(
                **dict(settings, scalar_chain=True, _structured=structured_)
            )
        else:
            plain_ = structured_
            if structured_ is not None and "_plain" in structured_:
                plain_ = dict(structured_["_plain"])
                plain_["_component_defs"] = structured_["_component_defs"]
            code = to_fortran(
                **dict(settings, scalar_chain=False, _structured=plain_)
            )
        if stats is not None:
            stats["scalar_chain_used"] = use
            stats["scalar_chain_estimate"] = estimate
        return code
    if scalar_chain == "cost":
        # keep the path whose scalar block (definitions and outputs without
        # loops) has fewer operations; the loops are the same by construction
        settings = dict(locals())
        stats_sc = {}
        code_sc = to_fortran(
            **dict(settings, scalar_chain=True, stats=stats_sc)
        )
        structured_ = stats_sc.pop("_structured", None)
        plain_, propagated = structured_, False
        if structured_ is not None and "_plain" in structured_:
            plain_ = dict(structured_["_plain"])
            plain_["_component_defs"] = structured_["_component_defs"]
            propagated = True
        stats_plain = {}
        code_plain = to_fortran(
            **dict(
                settings,
                scalar_chain=False,
                stats=stats_plain,
                _structured=plain_,
            )
        )
        stats_plain.pop("_structured", None)
        use = (
            propagated
            and stats_sc["scalar_operation_count"]
            < stats_plain["scalar_operation_count"]
        )
        chosen_code, chosen_stats = (
            (code_sc, stats_sc) if use else (code_plain, stats_plain)
        )
        if stats is not None:
            stats.update(chosen_stats)
            stats["scalar_chain_used"] = use
        return chosen_code
    module_symbols = set(module_symbols or ())
    integer_args = list(integer_args or [])

    base = _sanitize(name if name is not None else diff.name)
    sub_name = (
        _sanitize(subroutine_name) if subroutine_name else base + "_derivs"
    )

    if derivatives is None:
        candidates = getattr(diff, "derivatives", None)
        if candidates is None:
            candidates = _DERIV_ORDER
        selected = []
        for attr in _DERIV_ORDER:
            if attr not in candidates:
                continue
            value = getattr(diff, attr, None)
            if value is None or value == 0:
                continue
            selected.append(attr)
    else:
        resolved = resolve_derivatives(derivatives)
        selected = [attr for attr in _DERIV_ORDER if attr in resolved]
        missing = [
            attr for attr in selected if getattr(diff, attr, None) is None
        ]
        if missing:
            raise ValueError(
                "Derivatives not computed by the DiffPlz instance: "
                + ", ".join(missing)
                + ". Rebuild it with derivatives="
                + repr(list(resolved))
                + "."
            )

    resolver = _PlaceholderResolver(strict=strict)

    structured = None
    if chain_rule and _structured is not None:
        structured = _structured
    elif chain_rule:
        from thermodiff.chain import chain_rule_derivatives

        structured = chain_rule_derivatives(
            diff.expression,
            selected,
            diff.indexes,
            symmetric_params=tuple(symmetric_params or ()),
            split_sqrt=split_sqrt,
            split_sums=split_sums,
            pair_cse=pair_cse,
            pair_rates=pair_rates,
            nested_nodes=nested_nodes,
            table_layout=table_layout,
            scalar_chain=scalar_chain and not optional_derivatives,
        )

    if _structure_only:
        return structured

    def _derivative(attr):
        if structured is not None:
            return structured[attr]
        return getattr(diff, attr)

    value_expr = (
        structured["value"] if structured is not None else diff.expression
    )
    outputs = [("value", base, resolver.resolve(value_expr))]
    for attr in selected:
        outputs.append(
            (
                attr,
                "d" + base + _DERIV_SUFFIX[attr],
                resolver.resolve(_derivative(attr)),
            )
        )
    namer = _Namer()
    scalar_chain_defs = (structured or {}).get("_scalar_defs", ())
    sc_temp_names = set()
    if scalar_chain_defs:
        # the summation nodes read by the scalars of the outer function travel
        # with the outputs (one pseudo-output each, so the summation passes see
        # them as before); the scalars become definitions after the scalar CSE
        sc_nodes = structured.get("_scalar_nodes", ())
        sc_names = {
            symbol: namer("t")
            for symbol, _ in list(sc_nodes) + list(scalar_chain_defs)
        }
        outputs = [
            (attr, out_name, expr.xreplace(sc_names))
            for attr, out_name, expr in outputs
        ]
        sc_temp_names = {str(name) for name in sc_names.values()}
        for symbol, body in sc_nodes:
            outputs.append(
                (_SCALAR_CHAIN, str(sc_names[symbol]), resolver.resolve(body))
            )
        # the outputs without scalar_chain ride along (then dropped), so that
        # the summation passes see the same expressions as without the option
        # (first, so the shared summations come in the same order)
        plain = structured.get("_plain", {})
        outputs = [
            (_SCALAR_PLAIN, "_plain_" + attr, resolver.resolve(plain[attr]))
            for attr in ["value"] + list(selected)
            if attr in plain
        ] + outputs
        scalar_chain_defs = [
            (sc_names[symbol], resolver.resolve(body.xreplace(sc_names)))
            for symbol, body in scalar_chain_defs
        ]

    if cse_backend not in ("thermodiff", "sympy"):
        raise ValueError(f"unknown cse_backend {cse_backend!r}")
    if share_sums is None:
        share_sums = share
    if scalar_cse is None:
        scalar_cse = share
    symmetric_params = tuple(symmetric_params or ())
    if structured is not None:
        # symmetric pair functions behave as symmetric parameters
        symmetric_params = symmetric_params + tuple(
            name
            for name, _, rank, sym in structured.get("_component_defs", ())
            if rank == 2 and sym
        )

    _SYM_RANK3[0] = bool(sym_rank3)
    staged = []
    for attr, out_name, expr in outputs:
        expr = _kron_to_piecewise(expr)
        expr = expr.xreplace({nc: sp.Symbol("nc")})
        if float_canon:
            expr = _canonical_floats(expr)
        if param_symmetry and symmetric_params:
            expr = _canonical_symmetric(expr, symmetric_params)
        if sep_sums:
            expr = _separate_sums(expr, selective=(sep_sums == "cost"))
        staged.append((attr, out_name, expr))
    if dummy_canon:
        # arrays of rank >= 2 read inside sums with free indices by some output
        shared_bases = set()
        for _attr, _out, expr in staged:
            for total in expr.atoms(sp.Sum):
                if _free_indices(total):
                    shared_bases |= {
                        str(x.base)
                        for x in total.function.atoms(sp.Indexed)
                        if len(x.indices) >= 2
                    }
        staged = [
            (
                attr,
                out_name,
                _merge_dummy_images(
                    expr, symmetric_params, frozenset(shared_bases)
                ),
            )
            for attr, out_name, expr in staged
        ]
    if nest_sums:
        staged = _nest_double_sums(
            staged,
            diff.indexes,
            symmetric_params,
            shared_inner=nest_shared,
            general=nest_general,
            cost=nest_cost,
        )
    prepared = []
    for attr, out_name, expr in staged:
        if param_symmetry and symmetric_params:
            expr = _fold_symmetric_sums(expr, symmetric_params)
        free_order = _order_free(_free_indices(expr), diff.indexes)
        prepared.append((attr, out_name, expr, free_order))

    all_exprs = [expr for _, _, expr, _ in prepared]
    component_defs = []
    if structured is not None:
        for comp in structured.get("_component_defs", ()):
            comp_name, comp_body, comp_rank, comp_sym = comp
            comp_body = comp_body.xreplace({nc: sp.Symbol("nc")})
            if param_symmetry and symmetric_params:
                comp_body = _canonical_symmetric(comp_body, symmetric_params)
            component_defs.append((comp_name, comp_body, comp_rank, comp_sym))
    component_names = {c[0] for c in component_defs}
    scan_exprs = (
        all_exprs
        + [c[1] for c in component_defs]
        + [body for _, body in scalar_chain_defs]
    )
    placeholder_names = set(resolver.ranks)

    # ---- gather parameters --------------------------------------------
    # NOTE: ``n`` is compared by name because ``expr`` has been rebuilt by
    # the ``nc`` replacement above, which also rewrites the shape of the
    # ``n`` IndexedBase.
    n_name = str(n)
    uses_n = any(
        n_name in {str(base_) for base_ in expr.atoms(sp.IndexedBase)}
        for expr in scan_exprs
    )

    index_syms = set()
    indexed_bases = set()
    for expr in scan_exprs:
        for indexed in expr.atoms(sp.Indexed):
            index_syms.update(indexed.indices)
        for total in expr.atoms(sp.Sum):
            for limit in total.limits:
                index_syms.add(limit[0])
        index_syms.update(expr.atoms(sp.Idx))
        indexed_bases.update(expr.atoms(sp.IndexedBase))

    index_names = {str(idx) for idx in index_syms}
    base_names = {str(base_) for base_ in indexed_bases}

    param_scalars = set()
    for expr in scan_exprs:
        for sym in expr.atoms(sp.Symbol):
            name = str(sym)
            if name in index_names or name in base_names:
                continue
            if sym in (T, V, P) or name == "nc":
                continue
            if isinstance(sym, _LoopStart):
                continue
            if name in module_symbols or name in integer_args:
                continue
            if name in sc_temp_names:
                continue
            param_scalars.add(name)

    array_ranks = {}
    for expr in scan_exprs:
        for indexed in expr.atoms(sp.Indexed):
            base_name = str(indexed.base)
            rank = len(indexed.indices)
            array_ranks[base_name] = max(array_ranks.get(base_name, 0), rank)
    param_arrays = {
        base_name: rank
        for base_name, rank in array_ranks.items()
        if base_name != n_name
        and base_name not in placeholder_names
        and base_name not in component_names
    }
    if uses_n:
        param_arrays[n_name] = 1

    params = sorted(param_scalars - placeholder_names)
    placeholder_args = sorted(resolver.ranks)

    needs_nc = (
        uses_n
        or bool(param_arrays)
        or any(free for _, _, _, free in prepared)
        or any(sp.Symbol("nc") in expr.atoms(sp.Symbol) for expr in scan_exprs)
    )

    # ---- input declarations -------------------------------------------
    inputs = []
    if needs_nc:
        inputs.append("integer, intent(in) :: nc")
    if integer_args:
        inputs.append("integer, intent(in) :: " + ", ".join(integer_args))
    for var, label in ((V, "V"), (P, "P"), (T, "T")):
        if any(expr.has(var) for expr in scan_exprs):
            inputs.append(f"real(8), intent(in) :: {label}")
    for param in params:
        inputs.append(f"real(8), intent(in) :: {param}")
    in_array = (
        "real(8), contiguous, intent(in)"
        if contiguous_args
        else "real(8), intent(in)"
    )
    for arr in sorted(param_arrays):
        dims = ", ".join([":"] * param_arrays[arr])
        inputs.append(f"{in_array} :: {arr}({dims})")
    for arg in placeholder_args:
        rank = resolver.ranks[arg]
        if rank == 0:
            inputs.append(f"real(8), intent(in) :: {arg}")
        else:
            dims = ", ".join([":"] * rank)
            inputs.append(f"{in_array} :: {arg}({dims})")

    # ---- output declarations ------------------------------------------
    output_decls = [
        _declare_output(
            out_name,
            len(free_order),
            optional=optional_derivatives,
            contiguous=contiguous_args,
        )
        for _attr, out_name, _, free_order in prepared
        if _attr not in (_SCALAR_CHAIN, _SCALAR_PLAIN)
    ]

    # ---- body ----------------------------------------------------------
    arrays = {}
    component_array_defs = []
    if component_defs:
        from thermodiff.chain import COMPONENT_SLOT, PAIR_SLOTS

        for comp_name, comp_body, comp_rank, comp_sym in component_defs:
            arrays[comp_name] = comp_rank
            free_ = [COMPONENT_SLOT] if comp_rank == 1 else list(PAIR_SLOTS)
            base_ = sp.IndexedBase(comp_name, shape=(nc,) * comp_rank)
            component_array_defs.append(
                {
                    "target": sp.Indexed(base_, *free_),
                    "free": free_,
                    "limits": [],
                    "body": (
                        _canonical_floats(comp_body)
                        if float_canon
                        else comp_body
                    ),
                    "mirror": bool(comp_sym and param_symmetry),
                }
            )
    if array_cse:
        all_exprs, array_defs = _indexed_cse(
            all_exprs,
            namer,
            arrays,
            expensive_twice=cse_expensive,
            # with scalar_chain, only the plain outputs count (as without it)
            counted=(
                [entry[0] == _SCALAR_PLAIN for entry in prepared]
                if scalar_chain_defs
                else None
            ),
        )
    else:
        array_defs = []
    if param_symmetry and symmetric_params:
        for definition in array_defs:
            free = definition["free"]
            if len(free) != 2 or definition["limits"]:
                continue
            body = _canonical_symmetric(definition["body"], symmetric_params)
            if _swap(body, free[0], free[1], symmetric_params) == body:
                definition["mirror"] = True
                if nest_shared:
                    # a symmetric temporary behaves as a symmetric parameter
                    # for the later passes (sum sharing modulo renaming)
                    symmetric_params = symmetric_params + (
                        _temp_name(definition["target"]),
                    )
    reserved = (
        set(module_symbols)
        | set(integer_args)
        | {out_name for _, out_name, _, _ in prepared}
    )
    for expr in all_exprs:
        reserved.update(str(sym) for sym in expr.atoms(sp.Symbol))
        reserved.update(str(b) for b in expr.atoms(sp.IndexedBase))
    if fuse_temps:
        n_comp = len(component_array_defs)
        pool = component_array_defs + array_defs
        pool, all_exprs, fused = _fuse_temporaries(
            pool,
            all_exprs,
            dedup=nest_shared,
            guard=fuse_guard,
            keep_libm=keep_libm_tables,
        )
        component_array_defs = [d for d in pool if d in component_array_defs]
        array_defs = [d for d in pool if d not in component_array_defs]
        for name in fused:
            arrays.pop(name, None)
    if hoist_invariants:
        all_exprs, hoist_defs, _ = _hoist_invariants(
            all_exprs, namer, arrays, reserved=reserved
        )
        array_defs = array_defs + hoist_defs
    if share_sums:
        residuals, sum_defs = _share_sums(
            all_exprs,
            namer,
            diff.indexes,
            arrays,
            alpha=sum_alpha,
            free_alpha=sum_free_alpha,
            symmetric_params=symmetric_params if param_symmetry else (),
            transpose=sum_transpose,
        )
    else:
        residuals = list(all_exprs)
        sum_defs = []
    if hoist_invariants and sum_defs:
        _, _, sum_defs = _hoist_invariants(
            [], namer, arrays, sum_defs, reserved=reserved
        )
    if scalar_cse:
        if cse_backend == "sympy":
            residuals, cse_defs = _sympy_scalar_cse(residuals, namer, arrays)
        else:
            residuals, cse_defs = _common_scalar_cse(residuals, namer, arrays)
    else:
        cse_defs = []
    scalar_chain_definitions = []
    if scalar_chain_defs:
        keep = [
            k
            for k, entry in enumerate(prepared)
            if entry[0] not in (_SCALAR_CHAIN, _SCALAR_PLAIN)
        ]
        alias = {}
        for (attr, out_name, _, _), residual in zip(prepared, residuals):
            if attr != _SCALAR_CHAIN:
                continue
            if isinstance(residual, sp.Symbol):
                alias[sp.Symbol(out_name)] = residual
                namer.created[:] = [
                    x for x in namer.created if str(x) != out_name
                ]
            else:
                scalar_chain_definitions.append(
                    {
                        "target": sp.Symbol(out_name),
                        "free": [],
                        "limits": [],
                        "body": residual,
                    }
                )
        for target, body in scalar_chain_defs:
            body = (
                _kron_to_piecewise(body)
                .xreplace({nc: sp.Symbol("nc")})
                .xreplace(alias)
            )
            if float_canon:
                body = _canonical_floats(body)
            scalar_chain_definitions.append(
                {"target": target, "free": [], "limits": [], "body": body}
            )
        prepared = [prepared[k] for k in keep]
        residuals = [residuals[k] for k in keep]
        residuals = [residual.xreplace(alias) for residual in residuals]
        # drop what only the plain outputs read
        pools = [component_array_defs, array_defs, sum_defs, cse_defs]
        while True:
            read = set()
            for e in (
                residuals
                + [d["body"] for pool in pools for d in pool]
                + [d["body"] for d in scalar_chain_definitions]
            ):
                read |= {str(x) for x in e.free_symbols} | {
                    str(x.base) for x in e.atoms(sp.Indexed)
                }
            gone = {
                _temp_name(d["target"]) for pool in pools for d in pool
            } - read
            if not gone:
                break
            pools = [
                [d for d in pool if _temp_name(d["target"]) not in gone]
                for pool in pools
            ]
            namer.created[:] = [x for x in namer.created if str(x) not in gone]
            for name in gone:
                arrays.pop(name, None)
        component_array_defs, array_defs, sum_defs, cse_defs = pools
    rendered = [
        (
            attr,
            out_name,
            residual,
            _order_free(_free_indices(residual), diff.indexes),
        )
        for (attr, out_name, _, _), residual in zip(prepared, residuals)
    ]

    definitions = component_array_defs + array_defs + sum_defs + cse_defs
    if scalar_chain_definitions:
        # drop the scalars nothing reads
        while True:
            read = set()
            for e in residuals + [
                d["body"] for d in definitions + scalar_chain_definitions
            ]:
                read |= {str(x) for x in e.free_symbols}
            unread = [
                d
                for d in scalar_chain_definitions
                if str(d["target"]) not in read
            ]
            if not unread:
                break
            gone = {str(d["target"]) for d in unread}
            scalar_chain_definitions = [
                d
                for d in scalar_chain_definitions
                if str(d["target"]) not in gone
            ]
            namer.created[:] = [x for x in namer.created if str(x) not in gone]
        definitions = _dependency_order(definitions + scalar_chain_definitions)
    lowrank_outs = set()
    if lowrank_hess and not optional_derivatives:
        for position, (attr, out_name, residual, free) in enumerate(rendered):
            if len(free) != 2 or residual.has(sp.Sum):
                continue
            result = _lowrank_bilinear(residual, free, namer, arrays)
            if result is None:
                continue
            residual, new_defs = result
            definitions = definitions + new_defs
            rendered[position] = (attr, out_name, residual, free)
            residuals[position] = residual
            lowrank_outs.add(attr)
    if hoist_loops:
        definitions, new_residuals = _hoist_loop_invariants(
            definitions,
            [(residual, free) for _, _, residual, free in rendered],
            namer,
            arrays,
            reserved=reserved,
        )
        rendered = [
            (attr, out_name, residual, free)
            for (attr, out_name, _, free), residual in zip(
                rendered, new_residuals
            )
        ]
        residuals = new_residuals
    if drop_pair_tables and not optional_derivatives:
        definitions = _drop_pair_tables(
            definitions,
            [residual for _, _, residual, _ in rendered],
            namer,
            arrays,
        )
    if share_elementwise and not optional_derivatives:
        definitions, residuals = _share_elementwise(
            definitions,
            [residual for _, _, residual, _ in rendered],
            namer,
            arrays,
        )
        rendered = [
            (attr, out_name, residual, free)
            for (attr, out_name, _, free), residual in zip(rendered, residuals)
        ]
    if split_affine and not optional_derivatives:
        definitions, residuals = _split_affine(
            definitions,
            [residual for _, _, residual, _ in rendered],
            namer,
            arrays,
        )
        rendered = [
            (attr, out_name, residual, free)
            for (attr, out_name, _, free), residual in zip(rendered, residuals)
        ]
    if sym_sum2 and param_symmetry:
        for definition in definitions:
            free_ = definition["free"]
            if len(free_) != 2 or not definition["limits"]:
                continue
            body_ = _canonical_symmetric(definition["body"], symmetric_params)
            if body_.has(sp.Piecewise) or any(
                isinstance(x, _LoopStart) for x in body_.atoms(sp.Symbol)
            ):
                continue
            if _swap(body_, free_[0], free_[1], symmetric_params) == body_:
                definition["mirror_sum"] = True
    if col_major:
        for definition in definitions:
            if table_layout and _prefers_column_sum(
                definition, symmetric_params
            ):
                definition["colsum"] = True
                continue
            if definition.get("mirror_sum"):
                continue
            if not definition.get("symv") and _prefers_column_order(
                definition
            ):
                definition["colmajor"] = True
    if fuse_loops and not optional_derivatives:
        definitions = _fuse_loop_nests(definitions, namer)
    if sym_matvec and param_symmetry:
        for definition in definitions:
            parts = _symv_parts(definition, symmetric_params)
            if parts is not None:
                definition["symv"] = parts
                definition["symv_acc"] = namer("t")
    if sib_fuse and not optional_derivatives:
        definitions = _fuse_sibling_tables(definitions, namer)
    if diag_table and not optional_derivatives:
        definitions = _diagonal_from_tables(definitions, symmetric_params)
    if sym_perm and param_symmetry:
        definitions = _permute_symmetric_subscripts(
            definitions, symmetric_params
        )
    if comp_fuse and not optional_derivatives:
        definitions = _fuse_component_maps(definitions, namer)
    if fuse_outer and not optional_derivatives:
        definitions = _fuse_outer_loops(
            definitions,
            libm_guard=fuse_libm_guard,
            libm_reductions=fuse_libm_guard and drop_pair_tables,
        )
    if split_acc and not optional_derivatives:
        if int(split_acc) != 2:
            raise ValueError("split_acc supports 2 accumulators only")
        definitions = _split_accumulators(definitions, namer)
    if stats is not None:
        stats["operation_count"] = sum(
            sp.count_ops(residual, visual=False) for residual in residuals
        ) + sum(
            sp.count_ops(definition["body"], visual=False)
            for definition in definitions
        )
        stats["operation_count_tool"] = "sympy.count_ops(visual=False)"
        stats["n_temporaries"] = len(definitions)
    if stats is not None and (scalar_chain or _structured is not None):
        stats["scalar_operation_count"] = sum(
            sp.count_ops(residual, visual=False)
            for (_a, _o, _r, free_), residual in zip(rendered, residuals)
            if not free_
        ) + sum(
            sp.count_ops(definition["body"], visual=False)
            for definition in definitions
            if not definition["free"] and not definition["limits"]
        )
        if scalar_chain is True:
            stats["_structured"] = structured
    out_names = {attr: out_name for attr, out_name, _, _ in rendered}

    temp_names = {
        _temp_name(definition["target"]) for definition in definitions
    }
    consumers = {name: set() for name in temp_names}
    for attr, _out_name, residual, _free in rendered:
        for name in _referenced_temp_names(residual, temp_names):
            consumers[name].add(attr)
    for definition in reversed(definitions):
        name = _temp_name(definition["target"])
        for dependency in _referenced_temp_names(
            definition["body"], temp_names
        ):
            consumers[dependency] |= consumers[name]

    body = []
    for definition in definitions:
        if optional_derivatives:
            condition = _guard(
                consumers[_temp_name(definition["target"])], out_names
            )
        else:
            condition = ""
        body.extend(_guarded(_render_definition(definition), condition))
        body.append("")
    for attr, out_name, expr, free_order in rendered:
        if optional_derivatives:
            condition = f"present({out_name})"
        else:
            condition = ""
        mirror = (
            hessian_symmetry
            and attr == "dnidnj"
            and (attr not in lowrank_outs or lowrank_hess == "sym")
        )
        column = attr in lowrank_outs and lowrank_hess != "sym"
        body.extend(
            _guarded(
                _render_output(
                    out_name,
                    expr,
                    free_order,
                    namer,
                    mirror=mirror,
                    column=column,
                ),
                condition,
            )
        )
        body.append("")

    temps = sorted(
        {str(t) for t in namer.created if str(t) not in arrays} | {"res"}
    )
    loop_indices = sorted(
        {str(idx) for _, _, _, free in rendered for idx in free}
        | {
            str(idx)
            for definition in definitions
            for idx in definition["free"]
        }
        | {
            str(limit[0])
            for definition in definitions
            for limit in definition["limits"]
        }
        | {
            str(limit[0])
            for _attr, _out, residual, _free in rendered
            for total in residual.atoms(sp.Sum)
            for limit in total.limits
        }
    )

    # ---- assemble ------------------------------------------------------
    arg_names = []
    if needs_nc:
        arg_names.append("nc")
    arg_names.extend(integer_args)
    for var, label in ((V, "V"), (P, "P"), (T, "T")):
        if any(expr.has(var) for expr in scan_exprs):
            arg_names.append(label)
    arg_names.extend(params)
    arg_names.extend(sorted(param_arrays))
    arg_names.extend(placeholder_args)
    arg_names.extend(out_name for _, out_name, _, _ in prepared)

    while sub_name in arg_names:
        sub_name += "_s"

    lines = _wrap_signature(sub_name, arg_names)
    lines.append(f"{_INDENT}implicit none")
    for decl in inputs:
        lines.append(_INDENT + decl)
    for decl in output_decls:
        lines.append(_INDENT + decl)

    for line in _wrap_name_list("real(8) :: ", temps):
        lines.append(_INDENT + line)
    for rank in sorted(set(arrays.values())):
        dims = ", ".join(["nc"] * rank)
        if array_pad and rank >= 2:
            # padded leading dimension: avoids cache-set aliasing between the
            # many (nc, nc) temporaries for some values of nc
            dims = ", ".join([f"nc + {int(array_pad)}"] + ["nc"] * (rank - 1))
        names = [
            f"{name}({dims})"
            for name in sorted(arrays)
            if arrays[name] == rank
        ]
        for line in _wrap_name_list("real(8) :: ", names):
            lines.append(_INDENT + line)
    for line in _wrap_name_list("integer :: ", loop_indices):
        lines.append(_INDENT + line)

    lines.append("")
    for line in body:
        lines.append(_INDENT + line if line else "")

    lines.append(f"end subroutine {sub_name}")
    lines.append("")

    _SYM_RANK3[0] = False
    return "\n".join(lines)
