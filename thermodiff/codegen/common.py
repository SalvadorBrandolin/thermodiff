"""Shared helpers for the Fortran code generator."""

import re
from collections.abc import Callable, Container, Iterable
from collections.abc import Set as AbstractSet
from typing import TypeVar

import sympy as sp
from sympy.printing.fortran import fcode

from thermodiff.derivatives import DERIVATIVE_ATTRIBUTES
from thermodiff.thermovars import i, j, nc

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


_K = TypeVar("_K")
_V = TypeVar("_V")

#: Memo entry of :func:`_free_indices`: the ``Idx`` atoms of a subtree and
#: the summation variables of every ``Sum`` inside it.
_IndexInfo = tuple[frozenset[sp.Idx], frozenset[sp.Idx]]


def _node_index_info(
    node: sp.Basic, memo: dict[sp.Basic, _IndexInfo]
) -> _IndexInfo:
    """Combine the memoised index info of the children of ``node``."""
    indices: set[sp.Idx] = set()
    bound: set[sp.Idx] = set()
    for child in node.args:
        child_indices, child_bound = memo[child]
        indices |= child_indices
        bound |= child_bound
    if isinstance(node, sp.Idx):
        indices.add(node)
    if isinstance(node, sp.Sum):
        bound.update(limit[0] for limit in node.limits)
    return frozenset(indices), frozenset(bound)


def _fill_index_memo(expr: sp.Basic, memo: dict[sp.Basic, _IndexInfo]) -> None:
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


def _free_indices(
    expr: sp.Expr, memo: dict[sp.Basic, _IndexInfo] | None = None
) -> set[sp.Idx]:
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
    bound: AbstractSet[sp.Idx]
    if memo is None:
        bound = {
            limit[0] for total in expr.atoms(sp.Sum) for limit in total.limits
        }
        return {idx for idx in expr.atoms(sp.Idx) if idx not in bound}
    _fill_index_memo(expr, memo)
    indices, bound = memo[expr]
    return set(indices - bound)


def _order_free(free: set[sp.Idx], indexes: Iterable[sp.Idx]) -> list[sp.Idx]:
    """Order free indices deterministically.

    Derivative indices (``i`` and ``j``) come first, then the user supplied
    ``indexes`` in their given order, then any remaining index sorted by
    name.
    """
    ordered: list[sp.Idx] = []
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


class _Namer:
    """Generate sequential temporary variable names."""

    def __init__(self) -> None:
        self._count = 0
        self.created: list[sp.Symbol] = []

    def __call__(self, prefix: str = "t") -> sp.Symbol:
        self._count += 1
        symbol = sp.Symbol(f"{prefix}{self._count}")
        self.created.append(symbol)
        return symbol


def _print_expr(expr: sp.Expr) -> str:
    """Print ``expr`` as a single Fortran expression."""
    return str(fcode(expr, **_FCODE_SETTINGS).strip())


def _indent_lines(text: str, depth: int) -> list[str]:
    """Indent every non-empty line of ``text`` by ``depth`` levels."""
    prefix = _INDENT * depth
    return [prefix + line for line in text.splitlines() if line.strip()]


def _new_shared_temp(
    namer: _Namer, free_order: list[sp.Idx], arrays: dict[str, int]
) -> sp.Basic:
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


def _index_symbols(expr: sp.Basic) -> list[sp.Idx]:
    """Return the model indices appearing in ``expr``, sorted by name."""
    return sorted(expr.atoms(sp.Idx), key=str)


def _index_set(
    expr: sp.Basic, memo: dict[sp.Basic, frozenset[sp.Idx]]
) -> frozenset[sp.Idx]:
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


def _sorted_indices(
    expr: sp.Basic, memo: dict[sp.Basic, frozenset[sp.Idx]]
) -> list[sp.Idx]:
    """Like :func:`_index_symbols`, but reusing the ``memo`` index sets."""
    return sorted(_index_set(expr, memo), key=str)


def _memoised_call(
    function: Callable[[_K], _V], key: _K, memo: dict[_K, _V]
) -> _V:
    """Return ``function(key)``, cached in the per-call ``memo``."""
    if key not in memo:
        memo[key] = function(key)
    return memo[key]


def _is_expensive(expr: sp.Basic) -> bool:
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


def _canonical_floats(expr: sp.Basic) -> sp.Basic:
    """Round every floating-point constant to its double value.

    Each value gets one representation.  Constants that SymPy carried
    with different precisions (``5.96554738597035e-7`` and
    ``5.96554738597034e-7`` from the same product of literals) become
    the same object, so the common sub-expression passes can merge the
    expressions that contain them.
    """
    floats = expr.atoms(sp.Float)
    if not floats:
        return expr
    return expr.xreplace({f: sp.Float(repr(float(f)), 17) for f in floats})


def _calls_libm(expr: sp.Basic) -> bool:
    """Whether ``expr`` calls a vectorisable libm routine.

    The routines are ``exp``, ``log``, ``sqrt`` and a real power: a
    function other than ``Piecewise`` or a power whose exponent is not
    an integer.
    """
    for sub in sp.preorder_traversal(expr):
        if isinstance(sub, sp.Function) and not isinstance(sub, sp.Piecewise):
            return True
        if isinstance(sub, sp.Pow) and not sub.exp.is_Integer:
            return True
    return False


def _temp_name(target: sp.Basic) -> str:
    """Return the base name of a shared temporary."""
    if isinstance(target, sp.Indexed):
        return str(target.base)
    return str(target)


def _referenced_temp_names(
    expr: sp.Basic, temp_names: Container[str]
) -> set[str]:
    """Return the names of the temporaries referenced by ``expr``."""
    names: set[str] = set()
    for symbol in expr.atoms(sp.Symbol):
        if str(symbol) in temp_names:
            names.add(str(symbol))
    for indexed in expr.atoms(sp.Indexed):
        base = indexed.base
        if isinstance(base, sp.IndexedBase) and str(base) in temp_names:
            names.add(str(base))
    return names


class _LoopStart(sp.Symbol):  # type: ignore[misc]  # sympy is Any
    """Lower bound ``outer + 1`` of a folded triangular loop.

    It is an opaque symbol so that no common sub-expression pass can hoist
    the bound out of the loop header; it prints as ``outer + 1``.
    """

    def __new__(cls, outer: sp.Basic) -> "_LoopStart":
        obj: _LoopStart = super().__new__(cls, f"{outer} + 1")
        obj.outer = outer
        return obj
