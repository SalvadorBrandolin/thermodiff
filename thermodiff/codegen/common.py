"""Shared helpers for the Fortran code generator."""

import re
from collections.abc import Callable, Container, Iterable
from collections.abc import Set as AbstractSet
from typing import TypeVar

import sympy as sp
from sympy.core.function import DefinedFunction
from sympy.printing.fortran import fcode

from thermodiff.codegen.options import DEFAULT_REAL_KIND
from thermodiff.derivatives import (
    DERIVATIVE_ATTRIBUTES,
    SCALAR_PREFIX,
    scalar_suffix,
)
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

#: Order in which the derivatives are emitted (those with respect to the
#: extra scalar variables of the ``DiffPlz`` follow, in their own order).
_DERIV_ORDER = list(DERIVATIVE_ATTRIBUTES)


def _deriv_suffix(attribute: str) -> str:
    """Return the output name suffix of a derivative attribute.

    Parameters
    ----------
    attribute : str
        A ``DiffPlz`` derivative attribute, including the ones with
        respect to the extra scalar variables (``"var_dB"``).

    Returns
    -------
    str
        ``"dT"`` for ``"dt"``, ``"dB"`` for ``"var_dB"`` ...
    """
    if attribute.startswith(SCALAR_PREFIX):
        return scalar_suffix(attribute)
    return _DERIV_SUFFIX[attribute]


def _emission_order(attributes: Iterable[str]) -> list[str]:
    """Sort derivative attributes in emission order.

    The thermodynamic derivatives come first, in ``_DERIV_ORDER``; the
    others keep the order in which they are given.
    """
    given = list(attributes)
    return [a for a in _DERIV_ORDER if a in given] + [
        a for a in given if a not in _DERIV_ORDER
    ]


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


#: Names the generator itself declares as temporaries: ``t<number>`` and the
#: accumulator ``res``.  A user name matching it (in any case) is renamed.
_GENERATED_NAME = re.compile(r"(t\d+|res)", re.IGNORECASE)


def _fortran_key(name: str) -> str:
    """Return the key under which Fortran identifies ``name``.

    Fortran is case-insensitive and the generator sanitises names, so two
    names are the same variable when their keys are equal.
    """
    return _sanitize(name).lower()


def _case_clash_renames(
    fixed: Iterable[str], candidates: Iterable[str]
) -> dict[str, str]:
    """Rename the user names that would collide in Fortran.

    ``fixed`` names (module symbols, integer arguments, the outputs and the
    names the generator declares itself) keep their spelling; if two of
    them differ only in case the clash cannot be solved and a
    ``ValueError`` is raised.  A name of ``candidates`` (parameters,
    placeholders) that collides with a fixed name, with a candidate seen
    before it in sorted order, or with a generated name (``t1``, ``res``)
    is renamed to ``<name>_<k>`` with the smallest free ``k``.

    Parameters
    ----------
    fixed : iterable of str
        Names that cannot be renamed.
    candidates : iterable of str
        Names that can be renamed.

    Returns
    -------
    dict
        ``old name -> new name`` for every renamed candidate (empty when
        nothing clashes).

    Raises
    ------
    ValueError
        If two fixed names differ only in case.
    """
    taken: dict[str, str] = {}
    for name in sorted(set(fixed)):
        other = taken.setdefault(_fortran_key(name), name)
        if other != name:
            raise ValueError(
                f"Names {other!r} and {name!r} are the same identifier in "
                "Fortran (case-insensitive) and neither can be renamed "
                "(module symbol, integer argument, output or fixed name)."
            )
    renames: dict[str, str] = {}
    for name in sorted(set(candidates) - set(taken.values())):
        key = _fortran_key(name)
        if key not in taken and not _GENERATED_NAME.fullmatch(key):
            taken[key] = name
            continue
        count = 1
        while _fortran_key(f"{name}_{count}") in taken:
            count += 1
        renames[name] = f"{name}_{count}"
        taken[_fortran_key(renames[name])] = renames[name]
    return renames


def _rename_names(expr: sp.Basic, renames: dict[str, str]) -> sp.Basic:
    """Return ``expr`` with symbols, arrays and functions renamed.

    Parameters
    ----------
    expr : sympy.Basic
        Expression to rewrite.
    renames : dict
        ``old name -> new name``; empty returns ``expr`` itself.

    Returns
    -------
    sympy.Basic
        The renamed expression (assumptions and array shapes kept).
    """
    if not renames:
        return expr
    expr = expr.replace(
        lambda node: _is_placeholder(node) and node.func.__name__ in renames,
        lambda node: sp.Function(renames[node.func.__name__])(*node.args),
    )
    mapping: dict[sp.Basic, sp.Basic] = {}
    for base in expr.atoms(sp.IndexedBase):
        if str(base) in renames:
            mapping[base] = sp.IndexedBase(
                renames[str(base)], shape=base.shape
            )
    for sym in expr.atoms(sp.Symbol):
        if str(sym) in renames:
            mapping[sym] = sp.Symbol(renames[str(sym)], **sym.assumptions0)
    return expr.xreplace(mapping)


def _is_placeholder(func: sp.Basic) -> bool:
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


def unique_name(name: str, taken: Iterable[str]) -> str:
    """Append ``_`` to ``name`` until it is not in ``taken`` (any case).

    Parameters
    ----------
    name : str
        Wanted identifier.
    taken : iterable of str
        Identifiers that are already in use; Fortran ignores case.

    Returns
    -------
    str
        ``name`` or ``name`` followed by underscores.
    """
    lowered = {item.lower() for item in taken}
    while name.lower() in lowered:
        name += "_"
    return name


_REAL_8 = "real(8)"
_FLOAT_LITERAL = re.compile(r"(?<![\w.])(\d+\.\d*)d([+-]?\d+)")


def _kind_literal(match: re.Match[str]) -> str:
    """Rewrite a ``<mantissa>d<exp>`` literal with the kind suffix."""
    mantissa, exponent = match.groups()
    if int(exponent) == 0:
        return mantissa
    return f"{mantissa}e{exponent}"


def apply_real_kind(
    text: str, real_kind: str, kind_module: str | None = None
) -> str:
    """Re-express the ``real(8)`` source ``text`` in another real kind.

    Every ``real(8)`` becomes ``real(<real_kind>)`` and every ``d``
    floating-point literal ``<m>d<e>`` becomes ``<m>e<e>_<real_kind>``
    (``<m>_<real_kind>`` when ``e`` is 0).  With ``kind_module`` the line
    ``use <kind_module>, only: <real_kind>`` is added before the first
    ``implicit none``, which the internal procedures of ``work_array``
    inherit.

    Parameters
    ----------
    text : str
        Fortran source generated for ``real(8)``.
    real_kind : str
        Kind value (``"8"``, ``"16"``) or the name of a kind constant.
    kind_module : str, optional
        Module that exports ``real_kind``.

    Returns
    -------
    str
        The converted source; ``text`` itself for the default kind.
    """
    if real_kind == DEFAULT_REAL_KIND:
        return text
    text = text.replace(_REAL_8, f"real({real_kind})")
    text = _FLOAT_LITERAL.sub(
        lambda match: f"{_kind_literal(match)}_{real_kind}", text
    )
    if kind_module is None:
        return text
    use_line = f"{_INDENT}use {kind_module}, only: {real_kind}"
    marker = f"{_INDENT}implicit none"
    head, found, tail = text.partition(marker)
    return head + use_line + "\n" + found + tail if found else text


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
