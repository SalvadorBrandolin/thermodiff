"""Placeholder resolution for the Fortran code generator."""

from collections.abc import Iterable
from typing import NamedTuple, TYPE_CHECKING

import sympy as sp

from thermodiff.codegen.common import (
    _DERIV_ORDER,
    _is_placeholder,
    _sanitize,
)
from thermodiff.derivatives import resolve_derivatives
from thermodiff.thermovars import nc

if TYPE_CHECKING:
    from thermodiff.diffplz import DiffPlz


class _PlaceholderResolver:
    """Replace undefined functions and derivatives with input arguments.

    Parameters
    ----------
    strict : bool
        When ``True``, raise a ``ValueError`` if any placeholder is found
        instead of exposing it as an argument.
    """

    def __init__(self, strict: bool = False) -> None:
        self.strict = strict
        #: name -> rank (0 for scalars)
        self.ranks: dict[str, int] = {}

    def resolve(self, expr: sp.Expr) -> sp.Expr:
        """Return ``expr`` with placeholders replaced by plain variables."""
        derivative_set, function_set = _derivative_function_atoms(expr)
        derivatives = list(derivative_set)
        functions = [f for f in function_set if _is_placeholder(f)]

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
        if not replacements:
            # ``xreplace`` with no rule returns ``expr``: reuse its walk
            resolved = expr
            bare = set(functions)
        else:
            resolved = expr.xreplace(replacements)
            bare = {
                func
                for func in resolved.atoms(sp.Function)
                if _is_placeholder(func)
            }
        if not bare:
            return resolved
        replacements = {func: self._function_var(func) for func in bare}
        return resolved.xreplace(replacements)

    def _index_dims(self, func: sp.Function) -> list[str]:
        """Return the index labels used as array dimensions by ``func``."""
        dims: list[str] = []
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

    def _build(self, name: str, labels: list[str]) -> sp.Expr:
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


def _derivative_function_atoms(
    expr: sp.Basic,
) -> tuple[set[sp.Basic], set[sp.Basic]]:
    """Return ``expr.atoms(Derivative)`` and ``expr.atoms(Function)``.

    Both sets come from a single ``preorder_traversal`` of ``expr``.

    Parameters
    ----------
    expr : sympy.Basic
        Expression to scan.

    Returns
    -------
    tuple of set
        The Derivative nodes and the Function nodes.
    """
    derivatives = set()
    functions = set()
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Derivative):
            derivatives.add(node)
        if isinstance(node, sp.Function):
            functions.add(node)
    return derivatives, functions


class _ScanAtoms(NamedTuple):
    """Atoms of one expression gathered by :func:`_scan_atoms`."""

    indexed: set[sp.Indexed]
    sums: set[sp.Sum]
    idxs: set[sp.Idx]
    bases: set[sp.IndexedBase]
    symbols: set[sp.Symbol]


def _scan_atoms(expr: sp.Basic) -> _ScanAtoms:
    """Collect Indexed, Sum, Idx, IndexedBase and Symbol atoms in one walk.

    Each set equals ``expr.atoms(cls)`` for the matching class: the nodes of
    ``preorder_traversal`` (root included) that are instances of ``cls``.

    Parameters
    ----------
    expr : sympy.Basic
        Expression to scan.

    Returns
    -------
    _ScanAtoms
        The five atom sets.
    """
    scan = _ScanAtoms(set(), set(), set(), set(), set())
    targets = (
        (sp.Indexed, scan.indexed),
        (sp.Sum, scan.sums),
        (sp.Idx, scan.idxs),
        (sp.IndexedBase, scan.bases),
        (sp.Symbol, scan.symbols),
    )
    for node in sp.preorder_traversal(expr):
        for cls, found in targets:
            if isinstance(node, cls):
                found.add(node)
    return scan


def _select_derivatives(
    diff: "DiffPlz", derivatives: Iterable[str] | None
) -> list[str]:
    """Return the derivative attributes to emit, in canonical order.

    Parameters
    ----------
    diff : DiffPlz
        Instance holding the derivatives.
    derivatives : iterable of str or None
        Requested derivatives; ``None`` takes every non-zero one.

    Returns
    -------
    list of str
        Selected attributes.

    Raises
    ------
    ValueError
        If a requested derivative was not computed by ``diff``.
    """
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
        return selected
    resolved = resolve_derivatives(derivatives)
    selected = [attr for attr in _DERIV_ORDER if attr in resolved]
    missing = [attr for attr in selected if getattr(diff, attr, None) is None]
    if missing:
        raise ValueError(
            "Derivatives not computed by the DiffPlz instance: "
            + ", ".join(missing)
            + ". Rebuild it with derivatives="
            + repr(list(resolved))
            + "."
        )
    return selected
