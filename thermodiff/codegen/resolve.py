"""Placeholder resolution for the Fortran code generator."""

import warnings
from collections.abc import Iterable
from typing import NamedTuple, TYPE_CHECKING

import sympy as sp

from thermodiff.codegen.common import (
    _DERIV_ORDER,
    _emission_order,
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
    canonical : bool
        Name placeholder derivatives by the differentiation variables, not
        by the index labels (``dBdn(i)`` and ``dBdn(j)`` instead of
        ``dBdi(i)`` and ``dBdj(j)``), see :func:`_canonical_suffix`.
    symmetric_names : iterable of str
        Placeholders declared symmetric in their first two indices
        (``symmetric_params``); the placeholder and its derivatives without
        extra indices are recorded in :attr:`symmetric`.
    """

    def __init__(
        self,
        strict: bool = False,
        canonical: bool = False,
        symmetric_names: Iterable[str] = (),
        reported: set[str] | None = None,
    ) -> None:
        self.strict = strict
        self.canonical = canonical
        self.symmetric_names = frozenset(symmetric_names)
        #: name -> rank (0 for scalars)
        self.ranks: dict[str, int] = {}
        #: canonical second derivatives over two indexed variables of a
        #: placeholder without own indices: symmetric in their subscripts
        self.symmetric: set[str] = set()
        #: placeholders already reported by :func:`_warn_expression_args`
        #: shared by the resolvers of one ``to_fortran`` call
        self._reported: set[str] = set() if reported is None else reported

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

    def _check_arguments(self, func: sp.Function) -> None:
        """Warn once per placeholder whose arguments include expressions."""
        key = func.func.__name__
        if key not in self._reported:
            self._reported.add(key)
            _warn_expression_args(func)

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
            self._check_arguments(inner)
            base = _sanitize(inner.func.__name__)
            dims = self._index_dims(inner)
        else:
            base = _sanitize(str(inner))
            dims = []

        if self.canonical:
            return self._canonical_var(
                base, dims, deriv.variables, self._inherits(inner, dims)
            )
        suffix = ""
        for var in deriv.variables:
            if isinstance(var, sp.Indexed):
                label = str(var.indices[0])
                dims.append(label)
                suffix += "d" + label
            else:
                suffix += "d" + _sanitize(str(var))

        return self._build("d" + base + suffix, dims)

    def _inherits(self, inner: sp.Basic, dims: list[str]) -> bool:
        """Return whether derivatives of ``inner`` keep its symmetry."""
        return _is_placeholder(inner) and self._is_symmetric_placeholder(
            inner, dims
        )

    def _canonical_var(
        self,
        base: str,
        dims: list[str],
        variables: tuple[sp.Basic, ...],
        symmetric_inner: bool = False,
    ) -> sp.Expr:
        """Build a derivative variable named by its variables (canonical).

        The scalar variables come first, sorted by name, then one ``d<x>``
        per indexed variable ``x[i]``; the array subscripts are the
        placeholder's own indices followed by the derivative indices, in
        the order of ``variables``.  A derivative without indexed variables
        of a symmetric placeholder (``symmetric_inner``) stays symmetric.
        """
        suffix, labels = _canonical_suffix(variables)
        name = "d" + base + suffix
        if (not dims and len(labels) == 2) or (symmetric_inner and not labels):
            self.symmetric.add(name)
        return self._build(name, dims + labels)

    def _function_var(self, func: sp.Function) -> sp.Expr:
        self._check_arguments(func)
        base = _sanitize(func.func.__name__)
        dims = self._index_dims(func)
        if self._is_symmetric_placeholder(func, dims):
            self.symmetric.add(base)
        return self._build(base, dims)

    def _is_symmetric_placeholder(
        self, func: sp.Function, dims: list[str]
    ) -> bool:
        """Return whether ``func`` is a rank-2 symmetric placeholder.

        Declared with ``symmetric_params`` or ``idx_function(symmetric=)``.

        Raises
        ------
        ValueError
            If ``idx_function(symmetric=(0, 1))`` was used on a function
            whose first two arguments are not index arguments.
        """
        pair = getattr(func.func, "symmetric_args", None)
        if pair is not None:
            _check_symmetric_pair(func, pair)
        declared = (
            _sanitize(func.func.__name__) in self.symmetric_names
            or pair is not None
        )
        return declared and len(dims) == 2


def _check_symmetric_pair(func: sp.Function, pair: tuple[int, ...]) -> None:
    """Require that the arguments at ``pair`` of ``func`` are indices."""
    if all(
        position < len(func.args) and isinstance(func.args[position], sp.Idx)
        for position in pair
    ):
        return
    raise ValueError(
        f"Function '{func.func.__name__}' is declared symmetric in its "
        f"arguments {pair}, but they are not both index arguments: "
        f"{func.args}"
    )


def _expression_arguments(func: sp.Function) -> list[sp.Basic]:
    """Return the arguments of ``func`` that are not plain variables."""
    return [
        arg
        for arg in func.args
        if not isinstance(arg, (sp.Symbol, sp.Idx, sp.Indexed))
    ]


def expression_argument_placeholders(expr: sp.Basic) -> list[str]:
    """Return the names of the placeholders with expression arguments.

    Parameters
    ----------
    expr : sympy.Basic
        Expression to scan.

    Returns
    -------
    list of str
        Sorted names (empty when every argument is a plain variable).
    """
    return sorted(
        {
            func.func.__name__
            for func in expr.atoms(sp.Function)
            if _is_placeholder(func) and _expression_arguments(func)
        }
    )


def _warn_expression_args(func: sp.Function) -> None:
    """Warn when a placeholder takes an expression as an argument.

    The generator treats a placeholder as a function of the variables its
    arguments contain, and asks the caller for its derivatives with respect
    to those variables (total derivatives); it does not apply the chain
    rule through an argument such as ``delta(n, V)``.

    Parameters
    ----------
    func : sympy.Function
        The placeholder call.
    """
    expression_args = _expression_arguments(func)
    if not expression_args:
        return
    warnings.warn(
        f"placeholder '{func.func.__name__}' has the expression argument(s) "
        + ", ".join(str(arg) for arg in expression_args)
        + ": it is treated as a function of the variables they contain, "
        "so the caller must supply its total derivatives with respect to "
        "those variables (the chain rule through the arguments is not "
        "applied). To differentiate through them, give the arguments as "
        "scalar variables (DiffPlz(..., variables=[...])) and assemble "
        "the result; see the scalar core recipe in the documentation.",
        UserWarning,
        stacklevel=2,
    )


def _canonical_suffix(
    variables: tuple[sp.Basic, ...],
) -> tuple[str, list[str]]:
    """Return the canonical name suffix and the subscripts of a derivative.

    ``(T, n[i])`` and ``(n[i], T)`` both give ``("dTdn", ["i"])``;
    ``(n[i], n[j])`` gives ``("dndn", ["i", "j"])``.
    """
    scalars = sorted(
        _sanitize(str(var))
        for var in variables
        if not isinstance(var, sp.Indexed)
    )
    indexed = [var for var in variables if isinstance(var, sp.Indexed)]
    suffix = "".join("d" + name for name in scalars)
    suffix += "".join("d" + _sanitize(str(var.base)) for var in indexed)
    return suffix, [str(var.indices[0]) for var in indexed]


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
        for attr in _emission_order(candidates):
            value = getattr(diff, attr, None)
            if value is None or value == 0:
                continue
            selected.append(attr)
        return selected
    resolved = resolve_derivatives(
        derivatives,
        scalar_variables=[str(var) for var in getattr(diff, "variables", ())],
    )
    selected = _emission_order(resolved)
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
