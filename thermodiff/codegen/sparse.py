"""Zero guards for sparse parameters.

A parameter such as the group-count matrix ``v(l, k)`` of UNIFAC is mostly
zero, and the sums that multiply it,
``sum_l n(l) v(l, k) log(sum_m th(l, m) psi(m, k))``, only need the factor
``log(...)`` where ``v(l, k) /= 0``.  This module marks, on plain loop nests
(:class:`~thermodiff.codegen.ir.TempDef` without annotations):

* *term guards*: a reduction whose body has terms with a factor
  ``v(a, b)`` (``a`` and ``b`` loop indices of the nest, one of them bound by
  the reduction) accumulates those terms under ``if (v(a, b) /= 0)``;
* *table guards*: a rank-2 table ``T(a, b)`` is not evaluated where
  ``v(a, b) = 0`` when **every** read of ``T`` is a term carrying the same
  element ``v`` at the same subscripts (or the whole body of another table
  guarded by it).  The skipped entry is then never read, so it needs no
  initialisation; the proof is the read analysis of :func:`_covers`.

Operation-count guard: the test costs a branch per iteration, so a guard is
only emitted when the skipped work is a nested loop or calls a
transcendental function (:func:`_is_heavy`), or when it is what lets a table
guard exist.
"""

import itertools
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import sympy as sp

from thermodiff.codegen.ir import SparseGuard, SparseNest, TempDef

_PLAIN_KEYS = frozenset({"target", "free", "limits", "body"})
_TRANSCENDENTAL = (
    sp.log,
    sp.exp,
    sp.sin,
    sp.cos,
    sp.tan,
    sp.sinh,
    sp.cosh,
    sp.tanh,
    sp.asin,
    sp.acos,
    sp.atan,
    sp.erf,
    sp.erfc,
    sp.gamma,
    sp.loggamma,
)


@dataclass
class _TermSplit:
    """Split of a reduction body around one sparse element."""

    factor: sp.Indexed
    rest: sp.Basic
    guarded: sp.Basic


def _is_factor(term: sp.Basic, atom: sp.Indexed) -> bool:
    """Tell whether ``term`` is ``atom`` or a positive power of it."""
    if term == atom:
        return True
    return bool(
        term.is_Pow
        and term.base == atom
        and term.exp.is_Integer
        and term.exp > 0
    )


def split_by_factor(
    expr: sp.Basic, atom: sp.Indexed
) -> tuple[sp.Basic, sp.Basic]:
    """Split ``expr`` into ``rest + guarded`` around the factor ``atom``.

    Parameters
    ----------
    expr : sympy.Basic
        Expression to split.
    atom : sympy.Indexed
        Sparse element.

    Returns
    -------
    tuple
        ``(rest, guarded)`` with ``expr == rest + guarded``; every term of
        ``guarded`` has ``atom`` as a factor (so it vanishes with it), no
        term of ``rest`` is known to.  Products of a factor and a sum are
        distributed over the sum only when the sum holds the element.
    """
    if expr.is_Add:
        parts = [split_by_factor(arg, atom) for arg in expr.args]
        return (
            sp.Add(*(part[0] for part in parts)),
            sp.Add(*(part[1] for part in parts)),
        )
    if expr.is_Mul:
        return _split_product(expr, atom)
    if _is_factor(expr, atom):
        return sp.S.Zero, expr
    return expr, sp.S.Zero


def _split_product(
    product: sp.Basic, atom: sp.Indexed
) -> tuple[sp.Basic, sp.Basic]:
    """Split a product (see :func:`split_by_factor`)."""
    args = list(product.args)
    if any(_is_factor(arg, atom) for arg in args):
        return sp.S.Zero, product
    holders = [pos for pos, arg in enumerate(args) if arg.has(atom)]
    if len(holders) != 1 or not args[holders[0]].is_Add:
        return product, sp.S.Zero
    inner = args.pop(holders[0])
    others = sp.Mul(*args)
    rest, guarded = split_by_factor(inner, atom)
    return others * rest, others * guarded


def _is_heavy(expr: sp.Basic) -> bool:
    """Tell whether ``expr`` calls a transcendental function or a power."""
    if expr.has(*_TRANSCENDENTAL):
        return True
    return any(not power.exp.is_Integer for power in expr.atoms(sp.Pow))


def _is_plain(definition: TempDef) -> bool:
    """Tell whether ``definition`` is a simple loop nest.

    Annotations that are present but false (``mirror=False``) do not change
    how a definition is rendered.
    """
    annotations = definition.present_keys() - _PLAIN_KEYS
    return (
        type(definition) is TempDef
        and not any(getattr(definition, key) for key in annotations)
        and not definition.body.has(sp.Sum)
    )


def _sparse_atoms(
    expr: sp.Basic, names: Iterable[str], loop_indices: set[sp.Basic]
) -> list[sp.Indexed]:
    """Return the sparse elements of ``expr`` indexed by ``loop_indices``."""
    wanted = set(names)
    atoms = [
        atom
        for atom in expr.atoms(sp.Indexed)
        if str(atom.base) in wanted and set(atom.indices) <= loop_indices
    ]
    return sorted(atoms, key=sp.default_sort_key)


def _term_split(
    definition: TempDef, ranks: Mapping[str, int]
) -> _TermSplit | None:
    """Return the term split of a reduction, or ``None`` when none applies."""
    if not definition.limits:
        return None
    bound = {limit[0] for limit in definition.limits}
    loops = bound | set(definition.free)
    for atom in _sparse_atoms(definition.body, ranks, loops):
        if not set(atom.indices) & bound:
            continue
        rest, guarded = split_by_factor(definition.body, atom)
        if guarded != 0:
            return _TermSplit(atom, rest, guarded)
    return None


def _reads(expr: sp.Basic, base: sp.IndexedBase) -> list[sp.Indexed]:
    """Return the elements of ``base`` read in ``expr``."""
    return [atom for atom in expr.atoms(sp.Indexed) if atom.base == base]


def _covers(
    reads: Sequence[sp.Indexed],
    positions: Sequence[int],
    factor: sp.Indexed,
    name: str,
) -> bool:
    """Tell whether ``factor`` is the sparse element of every read.

    ``positions`` maps the subscripts of a read of the table to the
    subscripts of the sparse parameter ``name``.
    """
    if str(factor.base) != name:
        return False
    return all(
        tuple(read.indices[pos] for pos in positions) == factor.indices
        for read in reads
    )


@dataclass
class _Scope:
    """Facts about the definitions shared by the table analysis."""

    ranks: Mapping[str, int]
    splits: dict[int, _TermSplit]
    table_guards: dict[int, sp.Indexed]
    zero_fills: set[int]
    term_guards: dict[int, SparseGuard]
    outputs: set[sp.Basic]
    readers: dict[sp.Basic, list[TempDef]]
    bases: dict[str, sp.Basic]


def _expressions(definition: TempDef) -> list[sp.Basic]:
    """Return every expression held by ``definition`` and its members."""
    found: list[sp.Basic] = []
    stack: list[object] = [definition]
    while stack:
        item = stack.pop()
        if isinstance(item, sp.Basic):
            found.append(item)
        elif isinstance(item, TempDef):
            stack.extend(vars(item).values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return found


def _read_bases(definition: TempDef) -> set[sp.Basic]:
    """Return the indexed bases mentioned by the expressions of a definition.

    A plain definition's target is not a read; the other kinds are scanned
    whole, which over-reports and so only makes the analysis conservative.
    """
    if _is_plain(definition):
        return set(definition.body.atoms(sp.IndexedBase))
    return {
        base
        for expr in _expressions(definition)
        for base in expr.atoms(sp.IndexedBase)
    }


def _consumer_covers(
    consumer: TempDef,
    base: sp.IndexedBase,
    positions: Sequence[int],
    name: str,
    scope: _Scope,
) -> bool:
    """Tell whether every read of ``base`` in ``consumer`` is covered."""
    if not _is_plain(consumer):
        return False
    reads = _reads(consumer.body, base)
    table_guard = scope.table_guards.get(id(consumer))
    if table_guard is not None:
        return _covers(reads, positions, table_guard, name)
    split = scope.splits.get(id(consumer))
    if split is None or _reads(split.rest, base):
        return False
    guarded_reads = _reads(split.guarded, base)
    return _covers(guarded_reads, positions, split.factor, name)


def _table_templates(
    table: TempDef, ranks: Mapping[str, int]
) -> list[tuple[str, tuple[int, ...]]]:
    """Return the ``(parameter, positions)`` guards a table could take."""
    count = len(table.target.indices)
    return [
        (name, positions)
        for name, rank in ranks.items()
        for positions in itertools.permutations(range(count), rank)
    ]


def _is_table_candidate(table: TempDef) -> bool:
    """Tell whether ``table`` is a plain, costly, rank >= 1 table entry."""
    target = table.target
    return (
        _is_plain(table)
        and isinstance(target, sp.Indexed)
        and len(set(target.indices)) == len(target.indices)
        and set(target.indices) == set(table.free)
        and (bool(table.limits) or _is_heavy(table.body))
    )


def _table_guard(table: TempDef, scope: _Scope) -> sp.Indexed | None:
    """Return the element that can guard ``table``, or ``None``.

    Every read of the table, in every definition, must be covered by the
    element at the matching subscripts; a read in an output, in a
    definition of another kind or in an unguarded term refuses the guard.
    """
    if not _is_table_candidate(table):
        return None
    base = table.target.base
    consumers = [d for d in scope.readers.get(base, []) if d is not table]
    if base in scope.outputs or not consumers:
        return None
    for name, positions in _table_templates(table, scope.ranks):
        if name not in scope.bases:
            continue
        if all(
            _consumer_covers(consumer, base, positions, name, scope)
            for consumer in consumers
        ):
            indices = [table.target.indices[pos] for pos in positions]
            return sp.Indexed(scope.bases[name], *indices)
    return None


def _zero_fill_guard(
    table: TempDef, ranks: Mapping[str, int]
) -> sp.Indexed | None:
    """Return the element of a table whose body is a multiple of it.

    Such an entry is zero where the element is, whoever reads it: the guard
    computes the costly factor only where the element is not zero and sets
    the entry to zero elsewhere (``mat(l, k) = log(...)*v(l, k)``).
    """
    target = table.target
    if not (
        _is_plain(table)
        and not table.limits
        and isinstance(target, sp.Indexed)
        and _is_heavy(table.body)
    ):
        return None
    for atom in _sparse_atoms(table.body, ranks, set(table.free)):
        rest, guarded = split_by_factor(table.body, atom)
        if rest == 0 and set(atom.indices) == set(table.free):
            return atom
    return None


def _keeps_term_guard(split: _TermSplit, guarded_bases: set[sp.Basic]) -> bool:
    """Apply the operation-count guard to a term split."""
    if _is_heavy(split.guarded):
        return True
    return bool(split.guarded.atoms(sp.IndexedBase) & guarded_bases)


def _build_scope(
    definitions: Sequence[TempDef],
    outputs: Iterable[sp.Basic],
    ranks: Mapping[str, int],
) -> _Scope:
    """Collect the term splits, the readers and the sparse bases."""
    plain = [d for d in definitions if _is_plain(d)]
    splits = {id(d): _term_split(d, ranks) for d in plain}
    readers: dict[sp.Basic, list[TempDef]] = {}
    for definition in definitions:
        for base in _read_bases(definition):
            readers.setdefault(base, []).append(definition)
    bases = {
        str(atom.base): atom.base
        for d in plain
        for atom in d.body.atoms(sp.Indexed)
        if str(atom.base) in ranks
    }
    return _Scope(
        ranks,
        {key: split for key, split in splits.items() if split is not None},
        {},
        set(),
        {},
        {base for out in outputs for base in out.atoms(sp.IndexedBase)},
        readers,
        bases,
    )


def _decide_tables(definitions: Sequence[TempDef], scope: _Scope) -> None:
    """Decide the table guards, consumers first."""
    for table in reversed(definitions):
        guard = _table_guard(table, scope)
        if guard is None:
            guard = _zero_fill_guard(table, scope.ranks)
            if guard is not None:
                scope.zero_fills.add(id(table))
        if guard is not None:
            scope.table_guards[id(table)] = guard


def apply_sparse_guards(
    definitions: list[TempDef],
    outputs: Iterable[sp.Basic],
    ranks: Mapping[str, int],
) -> None:
    """Guard the definitions that profit from a sparse parameter.

    Parameters
    ----------
    definitions : list of TempDef
        Ordered definitions of the kernel; the guarded plain definitions
        are replaced in place by :class:`~thermodiff.codegen.ir.SparseNest`.
    outputs : iterable of sympy.Basic
        Residual expressions of the outputs.
    ranks : mapping
        Rank of every sparse array parameter, by name.
    """
    if not ranks:
        return
    scope = _build_scope(definitions, outputs, ranks)
    _decide_tables(definitions, scope)
    _apply_term_guards([d for d in definitions if _is_plain(d)], scope)
    for position, definition in enumerate(definitions):
        guard = _guard_of(definition, scope)
        if guard is not None:
            definitions[position] = _guarded(definition, guard)


def _guard_of(definition: TempDef, scope: _Scope) -> SparseGuard | None:
    """Return the guard decided for ``definition``, if any."""
    table_guard = scope.table_guards.get(id(definition))
    if table_guard is None:
        return scope.term_guards.get(id(definition))
    fill = id(definition) in scope.zero_fills
    return SparseGuard(table_guard, table=True, zero_fill=fill)


def _guarded(definition: TempDef, guard: SparseGuard) -> SparseNest:
    """Return the guarded replacement of a plain definition."""
    return SparseNest(
        target=definition.target,
        free=definition.free,
        limits=definition.limits,
        body=definition.body,
        guard=guard,
    )


def _apply_term_guards(plain: Sequence[TempDef], scope: _Scope) -> None:
    """Set the term guards that survive the operation-count guard."""
    guarded_bases = {
        d.target.base for d in plain if id(d) in scope.table_guards
    }
    for definition in plain:
        split = scope.splits.get(id(definition))
        if split is None or id(definition) in scope.table_guards:
            continue
        if _keeps_term_guard(split, guarded_bases):
            scope.term_guards[id(definition)] = SparseGuard(
                split.factor,
                table=False,
                rest=split.rest,
                guarded=split.guarded,
            )
