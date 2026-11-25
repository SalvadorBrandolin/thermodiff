"""Typed records for the temporaries emitted by the code generator.

A definition is one temporary: ``target`` is assigned ``body`` inside
loops over the ``free`` indexes and the reduction ``limits``.  The
optional annotations (``mirror``, ``colsum``, ``symv``, ...) and the
wrapper records (fused nests, sibling nests, ...) used to be string-keyed
dicts.  The records keep the semantics of those dicts exactly: they are
mutable, compare by value and are unhashable.

An annotation whose value is ``None`` is absent, any other value
(``False`` included) is present; :meth:`TempDef.present_keys` reports
the present ones under the old dict key names.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, ClassVar

import sympy as sp

Limit = tuple[sp.Basic, sp.Basic, sp.Basic]
"""Reduction limit ``(index, low, high)``."""

Namer = Callable[[str], sp.Symbol]
"""Fresh-symbol generator: maps a name prefix to a new symbol."""

_BASE_KEYS: Mapping[str, str] = {
    "target": "target",
    "free": "free",
    "limits": "limits",
    "body": "body",
    "mirror": "mirror",
    "mirror_sum": "mirror_sum",
    "colsum": "colsum",
    "colmajor": "colmajor",
    "symv": "symv",
    "symv_acc": "symv_acc",
    "factored": "factored",
    "acc": "acc",
}


@dataclass(eq=True)
class SparseGuard:
    """Zero test of a sparse parameter element around a definition.

    Attributes
    ----------
    factor : sympy.Indexed
        The element ``v(a, b)`` tested with ``/= 0``.
    table : bool
        ``True``: the whole assignment of the table entry is skipped where
        the element is zero (its readers never read the entry).  ``False``:
        only the terms of the reduction body that carry the element are
        skipped.
    zero_fill : bool
        With ``table``: the entry is set to zero where the element is zero
        (its body is a multiple of the element) instead of left unset.
    rest, guarded : sympy.Basic or None
        For ``table=False``: the part of the body accumulated always and the
        part accumulated under the test (``body = rest + guarded``).
    """

    factor: sp.Basic
    table: bool
    zero_fill: bool = False
    rest: sp.Basic | None = None
    guarded: sp.Basic | None = None


@dataclass(eq=True)
class TempDef:
    """One temporary: ``target`` = ``body`` over ``free`` and ``limits``.

    Attributes
    ----------
    target : sympy.Basic
        Indexed array element or scalar symbol being assigned.
    free : list
        Free (output) indexes.
    limits : list
        Reduction limits as ``(index, low, high)`` tuples.
    body : sympy.Basic
        Right-hand side expression.
    mirror, mirror_sum, colsum, colmajor, factored : bool or None
        Optional annotations; ``None`` means absent.
    symv : tuple or None
        ``(pair, rest)`` of a symmetric matrix-vector product.
    symv_acc, acc : sympy.Symbol or None
        Accumulator symbols of ``symv`` and ``factored`` temporaries.
    """

    target: sp.Basic
    free: list[sp.Basic]
    limits: list[Limit]
    body: sp.Basic
    mirror: bool | None = None
    mirror_sum: bool | None = None
    colsum: bool | None = None
    colmajor: bool | None = None
    symv: tuple[Any, Any] | None = None
    symv_acc: sp.Symbol | None = None
    factored: bool | None = None
    acc: sp.Symbol | None = None

    # Old dict key -> attribute, and keys that are always present.
    _KEY_TO_ATTR: ClassVar[Mapping[str, str]] = _BASE_KEYS
    _CONSTANT_KEYS: ClassVar[Mapping[str, Any]] = {}

    def present_keys(self) -> frozenset[str]:
        """Return the old dict key names that are present.

        Returns
        -------
        frozenset of str
            The base keys, every annotation that is not ``None`` and the
            wrapper keys of the subclass.
        """
        present = {
            key
            for key, attribute in self._KEY_TO_ATTR.items()
            if getattr(self, attribute) is not None
        }
        return frozenset(present | set(self._CONSTANT_KEYS))

    def copy(self, **changes: Any) -> TempDef:
        """Return a shallow copy with some fields replaced.

        Parameters
        ----------
        **changes
            Replacement values, keyed by attribute name.

        Returns
        -------
        TempDef
            A new record of the same class; lists are shared.

        Raises
        ------
        TypeError
            If a name is not a field of this record.
        """
        return replace(self, **changes)


@dataclass(eq=True, kw_only=True)
class FusedNest(TempDef):
    """Fused vector loop nest (old key ``fused``).

    Attributes
    ----------
    fused : list
        Member definitions fused into one loop nest.
    """

    fused: list[TempDef]

    _KEY_TO_ATTR: ClassVar[Mapping[str, str]] = {
        **_BASE_KEYS,
        "fused": "fused",
    }


@dataclass(eq=True, kw_only=True)
class ComponentFusedMap(TempDef):
    """Component-fused map (old key ``compfused``).

    Attributes
    ----------
    steps : list
        ``(kind, target, value)`` tuples.
    members : list
        ``(target, body)`` pairs (tuples, not definition records; compare
        ``OuterFusedNest.members``).
    """

    steps: list[tuple[Any, Any, Any]]
    members: list[tuple[Any, Any]]

    _KEY_TO_ATTR: ClassVar[Mapping[str, str]] = {
        **_BASE_KEYS,
        "steps": "steps",
        "members": "members",
    }
    _CONSTANT_KEYS: ClassVar[Mapping[str, Any]] = {"compfused": True}


@dataclass(eq=True, kw_only=True)
class OuterFusedNest(TempDef):
    """Outer-fused nest (old keys ``outerfused`` and ``outer``).

    Attributes
    ----------
    members : list
        Nested definition records (old key ``outerfused``); compare
        ``ComponentFusedMap.members``, which are ``(target, body)`` pairs.
    outer : sympy.Idx
        The shared outer index.
    """

    members: list[TempDef]
    outer: sp.Idx

    _KEY_TO_ATTR: ClassVar[Mapping[str, str]] = {
        **_BASE_KEYS,
        "outerfused": "members",
        "outer": "outer",
    }


@dataclass(eq=True, kw_only=True)
class SiblingNest(TempDef):
    """Sibling table nest, possibly with split accumulators.

    Attributes
    ----------
    sibling : list
        Mirror-sum table definitions sharing one loop nest.
    accs : list
        One accumulator symbol per sibling.
    accumulators : int or None
        Number of accumulators per table (old key ``split_acc``).
    accs2 : list or None
        Second accumulator symbols when split.
    """

    sibling: list[TempDef]
    accs: list[sp.Symbol]
    accumulators: int | None = None
    accs2: list[sp.Symbol] | None = None

    _KEY_TO_ATTR: ClassVar[Mapping[str, str]] = {
        **_BASE_KEYS,
        "sibling": "sibling",
        "accs": "accs",
        "split_acc": "accumulators",
        "accs2": "accs2",
    }


@dataclass(eq=True, kw_only=True)
class SparseNest(TempDef):
    """Plain loop nest guarded by the zero test of a sparse parameter.

    Replaces the plain definition it guards (``body`` is unchanged).

    Attributes
    ----------
    guard : SparseGuard
        The zero test and, for a term guard, the split of the body.
    """

    guard: SparseGuard

    _KEY_TO_ATTR: ClassVar[Mapping[str, str]] = {
        **_BASE_KEYS,
        "sparse": "guard",
    }
