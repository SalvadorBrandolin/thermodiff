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

from dataclasses import dataclass, replace
from typing import Any, ClassVar, Mapping

import sympy as sp

_BASE_KEYS: Mapping[str, str] = {
    name: name
    for name in (
        "target",
        "free",
        "limits",
        "body",
        "mirror",
        "mirror_sum",
        "colsum",
        "colmajor",
        "symv",
        "symv_acc",
        "factored",
        "acc",
    )
}


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
    free: list
    limits: list
    body: sp.Basic
    mirror: bool | None = None
    mirror_sum: bool | None = None
    colsum: bool | None = None
    colmajor: bool | None = None
    symv: tuple | None = None
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

    fused: list

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
        ``(target, body)`` tuples.
    """

    steps: list
    members: list

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
        Member definitions (old key ``outerfused``).
    outer : sympy.Idx
        The shared outer index.
    """

    members: list
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

    sibling: list
    accs: list
    accumulators: int | None = None
    accs2: list | None = None

    _KEY_TO_ATTR: ClassVar[Mapping[str, str]] = {
        **_BASE_KEYS,
        "sibling": "sibling",
        "accs": "accs",
        "split_acc": "accumulators",
        "accs2": "accs2",
    }
