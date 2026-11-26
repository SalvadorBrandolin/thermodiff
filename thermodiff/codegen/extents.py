"""Extents of the indices that run over a dimension other than ``nc``.

Summations over an extra dimension (``sum_m[1:ng]`` in the ``.model``
language, ``sum_custom(expr, m, 1, ng)`` in Python) bind an index whose range
is the dimension symbol.  The temporaries built over such an index must be
declared and looped with that bound instead of ``nc``.  The generator keeps
the range in the index itself: :func:`annotate_index_ranges` rebuilds every
summation index of an extra dimension as ``Idx(label, (1, dimension))``, so
the extent travels with the index through every pass, and
:func:`index_extent` reads it back when a loop or a declaration is emitted.
"""

from collections.abc import Iterable

import sympy as sp

from thermodiff.thermovars import nc

#: Name of the dimension that every index runs over unless it says otherwise.
COMPONENT_EXTENT = "nc"


def index_extent(index: sp.Basic) -> str:
    """Return the Fortran upper bound of the loop over ``index``.

    Parameters
    ----------
    index : sympy.Basic
        A loop index.  Only an ``Idx`` built by
        :func:`annotate_index_ranges` carries a dimension.

    Returns
    -------
    str
        The name of the dimension symbol, ``"nc"`` for every other index.
    """
    if isinstance(index, sp.Idx) and index.lower == 1:
        upper = index.upper
        if isinstance(upper, sp.Symbol):
            return str(upper)
    return COMPONENT_EXTENT


def is_component_index(index: sp.Basic) -> bool:
    """Return whether ``index`` runs over ``1..nc``.

    Parameters
    ----------
    index : sympy.Basic
        A loop index.

    Returns
    -------
    bool
        ``True`` unless ``index`` carries an extra dimension.
    """
    return index_extent(index) == COMPONENT_EXTENT


def with_range_of(index: sp.Idx, label: str) -> sp.Idx:
    """Return a new index named ``label`` with the range of ``index``.

    Parameters
    ----------
    index : sympy.Idx
        Index whose range (extra dimension or ``nc``) is inherited.
    label : str
        Name of the new index.

    Returns
    -------
    sympy.Idx
        ``Idx(label)`` for a component index, a ranged index otherwise.
    """
    if is_component_index(index):
        return sp.Idx(label)
    return sp.Idx(label, (index.lower, index.upper))


def slot_mapping(
    indices: list[sp.Idx], slots: list[sp.Idx]
) -> dict[sp.Idx, sp.Idx]:
    """Map ``indices`` to the leading auxiliary ``slots``, keeping ranges.

    Parameters
    ----------
    indices : list of sympy.Idx
        Indices of a sub-expression, in slot order.
    slots : list of sympy.Idx
        Auxiliary indices; the first ``len(indices)`` are used.

    Returns
    -------
    dict
        Each index to the slot at its position, as a ranged index when the
        index runs over an extra dimension.  The values are the free
        indices of the temporary that holds the sub-expression.
    """
    return {
        index: with_range_of(index, str(slots[position]))
        for position, index in enumerate(indices)
    }


def _dimension_symbol(high: sp.Basic) -> bool:
    """Whether a summation bound is a dimension symbol other than ``nc``."""
    return (
        isinstance(high, sp.Symbol)
        and high != nc
        and str(high) != COMPONENT_EXTENT
    )


def _summation_bounds(
    expressions: Iterable[sp.Basic],
) -> dict[sp.Idx, set[sp.Basic]]:
    """Return the upper bounds of every summation index, by index."""
    bounds: dict[sp.Idx, set[sp.Basic]] = {}
    for expression in expressions:
        for total in expression.atoms(sp.Sum):
            for index, _low, high in total.limits:
                if isinstance(index, sp.Idx):
                    bounds.setdefault(index, set()).add(high)
    return bounds


def annotate_index_ranges(
    expressions: Iterable[sp.Basic],
) -> dict[sp.Basic, sp.Basic]:
    """Map the summation indices of an extra dimension to ranged indices.

    Parameters
    ----------
    expressions : iterable of sympy expression
        Every expression of one generation.

    Returns
    -------
    dict
        ``xreplace`` mapping from ``Idx(label)`` to ``Idx(label, (1, dim))``
        for each index that is summed up to a dimension symbol other than
        ``nc``.  Empty when the model has no extra dimension.

    Raises
    ------
    ValueError
        If one index label is summed over two different bounds, one of them
        an extra dimension: a temporary indexed by it would have no single
        extent.  Rename one of the indices.
    """
    mapping: dict[sp.Basic, sp.Basic] = {}
    for index, highs in _summation_bounds(expressions).items():
        if not any(_dimension_symbol(high) for high in highs):
            continue
        if len(highs) > 1:
            names = ", ".join(sorted(str(high) for high in highs))
            raise ValueError(
                f"Index '{index}' is summed over several ranges ({names}); "
                "use a different index for each dimension."
            )
        (high,) = highs
        mapping[index] = sp.Idx(str(index), (1, high))
    return mapping


def _use_extents(
    name: str, expressions: Iterable[sp.Basic]
) -> dict[tuple[str, ...], tuple[sp.Idx, ...]]:
    """Return the subscripts that read array ``name``, by their extents."""
    uses: dict[tuple[str, ...], tuple[sp.Idx, ...]] = {}
    for expression in expressions:
        for element in expression.atoms(sp.Indexed):
            if str(element.base) != name:
                continue
            indices = tuple(element.indices)
            extents = tuple(index_extent(index) for index in indices)
            if all(isinstance(index, sp.Idx) for index in indices):
                uses.setdefault(extents, indices)
    return uses


def ranged_slots(
    name: str, slots: list[sp.Idx], expressions: Iterable[sp.Basic]
) -> dict[sp.Idx, sp.Idx]:
    """Give the slots of a component array the ranges of its readers.

    The body of a component array is written over fixed slot indices, which
    do not know the dimension the readers run over.

    Parameters
    ----------
    name : str
        Name of the component array.
    slots : list of sympy.Idx
        Slot indices of the definition, one per subscript.
    expressions : iterable of sympy expression
        Expressions that read the array with their (ranged) indices.

    Returns
    -------
    dict
        Slot to ranged slot; empty when every reader runs over ``nc``.

    Raises
    ------
    ValueError
        If the readers disagree on the extents of the array.

    Notes
    -----
    Readers over ``nc`` only count when no reader runs over an extra
    dimension: the bodies of the other component arrays read this one at
    their own slot indices, which carry no range yet.
    """
    uses = _use_extents(name, expressions)
    if len(uses) > 1:
        # a component body reads the other arrays at its own, unranged,
        # slots: only a reader over an extra dimension says something
        plain = (COMPONENT_EXTENT,) * len(slots)
        uses.pop(plain, None)
    if len(uses) > 1:
        raise ValueError(
            f"Function '{name}' is used over several extents "
            f"({sorted(uses)}); use a different function for each."
        )
    if not uses:
        return {}
    (indices,) = uses.values()
    mapping = {
        slot: with_range_of(index, str(slot))
        for slot, index in zip(slots, indices)
    }
    return {slot: new for slot, new in mapping.items() if new != slot}


def dimension_symbols(expression: sp.Basic) -> list[str]:
    """Return the names of the extra dimensions that bound summations.

    Parameters
    ----------
    expression : sympy expression
        Expression of the model.

    Returns
    -------
    list of str
        Sorted names of the symbols, other than ``nc``, that are a limit of
        a summation.  They are ``integer`` arguments of the kernel.
    """
    names: set[str] = set()
    for total in expression.atoms(sp.Sum):
        for _index, low, high in total.limits:
            names.update(
                str(bound) for bound in (low, high) if _dimension_symbol(bound)
            )
    return sorted(names)
