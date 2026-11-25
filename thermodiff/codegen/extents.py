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
