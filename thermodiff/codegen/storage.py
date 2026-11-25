"""Storage of the array temporaries: automatic, one work array or fixed.

By default every array temporary is an automatic array sized by ``nc``
(``t3(nc, nc)``).  Without ``-fstack-arrays`` gfortran allocates each one
on the heap, so every call pays one ``malloc``/``free`` pair per temporary
(issue #10).  Two opt-in layouts avoid that:

``work_array``
    One automatic ``work_(size)`` array holds every temporary.  The kernel
    becomes a thin wrapper that calls an internal procedure ``<name>_body``
    with the original arguments plus one element of ``work_`` per temporary
    (sequence association), so the body sees ordinary explicit-shape arrays
    ``t3(nc, nc)``.  N allocations become one.
``max_nc``
    Every temporary gets the compile-time extent ``max_nc`` (``t3(16, 16)``,
    used up to ``nc``), so it lives on the stack.  The kernel starts with an
    ``error stop`` when ``nc > max_nc``.
"""

import math
import warnings
from collections.abc import Mapping, Sequence

from thermodiff.codegen.common import _INDENT
from thermodiff.codegen.render import _wrap_name_list, _wrap_signature

#: gfortran's default ``-fmax-stack-var-size`` (bytes).  A larger local
#: array is made static (``SAVE``), which is not thread safe.
GFORTRAN_MAX_STACK_VAR_SIZE = 65536

_REAL_BYTES = 8


Shape = tuple[str, ...]
"""Extents of one temporary, by dimension name (``("nc", "ng")``)."""

_COMPONENT_EXTENT = "nc"


def _shape(extents: int | Shape) -> Shape:
    """Return the shape for a rank (an ``nc``-sized array) or a shape."""
    if isinstance(extents, int):
        return (_COMPONENT_EXTENT,) * extents
    return tuple(extents)


def _dims(shape: Shape, array_pad: int, max_nc: int | None) -> list[str]:
    """Return the extents of one temporary as Fortran expressions.

    ``max_nc`` replaces only the ``nc`` extents: an extra dimension (``ng``)
    has no compile-time bound and stays automatic.  The padding goes on the
    leading dimension of rank >= 2 temporaries (of every one with
    ``max_nc``).
    """
    dims = [
        (
            str(max_nc)
            if max_nc is not None and extent == _COMPONENT_EXTENT
            else extent
        )
        for extent in shape
    ]
    if array_pad and (max_nc is not None or len(dims) >= 2):
        lead = dims[0]
        dims[0] = (
            str(int(lead) + array_pad)
            if lead.isdigit()
            else f"{lead} + {int(array_pad)}"
        )
    return dims


def _warn_static(
    arrays: Mapping[str, Shape], max_nc: int, array_pad: int
) -> None:
    """Warn when a fixed-size temporary exceeds gfortran's stack limit."""
    sizes = []
    for shape in arrays.values():
        dims = _dims(shape, array_pad, max_nc)
        if all(dim.isdigit() for dim in dims):
            sizes.append((math.prod(int(dim) for dim in dims), len(dims)))
    if not sizes:
        return
    size, largest = max(sizes)
    size *= _REAL_BYTES
    if size > GFORTRAN_MAX_STACK_VAR_SIZE:
        warnings.warn(
            f"max_nc={max_nc}: a rank-{largest} temporary takes {size} "
            f"bytes, above gfortran's -fmax-stack-var-size default "
            f"({GFORTRAN_MAX_STACK_VAR_SIZE}); gfortran makes it static "
            "(not thread safe) unless -frecursive or -fopenmp is used",
            UserWarning,
            stacklevel=2,
        )


def _by_shape(arrays: Mapping[str, int | Shape]) -> dict[str, Shape]:
    """Normalise ``arrays`` to shapes."""
    return {name: _shape(extents) for name, extents in arrays.items()}


def array_declarations(
    arrays: Mapping[str, int | Shape],
    array_pad: int,
    max_nc: int | None,
) -> list[str]:
    """Return the declaration lines of the temporary arrays, by shape.

    Parameters
    ----------
    arrays : mapping
        Temporary name to its rank (``nc``-sized) or its shape, a tuple of
        extent names such as ``("nc", "ng")``.
    array_pad : int
        Extra rows of the leading dimension of rank >= 2 temporaries (with
        ``max_nc``, of every temporary).
    max_nc : int or None
        Compile-time extent of every ``nc`` dimension, ``None`` for
        ``nc``.  Extents over an extra dimension stay automatic.

    Returns
    -------
    list of str
        Indented declaration lines.
    """
    shapes_by_name = _by_shape(arrays)
    if max_nc is not None and shapes_by_name:
        _warn_static(shapes_by_name, max_nc, array_pad)
    lines = []
    for shape in sorted(set(shapes_by_name.values()), key=_shape_key):
        dims = ", ".join(_dims(shape, array_pad, max_nc))
        names = [
            f"{name}({dims})"
            for name in sorted(shapes_by_name)
            if shapes_by_name[name] == shape
        ]
        for line in _wrap_name_list("real(8) :: ", names):
            lines.append(_INDENT + line)
    return lines


def _shape_key(shape: Shape) -> tuple[int, Shape]:
    """Sort key of the declaration groups: rank, then extents."""
    return (len(shape), shape)


def max_nc_guard(sub_name: str, max_nc: int) -> list[str]:
    """Return the body lines that stop when ``nc`` exceeds ``max_nc``."""
    return [
        f"if (nc > {max_nc}) error stop "
        f"'{sub_name}: nc > max_nc = {max_nc}'",
        "",
    ]


def _ordered_arrays(arrays: Mapping[str, Shape]) -> list[tuple[str, Shape]]:
    """Return ``(name, shape)`` in declaration order (shape, then name)."""
    return sorted(
        arrays.items(), key=lambda item: (_shape_key(item[1]), item[0])
    )


def _size_expr(shape: Shape, array_pad: int) -> str:
    """Return the element count of one automatic temporary of ``shape``."""
    return "*".join(
        f"({dim})" if "+" in dim else dim
        for dim in _dims(shape, array_pad, None)
    )


def _sum_expr(counts: Mapping[Shape, int], array_pad: int) -> str:
    """Return the total size of ``counts[shape]`` temporaries per shape."""
    terms = []
    for shape in sorted(counts, key=_shape_key):
        size = _size_expr(shape, array_pad)
        terms.append(size if counts[shape] == 1 else f"{counts[shape]}*{size}")
    return " + ".join(terms)


def _work_slices(
    arrays: Mapping[str, Shape], array_pad: int, work: str
) -> tuple[list[str], str]:
    """Return the actual arguments ``work_(offset)`` and the total size."""
    counts: dict[Shape, int] = {}
    actuals = []
    for _name, shape in _ordered_arrays(arrays):
        offset = _sum_expr(counts, array_pad)
        actuals.append(f"{work}(1 + {offset})" if offset else f"{work}(1)")
        counts[shape] = counts.get(shape, 0) + 1
    return actuals, _sum_expr(counts, array_pad)


def _unique(name: str, taken: Sequence[str]) -> str:
    """Append ``_`` to ``name`` until it is not in ``taken`` (any case)."""
    lowered = {item.lower() for item in taken}
    while name.lower() in lowered:
        name += "_"
    return name


def work_array_kernel(
    sub_name: str,
    arg_names: list[str],
    header: list[str],
    local_decls: list[str],
    body: list[str],
    arrays: Mapping[str, int | Shape],
    array_pad: int,
) -> list[str]:
    """Return the lines of a kernel whose temporaries share one work array.

    Parameters
    ----------
    sub_name : str
        Subroutine name.
    arg_names : list of str
        Dummy arguments of the kernel.
    header : list of str
        Indented declarations of the dummy arguments.
    local_decls : list of str
        Indented declarations of the scalar temporaries, the array
        temporaries (explicit shape, ``nc``-sized) and the loop indices.
    body : list of str
        Body lines, indented once.
    arrays : mapping
        Temporary name to its rank (``nc``-sized) or its shape.  The work
        array holds the product of the extents of every temporary.
    array_pad : int
        Extra rows of the leading dimension of rank >= 2 temporaries.

    Returns
    -------
    list of str
        The subroutine, ending with ``end subroutine``.
    """
    shapes = _by_shape(arrays)
    taken = [sub_name, *arg_names, *shapes]
    work = _unique("work_", taken)
    inner = _unique(f"{sub_name}_body", [*taken, work])
    actuals, total = _work_slices(shapes, array_pad, work)
    dummies = [name for name, _dims_ in _ordered_arrays(shapes)]
    lines = _wrap_signature(sub_name, arg_names)
    lines.append(f"{_INDENT}implicit none")
    lines.extend(header)
    lines.append(f"{_INDENT}real(8) :: {work}({total})")
    lines.append("")
    call = _wrap_signature(inner, arg_names + actuals, keyword="call")
    lines.extend(_INDENT + line for line in call)
    lines.append("contains")
    inner_lines = _wrap_signature(inner, arg_names + dummies)
    inner_lines.append(f"{_INDENT}implicit none")
    inner_lines.extend(header)
    inner_lines.extend(local_decls)
    inner_lines.append("")
    inner_lines.extend(body)
    inner_lines.append(f"end subroutine {inner}")
    lines.extend(_INDENT + line if line else "" for line in inner_lines)
    lines.append(f"end subroutine {sub_name}")
    return lines
