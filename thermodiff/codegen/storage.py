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

import warnings
from collections.abc import Mapping, Sequence

from thermodiff.codegen.common import _INDENT
from thermodiff.codegen.render import _wrap_name_list, _wrap_signature

#: gfortran's default ``-fmax-stack-var-size`` (bytes).  A larger local
#: array is made static (``SAVE``), which is not thread safe.
GFORTRAN_MAX_STACK_VAR_SIZE = 65536

_REAL_BYTES = 8


def _fixed_dims(rank: int, max_nc: int, array_pad: int) -> list[str]:
    """Return the literal extents of a fixed-size temporary."""
    return [str(max_nc + array_pad)] + [str(max_nc)] * (rank - 1)


def _automatic_dims(rank: int, array_pad: int) -> list[str]:
    """Return the ``nc``-based extents of an automatic temporary."""
    if array_pad and rank >= 2:
        # padded leading dimension: avoids cache-set aliasing between the
        # many (nc, nc) temporaries for some values of nc
        return [f"nc + {int(array_pad)}"] + ["nc"] * (rank - 1)
    return ["nc"] * rank


def _warn_static(arrays: Mapping[str, int], max_nc: int, pad: int) -> None:
    """Warn when a fixed-size temporary exceeds gfortran's stack limit."""
    largest = max(arrays.values())
    size = (max_nc + pad) * max_nc ** (largest - 1) * _REAL_BYTES
    if size > GFORTRAN_MAX_STACK_VAR_SIZE:
        warnings.warn(
            f"max_nc={max_nc}: a rank-{largest} temporary takes {size} "
            f"bytes, above gfortran's -fmax-stack-var-size default "
            f"({GFORTRAN_MAX_STACK_VAR_SIZE}); gfortran makes it static "
            "(not thread safe) unless -frecursive or -fopenmp is used",
            UserWarning,
            stacklevel=2,
        )


def array_declarations(
    arrays: Mapping[str, int], array_pad: int, max_nc: int | None
) -> list[str]:
    """Return the declaration lines of the temporary arrays, by rank.

    Parameters
    ----------
    arrays : mapping
        Temporary name to rank.
    array_pad : int
        Extra rows of the leading dimension of rank >= 2 temporaries (with
        ``max_nc``, of every temporary).
    max_nc : int or None
        Compile-time extent of every dimension, ``None`` for ``nc``.

    Returns
    -------
    list of str
        Indented declaration lines.
    """
    if max_nc is not None and arrays:
        _warn_static(arrays, max_nc, array_pad)
    lines = []
    for rank in sorted(set(arrays.values())):
        if max_nc is None:
            dims = ", ".join(_automatic_dims(rank, array_pad))
        else:
            dims = ", ".join(_fixed_dims(rank, max_nc, array_pad))
        names = [
            f"{name}({dims})"
            for name in sorted(arrays)
            if arrays[name] == rank
        ]
        for line in _wrap_name_list("real(8) :: ", names):
            lines.append(_INDENT + line)
    return lines


def max_nc_guard(sub_name: str, max_nc: int) -> list[str]:
    """Return the body lines that stop when ``nc`` exceeds ``max_nc``."""
    return [
        f"if (nc > {max_nc}) error stop "
        f"'{sub_name}: nc > max_nc = {max_nc}'",
        "",
    ]


def _ordered_arrays(arrays: Mapping[str, int]) -> list[tuple[str, int]]:
    """Return ``(name, rank)`` in declaration order (rank, then name)."""
    return sorted(arrays.items(), key=lambda item: (item[1], item[0]))


def _size_expr(rank: int, array_pad: int) -> str:
    """Return the element count of one automatic temporary of ``rank``."""
    return "*".join(
        f"({dim})" if "+" in dim else dim
        for dim in _automatic_dims(rank, array_pad)
    )


def _sum_expr(counts: Mapping[int, int], array_pad: int) -> str:
    """Return the total size of ``counts[rank]`` temporaries per rank."""
    terms = []
    for rank in sorted(counts):
        size = _size_expr(rank, array_pad)
        terms.append(size if counts[rank] == 1 else f"{counts[rank]}*{size}")
    return " + ".join(terms)


def _work_slices(
    arrays: Mapping[str, int], array_pad: int, work: str
) -> tuple[list[str], str]:
    """Return the actual arguments ``work_(offset)`` and the total size."""
    counts: dict[int, int] = {}
    actuals = []
    for _name, rank in _ordered_arrays(arrays):
        offset = _sum_expr(counts, array_pad)
        actuals.append(f"{work}(1 + {offset})" if offset else f"{work}(1)")
        counts[rank] = counts.get(rank, 0) + 1
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
    arrays: Mapping[str, int],
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
        Temporary name to rank.
    array_pad : int
        Extra rows of the leading dimension of rank >= 2 temporaries.

    Returns
    -------
    list of str
        The subroutine, ending with ``end subroutine``.
    """
    taken = [sub_name, *arg_names, *arrays]
    work = _unique("work_", taken)
    inner = _unique(f"{sub_name}_body", [*taken, work])
    actuals, total = _work_slices(arrays, array_pad, work)
    dummies = [name for name, _rank in _ordered_arrays(arrays)]
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
