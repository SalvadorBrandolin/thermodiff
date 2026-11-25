"""Definition rendering for the Fortran code generator."""

import functools
from collections.abc import Collection, Iterable, Mapping
from typing import cast

import sympy as sp

from thermodiff.codegen.common import (
    _INDENT,
    _Namer,
    _emission_order,
    _free_indices,
    _indent_lines,
    _print_expr,
    _temp_name,
    fortran_code,
)
from thermodiff.codegen.extents import index_extent
from thermodiff.codegen.hoisting import _split_factored
from thermodiff.codegen.ir import (
    ComponentFusedMap,
    FusedNest,
    OuterFusedNest,
    SiblingNest,
    SparseGuard,
    SparseNest,
    TempDef,
)
from thermodiff.codegen.symmetry import _render_symv


def _enclosing_bound_indices(
    root: sp.Basic,
) -> dict[sp.Sum, set[sp.Idx]]:
    """Map every Sum in ``root`` to the indices bound by its ancestors.

    Parameters
    ----------
    root : sympy.Basic
        Expression that may contain nested summations.

    Returns
    -------
    dict
        For each ``Sum`` in ``root``, the indices bound by the sums that
        enclose it.
    """
    found: dict[sp.Sum, set[sp.Idx]] = {}
    _collect_enclosing_bound(root, frozenset(), found)
    return found


def _collect_enclosing_bound(
    sub: sp.Basic,
    bound: frozenset[sp.Idx],
    found: dict[sp.Sum, set[sp.Idx]],
) -> None:
    """Record into ``found`` the indices bound above each Sum of ``sub``.

    Parameters
    ----------
    sub : sympy.Basic
        Sub-expression being walked.
    bound : frozenset of sympy.Idx
        Indices bound by the sums enclosing ``sub``.
    found : dict
        Accumulator, updated in place.
    """
    if isinstance(sub, sp.Sum):
        found.setdefault(sub, set()).update(bound)
        bound = bound | {limit[0] for limit in sub.limits}
        _collect_enclosing_bound(sub.function, bound, found)
        return
    for arg in getattr(sub, "args", ()):
        _collect_enclosing_bound(arg, bound, found)


def _expand_sums(expr: sp.Expr, namer: _Namer) -> tuple[list[str], sp.Expr]:
    """Expand ``Sum`` sub-expressions into loop statements.

    Parameters
    ----------
    expr : sympy expression
        Expression that may contain summations.
    namer : _Namer
        Generator of temporary variable names.

    Returns
    -------
    statements : list of str
        The generated assignment and loop statements, innermost first.
    residual : sympy expression
        ``expr`` with every ``Sum`` replaced by a temporary symbol.
    """
    statements: list[str] = []
    residual = expr

    while True:
        outer = _enclosing_bound_indices(residual)
        # an innermost sum can be computed before everything else only if
        # it does not depend on an index bound by an enclosing sum
        innermost = [
            total
            for total in sorted(
                residual.atoms(sp.Sum), key=sp.default_sort_key
            )
            if not total.function.atoms(sp.Sum)
            and not (_free_indices(total) & outer.get(total, set()))
        ]
        if not innermost:
            break
        for total in innermost:
            tmp = namer("t")
            statements.extend(_sum_loop_lines(tmp, total, [], total.function))
            residual = residual.xreplace({total: tmp})

    # remaining sums contain inner sums that depend on their bound indices
    # (e.g. sum_i n_i/(sum_j n_j G_ji)): the inner sums are expanded inside
    # the loops of the enclosing one
    remaining = sorted(
        (
            total
            for total in residual.atoms(sp.Sum)
            if not _enclosing_bound_indices(residual).get(total)
        ),
        key=sp.default_sort_key,
    )
    for total in remaining:
        tmp = namer("t")
        inner_statements, body = _expand_sums(total.function, namer)
        statements.extend(_sum_loop_lines(tmp, total, inner_statements, body))
        residual = residual.xreplace({total: tmp})

    return statements, residual


def _sum_loop_lines(
    tmp: sp.Symbol,
    total: sp.Sum,
    inner_lines: list[str],
    body: sp.Expr,
) -> list[str]:
    """Return the Fortran loop nest that accumulates ``total`` in ``tmp``.

    Parameters
    ----------
    tmp : sympy.Symbol
        Temporary that receives the sum.
    total : sympy.Sum
        Summation whose limits define the ``do`` headers.
    inner_lines : list of str
        Statements emitted inside the innermost loop before the update.
    body : sympy expression
        Summand added to ``tmp`` at each iteration.

    Returns
    -------
    list of str
        Initialisation, loop headers, inner lines, update and closers.
    """
    lines = [f"{tmp} = 0.0d0"]
    for depth, (idx, low, high) in enumerate(total.limits):
        lines.append(
            f"{_INDENT * depth}do {idx} = "
            f"{_print_expr(low)}, {_print_expr(high)}"
        )
    depth = len(total.limits)
    lines.extend(_INDENT * depth + line for line in inner_lines)
    lines.append(f"{_INDENT * depth}{tmp} = {tmp} + {_print_expr(body)}")
    for depth in range(len(total.limits) - 1, -1, -1):
        lines.append(f"{_INDENT * depth}end do")
    return lines


def _declare_output(
    name: str, rank: int, optional: bool = False, contiguous: bool = False
) -> str:
    """Return the declaration line for an output argument.

    With ``contiguous=True`` assumed-shape outputs carry the ``contiguous``
    attribute (option ``contiguous_args``).
    """
    if optional:
        attributes = "real(8), optional, intent(out)"
    else:
        attributes = "real(8), intent(out)"
    if rank == 0:
        return f"{attributes} :: {name}"
    if contiguous:
        attributes = attributes.replace("real(8), ", "real(8), contiguous, ")
    dims = ", ".join([":"] * rank)
    return f"{attributes} :: {name}({dims})"


def _render_output(
    name: str,
    expr: sp.Expr,
    free_order: list[sp.Idx],
    namer: _Namer,
    mirror: bool = False,
    column: bool = False,
) -> list[str]:
    """Render the statements that compute a single output.

    With ``mirror=True`` (rank-2 outputs known to be symmetric) only the
    upper triangle is computed and every value is stored twice.  With
    ``column=True`` the loops are opened in reverse order (the first
    subscript innermost, column-major); the subscripts are unchanged.
    """
    statements, residual = _expand_sums(expr, namer)
    residual = residual.xreplace(
        {idx: sp.Symbol(str(idx)) for idx in free_order}
    )

    mirror = mirror and len(free_order) == 2
    lines = []
    depth = 0
    loop_order = (
        list(reversed(free_order)) if column and not mirror else free_order
    )
    for position, idx in enumerate(loop_order):
        low = str(free_order[0]) if (mirror and position == 1) else "1"
        lines.append(
            _INDENT * depth + f"do {idx} = {low}, {index_extent(idx)}"
        )
        depth += 1

    for statement in statements:
        lines.append(_INDENT * depth + statement)

    body = fortran_code(residual, assign_to="res")
    lines.extend(_indent_lines(body, depth))

    if free_order:
        subscript = ", ".join(str(idx) for idx in free_order)
        lines.append(_INDENT * depth + f"{name}({subscript}) = res")
        if mirror:
            transposed = ", ".join(str(idx) for idx in reversed(free_order))
            lines.append(_INDENT * depth + f"{name}({transposed}) = res")
    else:
        lines.append(_INDENT * depth + f"{name} = res")

    for _ in free_order:
        depth -= 1
        lines.append(_INDENT * depth + "end do")

    return lines


def _render_column_order(definition: TempDef) -> list[str]:
    """Loop nest with the first subscript innermost (column-major)."""
    target = definition.target
    free = definition.free
    limits = definition.limits
    target_s = _print_expr(target)
    body_s = _print_expr(definition.body)
    if limits:
        (f,) = free
        b, low, high = limits[0]
        return [
            f"do {f} = 1, {index_extent(f)}",
            f"{_INDENT}{target_s} = 0.0d0",
            "end do",
            f"do {b} = {_print_expr(low)}, {_print_expr(high)}",
            f"{_INDENT}do {f} = 1, {index_extent(f)}",
            f"{_INDENT * 2}{target_s} = {target_s} + {body_s}",
            f"{_INDENT}end do",
            "end do",
        ]
    p_, q_ = free
    return [
        f"do {q_} = 1, {index_extent(q_)}",
        f"{_INDENT}do {p_} = 1, {index_extent(p_)}",
        f"{_INDENT * 2}{target_s} = {body_s}",
        f"{_INDENT}end do",
        "end do",
    ]


def _render_column_sum(definition: TempDef) -> list[str]:
    """Rank-2 sum in ``jki`` order.

    For every column ``q``, axpy updates over the bound index; a
    symmetric (``mirror_sum``) sum fills ``p <= q`` and mirrors the
    row.
    """
    target = definition.target
    p_, q_ = definition.free
    b, low, high = definition.limits[0]
    target_s = _print_expr(target)
    upper = q_ if definition.mirror_sum else index_extent(p_)
    lines = [
        f"do {q_} = 1, {index_extent(q_)}",
        f"{_INDENT}do {p_} = 1, {upper}",
        f"{_INDENT * 2}{target_s} = 0.0d0",
        f"{_INDENT}end do",
        f"{_INDENT}do {b} = {_print_expr(low)}, {_print_expr(high)}",
        f"{_INDENT * 2}do {p_} = 1, {upper}",
        f"{_INDENT * 3}{target_s} = {target_s} + "
        f"{_print_expr(definition.body)}",
        f"{_INDENT * 2}end do",
        f"{_INDENT}end do",
    ]
    if definition.mirror_sum:
        transposed = _print_expr(sp.Indexed(target.base, q_, p_))
        lines += [
            f"{_INDENT}do {p_} = 1, {q_} - 1",
            f"{_INDENT * 2}{transposed} = {target_s}",
            f"{_INDENT}end do",
        ]
    lines.append("end do")
    return lines


def _render_fused(definition: FusedNest) -> list[str]:
    group = definition.fused
    f = group[0].free[0]
    b, low, high = group[0].limits[0]
    colmajor = bool(group[0].colmajor)
    extent = index_extent(f)
    lines = [f"do {f} = 1, {extent}"]
    for d in group:
        lines.append(f"{_INDENT}{_print_expr(d.target)} = 0.0d0")
    lines.append("end do")
    outer, inner = (b, f) if colmajor else (f, b)
    free_extent = sp.Symbol(extent)
    lo_o, hi_o = (low, high) if colmajor else (1, free_extent)
    lo_i, hi_i = (1, free_extent) if colmajor else (low, high)
    lines.append(
        f"do {outer} = {_print_expr(sp.sympify(lo_o))}, "
        f"{_print_expr(sp.sympify(hi_o))}"
    )
    lines.append(
        f"{_INDENT}do {inner} = {_print_expr(sp.sympify(lo_i))}, "
        f"{_print_expr(sp.sympify(hi_i))}"
    )
    for d in group:
        t = _print_expr(d.target)
        lines.append(f"{_INDENT * 2}{t} = {t} + {_print_expr(d.body)}")
    lines.append(f"{_INDENT}end do")
    lines.append("end do")
    return lines


def _render_component_fused(definition: ComponentFusedMap) -> list[str]:
    idx = definition.free[0]
    lines = [f"do {idx} = 1, {index_extent(idx)}"]
    for _kind, target, value in definition.steps:
        lines.append(f"{_INDENT}{_print_expr(target)} = {_print_expr(value)}")
    lines.append("end do")
    return lines


def _is_reduction_member(member: TempDef) -> bool:
    """Whether an outer-fused member is a reduction without free indices."""
    return not isinstance(member, ComponentFusedMap) and not member.free


def _reduction_run_length(members: list[TempDef], start: int) -> int:
    """Count the consecutive reductions that share the inner loops.

    Parameters
    ----------
    members : list
        Members of the outer-fused nest.
    start : int
        Position of the first reduction of the run.

    Returns
    -------
    int
        Length of the run, at least one.
    """
    first = members[start]
    length = 1
    while (
        start + length < len(members)
        and _is_reduction_member(members[start + length])
        and members[start + length].limits[1:] == first.limits[1:]
    ):
        length += 1
    return length


def _render_reduction_run(run: list[TempDef]) -> list[str]:
    """Render consecutive reductions that share their inner loops.

    Parameters
    ----------
    run : list
        Reduction definitions with the same inner limits (``limits[1:]``).

    Returns
    -------
    list of str
        The lines, indented one level inside the outer loop.
    """
    first = run[0]
    lines = []
    depth = 1
    for member in run:
        if member.factored:
            lines.append(_INDENT + f"{member.acc} = 0.0d0")
    for idx, low, high in first.limits[1:]:
        lines.append(
            _INDENT * depth
            + f"do {idx} = {_print_expr(low)}, {_print_expr(high)}"
        )
        depth += 1
    for member in run:
        if member.factored:
            _outer_factor, inner_body = _split_factored(member, member.limits)
            lines.append(
                _INDENT * depth
                + f"{member.acc} = {member.acc} + {_print_expr(inner_body)}"
            )
            continue
        target = _print_expr(member.target)
        lines.append(
            _INDENT * depth
            + f"{target} = {target} + {_print_expr(member.body)}"
        )
    for _ in first.limits[1:]:
        depth -= 1
        lines.append(_INDENT * depth + "end do")
    for member in run:
        if member.factored:
            target = _print_expr(member.target)
            outer_factor, _inner_body = _split_factored(member, member.limits)
            lines.append(
                _INDENT
                + f"{target} = {target} + "
                + f"{_print_expr(member.acc * outer_factor)}"
            )
    return lines


def _render_outer_fused(definition: OuterFusedNest) -> list[str]:
    """Render the members of an outer-fused nest in one loop over ``outer``.

    Parameters
    ----------
    definition : OuterFusedNest
        The nest; its members are component maps, free maps and reductions.

    Returns
    -------
    list of str
        The Fortran lines: the zeroing of the plain reductions, then the
        outer loop with every member in order.
    """
    outer = definition.outer
    lines = []
    for member in definition.members:
        if _is_reduction_member(member):
            lines.append(f"{_print_expr(member.target)} = 0.0d0")
    lines.append(f"do {outer} = 1, {index_extent(outer)}")
    members = definition.members
    position = 0
    while position < len(members):
        member = members[position]
        if isinstance(member, ComponentFusedMap):
            for _kind, target, value in member.steps:
                lines.append(
                    f"{_INDENT}{_print_expr(target)} = {_print_expr(value)}"
                )
            position += 1
        elif member.free:
            lines.append(
                f"{_INDENT}{_print_expr(member.target)} = "
                f"{_print_expr(member.body)}"
            )
            position += 1
        else:
            # consecutive reductions with the same inner loops share them
            stop = position + _reduction_run_length(members, position)
            lines.extend(_render_reduction_run(members[position:stop]))
            position = stop
    lines.append("end do")
    return lines


def _render_sibling(definition: SiblingNest) -> list[str]:
    a, b = definition.free
    lines = [
        f"do {b} = 1, {index_extent(b)}",
        f"{_INDENT}do {a} = 1, {b}",
    ]
    for acc in definition.accs:
        lines.append(f"{_INDENT * 2}{acc} = 0.0d0")
    m, lo, hi = definition.limits[0]
    if definition.accumulators:
        # two partial accumulators (even/odd m): a reduction gcc vectorises
        # without reassociation (reportes/brechas/ronda3-cmr-o3.md)
        accs2 = cast(list[sp.Symbol], definition.accs2)
        pairs = list(zip(definition.accs, accs2, definition.sibling))
        for _acc, acc2, _g in pairs:
            lines.append(f"{_INDENT * 2}{acc2} = 0.0d0")
        lines.append(
            f"{_INDENT * 2}do {m} = {_print_expr(lo)}, "
            f"{_print_expr(hi - 1)}, 2"
        )
        for acc, acc2, g in pairs:
            lines.append(f"{_INDENT * 3}{acc} = {acc} + {_print_expr(g.body)}")
            lines.append(
                f"{_INDENT * 3}{acc2} = {acc2} + "
                f"{_print_expr(g.body.xreplace({m: m + 1}))}"
            )
        lines.append(f"{_INDENT * 2}end do")
        # remainder: at most one iteration, from the first m not done
        lines.append(f"{_INDENT * 2}do {m} = {m}, {_print_expr(hi)}")
        for _acc, acc2, g in pairs:
            lines.append(
                f"{_INDENT * 3}{acc2} = {acc2} + {_print_expr(g.body)}"
            )
        lines.append(f"{_INDENT * 2}end do")
        for acc, acc2, _g in pairs:
            lines.append(f"{_INDENT * 2}{acc} = {acc} + {acc2}")
    else:
        lines.append(
            f"{_INDENT * 2}do {m} = {_print_expr(lo)}, {_print_expr(hi)}"
        )
        for acc, g in zip(definition.accs, definition.sibling):
            lines.append(f"{_INDENT * 3}{acc} = {acc} + {_print_expr(g.body)}")
        lines.append(f"{_INDENT * 2}end do")
    for acc, g in zip(definition.accs, definition.sibling):
        t = g.target
        lines.append(f"{_INDENT * 2}{_print_expr(t)} = {acc}")
        lines.append(
            f"{_INDENT * 2}{_print_expr(sp.Indexed(t.base, b, a))} = {acc}"
        )
    lines += [f"{_INDENT}end do", "end do"]
    return lines


def _render_mirror_sum_table(definition: TempDef) -> list[str]:
    """Upper triangle of a symmetric rank-2 summation, then its mirror."""
    a, b = definition.free
    target = definition.target
    target_s = _print_expr(target)
    transposed = _print_expr(sp.Indexed(target.base, b, a))
    out = [
        f"do {b} = 1, {index_extent(b)}",
        f"{_INDENT}do {a} = 1, {b}",
        f"{_INDENT * 2}{target_s} = 0.0d0",
    ]
    depth = 2
    for idx, low, high in definition.limits:
        out.append(
            _INDENT * depth
            + f"do {idx} = {_print_expr(low)}, {_print_expr(high)}"
        )
        depth += 1
    out.append(
        _INDENT * depth
        + f"{target_s} = {target_s} + {_print_expr(definition.body)}"
    )
    for _ in definition.limits:
        depth -= 1
        out.append(_INDENT * depth + "end do")
    out.append(_INDENT * 2 + f"{transposed} = {target_s}")
    out += [f"{_INDENT}end do", "end do"]
    return out


def _render_mirror_triangle(definition: TempDef) -> list[str]:
    """Symmetric rank-2 temporary: fill the upper triangle and mirror."""
    free = definition.free
    target = definition.target
    body_s = _print_expr(definition.body)
    target_s = _print_expr(target)
    transposed = _print_expr(sp.Indexed(target.base, free[1], free[0]))
    return [
        f"do {free[0]} = 1, {index_extent(free[0])}",
        f"{_INDENT}do {free[1]} = {free[0]}, {index_extent(free[1])}",
        _INDENT * 2 + f"{target_s} = {body_s}",
        _INDENT * 2 + f"{transposed} = {target_s}",
        _INDENT + "end do",
        "end do",
    ]


def _render_factored_reduction(definition: TempDef) -> list[str]:
    """Double reduction whose inner sum is accumulated in a temporary."""
    limits = definition.limits
    target_s = _print_expr(definition.target)
    outer_f, inner_body = _split_factored(definition, limits)
    acc = definition.acc
    (o, o_lo, o_hi), (x, x_lo, x_hi) = limits
    return [
        f"{target_s} = 0.0d0",
        f"do {o} = {_print_expr(o_lo)}, {_print_expr(o_hi)}",
        f"{_INDENT}{acc} = 0.0d0",
        f"{_INDENT}do {x} = {_print_expr(x_lo)}, {_print_expr(x_hi)}",
        f"{_INDENT * 2}{acc} = {acc} + {_print_expr(inner_body)}",
        f"{_INDENT}end do",
        f"{_INDENT}{target_s} = {target_s} + {_print_expr(acc * outer_f)}",
        "end do",
    ]


def _render_loop_nest(definition: TempDef) -> list[str]:
    """Free loops around an assignment or a reduction."""
    target_s = _print_expr(definition.target)
    limits = definition.limits
    lines = []
    depth = 0
    for idx in definition.free:
        lines.append(_INDENT * depth + f"do {idx} = 1, {index_extent(idx)}")
        depth += 1
    if limits:
        lines.append(_INDENT * depth + f"{target_s} = 0.0d0")
        for idx, low, high in limits:
            low_s = _print_expr(low)
            high_s = _print_expr(high)
            lines.append(_INDENT * depth + f"do {idx} = {low_s}, {high_s}")
            depth += 1
        body_s = _print_expr(definition.body)
        lines.append(_INDENT * depth + f"{target_s} = {target_s} + {body_s}")
        for _ in limits:
            depth -= 1
            lines.append(_INDENT * depth + "end do")
    else:
        body_s = _print_expr(definition.body)
        lines.append(_INDENT * depth + f"{target_s} = {body_s}")

    for _ in definition.free:
        depth -= 1
        lines.append(_INDENT * depth + "end do")
    return lines


def _sparse_test(guard: SparseGuard) -> str:
    """Return the Fortran condition that the sparse element is non-zero."""
    return f"{_print_expr(guard.factor)} /= 0.0d0"


def _render_guarded_reduction(
    definition: SparseNest, free_depth: int
) -> list[str]:
    """Reduce with the terms that carry the sparse element under its test."""
    guard = definition.guard
    target_s = _print_expr(definition.target)
    depth = free_depth
    lines = [_INDENT * depth + f"{target_s} = 0.0d0"]
    for idx, low, high in definition.limits:
        low_s = _print_expr(low)
        high_s = _print_expr(high)
        lines.append(_INDENT * depth + f"do {idx} = {low_s}, {high_s}")
        depth += 1
    accumulate = _INDENT * depth + f"{target_s} = {target_s} + "
    if guard.rest != 0:
        rest_s = _print_expr(guard.rest)
        if rest_s.startswith("-"):
            rest_s = f"({rest_s})"
        lines.append(accumulate + rest_s)
    lines.append(_INDENT * depth + f"if ({_sparse_test(guard)}) then")
    lines.append(_INDENT + accumulate + _print_expr(guard.guarded))
    lines.append(_INDENT * depth + "end if")
    for _ in definition.limits:
        depth -= 1
        lines.append(_INDENT * depth + "end do")
    return lines


def _render_sparse_nest(definition: SparseNest) -> list[str]:
    """Free loops around a guarded table entry or a guarded reduction."""
    guard = definition.guard
    lines = []
    depth = 0
    for idx in definition.free:
        lines.append(_INDENT * depth + f"do {idx} = 1, {index_extent(idx)}")
        depth += 1
    if guard.table:
        entry = definition.copy(free=[])
        lines.append(_INDENT * depth + f"if ({_sparse_test(guard)}) then")
        inner = "\n".join(_render_loop_nest(entry))
        lines.extend(_indent_lines(inner, depth + 1))
        if guard.zero_fill:
            target_s = _print_expr(definition.target)
            lines.append(_INDENT * depth + "else")
            lines.append(_INDENT * (depth + 1) + f"{target_s} = 0.0d0")
        lines.append(_INDENT * depth + "end if")
    else:
        lines.extend(_render_guarded_reduction(definition, depth))
    for _ in definition.free:
        depth -= 1
        lines.append(_INDENT * depth + "end do")
    return lines


@functools.singledispatch
def _render_definition(definition: TempDef) -> list[str]:
    """Render the statements that fill one shared temporary.

    The base handler covers plain and annotated temporaries; the
    annotations are tested in a fixed order (``symv``, ``colmajor``,
    ``colsum``, ``mirror_sum``, ``mirror``, ``factored``, ``limits``).
    The wrapper records have their own registered handlers.
    """
    if definition.symv:
        return _render_symv(definition)
    if definition.colmajor:
        return _render_column_order(definition)
    if definition.colsum:
        return _render_column_sum(definition)
    free = definition.free
    if definition.mirror_sum and len(free) == 2 and definition.limits:
        return _render_mirror_sum_table(definition)
    if definition.mirror and len(free) == 2 and not definition.limits:
        return _render_mirror_triangle(definition)
    if definition.factored and not free:
        return _render_factored_reduction(definition)
    return _render_loop_nest(definition)


@_render_definition.register
def _(definition: SparseNest) -> list[str]:
    """Render a nest guarded by a sparse parameter."""
    return _render_sparse_nest(definition)


@_render_definition.register
def _(definition: SiblingNest) -> list[str]:
    """Render a sibling table nest."""
    return _render_sibling(definition)


@_render_definition.register
def _(definition: OuterFusedNest) -> list[str]:
    """Render an outer-fused nest."""
    return _render_outer_fused(definition)


@_render_definition.register
def _(definition: ComponentFusedMap) -> list[str]:
    """Render a component-fused map."""
    return _render_component_fused(definition)


@_render_definition.register
def _(definition: FusedNest) -> list[str]:
    """Render a fused vector nest."""
    return _render_fused(definition)


def _dependency_order(definitions: list[TempDef]) -> list[TempDef]:
    """Stable topological order of the definitions.

    A definition goes after every one it reads, otherwise in its
    original position (lowest index first).
    """
    import heapq

    index = {_temp_name(d.target): k for k, d in enumerate(definitions)}
    deps: list[set[int]] = []
    for k, d in enumerate(definitions):
        body = d.body
        names = {str(x) for x in body.free_symbols} | {
            str(x.base) for x in body.atoms(sp.Indexed)
        }
        deps.append(
            {index[nm] for nm in names if nm in index and index[nm] != k}
        )
    users: dict[int, list[int]] = {k: [] for k in range(len(definitions))}
    for k, ds in enumerate(deps):
        for dep in ds:
            users[dep].append(k)
    pending = [len(ds) for ds in deps]
    ready = [k for k, count in enumerate(pending) if count == 0]
    heapq.heapify(ready)
    order: list[TempDef] = []
    while ready:
        k = heapq.heappop(ready)
        order.append(definitions[k])
        for user in users[k]:
            pending[user] -= 1
            if pending[user] == 0:
                heapq.heappush(ready, user)
    if len(order) != len(definitions):
        raise ValueError("cyclic temporaries")
    return order


def _guard(attributes: Collection[str], out_names: Mapping[str, str]) -> str:
    """Return the ``present(...)`` condition for a group of outputs.

    Returns an empty string when nothing depends on the group.
    """
    if not attributes:
        return ""
    names: list[str] = []
    if "value" in attributes and "value" in out_names:
        names.append(out_names["value"])
    names.extend(
        out_names[attr]
        for attr in _emission_order(out_names)
        if attr in attributes and attr != "value"
    )
    return " .or. ".join(f"present({name})" for name in names)


def _wrap_if(condition: str, limit: int = 120) -> list[str]:
    """Wrap an ``if (condition) then`` line with continuations."""
    pieces = condition.split(" .or. ")
    lines = []
    current = "if (" + pieces[0]
    for piece in pieces[1:]:
        token = " .or. " + piece
        if len(current) + len(token) + len(") then") > limit:
            lines.append(current + " .or. &")
            current = "     " + piece
        else:
            current += token
    lines.append(current + ") then")
    return lines


def _guarded(lines: Iterable[str], condition: str) -> list[str]:
    """Wrap ``lines`` in an ``if (condition)`` block when needed."""
    if not condition:
        return list(lines)
    block = _wrap_if(condition)
    block.extend((_INDENT + line) if line else line for line in lines)
    block.append("end if")
    return block


def _wrap_name_list(
    keyword: str, names: Iterable[str], limit: int = 120
) -> list[str]:
    """Split ``keyword`` plus names into continuation-free lines.

    Parameters
    ----------
    keyword : str
        Declaration prefix including the ``::``, for example
        ``"real(8) :: "``.
    names : list of str
        Names to declare.
    limit : int, optional
        Maximum line length.

    Returns
    -------
    list of str
        One or more declaration lines.
    """
    lines: list[str] = []
    current: str | None = None
    for name in names:
        if current is None:
            current = name
        elif len(keyword) + len(current) + len(name) + 2 > limit:
            lines.append(keyword + current)
            current = name
        else:
            current += ", " + name
    if current is not None:
        lines.append(keyword + current)
    return lines


def _wrap_signature(
    sub_name: str,
    arg_names: list[str],
    limit: int = 120,
    keyword: str = "subroutine",
) -> list[str]:
    """Return the subroutine (or call) statement, wrapped if necessary.

    Parameters
    ----------
    sub_name : str
        Subroutine name.
    arg_names : list of str
        Argument names.
    limit : int, optional
        Maximum line length.
    keyword : str, optional
        Leading keyword, ``"subroutine"`` or ``"call"``.

    Returns
    -------
    list of str
        The (possibly continued) statement.
    """
    head = f"{keyword} {sub_name}("
    if len(head) + len(", ".join(arg_names)) <= limit:
        return [head + ", ".join(arg_names) + ")"]

    continuation = _INDENT + "  "
    lines: list[str] = []
    current = ""
    for arg in arg_names:
        candidate = arg if not current else current + ", " + arg
        if current and len(head) + len(candidate) > limit:
            lines.append(head + current + ", &")
            head = continuation
            current = arg
        else:
            current = candidate
    lines.append(head + current + ")")
    return lines
