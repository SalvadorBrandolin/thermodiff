"""Definition rendering for the Fortran code generator."""

import sympy as sp
from sympy.printing.fortran import fcode

from thermodiff.codegen.common import (
    _DERIV_ORDER,
    _FCODE_SETTINGS,
    _INDENT,
    _Namer,
    _free_indices,
    _indent_lines,
    _print_expr,
    _temp_name,
)
from thermodiff.codegen.hoisting import _split_factored
from thermodiff.codegen.symmetry import _render_symv


def _expand_sums(expr: sp.Expr, namer: _Namer):
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
    statements = []
    residual = expr

    def enclosing_bound(root):
        """Map every Sum in ``root`` to the indices bound by its ancestors."""
        found = {}

        def walk(sub, bound):
            if isinstance(sub, sp.Sum):
                found.setdefault(sub, set()).update(bound)
                bound = bound | {limit[0] for limit in sub.limits}
                walk(sub.function, bound)
                return
            for arg in getattr(sub, "args", ()):
                walk(arg, bound)

        walk(root, frozenset())
        return found

    while True:
        outer = enclosing_bound(residual)
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
            if not enclosing_bound(residual).get(total)
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
    inner_lines: list,
    body: sp.Expr,
) -> list:
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
    attribute (option ``contiguous_args``)."""
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
    free_order: list,
    namer,
    mirror=False,
    column=False,
) -> list:
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
        lines.append(_INDENT * depth + f"do {idx} = {low}, nc")
        depth += 1

    for statement in statements:
        lines.append(_INDENT * depth + statement)

    body = fcode(residual, assign_to="res", **_FCODE_SETTINGS)
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


def _render_column_order(definition) -> list:
    """Loop nest with the first subscript innermost (column-major)."""
    target = definition["target"]
    free = definition["free"]
    limits = definition["limits"]
    target_s = _print_expr(target)
    body_s = _print_expr(definition["body"])
    if limits:
        (f,) = free
        b, low, high = limits[0]
        return [
            f"do {f} = 1, nc",
            f"{_INDENT}{target_s} = 0.0d0",
            "end do",
            f"do {b} = {_print_expr(low)}, {_print_expr(high)}",
            f"{_INDENT}do {f} = 1, nc",
            f"{_INDENT * 2}{target_s} = {target_s} + {body_s}",
            f"{_INDENT}end do",
            "end do",
        ]
    p_, q_ = free
    return [
        f"do {q_} = 1, nc",
        f"{_INDENT}do {p_} = 1, nc",
        f"{_INDENT * 2}{target_s} = {body_s}",
        f"{_INDENT}end do",
        "end do",
    ]


def _render_column_sum(definition) -> list:
    """Rank-2 sum in ``jki`` order: for every column ``q``, axpy updates over
    the bound index; a symmetric (``mirror_sum``) sum fills ``p <= q`` and
    mirrors the row."""
    target = definition["target"]
    p_, q_ = definition["free"]
    b, low, high = definition["limits"][0]
    target_s = _print_expr(target)
    upper = q_ if definition.get("mirror_sum") else "nc"
    lines = [
        f"do {q_} = 1, nc",
        f"{_INDENT}do {p_} = 1, {upper}",
        f"{_INDENT * 2}{target_s} = 0.0d0",
        f"{_INDENT}end do",
        f"{_INDENT}do {b} = {_print_expr(low)}, {_print_expr(high)}",
        f"{_INDENT * 2}do {p_} = 1, {upper}",
        f"{_INDENT * 3}{target_s} = {target_s} + {_print_expr(definition['body'])}",
        f"{_INDENT * 2}end do",
        f"{_INDENT}end do",
    ]
    if definition.get("mirror_sum"):
        transposed = _print_expr(sp.Indexed(target.base, q_, p_))
        lines += [
            f"{_INDENT}do {p_} = 1, {q_} - 1",
            f"{_INDENT * 2}{transposed} = {target_s}",
            f"{_INDENT}end do",
        ]
    lines.append("end do")
    return lines


def _render_fused(definition) -> list:
    group = definition["fused"]
    f = group[0]["free"][0]
    b, low, high = group[0]["limits"][0]
    colmajor = bool(group[0].get("colmajor"))
    lines = [f"do {f} = 1, nc"]
    for d in group:
        lines.append(f"{_INDENT}{_print_expr(d['target'])} = 0.0d0")
    lines.append("end do")
    outer, inner = (b, f) if colmajor else (f, b)
    lo_o, hi_o = (low, high) if colmajor else (1, sp.Symbol("nc"))
    lo_i, hi_i = (1, sp.Symbol("nc")) if colmajor else (low, high)
    lines.append(
        f"do {outer} = {_print_expr(sp.sympify(lo_o))}, {_print_expr(sp.sympify(hi_o))}"
    )
    lines.append(
        f"{_INDENT}do {inner} = {_print_expr(sp.sympify(lo_i))}, {_print_expr(sp.sympify(hi_i))}"
    )
    for d in group:
        t = _print_expr(d["target"])
        lines.append(f"{_INDENT * 2}{t} = {t} + {_print_expr(d['body'])}")
    lines.append(f"{_INDENT}end do")
    lines.append("end do")
    return lines


def _render_component_fused(definition) -> list:
    idx = definition["free"][0]
    lines = [f"do {idx} = 1, nc"]
    for _kind, target, value in definition["steps"]:
        lines.append(f"{_INDENT}{_print_expr(target)} = {_print_expr(value)}")
    lines.append("end do")
    return lines


def _render_outer_fused(definition) -> list:
    x = definition["outer"]
    lines = []
    for g in definition["outerfused"]:
        if not g.get("compfused") and not g["free"]:
            lines.append(f"{_print_expr(g['target'])} = 0.0d0")
    lines.append(f"do {x} = 1, nc")
    members = definition["outerfused"]
    k = 0
    while k < len(members):
        g = members[k]
        if g.get("compfused"):
            for _k, t, v in g["steps"]:
                lines.append(f"{_INDENT}{_print_expr(t)} = {_print_expr(v)}")
            k += 1
        elif g["free"]:
            lines.append(
                f"{_INDENT}{_print_expr(g['target'])} = {_print_expr(g['body'])}"
            )
            k += 1
        else:
            # consecutive reductions with the same inner loops share them
            run = [g]
            while (
                k + len(run) < len(members)
                and not members[k + len(run)].get("compfused")
                and not members[k + len(run)]["free"]
                and members[k + len(run)]["limits"][1:] == g["limits"][1:]
            ):
                run.append(members[k + len(run)])
            depth = 1
            for r in run:
                if r.get("factored"):
                    lines.append(_INDENT + f"{r['acc']} = 0.0d0")
            for idx, low, high in g["limits"][1:]:
                lines.append(
                    _INDENT * depth
                    + f"do {idx} = {_print_expr(low)}, {_print_expr(high)}"
                )
                depth += 1
            for r in run:
                if r.get("factored"):
                    _o, inner_body = _split_factored(r, r["limits"])
                    lines.append(
                        _INDENT * depth
                        + f"{r['acc']} = {r['acc']} + {_print_expr(inner_body)}"
                    )
                    continue
                t = _print_expr(r["target"])
                lines.append(
                    _INDENT * depth + f"{t} = {t} + {_print_expr(r['body'])}"
                )
            for _ in g["limits"][1:]:
                depth -= 1
                lines.append(_INDENT * depth + "end do")
            for r in run:
                if r.get("factored"):
                    t = _print_expr(r["target"])
                    outer_f, _i = _split_factored(r, r["limits"])
                    lines.append(
                        _INDENT
                        + f"{t} = {t} + {_print_expr(r['acc'] * outer_f)}"
                    )
            k += len(run)
    lines.append("end do")
    return lines


def _render_sibling(definition) -> list:
    a, b = definition["free"]
    lines = [f"do {b} = 1, nc", f"{_INDENT}do {a} = 1, {b}"]
    for acc in definition["accs"]:
        lines.append(f"{_INDENT * 2}{acc} = 0.0d0")
    m, lo, hi = definition["limits"][0]
    if definition.get("split_acc"):
        # two partial accumulators (even/odd m): a reduction gcc vectorises
        # without reassociation (reportes/brechas/ronda3-cmr-o3.md)
        pairs = list(
            zip(definition["accs"], definition["accs2"], definition["sibling"])
        )
        for _acc, acc2, _g in pairs:
            lines.append(f"{_INDENT * 2}{acc2} = 0.0d0")
        lines.append(
            f"{_INDENT * 2}do {m} = {_print_expr(lo)}, {_print_expr(hi - 1)}, 2"
        )
        for acc, acc2, g in pairs:
            lines.append(
                f"{_INDENT * 3}{acc} = {acc} + {_print_expr(g['body'])}"
            )
            lines.append(
                f"{_INDENT * 3}{acc2} = {acc2} + "
                f"{_print_expr(g['body'].xreplace({m: m + 1}))}"
            )
        lines.append(f"{_INDENT * 2}end do")
        # remainder: at most one iteration, from the first m not done
        lines.append(f"{_INDENT * 2}do {m} = {m}, {_print_expr(hi)}")
        for _acc, acc2, g in pairs:
            lines.append(
                f"{_INDENT * 3}{acc2} = {acc2} + {_print_expr(g['body'])}"
            )
        lines.append(f"{_INDENT * 2}end do")
        for acc, acc2, _g in pairs:
            lines.append(f"{_INDENT * 2}{acc} = {acc} + {acc2}")
    else:
        lines.append(
            f"{_INDENT * 2}do {m} = {_print_expr(lo)}, {_print_expr(hi)}"
        )
        for acc, g in zip(definition["accs"], definition["sibling"]):
            lines.append(
                f"{_INDENT * 3}{acc} = {acc} + {_print_expr(g['body'])}"
            )
        lines.append(f"{_INDENT * 2}end do")
    for acc, g in zip(definition["accs"], definition["sibling"]):
        t = g["target"]
        lines.append(f"{_INDENT * 2}{_print_expr(t)} = {acc}")
        lines.append(
            f"{_INDENT * 2}{_print_expr(sp.Indexed(t.base, b, a))} = {acc}"
        )
    lines += [f"{_INDENT}end do", "end do"]
    return lines


def _render_definition(definition) -> list:
    """Render the statements that fill one shared temporary."""
    if definition.get("sibling"):
        return _render_sibling(definition)
    if definition.get("outerfused"):
        return _render_outer_fused(definition)
    if definition.get("compfused"):
        return _render_component_fused(definition)
    if definition.get("fused"):
        return _render_fused(definition)
    if definition.get("symv"):
        return _render_symv(definition)
    if definition.get("colmajor"):
        return _render_column_order(definition)
    if definition.get("colsum"):
        return _render_column_sum(definition)
    target = definition["target"]
    free = definition["free"]
    lines = []
    depth = 0
    for idx in free:
        lines.append(_INDENT * depth + f"do {idx} = 1, nc")
        depth += 1

    if (
        definition.get("mirror_sum")
        and len(free) == 2
        and definition["limits"]
    ):
        # symmetric rank-2 summation: upper triangle, then mirror
        a, b = free
        target_s = _print_expr(target)
        transposed = _print_expr(sp.Indexed(target.base, b, a))
        out = [
            f"do {b} = 1, nc",
            f"{_INDENT}do {a} = 1, {b}",
            f"{_INDENT * 2}{target_s} = 0.0d0",
        ]
        depth = 2
        for idx, low, high in definition["limits"]:
            out.append(
                _INDENT * depth
                + f"do {idx} = {_print_expr(low)}, {_print_expr(high)}"
            )
            depth += 1
        out.append(
            _INDENT * depth
            + f"{target_s} = {target_s} + {_print_expr(definition['body'])}"
        )
        for _ in definition["limits"]:
            depth -= 1
            out.append(_INDENT * depth + "end do")
        out.append(_INDENT * 2 + f"{transposed} = {target_s}")
        out += [f"{_INDENT}end do", "end do"]
        return out
    mirror = (
        definition.get("mirror", False)
        and len(free) == 2
        and not (definition["limits"])
    )
    if mirror:
        # Symmetric rank-2 temporary: fill the upper triangle and mirror.
        lines = [
            f"do {free[0]} = 1, nc",
            f"{_INDENT}do {free[1]} = {free[0]}, nc",
        ]
        body_s = _print_expr(definition["body"])
        target_s = _print_expr(target)
        transposed = _print_expr(sp.Indexed(target.base, free[1], free[0]))
        lines.append(_INDENT * 2 + f"{target_s} = {body_s}")
        lines.append(_INDENT * 2 + f"{transposed} = {target_s}")
        lines.append(_INDENT + "end do")
        lines.append("end do")
        return lines

    target_s = _print_expr(target)
    limits = definition["limits"]
    if definition.get("factored") and not free:
        outer_f, inner_body = _split_factored(definition, limits)
        acc = definition["acc"]
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
    if limits:
        lines.append(_INDENT * depth + f"{target_s} = 0.0d0")
        for idx, low, high in limits:
            low_s = _print_expr(low)
            high_s = _print_expr(high)
            lines.append(_INDENT * depth + f"do {idx} = {low_s}, {high_s}")
            depth += 1
        body_s = _print_expr(definition["body"])
        lines.append(_INDENT * depth + f"{target_s} = {target_s} + {body_s}")
        for _ in limits:
            depth -= 1
            lines.append(_INDENT * depth + "end do")
    else:
        body_s = _print_expr(definition["body"])
        lines.append(_INDENT * depth + f"{target_s} = {body_s}")

    for _ in free:
        depth -= 1
        lines.append(_INDENT * depth + "end do")
    return lines


def _dependency_order(definitions):
    """Stable topological order: a definition goes after every one it reads,
    otherwise in its original position (lowest index first)."""
    import heapq

    index = {_temp_name(d["target"]): k for k, d in enumerate(definitions)}
    deps = []
    for k, d in enumerate(definitions):
        body = d["body"]
        names = {str(x) for x in body.free_symbols} | {
            str(x.base) for x in body.atoms(sp.Indexed)
        }
        deps.append(
            {index[nm] for nm in names if nm in index and index[nm] != k}
        )
    users = {k: [] for k in range(len(definitions))}
    for k, ds in enumerate(deps):
        for dep in ds:
            users[dep].append(k)
    pending = [len(ds) for ds in deps]
    ready = [k for k, count in enumerate(pending) if count == 0]
    heapq.heapify(ready)
    order = []
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


def _guard(attributes, out_names) -> str:
    """Return the ``present(...)`` condition for a group of outputs.

    Returns an empty string when nothing depends on the group.
    """
    if not attributes:
        return ""
    names = []
    if "value" in attributes and "value" in out_names:
        names.append(out_names["value"])
    names.extend(
        out_names[attr]
        for attr in _DERIV_ORDER
        if attr in attributes and attr in out_names
    )
    return " .or. ".join(f"present({name})" for name in names)


def _wrap_if(condition, limit=120) -> list:
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


def _guarded(lines, condition) -> list:
    """Wrap ``lines`` in an ``if (condition)`` block when needed."""
    if not condition:
        return list(lines)
    block = _wrap_if(condition)
    block.extend((_INDENT + line) if line else line for line in lines)
    block.append("end if")
    return block


def _wrap_name_list(keyword, names, limit=120):
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
    lines = []
    current = None
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


def _wrap_signature(sub_name, arg_names, limit=120):
    """Return the subroutine statement, wrapped if necessary.

    Parameters
    ----------
    sub_name : str
        Subroutine name.
    arg_names : list of str
        Argument names.
    limit : int, optional
        Maximum line length.

    Returns
    -------
    list of str
        The (possibly continued) subroutine statement.
    """
    head = f"subroutine {sub_name}("
    if len(head) + len(", ".join(arg_names)) <= limit:
        return [head + ", ".join(arg_names) + ")"]

    continuation = _INDENT + "  "
    lines = []
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
