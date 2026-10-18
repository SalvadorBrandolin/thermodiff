"""Fortran code generation for thermodynamic models.

This module turns a :class:`~thermodiff.diffplz.DiffPlz` instance into a
free-form Fortran subroutine that evaluates the original function together
with all its available derivatives.

The generated subroutine takes the independent variables (mole numbers
``n``, volume ``V``, pressure ``P``, temperature ``T``), the model
parameters, and returns the function value and the derivatives as ``intent
(out)`` arguments, for example::

    subroutine f(n, V, T, f, dfdT, dfdV, dfdn)
      implicit none
      integer, intent(in) :: nc
      real(8), intent(in) :: n(:)
      real(8), intent(in) :: V, T
      real(8), intent(out) :: f, dfdT, dfdV
      real(8), intent(out) :: dfdn(:)
      ...
    end subroutine

Summations produced with :func:`~thermodiff.core.easy_sums.sum_components`
are expanded into explicit ``do`` loops and the Kronecker deltas handled by
``DiffPlz`` become ``if``/``else`` blocks.  Internal functions created with
:func:`~thermodiff.core.idxfunction.idx_function` (and their derivatives)
are exposed as extra input arguments by default; pass ``strict=True`` to
raise an error instead.
"""

import re

import sympy as sp
from sympy.core.function import DefinedFunction
from sympy.printing.fortran import fcode

from thermodiff.derivatives import DERIVATIVE_ATTRIBUTES, resolve_derivatives
from thermodiff.thermovars import P, T, V, i, j, n, nc

__all__ = ["to_fortran"]


#: Mapping of ``DiffPlz`` derivative attributes to the suffix used in the
#: generated Fortran variable names.
_DERIV_SUFFIX = {
    "dt": "dT",
    "dv": "dV",
    "dp": "dP",
    "dni": "dn",
    "dt2": "dT2",
    "dv2": "dV2",
    "dp2": "dP2",
    "dnidnj": "dndn",
    "dtdv": "dTdV",
    "dtdp": "dTdP",
    "dtdni": "dndT",
    "dvdni": "dndV",
    "dvdp": "dVdP",
    "dpdni": "dndP",
}

#: Order in which the derivatives are emitted.
_DERIV_ORDER = list(DERIVATIVE_ATTRIBUTES)

#: Settings passed to :func:`sympy.printing.fortran.fcode`.
_FCODE_SETTINGS = {
    "source_format": "free",
    "standard": 95,
    "contract": False,
}

_INDENT = "   "


def _sanitize(name: str) -> str:
    r"""Turn an arbitrary symbol name into a valid Fortran identifier.

    Parameters
    ----------
    name : str
        Symbol name, potentially containing LaTeX pieces such as
        ``\\tau_{lk}`` or ``G^E``.

    Returns
    -------
    str
        A valid Fortran identifier.
    """
    clean = str(name).replace("\\", "")
    clean = clean.replace("{", "_").replace("}", "")
    clean = clean.replace("^", "_")
    clean = re.sub(r"[^0-9A-Za-z_]", "_", clean)
    clean = re.sub(r"_+", "_", clean).strip("_")
    if not clean:
        clean = "var"
    if clean[0].isdigit():
        clean = "v_" + clean
    return clean


def _free_indices(expr: sp.Expr) -> set:
    """Return the indices that are free in ``expr``.

    An index is free when it is not the summation index of any
    :class:`sympy.Sum` inside ``expr``.
    """
    bound = {
        limit[0] for total in expr.atoms(sp.Sum) for limit in total.limits
    }
    return {idx for idx in expr.atoms(sp.Idx) if idx not in bound}


def _order_free(free: set, indexes) -> list:
    """Order free indices deterministically.

    Derivative indices (``i`` and ``j``) come first, then the user supplied
    ``indexes`` in their given order, then any remaining index sorted by
    name.
    """
    ordered = []
    for idx in (i, j):
        if idx in free and idx not in ordered:
            ordered.append(idx)
    for idx in indexes:
        if idx in free and idx not in ordered:
            ordered.append(idx)
    for idx in sorted(free, key=str):
        if idx not in ordered:
            ordered.append(idx)
    return ordered


def _kron_to_piecewise(expr: sp.Expr) -> sp.Expr:
    """Replace ``KroneckerDelta`` by a two branch ``Piecewise``."""
    replacements = {}
    for delta in expr.atoms(sp.KroneckerDelta):
        first, second = delta.args
        replacements[delta] = sp.Piecewise(
            (sp.Float(1), sp.Eq(first, second)), (sp.Float(0), True)
        )
    return expr.xreplace(replacements)


def _is_placeholder(func) -> bool:
    """Return whether ``func`` is a user defined placeholder function.

    Placeholder functions are the ones created with
    :func:`~thermodiff.core.idxfunction.idx_function` or plain undefined
    :class:`sympy.Function` objects.  SymPy provided functions (``exp``,
    ``KroneckerDelta``, ``Piecewise`` ...) are defined functions and are
    handled by the Fortran printer directly.
    """
    return isinstance(func, sp.Function) and not isinstance(
        func, DefinedFunction
    )


class _PlaceholderResolver:
    """Replace undefined functions and derivatives with input arguments.

    Parameters
    ----------
    strict : bool
        When ``True``, raise a ``ValueError`` if any placeholder is found
        instead of exposing it as an argument.
    """

    def __init__(self, strict: bool = False):
        self.strict = strict
        #: name -> rank (0 for scalars)
        self.ranks: dict = {}

    def resolve(self, expr: sp.Expr) -> sp.Expr:
        """Return ``expr`` with placeholders replaced by plain variables."""
        derivatives = list(expr.atoms(sp.Derivative))
        functions = [f for f in expr.atoms(sp.Function) if _is_placeholder(f)]

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
        resolved = expr.xreplace(replacements)

        bare = {
            func
            for func in resolved.atoms(sp.Function)
            if _is_placeholder(func)
        }
        replacements = {func: self._function_var(func) for func in bare}
        return resolved.xreplace(replacements)

    def _index_dims(self, func: sp.Function) -> list:
        """Return the index labels used as array dimensions by ``func``."""
        dims = []
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

    def _build(self, name: str, labels: list) -> sp.Expr:
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
            base = _sanitize(inner.func.__name__)
            dims = self._index_dims(inner)
        else:
            base = _sanitize(str(inner))
            dims = []

        suffix = ""
        for var in deriv.variables:
            if isinstance(var, sp.Indexed):
                label = str(var.indices[0])
                dims.append(label)
                suffix += "d" + label
            else:
                suffix += "d" + _sanitize(str(var))

        return self._build("d" + base + suffix, dims)

    def _function_var(self, func: sp.Function) -> sp.Expr:
        base = _sanitize(func.func.__name__)
        dims = self._index_dims(func)
        return self._build(base, dims)


class _Namer:
    """Generate sequential temporary variable names."""

    def __init__(self):
        self._count = 0
        self.created = []

    def __call__(self, prefix: str = "t") -> sp.Symbol:
        self._count += 1
        symbol = sp.Symbol(f"{prefix}{self._count}")
        self.created.append(symbol)
        return symbol


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

    while True:
        innermost = [
            total
            for total in residual.atoms(sp.Sum)
            if not total.function.atoms(sp.Sum)
        ]
        if not innermost:
            break
        for total in innermost:
            tmp = namer("t")
            statements.append(f"{tmp} = 0.0d0")
            depth = 0
            for idx, low, high in total.limits:
                low_s = _print_expr(low)
                high_s = _print_expr(high)
                statements.append(
                    f"{_INDENT * depth}do {idx} = {low_s}, {high_s}"
                )
                depth += 1
            body_s = _print_expr(total.function)
            statements.append(f"{_INDENT * depth}{tmp} = {tmp} + {body_s}")
            for _ in total.limits:
                depth -= 1
                statements.append(f"{_INDENT * depth}end do")
            residual = residual.xreplace({total: tmp})

    return statements, residual


def _print_expr(expr: sp.Expr) -> str:
    """Print ``expr`` as a single Fortran expression."""
    return fcode(expr, **_FCODE_SETTINGS).strip()


def _indent_lines(text: str, depth: int) -> list:
    """Indent every non-empty line of ``text`` by ``depth`` levels."""
    prefix = _INDENT * depth
    return [prefix + line for line in text.splitlines() if line.strip()]


def _declare_output(name: str, rank: int, optional: bool = False) -> str:
    """Return the declaration line for an output argument."""
    if optional:
        attributes = "real(8), optional, intent(out)"
    else:
        attributes = "real(8), intent(out)"
    if rank == 0:
        return f"{attributes} :: {name}"
    dims = ", ".join([":"] * rank)
    return f"{attributes} :: {name}({dims})"


def _render_output(name: str, expr: sp.Expr, free_order: list, namer) -> list:
    """Render the statements that compute a single output."""
    statements, residual = _expand_sums(expr, namer)
    residual = residual.xreplace(
        {idx: sp.Symbol(str(idx)) for idx in free_order}
    )

    lines = []
    depth = 0
    for idx in free_order:
        lines.append(_INDENT * depth + f"do {idx} = 1, nc")
        depth += 1

    for statement in statements:
        lines.append(_INDENT * depth + statement)

    body = fcode(residual, assign_to="res", **_FCODE_SETTINGS)
    lines.extend(_indent_lines(body, depth))

    if free_order:
        subscript = ", ".join(str(idx) for idx in free_order)
        lines.append(_INDENT * depth + f"{name}({subscript}) = res")
    else:
        lines.append(_INDENT * depth + f"{name} = res")

    for _ in free_order:
        depth -= 1
        lines.append(_INDENT * depth + "end do")

    return lines


def _new_shared_temp(namer, free_order, arrays):
    """Create a temporary for a shared sub-expression.

    Parameters
    ----------
    namer : _Namer
        Generator of temporary variable names.
    free_order : list
        Free indices the temporary depends on; empty for a scalar.
    arrays : dict
        Records the rank of every array temporary created.

    Returns
    -------
    sympy expression
        A ``Symbol`` for a scalar temporary, or an ``Indexed`` (indexed by
        ``free_order``) for an array temporary.
    """
    name = str(namer("t"))
    if not free_order:
        return sp.Symbol(name)
    arrays[name] = len(free_order)
    base = sp.IndexedBase(name, shape=(nc,) * len(free_order))
    return sp.Indexed(base, *free_order)


def _share_sums(expressions, namer, indexes, arrays):
    """Replace every ``Sum`` by a temporary that is computed once.

    Identical summations reused by several outputs are evaluated a single
    time.  Sums that appear in only one output are hoisted too, which keeps
    the output residuals free of loop nests and lets the scalar common
    sub-expression pass below see through them.

    Parameters
    ----------
    expressions : list of sympy expressions
        Expressions that may contain summations.
    namer : _Namer
        Generator of temporary variable names.
    indexes : list
        User index order, used to sort the free indices of a temporary.
    arrays : dict
        Records the rank of every array temporary created.

    Returns
    -------
    residuals : list of sympy expressions
        ``expressions`` with every ``Sum`` replaced by a temporary.
    definitions : list of dict
        How to compute each temporary, in dependency order.
    """
    exprs = list(expressions)
    definitions = []
    while True:
        innermost = []
        seen = set()
        for expr in exprs:
            for total in expr.atoms(sp.Sum):
                if total.function.atoms(sp.Sum) or total in seen:
                    continue
                seen.add(total)
                innermost.append(total)
        if not innermost:
            break
        replacements = {}
        for total in innermost:
            free = _order_free(_free_indices(total), indexes)
            target = _new_shared_temp(namer, free, arrays)
            definitions.append(
                {
                    "target": target,
                    "free": free,
                    "limits": list(total.limits),
                    "body": total.function,
                }
            )
            replacements[total] = target
        exprs = [expr.xreplace(replacements) for expr in exprs]
    return exprs, definitions


def _common_scalar_cse(expressions, namer, arrays):
    """Extract scalar sub-expressions shared by several outputs.

    Only sub-expressions without free indices are hoisted, so every
    temporary can be computed once, before the output loops, without
    worrying about the index scope of a particular output.

    Parameters
    ----------
    expressions : list of sympy expressions
        Sum free expressions to optimise.
    namer : _Namer
        Generator of temporary variable names.
    arrays : dict
        Records the rank of every array temporary created.

    Returns
    -------
    residuals : list of sympy expressions
        ``expressions`` with the shared sub-expressions replaced.
    definitions : list of dict
        How to compute each shared scalar, in dependency order.
    """
    exprs = list(expressions)
    definitions = []
    while True:
        counts = {}
        for expr in exprs:
            for sub in sp.preorder_traversal(expr):
                if not isinstance(sub, sp.Expr) or sub.is_Atom:
                    continue
                if _free_indices(sub):
                    continue
                counts[sub] = counts.get(sub, 0) + 1
        repeated = [sub for sub, count in counts.items() if count > 1]
        if not repeated:
            break
        best = max(repeated, key=lambda sub: (counts[sub], sp.count_ops(sub)))
        target = _new_shared_temp(namer, [], arrays)
        definitions.append(
            {
                "target": target,
                "free": [],
                "limits": [],
                "body": best,
            }
        )
        exprs = [expr.xreplace({best: target}) for expr in exprs]
    return exprs, definitions


def _render_definition(definition) -> list:
    """Render the statements that fill one shared temporary."""
    target = definition["target"]
    free = definition["free"]
    lines = []
    depth = 0
    for idx in free:
        lines.append(_INDENT * depth + f"do {idx} = 1, nc")
        depth += 1

    target_s = _print_expr(target)
    limits = definition["limits"]
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


def _temp_name(target) -> str:
    """Return the base name of a shared temporary."""
    if isinstance(target, sp.Indexed):
        return str(target.base)
    return str(target)


def _referenced_temp_names(expr, temp_names) -> set:
    """Return the names of the temporaries referenced by ``expr``."""
    names = set()
    for symbol in expr.atoms(sp.Symbol):
        if str(symbol) in temp_names:
            names.add(str(symbol))
    for indexed in expr.atoms(sp.Indexed):
        base = indexed.base
        if isinstance(base, sp.IndexedBase) and str(base) in temp_names:
            names.add(str(base))
    return names


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


def to_fortran(
    diff,
    derivatives=None,
    name=None,
    subroutine_name=None,
    strict=False,
    integer_args=None,
    module_symbols=None,
    optional_derivatives=False,
):
    """Generate a Fortran subroutine for a :class:`DiffPlz` instance.

    Parameters
    ----------
    diff : DiffPlz
        The differentiated thermodynamic expression.
    derivatives : str or list of str, optional
        Derivatives to emit, for example ``["dT", "dV", "dn"]``.  Both the
        friendly names and the derivative attributes (``["dt", "dv",
        "dni"]``) are accepted, as well as the groups ``all``, ``none``,
        ``first`` and ``second`` (see
        :func:`thermodiff.derivatives.resolve_derivatives`).  By default
        every non-zero derivative computed by ``diff`` is emitted.
    name : str, optional
        Base name used for the generated variables.  Defaults to the
        ``name`` attribute of ``diff``.
    subroutine_name : str, optional
        Name of the generated subroutine.  Defaults to the base name.
    strict : bool, optional
        When ``True``, raise a ``ValueError`` if the expression contains
        internal functions or unresolved derivatives instead of exposing
        them as input arguments.  By default ``False``.
    integer_args : list of str, optional
        Names of extra arguments that must be declared as
        ``integer, intent(in)`` (for example, extra dimensions).
    module_symbols : list of str, optional
        Names that are provided by a ``use`` statement of the enclosing
        module.  They are neither declared as arguments nor included in
        the subroutine signature.
    optional_derivatives : bool, optional
        When ``True``, every output - the function value included - is
        declared as a Fortran ``optional`` argument and computed only
        when the caller passes it.  A shared temporary is skipped
        whenever none of the outputs that need it is present, so calling
        the subroutine with a subset of the outputs also skips the work.

    Returns
    -------
    str
        The free-form Fortran source code.
    """
    module_symbols = set(module_symbols or ())
    integer_args = list(integer_args or [])

    base = _sanitize(name if name is not None else diff.name)
    sub_name = (
        _sanitize(subroutine_name) if subroutine_name else base + "_derivs"
    )

    if derivatives is None:
        candidates = getattr(diff, "derivatives", None)
        if candidates is None:
            candidates = _DERIV_ORDER
        selected = []
        for attr in _DERIV_ORDER:
            if attr not in candidates:
                continue
            value = getattr(diff, attr, None)
            if value is None or value == 0:
                continue
            selected.append(attr)
    else:
        resolved = resolve_derivatives(derivatives)
        selected = [attr for attr in _DERIV_ORDER if attr in resolved]
        missing = [
            attr for attr in selected if getattr(diff, attr, None) is None
        ]
        if missing:
            raise ValueError(
                "Derivatives not computed by the DiffPlz instance: "
                + ", ".join(missing)
                + ". Rebuild it with derivatives="
                + repr(list(resolved))
                + "."
            )

    resolver = _PlaceholderResolver(strict=strict)

    outputs = [("value", base, resolver.resolve(diff.expression))]
    for attr in selected:
        outputs.append(
            (
                attr,
                "d" + base + _DERIV_SUFFIX[attr],
                resolver.resolve(getattr(diff, attr)),
            )
        )

    prepared = []
    for attr, out_name, expr in outputs:
        expr = _kron_to_piecewise(expr)
        expr = expr.xreplace({nc: sp.Symbol("nc")})
        free_order = _order_free(_free_indices(expr), diff.indexes)
        prepared.append((attr, out_name, expr, free_order))

    all_exprs = [expr for _, _, expr, _ in prepared]
    placeholder_names = set(resolver.ranks)

    # ---- gather parameters --------------------------------------------
    # NOTE: ``n`` is compared by name because ``expr`` has been rebuilt by
    # the ``nc`` replacement above, which also rewrites the shape of the
    # ``n`` IndexedBase.
    n_name = str(n)
    uses_n = any(
        n_name in {str(base_) for base_ in expr.atoms(sp.IndexedBase)}
        for expr in all_exprs
    )

    index_syms = set()
    indexed_bases = set()
    for expr in all_exprs:
        for indexed in expr.atoms(sp.Indexed):
            index_syms.update(indexed.indices)
        for total in expr.atoms(sp.Sum):
            for limit in total.limits:
                index_syms.add(limit[0])
        index_syms.update(expr.atoms(sp.Idx))
        indexed_bases.update(expr.atoms(sp.IndexedBase))

    index_names = {str(idx) for idx in index_syms}
    base_names = {str(base_) for base_ in indexed_bases}

    param_scalars = set()
    for expr in all_exprs:
        for sym in expr.atoms(sp.Symbol):
            name = str(sym)
            if name in index_names or name in base_names:
                continue
            if sym in (T, V, P) or name == "nc":
                continue
            if name in module_symbols or name in integer_args:
                continue
            param_scalars.add(name)

    array_ranks = {}
    for expr in all_exprs:
        for indexed in expr.atoms(sp.Indexed):
            base_name = str(indexed.base)
            rank = len(indexed.indices)
            array_ranks[base_name] = max(array_ranks.get(base_name, 0), rank)
    param_arrays = {
        base_name: rank
        for base_name, rank in array_ranks.items()
        if base_name != n_name and base_name not in placeholder_names
    }
    if uses_n:
        param_arrays[n_name] = 1

    params = sorted(param_scalars - placeholder_names)
    placeholder_args = sorted(resolver.ranks)

    needs_nc = (
        uses_n
        or bool(param_arrays)
        or any(free for _, _, _, free in prepared)
        or any(sp.Symbol("nc") in expr.atoms(sp.Symbol) for expr in all_exprs)
    )

    # ---- input declarations -------------------------------------------
    inputs = []
    if needs_nc:
        inputs.append("integer, intent(in) :: nc")
    if integer_args:
        inputs.append("integer, intent(in) :: " + ", ".join(integer_args))
    for var, label in ((V, "V"), (P, "P"), (T, "T")):
        if any(expr.has(var) for expr in all_exprs):
            inputs.append(f"real(8), intent(in) :: {label}")
    for param in params:
        inputs.append(f"real(8), intent(in) :: {param}")
    for arr in sorted(param_arrays):
        dims = ", ".join([":"] * param_arrays[arr])
        inputs.append(f"real(8), intent(in) :: {arr}({dims})")
    for arg in placeholder_args:
        rank = resolver.ranks[arg]
        if rank == 0:
            inputs.append(f"real(8), intent(in) :: {arg}")
        else:
            dims = ", ".join([":"] * rank)
            inputs.append(f"real(8), intent(in) :: {arg}({dims})")

    # ---- output declarations ------------------------------------------
    output_decls = [
        _declare_output(
            out_name,
            len(free_order),
            optional=optional_derivatives,
        )
        for _attr, out_name, _, free_order in prepared
    ]

    # ---- body ----------------------------------------------------------
    namer = _Namer()
    arrays = {}
    residuals, sum_defs = _share_sums(all_exprs, namer, diff.indexes, arrays)
    residuals, cse_defs = _common_scalar_cse(residuals, namer, arrays)
    rendered = [
        (
            attr,
            out_name,
            residual,
            _order_free(_free_indices(residual), diff.indexes),
        )
        for (attr, out_name, _, _), residual in zip(prepared, residuals)
    ]

    definitions = sum_defs + cse_defs
    out_names = {attr: out_name for attr, out_name, _, _ in rendered}

    temp_names = {
        _temp_name(definition["target"]) for definition in definitions
    }
    consumers = {name: set() for name in temp_names}
    for attr, _out_name, residual, _free in rendered:
        for name in _referenced_temp_names(residual, temp_names):
            consumers[name].add(attr)
    for definition in reversed(definitions):
        name = _temp_name(definition["target"])
        for dependency in _referenced_temp_names(
            definition["body"], temp_names
        ):
            consumers[dependency] |= consumers[name]

    body = []
    for definition in definitions:
        if optional_derivatives:
            condition = _guard(
                consumers[_temp_name(definition["target"])], out_names
            )
        else:
            condition = ""
        body.extend(_guarded(_render_definition(definition), condition))
        body.append("")
    for _attr, out_name, expr, free_order in rendered:
        if optional_derivatives:
            condition = f"present({out_name})"
        else:
            condition = ""
        body.extend(
            _guarded(
                _render_output(out_name, expr, free_order, namer), condition
            )
        )
        body.append("")

    temps = sorted(
        {str(t) for t in namer.created if str(t) not in arrays} | {"res"}
    )
    loop_indices = sorted(
        {str(idx) for _, _, _, free in rendered for idx in free}
        | {
            str(idx)
            for definition in definitions
            for idx in definition["free"]
        }
        | {
            str(limit[0])
            for definition in definitions
            for limit in definition["limits"]
        }
    )

    # ---- assemble ------------------------------------------------------
    arg_names = []
    if needs_nc:
        arg_names.append("nc")
    arg_names.extend(integer_args)
    for var, label in ((V, "V"), (P, "P"), (T, "T")):
        if any(expr.has(var) for expr in all_exprs):
            arg_names.append(label)
    arg_names.extend(params)
    arg_names.extend(sorted(param_arrays))
    arg_names.extend(placeholder_args)
    arg_names.extend(out_name for _, out_name, _, _ in prepared)

    while sub_name in arg_names:
        sub_name += "_s"

    lines = _wrap_signature(sub_name, arg_names)
    lines.append(f"{_INDENT}implicit none")
    for decl in inputs:
        lines.append(_INDENT + decl)
    for decl in output_decls:
        lines.append(_INDENT + decl)

    for line in _wrap_name_list("real(8) :: ", temps):
        lines.append(_INDENT + line)
    for rank in sorted(set(arrays.values())):
        dims = ", ".join(["nc"] * rank)
        names = [
            f"{name}({dims})"
            for name in sorted(arrays)
            if arrays[name] == rank
        ]
        for line in _wrap_name_list("real(8) :: ", names):
            lines.append(_INDENT + line)
    for line in _wrap_name_list("integer :: ", loop_indices):
        lines.append(_INDENT + line)

    lines.append("")
    for line in body:
        lines.append(_INDENT + line if line else "")

    lines.append(f"end subroutine {sub_name}")
    lines.append("")

    return "\n".join(lines)
