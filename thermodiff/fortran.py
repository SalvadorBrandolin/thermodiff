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
_DERIV_ORDER = list(_DERIV_SUFFIX)

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
    bound = {total.limits[0][0] for total in expr.atoms(sp.Sum)}
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
            idx, low, high = total.limits[0]
            tmp = namer("t")
            low_s = _print_expr(low)
            high_s = _print_expr(high)
            body_s = _print_expr(total.function)
            statements.append(f"{tmp} = 0.0d0")
            statements.append(f"do {idx} = {low_s}, {high_s}")
            statements.append(f"{_INDENT}{tmp} = {tmp} + {body_s}")
            statements.append("end do")
            residual = residual.xreplace({total: tmp})

    return statements, residual


def _print_expr(expr: sp.Expr) -> str:
    """Print ``expr`` as a single Fortran expression."""
    return fcode(expr, **_FCODE_SETTINGS).strip()


def _indent_lines(text: str, depth: int) -> list:
    """Indent every non-empty line of ``text`` by ``depth`` levels."""
    prefix = _INDENT * depth
    return [prefix + line for line in text.splitlines() if line.strip()]


def _declare_output(name: str, rank: int) -> str:
    """Return the declaration line for an output argument."""
    if rank == 0:
        return f"real(8), intent(out) :: {name}"
    dims = ", ".join([":"] * rank)
    return f"real(8), intent(out) :: {name}({dims})"


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


def to_fortran(
    diff,
    derivatives=None,
    name=None,
    subroutine_name=None,
    strict=False,
):
    """Generate a Fortran subroutine for a :class:`DiffPlz` instance.

    Parameters
    ----------
    diff : DiffPlz
        The differentiated thermodynamic expression.
    derivatives : list of str, optional
        Names of the derivative attributes to emit, for example
        ``["dt", "dv", "dni", "dnidnj"]``.  By default every non-zero
        derivative computed by ``diff`` is emitted.
    name : str, optional
        Base name used for the generated variables.  Defaults to the
        ``name`` attribute of ``diff``.
    subroutine_name : str, optional
        Name of the generated subroutine.  Defaults to the base name.
    strict : bool, optional
        When ``True``, raise a ``ValueError`` if the expression contains
        internal functions or unresolved derivatives instead of exposing
        them as input arguments.  By default ``False``.

    Returns
    -------
    str
        The free-form Fortran source code.
    """
    base = _sanitize(name if name is not None else diff.name)
    sub_name = (
        _sanitize(subroutine_name) if subroutine_name else base + "_derivs"
    )

    if derivatives is None:
        selected = [
            attr
            for attr in _DERIV_ORDER
            if getattr(diff, attr, sp.Integer(0)) != 0
        ]
    else:
        unknown = set(derivatives) - set(_DERIV_SUFFIX)
        if unknown:
            raise ValueError(
                f"Unknown derivatives: {sorted(unknown)}. "
                f"Valid keys: {_DERIV_ORDER}"
            )
        selected = [attr for attr in _DERIV_ORDER if attr in derivatives]

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
    uses_n = any(n in expr.atoms(sp.IndexedBase) for expr in all_exprs)

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
    param_arrays = set()
    for expr in all_exprs:
        for sym in expr.atoms(sp.Symbol):
            name = str(sym)
            if name in index_names or name in base_names:
                continue
            if sym in (T, V, P) or name == "nc":
                continue
            param_scalars.add(name)
    for base_ in indexed_bases:
        if base_ == n or str(base_) in placeholder_names:
            continue
        param_arrays.add(str(base_))

    params = sorted(param_scalars - param_arrays - placeholder_names)
    placeholder_args = sorted(resolver.ranks)

    needs_nc = (
        uses_n or bool(param_arrays) or any(free for _, _, _, free in prepared)
    )

    # ---- input declarations -------------------------------------------
    inputs = []
    if needs_nc:
        inputs.append("integer, intent(in) :: nc")
    if uses_n:
        inputs.append("real(8), intent(in) :: n(:)")
    for var, label in ((V, "V"), (P, "P"), (T, "T")):
        if any(expr.has(var) for expr in all_exprs):
            inputs.append(f"real(8), intent(in) :: {label}")
    for param in params:
        inputs.append(f"real(8), intent(in) :: {param}")
    for arr in sorted(param_arrays):
        inputs.append(f"real(8), intent(in) :: {arr}(:)")
    for arg in placeholder_args:
        rank = resolver.ranks[arg]
        if rank == 0:
            inputs.append(f"real(8), intent(in) :: {arg}")
        else:
            dims = ", ".join([":"] * rank)
            inputs.append(f"real(8), intent(in) :: {arg}({dims})")

    # ---- output declarations ------------------------------------------
    output_decls = [
        _declare_output(out_name, len(free_order))
        for _, out_name, _, free_order in prepared
    ]

    # ---- body ----------------------------------------------------------
    namer = _Namer()
    body = []
    for _, out_name, expr, free_order in prepared:
        body.extend(_render_output(out_name, expr, free_order, namer))
        body.append("")

    temps = sorted({str(t) for t in namer.created} | {"res"})
    loop_indices = sorted(
        {str(idx) for _, _, _, free in prepared for idx in free}
        | {
            str(limit[0])
            for _, _, expr, _ in prepared
            for total in expr.atoms(sp.Sum)
            for limit in total.limits
        }
    )

    # ---- assemble ------------------------------------------------------
    arg_names = []
    if needs_nc:
        arg_names.append("nc")
    if uses_n:
        arg_names.append("n")
    for var, label in ((V, "V"), (P, "P"), (T, "T")):
        if any(expr.has(var) for expr in all_exprs):
            arg_names.append(label)
    arg_names.extend(params)
    arg_names.extend(sorted(param_arrays))
    arg_names.extend(placeholder_args)
    arg_names.extend(out_name for _, out_name, _, _ in prepared)

    while sub_name in arg_names:
        sub_name += "_s"

    lines = [f"subroutine {sub_name}({', '.join(arg_names)})"]
    lines.append(f"{_INDENT}implicit none")
    for decl in inputs:
        lines.append(_INDENT + decl)
    for decl in output_decls:
        lines.append(_INDENT + decl)

    if temps:
        lines.append(_INDENT + "real(8) :: " + ", ".join(temps))
    if loop_indices:
        lines.append(_INDENT + "integer :: " + ", ".join(loop_indices))

    lines.append("")
    for line in body:
        lines.append(_INDENT + line if line else "")

    lines.append(f"end subroutine {sub_name}")
    lines.append("")

    return "\n".join(lines)
