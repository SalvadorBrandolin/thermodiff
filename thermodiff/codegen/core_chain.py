"""Chain-rule assembly routine of a scalar core.

A model written as a *scalar core* ``F(x_1, ..., x_m; V, T)`` (see
``DiffPlz(..., variables=[...])``) whose scalars ``x_r`` depend on the mole
numbers through mixing rules (and, optionally, on ``T`` and ``V``) needs a
final step: the total derivatives of ``Ar(n, V, T) = F(x(n, V, T), V, T)``
from the derivatives of ``F`` and of the scalars.  :func:`scalar_chain_routine`
emits that step as a Fortran subroutine.

Why the loops are generated directly
------------------------------------
The composition is the second-order chain rule, a fixed contraction over the
scalar index ``r`` that SymPy cannot express with an unknown number of
scalars and array-valued derivatives (the generator does not apply the chain
rule through placeholder arguments, it warns).  The formulas below are the
total second derivatives written once, with the work per output element
reduced to ``O(m)``, and are checked numerically against the
full-composition kernel in ``tests/test_scalar_variables.py``.

Formulas
--------
With ``F_r = dF/dx_r``, ``F_rs``, ``F_rW`` (``W`` is ``T`` or ``V``) the
derivatives of the core, ``x_r,i``, ``x_r,ij``, ``x_r,W``, ``x_r,WW'``,
``x_r,iW`` those of the scalars (``x_r,W`` only for the scalars that
declare they depend on ``W``) and the *total partial*
``q_W,r = F_rW + sum_s F_rs x_s,W``::

    Ar_W      = F_W + sum_r F_r x_r,W
    Ar_WW'    = F_WW' + sum_r x_r,W q_W',r + sum_r F_rW x_r,W'
                + sum_r F_r x_r,WW'
    Ar_i      = sum_r F_r x_r,i
    Ar_iW     = sum_r q_W,r x_r,i + sum_r F_r x_r,iW
    Ar_ij     = sum_r y_r,j x_r,i + sum_r F_r x_r,ij
    y_r,j     = sum_s F_rs x_s,j

The Hessian is computed on the lower triangle and mirrored.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from typing import TYPE_CHECKING

import sympy as sp

from thermodiff.codegen.common import apply_real_kind
from thermodiff.codegen.options import CodegenOptions
from thermodiff.derivatives import resolve_derivatives
from thermodiff.thermovars import T, V

if TYPE_CHECKING:
    from thermodiff.diffplz import DiffPlz

__all__ = ["scalar_chain_routine", "scalar_chain_core_derivatives"]

#: Factor that stands for the composition ``n`` in a derivative.
COMPOSITION = "n"
#: Thermodynamic variables the scalars and the core may depend on.
THERMO_VARIABLES = ("T", "V")
#: Attribute of ``DiffPlz`` -> factors of the derivative it names.
_ATTRIBUTE_FACTORS: dict[str, tuple[str, ...]] = {
    "dt": ("T",),
    "dv": ("V",),
    "dt2": ("T", "T"),
    "dv2": ("V", "V"),
    "dtdv": ("T", "V"),
    "dni": (COMPOSITION,),
    "dtdni": (COMPOSITION, "T"),
    "dvdni": (COMPOSITION, "V"),
    "dnidnj": (COMPOSITION, COMPOSITION),
}
_INDENT = "   "
_LINE_WIDTH = 72
_ZERO = "0.0d0"
_THERMO_RANK_OFFSET = 1000
_VECTOR_RANK = 1
_MATRIX_RANK = 2
_LOOP_INDEX = "i"
_LOOP_COLUMN = "j"
_EXTENT = "nc"


def _derivative_name(factors: Sequence[str]) -> str:
    """Return the suffix of a derivative name: ``dB``, ``dB2``, ``dBdD``."""
    pair = len(factors) == 2 and factors[0] == factors[1]
    if pair and factors[0] != COMPOSITION:
        return f"d{factors[0]}2"
    return "".join(f"d{factor}" for factor in factors)


@dataclass(frozen=True)
class _Model:
    """Declared structure of the scalar core.

    Attributes
    ----------
    scalars : tuple of str
        Names of the scalars, in the declaration order of the core.
    depends : mapping
        Scalar -> thermodynamic variables (``T``, ``V``) it depends on.
    linear : frozenset of str
        Scalars whose Hessian with respect to ``n`` is zero.
    total : frozenset of str
        Scalars equal to the total moles: gradient 1, no other dependence.
    core, out : str
        Prefixes (``f`` and ``ar``) of the core and total derivatives.
    zero : frozenset of tuple of str
        Derivatives of the core that vanish identically, as factors in
        canonical order (scalars in declaration order, then ``T``, ``V``).
    """

    scalars: tuple[str, ...]
    depends: Mapping[str, frozenset[str]]
    linear: frozenset[str]
    core: str
    out: str
    zero: frozenset[tuple[str, ...]] = frozenset()
    total: frozenset[str] = frozenset()

    def dependent(self, thermo: str) -> list[str]:
        """Return the scalars that depend on the variable ``thermo``."""
        return [x for x in self.scalars if thermo in self.depends[x]]

    def rank(self, factor: str) -> int:
        """Return the position of a factor in the canonical order."""
        if factor in self.scalars:
            return self.scalars.index(factor)
        return _THERMO_RANK_OFFSET + THERMO_VARIABLES.index(factor)

    def vanishes(self, *factors: str) -> bool:
        """Return whether the core derivative of ``factors`` is zero."""
        return tuple(sorted(factors, key=self.rank)) in self.zero

    def q_vanishes(self, thermo: str, scalar: str) -> bool:
        """Return whether the total partial ``q_W,r`` is identically zero."""
        if not self.vanishes(scalar, thermo):
            return False
        return all(self.vanishes(scalar, o) for o in self.dependent(thermo))

    def active_y(self) -> list[str]:
        """Return the scalars ``r`` whose ``y_r,j`` is not identically zero."""
        return [
            r
            for r in self.scalars
            if not all(self.vanishes(r, s) for s in self.scalars)
        ]


@dataclass
class _Inputs:
    """Names of the inputs used by the generated code, in first-use order.

    ``core`` maps the name of a derivative of the core to its sort key;
    ``scalar`` maps the name of a derivative of a scalar to its rank
    (0 scalar, 1 vector over ``nc``, 2 matrix).
    """

    model: _Model
    core: dict[str, tuple[int, ...]] = field(default_factory=dict)
    scalar: dict[str, int] = field(default_factory=dict)

    def core_name(self, *factors: str) -> str:
        """Return the input name of ``d^k F / d(factors)``."""
        ordered = sorted(factors, key=self.model.rank)
        name = f"d{self.model.core}" + _derivative_name(ordered)
        key = (len(ordered), *(self.model.rank(f) for f in ordered))
        self.core[name] = key
        return name

    def scalar_name(self, scalar: str, *factors: str, rank: int = 0) -> str:
        """Return the input name of a derivative of ``scalar``."""
        name = f"d{scalar}" + _derivative_name(factors)
        self.scalar[name] = rank
        return name

    def gradient(self, scalar: str) -> str:
        """Return the name of ``dx/dn`` (a vector input)."""
        return self.scalar_name(scalar, COMPOSITION, rank=_VECTOR_RANK)

    def mixed(self, scalar: str, thermo: str) -> str:
        """Return the name of ``d2x/dn dW`` (a vector input)."""
        return self.scalar_name(scalar, COMPOSITION, thermo, rank=_VECTOR_RANK)

    def hessian(self, scalar: str) -> str:
        """Return the name of ``d2x/dn dn`` (a matrix input)."""
        return self.scalar_name(
            scalar, COMPOSITION, COMPOSITION, rank=_MATRIX_RANK
        )


def _output_name(model: _Model, factors: Sequence[str]) -> str:
    """Return the output name, for example ``dardndT``."""
    return f"d{model.out}" + _derivative_name(factors)


def _wrap(lhs: str, terms: Sequence[str], indent: str) -> list[str]:
    """Return the lines of ``lhs = t1 + t2 + ...`` wrapped at the width."""
    terms = list(terms) or [_ZERO]
    lines: list[str] = []
    current = f"{indent}{lhs} = {terms[0]}"
    for term in terms[1:]:
        if len(current) + len(term) + 5 > _LINE_WIDTH:
            lines.append(current + " &")
            current = f"{indent}{_INDENT}+ {term}"
        else:
            current += f" + {term}"
    return lines + [current]


def _q_name(thermo: str, scalar: str) -> str:
    """Return the local name of the total partial ``q_W,r``."""
    return f"q{thermo}_{scalar}"


def _q_terms(inputs: _Inputs, thermo: str, scalar: str) -> list[str]:
    """Return the terms of ``q_W,r = F_rW + sum_s F_rs x_s,W``."""
    model = inputs.model
    terms = []
    if not model.vanishes(scalar, thermo):
        terms.append(inputs.core_name(scalar, thermo))
    for other in model.dependent(thermo):
        if not model.vanishes(scalar, other):
            terms.append(
                f"{inputs.core_name(scalar, other)}"
                f"*{inputs.scalar_name(other, thermo)}"
            )
    return terms


def _first_thermo_terms(inputs: _Inputs, thermo: str) -> list[str]:
    """Return the terms of ``Ar_W = F_W + sum_r F_r x_r,W``."""
    model = inputs.model
    terms = []
    if not model.vanishes(thermo):
        terms.append(inputs.core_name(thermo))
    for scalar in model.dependent(thermo):
        if not model.vanishes(scalar):
            terms.append(
                f"{inputs.core_name(scalar)}"
                f"*{inputs.scalar_name(scalar, thermo)}"
            )
    return terms


def _second_thermo_terms(
    inputs: _Inputs, first: str, second: str
) -> list[str]:
    """Return the terms of ``Ar_WW'`` (``W`` = first, ``W'`` = second)."""
    model = inputs.model
    terms = []
    if not model.vanishes(first, second):
        terms.append(inputs.core_name(first, second))
    for scalar in model.dependent(first):
        if not model.q_vanishes(second, scalar):
            terms.append(
                f"{inputs.scalar_name(scalar, first)}"
                f"*{_q_name(second, scalar)}"
            )
    for scalar in model.dependent(second):
        if not model.vanishes(scalar, first):
            terms.append(
                f"{inputs.core_name(scalar, first)}"
                f"*{inputs.scalar_name(scalar, second)}"
            )
    for scalar in model.dependent(first):
        if second in model.depends[scalar] and not model.vanishes(scalar):
            terms.append(
                f"{inputs.core_name(scalar)}"
                f"*{inputs.scalar_name(scalar, first, second)}"
            )
    return terms


def _vector_terms(
    inputs: _Inputs, thermo: str | None, index: str
) -> list[str]:
    """Return the terms of ``Ar_i`` (``thermo`` None) or ``Ar_iW``."""
    model = inputs.model
    terms = []
    for scalar in model.scalars:
        if thermo is None and not model.vanishes(scalar):
            weight = inputs.core_name(scalar)
        elif thermo is not None and not model.q_vanishes(thermo, scalar):
            weight = _q_name(thermo, scalar)
        else:
            continue
        terms.append(_gradient_term(inputs, weight, scalar, index))
    if thermo is not None:
        for scalar in model.dependent(thermo):
            if not model.vanishes(scalar):
                terms.append(
                    f"{inputs.core_name(scalar)}"
                    f"*{inputs.mixed(scalar, thermo)}({index})"
                )
    return terms


def _gradient_term(
    inputs: _Inputs, weight: str, scalar: str, index: str
) -> str:
    """Return ``weight * dx/dn(index)``; the gradient of ``nT`` is one."""
    if scalar in inputs.model.total:
        return weight
    return f"{weight}*{inputs.gradient(scalar)}({index})"


def _is_thermo(factors: Sequence[str]) -> bool:
    """Return whether no factor is the composition."""
    return COMPOSITION not in factors


def _q_requirements(
    model: _Model, requested: Sequence[tuple[str, ...]]
) -> dict[str, list[str]]:
    """Return, for each ``W``, the scalars whose ``q_W,r`` are used."""
    needed: dict[str, set[str]] = {}
    for factors in requested:
        if factors[0] == COMPOSITION and len(factors) == 2:
            if factors[1] != COMPOSITION:
                needed.setdefault(factors[1], set()).update(model.scalars)
        elif len(factors) == 2:
            needed.setdefault(factors[1], set()).update(
                model.dependent(factors[0])
            )
    return {
        thermo: [
            x
            for x in model.scalars
            if x in needed[thermo] and not model.q_vanishes(thermo, x)
        ]
        for thermo in THERMO_VARIABLES
        if thermo in needed
    }


def _q_lines(
    inputs: _Inputs, needed: Mapping[str, Sequence[str]]
) -> list[str]:
    """Return the assignments of the total partials ``q_W,r``."""
    lines: list[str] = []
    for thermo, scalars in needed.items():
        for scalar in scalars:
            terms = _q_terms(inputs, thermo, scalar)
            lines += _wrap(_q_name(thermo, scalar), terms, "")
    return lines


def _hessian_lines(inputs: _Inputs, output: str) -> list[str]:
    """Return the triangle loop of ``Ar_ij`` with its mirror."""
    model = inputs.model
    lines = [f"do {_LOOP_COLUMN} = 1, {_EXTENT}"]
    for scalar in model.active_y():
        terms = [
            _gradient_term(
                inputs,
                inputs.core_name(scalar, other),
                other,
                _LOOP_COLUMN,
            )
            for other in model.scalars
            if not model.vanishes(scalar, other)
        ]
        lines += _wrap(f"y_{scalar}", terms, _INDENT)
    lines.append(f"{_INDENT}do {_LOOP_INDEX} = {_LOOP_COLUMN}, {_EXTENT}")
    terms = [
        _gradient_term(inputs, f"y_{scalar}", scalar, _LOOP_INDEX)
        for scalar in model.active_y()
    ]
    cell = f"{output}({_LOOP_INDEX}, {_LOOP_COLUMN})"
    for scalar in model.scalars:
        if scalar not in model.linear and not model.vanishes(scalar):
            terms.append(
                f"{inputs.core_name(scalar)}*{inputs.hessian(scalar)}"
                f"({_LOOP_INDEX}, {_LOOP_COLUMN})"
            )
    lines += _wrap(cell, terms, _INDENT * 2)
    lines.append(
        f"{_INDENT * 2}{output}({_LOOP_COLUMN}, {_LOOP_INDEX}) = {cell}"
    )
    return lines + [f"{_INDENT}end do", "end do"]


def _output_lines(inputs: _Inputs, factors: tuple[str, ...]) -> list[str]:
    """Return the statements that compute one output."""
    output = _output_name(inputs.model, factors)
    if factors == (COMPOSITION, COMPOSITION):
        return _hessian_lines(inputs, output)
    if _is_thermo(factors):
        if len(factors) == 1:
            terms = _first_thermo_terms(inputs, factors[0])
        else:
            terms = _second_thermo_terms(inputs, factors[0], factors[1])
        return _wrap(output, terms, "")
    thermo = factors[1] if len(factors) == 2 else None
    terms = _vector_terms(inputs, thermo, _LOOP_INDEX)
    lines = [f"do {_LOOP_INDEX} = 1, {_EXTENT}"]
    lines += _wrap(f"{output}({_LOOP_INDEX})", terms, _INDENT)
    return lines + ["end do"]


def _indented(lines: Iterable[str], levels: int = 1) -> list[str]:
    """Indent ``lines`` by ``levels`` steps."""
    return [_INDENT * levels + line for line in lines]


def _guarded(lines: list[str], conditions: Sequence[str]) -> list[str]:
    """Wrap ``lines`` in ``if (present(...) .or. ...)`` if requested."""
    if not conditions or not lines:
        return lines
    test = " .or. ".join(f"present({name})" for name in conditions)
    return [f"if ({test}) then", *_indented(lines), "end if"]


def _body(
    inputs: _Inputs,
    requested: Sequence[tuple[str, ...]],
    optional: bool,
) -> list[str]:
    """Return the executable statements of the routine."""
    model = inputs.model
    needed = _q_requirements(model, requested)
    users = [
        _output_name(model, f)
        for f in requested
        if len(f) == 2 and f != (COMPOSITION, COMPOSITION)
    ]
    body = _guarded(_q_lines(inputs, needed), users if optional else [])
    for factors in requested:
        name = _output_name(model, factors)
        lines = _output_lines(inputs, factors)
        body += _guarded(lines, [name] if optional else [])
    return body


def _declarations(
    inputs: _Inputs,
    requested: Sequence[tuple[str, ...]],
    optional: bool,
) -> list[str]:
    """Return the declarations of the dummies and the locals."""
    model = inputs.model
    attribute = "optional, " if optional else ""
    lines = [f"integer, intent(in) :: {_EXTENT}"]
    core_names = sorted(inputs.core, key=lambda name: inputs.core[name])
    scalar_names = list(inputs.scalar)
    shapes = {
        0: "",
        _VECTOR_RANK: f"({_EXTENT})",
        _MATRIX_RANK: f"({_EXTENT}, {_EXTENT})",
    }
    for name in core_names:
        lines.append(f"real(8), {attribute}intent(in) :: {name}")
    for name in scalar_names:
        shape = shapes[inputs.scalar[name]]
        lines.append(f"real(8), {attribute}intent(in) :: {name}{shape}")
    for factors in requested:
        rank = factors.count(COMPOSITION)
        name = _output_name(model, factors)
        lines.append(
            f"real(8), {attribute}intent(out) :: {name}{shapes[rank]}"
        )
    return lines


def _locals(
    inputs: _Inputs,
    requested: Sequence[tuple[str, ...]],
) -> list[str]:
    """Return the declarations of the local variables."""
    model = inputs.model
    names = [
        _q_name(thermo, scalar)
        for thermo, scalars in _q_requirements(model, requested).items()
        for scalar in scalars
    ]
    if (COMPOSITION, COMPOSITION) in requested:
        names += [f"y_{scalar}" for scalar in model.active_y()]
    lines = [f"real(8) :: {', '.join(names)}"] if names else []
    return lines + [f"integer :: {_LOOP_INDEX}, {_LOOP_COLUMN}"]


def _check_unique(names: Iterable[str]) -> None:
    """Raise if two names of the routine are equal ignoring case."""
    seen: dict[str, str] = {}
    for name in names:
        other = seen.setdefault(name.lower(), name)
        if other != name:
            raise ValueError(
                f"names {other!r} and {name!r} clash because Fortran is "
                "case-insensitive; rename a scalar"
            )


def _candidate_factors(scalars: Sequence[str]) -> list[tuple[str, ...]]:
    """Return the first and second core derivatives, in canonical order."""
    ordered = [*scalars, *THERMO_VARIABLES]
    return [
        factors
        for order in (1, 2)
        for factors in combinations_with_replacement(ordered, order)
    ]


def _named_zero_factors(
    scalars: Sequence[str], names: Iterable[str]
) -> set[tuple[str, ...]]:
    """Translate derivative names (``dD2``, ``dBdS``) to factor tuples."""
    known: dict[str, tuple[str, ...]] = {}
    for factors in _candidate_factors(scalars):
        known[_derivative_name(factors)] = factors
        known[_derivative_name(factors[::-1])] = factors
    unknown = sorted(set(names) - set(known))
    if unknown:
        raise ValueError(f"unknown derivatives in zero: {unknown}")
    return {known[name] for name in names}


def _symbolic_zero_factors(
    scalars: Sequence[str], core: "DiffPlz"
) -> set[tuple[str, ...]]:
    """Return the derivatives of the core that are identically zero."""
    symbols = {str(variable): variable for variable in core.variables}
    missing = [x for x in scalars if x not in symbols]
    if missing:
        raise ValueError(f"core has no variables {missing}")
    symbols.update({"T": T, "V": V})
    zero = set()
    for factors in _candidate_factors(scalars):
        value = core.expression
        for factor in factors:
            value = sp.diff(value, symbols[factor])
        if value == 0:
            zero.add(factors)
    return zero


def _build_model(
    scalars: Sequence[str],
    depends_on: Mapping[str, Iterable[str]],
    linear: Iterable[str],
    core_name: str,
    name: str,
    zero: frozenset[tuple[str, ...]] = frozenset(),
    total: Iterable[str] = (),
) -> _Model:
    """Validate the declarations and return the model."""
    names = tuple(str(scalar) for scalar in scalars)
    if not names or len(set(names)) != len(names):
        raise ValueError("scalars must be a non-empty list of unique names")
    total = frozenset(total)
    unknown = set(depends_on) | set(linear) | total
    if unknown - set(names):
        raise ValueError(f"unknown scalars: {sorted(unknown - set(names))}")
    depends = {x: frozenset(depends_on.get(x, ())) for x in names}
    for scalar, thermo in depends.items():
        if thermo - set(THERMO_VARIABLES):
            raise ValueError(
                f"{scalar!r} depends on {sorted(thermo)}; "
                f"only {list(THERMO_VARIABLES)} are supported"
            )
    clash = [x for x in names if x in total and depends[x]]
    if clash:
        raise ValueError(f"total_moles scalars {clash} cannot depend on T, V")
    linear = frozenset(linear) | total
    return _Model(names, depends, linear, core_name, name, zero, total)


def _zero_factors(
    scalars: Sequence[str],
    core: "DiffPlz | None",
    zero: Iterable[str] | None,
) -> frozenset[tuple[str, ...]]:
    """Merge the declared and the symbolic zero derivatives of the core."""
    names = tuple(str(scalar) for scalar in scalars)
    found = _named_zero_factors(names, zero or ())
    if core is not None:
        found |= _symbolic_zero_factors(names, core)
    return frozenset(found)


def _requested_factors(
    derivatives: object, model: _Model, thermo: Sequence[str]
) -> list[tuple[str, ...]]:
    """Resolve ``derivatives`` to the factor tuples of the outputs."""
    declared = {COMPOSITION, *thermo}
    used = {t for x in model.scalars for t in model.depends[x]}
    if used - declared:
        raise ValueError(
            f"scalars depend on undeclared {sorted(used - declared)}"
        )
    attributes = resolve_derivatives(derivatives, declared_variables=declared)
    return [_ATTRIBUTE_FACTORS[attribute] for attribute in attributes]


def _assemble(
    model: _Model,
    requested: Sequence[tuple[str, ...]],
    options: CodegenOptions,
    subroutine_name: str,
) -> tuple[str, _Inputs]:
    """Return the routine text and the inputs it uses."""
    inputs = _Inputs(model)
    optional = options.optional_derivatives
    body = _body(inputs, requested, optional)
    declarations = _declarations(inputs, requested, optional)
    local_lines = _locals(inputs, requested)
    dummies = [
        line.split("::")[1].split("(")[0].strip() for line in declarations
    ]
    _check_unique([*dummies, *(n for n in _local_names(local_lines))])
    arguments = ", ".join(dummies)
    head = f"subroutine {subroutine_name}({arguments})"
    lines = [
        head,
        *_indented(["implicit none", *declarations, *local_lines]),
        "",
        *_indented(body),
        f"end subroutine {subroutine_name}",
    ]
    text = "\n".join(lines) + "\n"
    return (
        apply_real_kind(text, options.real_kind, options.kind_module),
        inputs,
    )


def _local_names(lines: Sequence[str]) -> list[str]:
    """Return the variable names declared in ``lines``."""
    names: list[str] = []
    for line in lines:
        names += [word.strip() for word in line.split("::")[-1].split(",")]
    return names


def scalar_chain_routine(
    scalars: Sequence[str],
    depends_on: Mapping[str, Iterable[str]] | None = None,
    *,
    linear: Iterable[str] = (),
    total_moles: Iterable[str] = (),
    core: "DiffPlz | None" = None,
    zero: Iterable[str] | None = None,
    thermo: Sequence[str] = THERMO_VARIABLES,
    derivatives: object = "all",
    subroutine_name: str | None = None,
    core_name: str = "f",
    name: str = "ar",
    options: CodegenOptions | None = None,
) -> str:
    """Emit the chain-rule assembly of a scalar core as a Fortran routine.

    The core ``F(x_1, ..., x_m; V, T)`` is the kernel generated with
    ``DiffPlz(..., variables=[...])`` and ``scalars`` in the same order.
    Every scalar depends on the composition ``n`` (its mixing rule), and
    optionally on ``T`` and/or ``V``.  The routine takes the derivatives of
    the core (named as the core kernel emits them, ``dfdB``, ``dfdBdD``,
    ``dfdBdT``, ``dfdT``, ``dfdT2``, ``dfdTdV`` ...) and those of the
    scalars, and returns the total derivatives of ``Ar``.

    Inputs (``nc`` is an integer argument; arrays are explicit-shape):

    * ``dXdn(nc)``, ``dXdndn(nc, nc)`` (not for ``linear`` scalars);
    * if ``X`` depends on ``T``: ``dXdT``, ``dXdT2`` and ``dXdndT(nc)``
      (``dXdTdV`` if it also depends on ``V``); likewise for ``V``.

    Only the inputs the requested outputs need are arguments.  Use
    :func:`scalar_chain_core_derivatives` for the derivatives to request
    from the core kernel.  Outputs have the usual names with prefix
    ``d<name>``: ``dardT``, ``dardV``, ``dardT2``, ``dardTdV``, ``dardV2``,
    ``dardn``, ``dardndT``, ``dardndV``, ``dardndn``.  The Hessian does
    ``O(m)`` work per element on the lower triangle and mirrors it.

    Parameters
    ----------
    scalars : sequence of str
        Names of the scalars, in the declaration order of the core.
    depends_on : mapping, optional
        Scalar -> iterable of ``"T"``/``"V"`` it depends on besides ``n``.
    linear : iterable of str, optional
        Scalars linear in ``n`` (their Hessian input is omitted).
    total_moles : iterable of str, optional
        Scalars equal to the total number of moles, ``nT = sum(n)``: their
        gradient is one and their Hessian zero, so they have no input at
        all and the sums over the scalars add the core derivative
        directly (``F_S`` instead of ``F_S * dSdn(i)``).  They must not
        depend on ``T`` or ``V`` and imply ``linear``.  The name is
        ``total_moles`` and not a general ``unit_gradient`` because a
        scalar with unit gradient and no ``T``/``V`` dependence is ``nT``
        up to a constant, which no derivative sees.
    core : DiffPlz, optional
        The scalar core (``DiffPlz(..., variables=[...])``).  Its first and
        second derivatives that are identically zero (SymPy ``== 0``; the
        core kernel does not emit them) are dropped: they are not inputs
        and no generated statement reads them.
    zero : iterable of str, optional
        The same, declared by name (``"dD2"``, ``"dBdS"``, ``"dT2"``; the
        order of the factors is free), merged with those found in ``core``.
    thermo : sequence of str, optional
        Thermodynamic variables the model uses (default ``T`` and ``V``).
    derivatives : str or list of str, optional
        Outputs, as in ``to_fortran`` (default ``"all"`` of the applicable
        ones; pressure derivatives are not supported).
    subroutine_name : str, optional
        Name of the routine (default ``<name>_chain``).
    core_name : str, optional
        Name of the core, prefix of its derivatives (default ``f``).
    name : str, optional
        Name of the total function, prefix of the outputs (default ``ar``).
    options : CodegenOptions, optional
        ``real_kind``, ``kind_module`` and ``optional_derivatives`` apply:
        with the latter every output and every input is ``optional`` and
        an output is computed if present (its inputs must then be given).

    Returns
    -------
    str
        The subroutine.

    Raises
    ------
    ValueError
        If the declarations are inconsistent or two names clash ignoring
        case.
    """
    model = _build_model(
        scalars,
        depends_on or {},
        linear,
        core_name,
        name,
        _zero_factors(scalars, core, zero),
        total_moles,
    )
    requested = _requested_factors(derivatives, model, thermo)
    text, _ = _assemble(
        model,
        requested,
        options or CodegenOptions(),
        subroutine_name or f"{name}_chain",
    )
    return text


def scalar_chain_core_derivatives(
    scalars: Sequence[str],
    depends_on: Mapping[str, Iterable[str]] | None = None,
    *,
    linear: Iterable[str] = (),
    total_moles: Iterable[str] = (),
    core: "DiffPlz | None" = None,
    zero: Iterable[str] | None = None,
    thermo: Sequence[str] = THERMO_VARIABLES,
    derivatives: object = "all",
    core_name: str = "f",
    name: str = "ar",
) -> list[str]:
    """Return the core derivatives that :func:`scalar_chain_routine` reads.

    Parameters
    ----------
    scalars, depends_on, linear, total_moles, core, zero, thermo, derivatives
    core_name
        As in :func:`scalar_chain_routine`; the zero derivatives of the
        core are not returned.
    name
        As in :func:`scalar_chain_routine`.

    Returns
    -------
    list of str
        Names for ``to_fortran(core, derivatives=...)`` (``"dB"``,
        ``"dBdD"``, ``"dT"`` ...), in the order of the routine's inputs.
    """
    model = _build_model(
        scalars,
        depends_on or {},
        linear,
        core_name,
        name,
        _zero_factors(scalars, core, zero),
        total_moles,
    )
    requested = _requested_factors(derivatives, model, thermo)
    _, inputs = _assemble(model, requested, CodegenOptions(), "unused")
    prefix = f"d{core_name}"
    ordered = sorted(inputs.core, key=lambda item: inputs.core[item])
    return [item[len(prefix) :] for item in ordered]
