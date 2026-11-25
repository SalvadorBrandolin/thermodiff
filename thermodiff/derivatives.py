"""Resolution of user supplied derivative selections.

Users select the derivatives they need with friendly names such as ``dT``
or ``dndn``, optionally grouped as ``all``, ``none``, ``first`` or
``second``.  The generators, on the other hand, work with the attribute
names of :class:`~thermodiff.diffplz.DiffPlz` (``dt``, ``dni`` ...).  This
module maps one vocabulary onto the other and checks that every requested
derivative is applicable to the variables declared by the model.

The names are ::

    dT   dV   dP   dn
    dT2  dV2  dP2  dndn
    dTdV dTdP dVdP  dndT dndV dndP

A leading ``df`` is optional (``dfdn`` is the same as ``dn``) and the
``DiffPlz`` attribute names are accepted as aliases, so nothing that used
to work stops working.

Extra scalar variables
----------------------
A :class:`~thermodiff.diffplz.DiffPlz` can declare extra scalar
differentiation variables (``variables=[B, D]``).  Their derivatives are
spelled like the thermodynamic ones, ``d`` followed by the name of each
variable: ``dB``, ``dB2``, ``dBdD`` and the mixed ones with the
thermodynamic variables, ``dBdT``, ``dBdV``, ``dBdP`` and ``dndB`` (an
array over ``n[i]``).  Names are case sensitive.  Within a mixed
derivative the factors are written in the canonical order ``n``, the
declared variables (in declaration order), ``T``, ``V``, ``P``; other
orders are accepted and normalised.  The groups ``first``, ``second`` and
``all`` include them when the variables are declared.

The attribute of such a derivative is :data:`SCALAR_PREFIX` followed by its
canonical name (``var_dBdD``).
"""

from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations_with_replacement

__all__ = [
    "DERIVATIVE_ATTRIBUTES",
    "FIRST_ORDER",
    "SECOND_ORDER",
    "GROUPS",
    "SCALAR_PREFIX",
    "resolve_derivatives",
    "scalar_derivative_specs",
    "scalar_suffix",
]

#: Derivative attributes, in the order in which they are emitted.
DERIVATIVE_ATTRIBUTES = (
    "dt",
    "dv",
    "dp",
    "dni",
    "dt2",
    "dv2",
    "dp2",
    "dnidnj",
    "dtdv",
    "dtdp",
    "dtdni",
    "dvdni",
    "dvdp",
    "dpdni",
)

#: First order derivatives.
FIRST_ORDER = ("dt", "dv", "dp", "dni")

#: Second order derivatives.
SECOND_ORDER = (
    "dt2",
    "dv2",
    "dp2",
    "dnidnj",
    "dtdv",
    "dtdp",
    "dtdni",
    "dvdni",
    "dvdp",
    "dpdni",
)

#: Prefix of the attributes of the derivatives with respect to the extra
#: scalar variables (``var_dB``); it keeps them apart from the others.
SCALAR_PREFIX = "var_"

#: Names reserved for the thermodynamic variables; an extra scalar variable
#: cannot use them, in any case.
RESERVED_VARIABLE_NAMES = frozenset({"t", "v", "p", "n"})

#: Rank of the thermodynamic factors in a canonical scalar derivative name:
#: ``n`` first, then the declared variables, then ``T``, ``V``, ``P``.
_COMPOSITION_FACTOR = "n"
_THERMO_NAMES = frozenset({"T", "V", "P", "n"})
_THERMO_FACTOR_RANK = {_COMPOSITION_FACTOR: 0, "T": 1000, "V": 1001, "P": 1002}

#: Group names accepted together with the individual derivative names.
GROUPS = ("all", "none", "first", "second")

#: Friendly spelling of each attribute, used in error messages.
_FRIENDLY = {
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

#: Normalised name -> derivative attribute.  The attribute names
#: themselves are accepted, together with a few friendly aliases.
_ALIASES = {attribute: attribute for attribute in DERIVATIVE_ATTRIBUTES}
_ALIASES.update(
    {
        "dn": "dni",
        "dndn": "dnidnj",
        "dn2": "dnidnj",
        "dndt": "dtdni",
        "dtdn": "dtdni",
        "dndv": "dvdni",
        "dvdn": "dvdni",
        "dndp": "dpdni",
        "dpdn": "dpdni",
    }
)

#: Variables each derivative needs in order to make sense.
_REQUIRED = {
    "dt": frozenset({"T"}),
    "dv": frozenset({"V"}),
    "dp": frozenset({"P"}),
    "dni": frozenset({"n"}),
    "dt2": frozenset({"T"}),
    "dv2": frozenset({"V"}),
    "dp2": frozenset({"P"}),
    "dnidnj": frozenset({"n"}),
    "dtdv": frozenset({"T", "V"}),
    "dtdp": frozenset({"T", "P"}),
    "dtdni": frozenset({"T", "n"}),
    "dvdni": frozenset({"V", "n"}),
    "dvdp": frozenset({"V", "P"}),
    "dpdni": frozenset({"P", "n"}),
}


def _tokens(derivatives):
    """Split ``derivatives`` into a list of raw tokens."""
    if isinstance(derivatives, str):
        chunks = [derivatives]
    else:
        chunks = [str(item) for item in derivatives]
    tokens = []
    for chunk in chunks:
        tokens.extend(chunk.replace(",", " ").split())
    return tokens


def _normalise(token):
    """Return the lower case token without a leading ``df``."""
    token = token.strip().lower()
    if len(token) > 2 and token.startswith("df"):
        token = token[2:]
    return token


def scalar_suffix(attribute: str) -> str:
    """Return the name suffix of a scalar derivative attribute.

    Parameters
    ----------
    attribute : str
        An attribute starting with :data:`SCALAR_PREFIX`, for example
        ``"var_dBdD"``.

    Returns
    -------
    str
        The suffix used in the generated names, ``"dBdD"``.
    """
    return attribute[len(SCALAR_PREFIX) :]


def _canonical_scalar_attribute(factors: tuple[str, ...]) -> str:
    """Return the attribute of the derivative with respect to ``factors``."""
    if len(factors) == 2 and factors[0] == factors[1]:
        return f"{SCALAR_PREFIX}d{factors[0]}2"
    return SCALAR_PREFIX + "".join("d" + factor for factor in factors)


def scalar_derivative_specs(
    variables: Sequence[str],
    declared_variables: Iterable[str] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Return the derivatives with respect to the extra scalar variables.

    Parameters
    ----------
    variables : sequence of str
        Names of the extra scalar variables, in declaration order.
    declared_variables : iterable of str, optional
        Thermodynamic variables the model declares; the mixed derivatives
        only involve those.  By default all of ``T``, ``V``, ``P`` and
        ``n``.

    Returns
    -------
    dict
        ``attribute -> factors`` for every first and second derivative that
        involves at least one extra variable, in emission order: first the
        first derivatives, then the second ones; the factors of each are
        in the canonical order ``n``, the variables, ``T``, ``V``, ``P``.
    """
    declared = None if declared_variables is None else set(declared_variables)
    thermo = [
        name
        for name in _THERMO_FACTOR_RANK
        if declared is None or name in declared
    ]
    names = [name for name in thermo if name == _COMPOSITION_FACTOR]
    names += list(variables)
    names += [name for name in thermo if name != _COMPOSITION_FACTOR]
    specs: dict[str, tuple[str, ...]] = {}
    for order in (1, 2):
        for factors in combinations_with_replacement(names, order):
            if any(factor in variables for factor in factors):
                specs[_canonical_scalar_attribute(factors)] = factors
    return specs


def _factor_key(factor: str, rank: Mapping[str, int]) -> int:
    """Sort key of a factor: ``n``, the variables, then ``T``, ``V``, ``P``."""
    if factor in rank:
        return rank[factor] + 1
    return _THERMO_FACTOR_RANK[factor]


def _split_factors(
    text: str, names: Mapping[str, str]
) -> list[tuple[str, ...]]:
    """Split ``text`` (``dBdT``) into every sequence of ``d<name>`` factors."""
    if not text:
        return [()]
    if not text.startswith("d"):
        return []
    rest = text[1:]
    return [
        (canonical,) + tail
        for name, canonical in names.items()
        if rest.startswith(name)
        for tail in _split_factors(rest[len(name) :], names)
    ]


def _parse_scalar_token(
    token: str, variables: Sequence[str]
) -> tuple[str, ...] | None:
    """Return the canonical factors of a scalar derivative name.

    Parameters
    ----------
    token : str
        Raw token, for example ``"dBdD"``, ``"dfdB2"`` or ``"dndB"``.
    variables : sequence of str
        Names of the extra scalar variables.

    Returns
    -------
    tuple of str or None
        The factors in canonical order, or ``None`` when ``token`` is not
        a derivative with respect to the variables.

    Raises
    ------
    ValueError
        If the token can be read in more than one way (the names of two
        variables run together).
    """
    names = {name: name for name in variables}
    names.update({"T": "T", "V": "V", "P": "P", "n": "n"})
    names.update({"t": "T", "v": "V", "p": "P"})
    rank = {name: position for position, name in enumerate(variables)}
    texts = [token] + ([token[2:]] if token.startswith("df") else [])
    readings: set[tuple[str, ...]] = set()
    for text in texts:
        parsed = _split_factors(text, names)
        if text.endswith("2"):
            parsed += [
                (single[0], single[0])
                for single in _split_factors(text[:-1], names)
                if len(single) == 1
            ]
        readings.update(
            tuple(sorted(factors, key=lambda f: _factor_key(f, rank)))
            for factors in parsed
            if len(factors) in (1, 2)
            and any(factor in rank for factor in factors)
        )
    if len(readings) > 1:
        raise ValueError(
            f"Ambiguous derivative name {token!r}: it can be read as "
            + " or ".join(sorted(repr("".join(r)) for r in readings))
        )
    return next(iter(readings), None)


def resolve_derivatives(
    derivatives, declared_variables=None, scalar_variables=()
):
    """Resolve user supplied derivative names to derivative attributes.

    Parameters
    ----------
    derivatives : str or iterable of str
        Derivative names.  Accepts the friendly names (``dT``, ``dn``,
        ``dndT`` ...), the :class:`~thermodiff.diffplz.DiffPlz` attribute
        names (``dt``, ``dni`` ...), names prefixed with ``df`` and the
        groups ``all``, ``none``, ``first`` and ``second``.
    declared_variables : iterable of str, optional
        Variables declared by the model.  When given, requested
        derivatives are checked against them and the groups only expand
        to derivatives that are applicable.
    scalar_variables : iterable of str, optional
        Names of the extra scalar variables of the model (see the module
        docstring).  Enables the names of their derivatives (``dB``,
        ``dBdD``, ``dndB`` ...) and adds them to the groups.

    Returns
    -------
    list of str
        The selected derivative attributes, in emission order: the
        thermodynamic ones, then those of the scalar variables.

    Raises
    ------
    ValueError
        If a name is unknown or is not applicable to the declared
        variables.
    """
    scalar_variables = tuple(scalar_variables)
    requested, groups, unknown = _classify_tokens(
        derivatives, scalar_variables
    )
    if unknown:
        valid = ", ".join(GROUPS + DERIVATIVE_ATTRIBUTES)
        if scalar_variables:
            valid += ", d<variable> and d<variable>d<variable> of " + (
                ", ".join(scalar_variables)
            )
        raise ValueError(
            f"Unknown derivatives: {', '.join(sorted(set(unknown)))}. "
            f"Valid names: {valid} (a leading 'df' is optional)."
        )

    declared = None
    if declared_variables is not None:
        declared = {str(variable) for variable in declared_variables}
    specs = scalar_derivative_specs(scalar_variables)
    required = {
        attribute: _required_variables(attribute, specs)
        for attribute in DERIVATIVE_ATTRIBUTES + tuple(specs)
    }
    if declared is not None:
        _check_declared(requested, required, declared)
        applicable = [a for a in required if required[a] <= declared]
    else:
        applicable = list(required)

    selected = set(requested)
    if "all" in groups:
        selected.update(applicable)
    if "first" in groups:
        selected.update(a for a in applicable if _order(a, specs) == 1)
    if "second" in groups:
        selected.update(a for a in applicable if _order(a, specs) == 2)
    return [a for a in required if a in selected]


def _classify_tokens(
    derivatives, scalar_variables: tuple[str, ...]
) -> tuple[list[str], set[str], list[str]]:
    """Split the tokens into attributes, groups and unknown names."""
    requested: list[str] = []
    groups: set[str] = set()
    unknown: list[str] = []
    for raw in _tokens(derivatives):
        token = _normalise(raw)
        if token in GROUPS:
            groups.add(token)
            continue
        attribute = _ALIASES.get(token)
        if attribute is None and scalar_variables:
            factors = _parse_scalar_token(raw.strip(), scalar_variables)
            if factors is not None:
                attribute = _canonical_scalar_attribute(factors)
        if attribute is None:
            unknown.append(raw)
        elif attribute not in requested:
            requested.append(attribute)
    return requested, groups, unknown


def _required_variables(
    attribute: str, specs: Mapping[str, tuple[str, ...]]
) -> frozenset[str]:
    """Return the thermodynamic variables a derivative needs."""
    if attribute in specs:
        return frozenset(
            factor for factor in specs[attribute] if factor in _THERMO_NAMES
        )
    return _REQUIRED[attribute]


def _order(attribute: str, specs: Mapping[str, tuple[str, ...]]) -> int:
    """Return the order (1 or 2) of a derivative attribute."""
    if attribute in specs:
        return len(specs[attribute])
    return 1 if attribute in FIRST_ORDER else 2


def _check_declared(
    requested: Iterable[str],
    required: Mapping[str, frozenset[str]],
    declared: set[str],
) -> None:
    """Raise if a requested derivative needs an undeclared variable."""
    for attribute in requested:
        missing = required[attribute] - declared
        if missing:
            name = _FRIENDLY.get(attribute) or scalar_suffix(attribute)
            raise ValueError(
                f"derivative {name!r} requires the variable(s) "
                + ", ".join(sorted(missing))
            )
