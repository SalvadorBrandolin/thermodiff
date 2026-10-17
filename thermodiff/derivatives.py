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
"""

__all__ = [
    "DERIVATIVE_ATTRIBUTES",
    "FIRST_ORDER",
    "SECOND_ORDER",
    "GROUPS",
    "resolve_derivatives",
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


def resolve_derivatives(derivatives, declared_variables=None):
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

    Returns
    -------
    list of str
        The selected derivative attributes, in emission order.

    Raises
    ------
    ValueError
        If a name is unknown or is not applicable to the declared
        variables.
    """
    requested = []
    groups = set()
    unknown = []
    for raw in _tokens(derivatives):
        token = _normalise(raw)
        if token in GROUPS:
            groups.add(token)
            continue
        attribute = _ALIASES.get(token)
        if attribute is None:
            unknown.append(raw)
            continue
        if attribute not in requested:
            requested.append(attribute)

    if unknown:
        valid = ", ".join(GROUPS + DERIVATIVE_ATTRIBUTES)
        raise ValueError(
            f"Unknown derivatives: {', '.join(sorted(set(unknown)))}. "
            f"Valid names: {valid} (a leading 'df' is optional)."
        )

    declared = None
    if declared_variables is not None:
        declared = {str(variable) for variable in declared_variables}
        for attribute in requested:
            missing = _REQUIRED[attribute] - declared
            if missing:
                raise ValueError(
                    f"derivative {_FRIENDLY[attribute]!r} requires the "
                    "variable(s) " + ", ".join(sorted(missing))
                )

    if declared is None:
        applicable = list(DERIVATIVE_ATTRIBUTES)
    else:
        applicable = [
            attribute
            for attribute in DERIVATIVE_ATTRIBUTES
            if _REQUIRED[attribute] <= declared
        ]

    selected = set(requested)
    if "all" in groups:
        selected.update(applicable)
    if "first" in groups:
        selected.update(a for a in applicable if a in FIRST_ORDER)
    if "second" in groups:
        selected.update(a for a in applicable if a in SECOND_ORDER)

    return [a for a in DERIVATIVE_ATTRIBUTES if a in selected]
