"""Typed, immutable options of the Fortran generator.

:class:`CodegenOptions` groups the *behaviour* options of
:func:`thermodiff.fortran.to_fortran`.  Per-call context (the differentiated
object, the selected derivatives, names, extra arguments and so on) is not
part of it.  The legacy keyword values (``False``/``True``/strings) are
mapped to enums by :meth:`CodegenOptions.from_legacy` and back by
:meth:`CodegenOptions.to_legacy`.
"""

from __future__ import annotations

import dataclasses
import enum
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, NamedTuple


class ScalarChainMode(enum.Enum):
    """Legacy ``scalar_chain``: ``False | True | "cost" | "estimate"``."""

    OFF = False
    ON = True
    COST = "cost"
    ESTIMATE = "estimate"


class SepSumsMode(enum.Enum):
    """Legacy ``sep_sums``: ``False | True | "cost"``."""

    OFF = False
    ALL = True
    COST = "cost"


class LowrankHessMode(enum.Enum):
    """Legacy ``lowrank_hess``: ``False | True | "sym"``."""

    OFF = False
    ON = True
    SYM = "sym"


class CseBackend(enum.Enum):
    """Algorithm of the scalar CSE pass."""

    THERMODIFF = "thermodiff"
    SYMPY = "sympy"


class Preset(enum.Enum):
    """Named option sets, see :data:`PRESET_SETTINGS`."""

    MINIMAL = "minimal"
    STANDARD = "standard"
    FASTEST = "fastest"


_REQUIRED_SPLIT_ACC = 2


def _coerce_mode(
    enum_type: type[enum.Enum], truthy_fallback: enum.Enum, name: str
) -> Callable[[Any], enum.Enum]:
    """Build the coercer of a legacy-valued enum option.

    Parameters
    ----------
    enum_type : type
        Enum whose values are the legacy values.
    truthy_fallback : enum.Enum
        Member for an unknown truthy value (today it acts like ``True``).
    name : str
        Option name, for the warning text.

    Returns
    -------
    callable
        Function mapping a member or legacy value to a member.
    """
    off = enum_type(False)

    def coerce(value: Any) -> enum.Enum:
        if isinstance(value, enum_type):
            return value
        try:
            return enum_type(value)
        except ValueError:
            pass
        if not value:
            return off
        warnings.warn(
            f"{name}={value!r} is not a known mode; treated as "
            f"{truthy_fallback.value!r}",
            UserWarning,
            stacklevel=4,
        )
        return truthy_fallback

    return coerce


def _coerce_cse_backend(value: Any) -> CseBackend:
    """Map a legacy ``cse_backend`` string to :class:`CseBackend`."""
    if isinstance(value, CseBackend):
        return value
    try:
        return CseBackend(value)
    except ValueError:
        raise ValueError(f"unknown cse_backend {value!r}") from None


def _coerce_split_acc(value: Any) -> int | None:
    """Map a legacy ``split_acc`` (False/None/True/int/str) to int or None."""
    if value is None or value is False:
        return None
    return int(value)


def _coerce_optional_bool(value: Any) -> bool | None:
    """Keep ``None`` (inherit ``share``), otherwise ``bool(value)``."""
    return None if value is None else bool(value)


_COERCERS: Mapping[str, Callable[[Any], Any]] = MappingProxyType(
    {
        "share_sums": _coerce_optional_bool,
        "scalar_cse": _coerce_optional_bool,
        "cse_backend": _coerce_cse_backend,
        "array_pad": int,
        "sep_sums": _coerce_mode(SepSumsMode, SepSumsMode.ALL, "sep_sums"),
        "lowrank_hess": _coerce_mode(
            LowrankHessMode, LowrankHessMode.ON, "lowrank_hess"
        ),
        "scalar_chain": _coerce_mode(
            ScalarChainMode, ScalarChainMode.ON, "scalar_chain"
        ),
        "split_acc": _coerce_split_acc,
    }
)

# Fields whose legacy value is not the field value itself.
_ENUM_FIELDS = (
    "cse_backend",
    "sep_sums",
    "lowrank_hess",
    "scalar_chain",
)


@dataclass(frozen=True)
class CodegenOptions:
    """Behaviour options of :func:`thermodiff.fortran.to_fortran`.

    Every field has the name and default of the keyword of the same name in
    ``to_fortran``; see its docstring for the meaning of each one.  Values
    are coerced on construction (legacy ``False``/``True``/strings are
    accepted for the enum-valued fields).

    Notes
    -----
    ``scalar_chain``, ``sep_sums`` and ``lowrank_hess`` receive
    :class:`ScalarChainMode`, :class:`SepSumsMode` and
    :class:`LowrankHessMode`; ``split_acc`` is ``None`` or ``2``.
    """

    strict: bool = False
    optional_derivatives: bool = False
    array_cse: bool = True
    share: bool = True
    share_sums: bool | None = None
    scalar_cse: bool | None = None
    param_symmetry: bool = False
    hessian_symmetry: bool = False
    cse_backend: CseBackend = CseBackend.THERMODIFF
    sum_alpha: bool = False
    array_pad: int = 0
    hoist_invariants: bool = False
    chain_rule: bool = False
    hoist_loops: bool = False
    sum_free_alpha: bool = False
    fuse_temps: bool = False
    nest_sums: bool = False
    split_sqrt: bool = False
    split_sums: bool = False
    nest_shared: bool = False
    sym_matvec: bool = False
    col_major: bool = False
    pair_cse: bool = False
    pair_rates: bool = False
    sum_transpose: bool = False
    sep_sums: SepSumsMode = SepSumsMode.OFF
    sym_rank3: bool = False
    nest_general: bool = False
    sym_sum2: bool = False
    cse_expensive: bool = False
    fuse_loops: bool = False
    nested_nodes: bool = False
    sym_perm: bool = False
    fuse_guard: bool = False
    nest_cost: bool = False
    comp_fuse: bool = False
    fuse_outer: bool = False
    fuse_libm_guard: bool = False
    keep_libm_tables: bool = False
    diag_table: bool = False
    sib_fuse: bool = False
    dummy_canon: bool = False
    float_canon: bool = False
    contiguous_args: bool = False
    table_layout: bool = False
    lowrank_hess: LowrankHessMode = LowrankHessMode.OFF
    drop_pair_tables: bool = False
    share_elementwise: bool = False
    scalar_chain: ScalarChainMode = ScalarChainMode.OFF
    split_affine: bool = False
    split_acc: int | None = None

    def __post_init__(self) -> None:
        """Coerce every field to its canonical type."""
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            coerce = _COERCERS.get(field.name, bool)
            object.__setattr__(self, field.name, coerce(value))

    @classmethod
    def from_legacy(cls, **kwargs: Any) -> CodegenOptions:
        """Build options from ``to_fortran`` keyword values.

        Parameters
        ----------
        **kwargs
            Behaviour keywords of ``to_fortran`` with their legacy values.

        Returns
        -------
        CodegenOptions
            The coerced options.

        Raises
        ------
        TypeError
            For an unknown keyword.
        ValueError
            For an unknown ``cse_backend`` or a non-integer ``split_acc``.
        """
        return cls(**kwargs)

    def to_legacy(self) -> dict[str, Any]:
        """Return the keyword arguments of today's ``to_fortran``.

        Returns
        -------
        dict
            Option name to legacy value (enums become their values and
            ``split_acc=None`` becomes ``False``).
        """
        legacy = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if field.name in _ENUM_FIELDS:
                value = value.value
            elif field.name == "split_acc" and value is None:
                value = False
            legacy[field.name] = value
        return legacy

    @classmethod
    def from_preset(
        cls, preset: Preset | str, **overrides: Any
    ) -> CodegenOptions:
        """Build a preset and apply overrides on top of it.

        Presets
        -------
        ``"minimal"``
            Today's ``to_fortran`` defaults.
        ``"standard"``
            The recommended default: the e8 settings of the met4 campaign
            (chain rule, symmetry, nesting and sum sharing).
        ``"fastest"``
            The p33 settings: the fastest generated code of the met4
            campaign.  Meant for ``optional_derivatives=False``, because a
            kernel specialised to the needed derivative set is what pays
            off.  Generation is slower and reassociation changes the last
            bits of the results.

        Parameters
        ----------
        preset : Preset or str
            Preset member or its value.
        **overrides
            Options replacing the preset's (legacy values or enums).

        Returns
        -------
        CodegenOptions
            The resulting options.
        """
        settings = PRESET_SETTINGS[Preset(preset)]
        return cls(**{**settings, **overrides})

    def replace(self, **changes: Any) -> CodegenOptions:
        """Return a copy with some fields changed (values are coerced).

        Parameters
        ----------
        **changes
            Field names and their new values.

        Returns
        -------
        CodegenOptions
            The modified copy.
        """
        return dataclasses.replace(self, **changes)

    def validate(self) -> tuple[CodegenOptions, tuple[str, ...]]:
        """Disable options that today's generator silently ignores.

        Returns
        -------
        tuple
            The normalised options and the names of the ignored options.
            A single ``UserWarning`` lists them.

        Raises
        ------
        ValueError
            If ``split_acc`` is not 2 and ``optional_derivatives`` is off
            (the only case where ``to_fortran`` raises).
        """
        return _validate(self)


class _Rule(NamedTuple):
    """Options that have no effect when ``condition`` holds."""

    condition: Callable[[CodegenOptions], bool]
    names: tuple[str, ...]


def _effective_share_sums(options: CodegenOptions) -> bool:
    """Return ``share_sums`` resolved against ``share``."""
    if options.share_sums is None:
        return options.share
    return options.share_sums


_RULES: tuple[_Rule, ...] = (
    _Rule(
        lambda o: o.optional_derivatives,
        (
            "lowrank_hess",
            "drop_pair_tables",
            "share_elementwise",
            "split_affine",
            "fuse_loops",
            "sib_fuse",
            "diag_table",
            "comp_fuse",
            "fuse_outer",
            "split_acc",
            "scalar_chain",
        ),
    ),
    _Rule(
        lambda o: not o.chain_rule,
        (
            "split_sqrt",
            "split_sums",
            "pair_cse",
            "pair_rates",
            "nested_nodes",
            "scalar_chain",
        ),
    ),
    _Rule(
        lambda o: not o.chain_rule and not o.col_major,
        ("table_layout",),
    ),
    _Rule(
        lambda o: not _effective_share_sums(o),
        ("sum_alpha", "sum_free_alpha", "sum_transpose"),
    ),
    _Rule(lambda o: not o.array_cse, ("cse_expensive",)),
    _Rule(lambda o: not o.fuse_temps, ("fuse_guard", "keep_libm_tables")),
    _Rule(lambda o: not o.nest_sums, ("nest_general", "nest_cost")),
    _Rule(
        lambda o: not (o.fuse_temps or o.nest_sums or o.param_symmetry),
        ("nest_shared",),
    ),
    _Rule(
        lambda o: not o.param_symmetry,
        ("sym_matvec", "sym_sum2", "sym_perm"),
    ),
    _Rule(lambda o: not o.fuse_outer, ("fuse_libm_guard",)),
)

# Of the ``scalar_chain`` values, only these are coerced to off today; ``ON``
# also fills ``stats`` keys, so it is never reported as ignored.
_COERCED_SCALAR_CHAIN = (ScalarChainMode.COST, ScalarChainMode.ESTIMATE)


def _is_active(options: CodegenOptions, name: str) -> bool:
    """Tell whether ``name`` is set to something other than its default."""
    value = getattr(options, name)
    if name == "scalar_chain":
        return value in _COERCED_SCALAR_CHAIN
    return value != getattr(CodegenOptions(), name)


def _ignored_names(options: CodegenOptions) -> list[str]:
    """Return the active options disabled by some rule on ``options``."""
    names = {
        name
        for rule in _RULES
        if rule.condition(options)
        for name in rule.names
        if _is_active(options, name)
    }
    return sorted(names, key=_FIELD_ORDER.index)


def _apply_rules(options: CodegenOptions) -> tuple[CodegenOptions, list[str]]:
    """Apply the rules to a fixed point.

    Returns
    -------
    tuple
        The normalised options and the names ignored by the rules that
        fire on the options as given (names disabled only as a consequence
        of an earlier disabling are not listed).
    """
    defaults = CodegenOptions()
    listed = _ignored_names(options)
    pending = listed
    while pending:
        options = options.replace(
            **{name: getattr(defaults, name) for name in pending}
        )
        pending = _ignored_names(options)
    return options, listed


_FIELD_ORDER = tuple(
    field.name for field in dataclasses.fields(CodegenOptions)
)


def _check_split_acc(options: CodegenOptions) -> None:
    """Raise when ``split_acc`` is illegal where ``to_fortran`` raises."""
    if options.optional_derivatives:
        return
    if options.split_acc not in (None, _REQUIRED_SPLIT_ACC):
        raise ValueError("split_acc supports 2 accumulators only")


def _validate(
    options: CodegenOptions,
) -> tuple[CodegenOptions, tuple[str, ...]]:
    """Implement :meth:`CodegenOptions.validate`."""
    _check_split_acc(options)
    normalised, ignored = _apply_rules(options)
    if ignored:
        warnings.warn(
            "options ignored by the current combination and disabled: "
            + ", ".join(ignored),
            UserWarning,
            stacklevel=3,
        )
    return normalised, tuple(ignored)


def _preset_settings() -> Mapping[Preset, Mapping[str, object]]:
    """Build the immutable preset table."""
    standard = dict.fromkeys(
        (
            "chain_rule",
            "fuse_temps",
            "hessian_symmetry",
            "hoist_invariants",
            "hoist_loops",
            "nest_shared",
            "nest_sums",
            "param_symmetry",
            "split_sqrt",
            "split_sums",
            "sum_alpha",
            "sum_free_alpha",
            "sym_matvec",
        ),
        True,
    )
    fastest = dict.fromkeys(
        (
            "chain_rule",
            "col_major",
            "comp_fuse",
            "contiguous_args",
            "cse_expensive",
            "diag_table",
            "drop_pair_tables",
            "dummy_canon",
            "float_canon",
            "fuse_guard",
            "fuse_libm_guard",
            "fuse_loops",
            "fuse_outer",
            "fuse_temps",
            "hessian_symmetry",
            "hoist_invariants",
            "hoist_loops",
            "keep_libm_tables",
            "nest_cost",
            "nest_general",
            "nest_shared",
            "nest_sums",
            "nested_nodes",
            "pair_cse",
            "pair_rates",
            "param_symmetry",
            "share_elementwise",
            "sib_fuse",
            "split_affine",
            "split_sqrt",
            "split_sums",
            "sum_alpha",
            "sum_free_alpha",
            "sum_transpose",
            "sym_matvec",
            "sym_perm",
            "sym_rank3",
            "sym_sum2",
            "table_layout",
        ),
        True,
    )
    fastest.update(
        lowrank_hess=LowrankHessMode.ON,
        scalar_chain=ScalarChainMode.ESTIMATE,
        sep_sums=SepSumsMode.COST,
        split_acc=_REQUIRED_SPLIT_ACC,
    )
    return MappingProxyType(
        {
            Preset.MINIMAL: MappingProxyType({}),
            Preset.STANDARD: MappingProxyType(standard),
            Preset.FASTEST: MappingProxyType(fastest),
        }
    )


PRESET_SETTINGS: Mapping[Preset, Mapping[str, object]] = _preset_settings()
