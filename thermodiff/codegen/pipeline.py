"""The ``to_fortran`` pipeline of the Fortran code generator."""

import dataclasses
import inspect
import os
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import FrameType
from typing import Any, NamedTuple, cast

import sympy as sp

from thermodiff.codegen import (
    CodegenOptions,
    CseBackend,
    LowrankHessMode,
    ScalarChainMode,
    SepSumsMode,
)
from thermodiff.codegen.common import (
    _DERIV_SUFFIX,
    _INDENT,
    _LoopStart,
    _Namer,
    _canonical_floats,
    _case_clash_renames,
    _free_indices,
    _is_placeholder,
    _kron_to_piecewise,
    _order_free,
    _referenced_temp_names,
    _rename_names,
    _sanitize,
    _temp_name,
)
from thermodiff.codegen.cse import (
    _common_scalar_cse,
    _indexed_cse,
    _share_sums,
    _sympy_scalar_cse,
)
from thermodiff.codegen.hoisting import (
    _hoist_invariants,
    _hoist_loop_invariants,
    _lowrank_bilinear,
    _separate_sums,
    _split_affine,
)
from thermodiff.codegen.ir import TempDef
from thermodiff.codegen.loops import (
    Staged,
    _diagonal_from_tables,
    _drop_pair_tables,
    _fuse_component_maps,
    _fuse_loop_nests,
    _fuse_outer_loops,
    _fuse_sibling_tables,
    _fuse_temporaries,
    _nest_double_sums,
    _permute_symmetric_subscripts,
    _prefers_column_order,
    _prefers_column_sum,
    _share_elementwise,
    _split_accumulators,
)
from thermodiff.codegen.render import (
    _declare_output,
    _dependency_order,
    _guard,
    _guarded,
    _render_definition,
    _render_output,
    _wrap_name_list,
    _wrap_signature,
)
from thermodiff.codegen.resolve import (
    _PlaceholderResolver,
    _scan_atoms,
    _select_derivatives,
)
from thermodiff.codegen.symmetry import (
    _canonical_symmetric,
    _fold_symmetric_sums,
    _merge_dummy_images,
    _swap,
    _symv_parts,
)
from thermodiff.thermovars import P, T, V, n, nc

#: ``attr`` of the pseudo-outputs that carry the scalars of ``scalar_chain``
_SCALAR_CHAIN = "_scalar_chain"
#: ``attr`` of the pseudo-outputs that carry the plain outputs (dropped)
_SCALAR_PLAIN = "_scalar_plain"


# ``estimate`` keeps propagation unless it is estimated to cost more than
# this factor over the plain path
_PROPAGATION_SLACK = 1.1
_REQUIRED_ACCUMULATORS = 2
_ESTIMATE_OR_COST = (ScalarChainMode.COST, ScalarChainMode.ESTIMATE)
# ``stats`` entry through which the structure leaks (``scalar_chain`` on)
_STRUCTURE_STATS_KEY = "_structured"

#: Result of the chain-rule step (see :func:`thermodiff.chain`)
Structure = dict[str, Any]
#: Generation statistics filled while emitting
Stats = dict[str, Any]
#: ``(attr, out_name, expr)`` of an output
Output = tuple[str, str, sp.Expr]
#: ``(attr, out_name, expr, free_order)`` of an output
PreparedOutput = tuple[str, str, sp.Expr, list[sp.Idx]]
#: ``(name, body, rank, symmetric)`` of a component array
ComponentDef = tuple[str, sp.Expr, int, bool]
#: ``(name, body)`` of a scalar-chain scalar
ScalarDef = tuple[sp.Symbol, sp.Expr]


@dataclasses.dataclass(frozen=True)
class _EmitContext:
    """Inputs resolved once and shared by the stages of the generator.

    Attributes
    ----------
    diff : object
        The ``DiffPlz`` instance.
    selected : tuple of str
        Derivative attributes to emit, in canonical order.
    base : str
        Sanitised name of the value output.
    sub_name : str
        Sanitised subroutine name.
    integer_args : tuple of str
        Integer dummy arguments.
    module_symbols : frozenset of str
        Names provided by the enclosing module.
    symmetric_params : tuple
        Parameters declared symmetric.
    """

    diff: Any
    selected: tuple[str, ...]
    base: str
    sub_name: str
    integer_args: tuple[str, ...]
    module_symbols: frozenset[str]
    symmetric_params: tuple[str, ...]


class _RenamedDiff:
    """View of a ``DiffPlz`` whose symbol names were made case-unique.

    Parameters
    ----------
    diff : DiffPlz
        The instance to wrap.
    renames : dict
        ``old name -> new name`` applied to every expression read.
    """

    def __init__(self, diff: Any, renames: dict[str, str]) -> None:
        self._diff = diff
        self._renames = renames
        self.name = diff.name
        self.indexes = diff.indexes
        self.expression = _rename_names(diff.expression, renames)

    def __getattr__(self, attr: str) -> Any:
        """Return the derivative ``attr`` with the names renamed."""
        return _rename_names(getattr(self._diff, attr), self._renames)


def _expression_names(expr: sp.Basic) -> tuple[set[str], set[str], set[str]]:
    """Return the symbol, array and placeholder names used in ``expr``.

    Returns
    -------
    tuple of set
        ``(user names, index labels, thermodynamic symbols present)``:
        the names of the symbols, arrays and placeholder functions that
        are not ``T``, ``V``, ``P``, ``nc`` or ``n``; the labels of the
        indices; and the names of those fixed symbols that appear.
    """
    names = {str(sym) for sym in expr.atoms(sp.Symbol)}
    names.update(str(base) for base in expr.atoms(sp.IndexedBase))
    names.update(
        func.func.__name__
        for func in expr.atoms(sp.Function)
        if _is_placeholder(func)
    )
    fixed = {str(sym) for sym in (nc, n, P, T, V)}
    indices = {str(idx) for idx in expr.atoms(sp.Idx)}
    return names - fixed - indices, indices, names & fixed


def _case_renames(
    diff: Any,
    selected: Sequence[str],
    base: str,
    external: set[str],
    symmetric_params: Iterable[str],
) -> dict[str, str]:
    """Return the renames that make the names unique in Fortran.

    Fortran is case-insensitive: ``R`` and ``r`` are one variable.  The
    names of the parameters and placeholders are renamed (``r`` becomes
    ``r_1``); module symbols, integer arguments, outputs, ``n``, ``nc``,
    ``T``, ``V``, ``P`` and the indices keep their names.

    Parameters
    ----------
    diff : DiffPlz
        Instance holding the expression.
    selected : sequence of str
        Derivative attributes to emit.
    base : str
        Name of the value output.
    external : set of str
        Module symbols and integer arguments.
    symmetric_params : iterable of str
        Names that are parameters even if the expression hides them.

    Returns
    -------
    dict
        ``old name -> new name``, empty when no name clashes.

    Raises
    ------
    ValueError
        If two names that cannot be renamed differ only in case.
    """
    candidates, indices, thermo = _expression_names(diff.expression)
    outputs = {base} | {"d" + base + _DERIV_SUFFIX[attr] for attr in selected}
    fixed = external | outputs | indices | thermo
    return _case_clash_renames(
        fixed, (candidates | set(symmetric_params)) - external
    )


def _resolve_context(
    diff: Any,
    derivatives: Any,
    name: str | None,
    subroutine_name: str | None,
    integer_args: Iterable[str] | None,
    module_symbols: Iterable[str] | None,
    symmetric_params: Iterable[str] | None,
) -> _EmitContext:
    """Resolve names and the derivative selection into an immutable context.

    Parameters
    ----------
    diff : DiffPlz
        Instance holding the expression and derivatives.
    derivatives, name, subroutine_name, integer_args, module_symbols
        As in :func:`to_fortran`.
    symmetric_params : iterable
        Parameters declared symmetric.

    Returns
    -------
    _EmitContext
        The resolved context.
    """
    base = _sanitize(name if name is not None else diff.name)
    sub_name = (
        _sanitize(subroutine_name) if subroutine_name else base + "_derivs"
    )
    selected = tuple(_select_derivatives(diff, derivatives))
    integer_names = tuple(integer_args or ())
    module_names = frozenset(module_symbols or ())
    renames = _case_renames(
        diff,
        selected,
        base,
        set(integer_names) | module_names,
        symmetric_params or (),
    )
    return _EmitContext(
        diff=_RenamedDiff(diff, renames) if renames else diff,
        selected=selected,
        base=base,
        sub_name=sub_name,
        integer_args=integer_names,
        module_symbols=module_names,
        symmetric_params=tuple(
            renames.get(param, param) for param in symmetric_params or ()
        ),
    )


def _chain_structure(
    ctx: _EmitContext, opts: CodegenOptions
) -> Structure | None:
    """Run the chain-rule step, or return ``None`` without ``chain_rule``.

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    opts : CodegenOptions
        Options; ``scalar_chain`` is read as on or off.

    Returns
    -------
    dict or None
        The structured derivatives of the chain-rule step.
    """
    if not opts.chain_rule:
        return None
    from thermodiff.chain import chain_rule_derivatives

    scalar_chain = (
        opts.scalar_chain is ScalarChainMode.ON
        and not opts.optional_derivatives
    )
    structure: Structure = chain_rule_derivatives(  # type: ignore[no-untyped-call]  # noqa: E501
        ctx.diff.expression,
        list(ctx.selected),
        ctx.diff.indexes,
        symmetric_params=ctx.symmetric_params,
        split_sqrt=opts.split_sqrt,
        split_sums=opts.split_sums,
        pair_cse=opts.pair_cse,
        pair_rates=opts.pair_rates,
        nested_nodes=opts.nested_nodes,
        table_layout=opts.table_layout,
        scalar_chain=scalar_chain,
    )
    return structure


def _plain_structure(
    ctx: _EmitContext, opts: CodegenOptions, structured: Structure | None
) -> tuple[Structure | None, bool]:
    """Return the structure of the path without scalar-chain propagation.

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    opts : CodegenOptions
        Options of the plain path (``scalar_chain`` off).
    structured : dict or None
        Structure made with ``scalar_chain`` on.

    Returns
    -------
    plain : dict or None
        The ``_plain`` outputs of ``structured`` with its component
        definitions; ``structured`` itself when it carries none; a new
        chain-rule run when ``structured`` is ``None``.
    propagated : bool
        Whether ``structured`` carried a ``_plain`` set.
    """
    from thermodiff.chain import NAME_SCHEME_KEY

    if structured is None:
        return _chain_structure(ctx, opts), False
    if "_plain" not in structured:
        return structured, False
    plain = dict(structured["_plain"])
    plain["_component_defs"] = structured["_component_defs"]
    plain[NAME_SCHEME_KEY] = structured[NAME_SCHEME_KEY]
    return plain, True


def _generate_estimate(
    ctx: _EmitContext, opts: CodegenOptions, stats: Stats | None
) -> str:
    """Emit with ``scalar_chain="estimate"``: one structure, one emission.

    The path is chosen from the operation estimates of the two scalar
    blocks made by the chain-rule step.  Ties and near ties go to
    propagation: after the scalar CSE of the pipeline it is the cheaper
    path in every case checked (7 models).

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    opts : CodegenOptions
        Options with ``scalar_chain`` set to ``ESTIMATE``.
    stats : dict or None
        Receives the statistics, ``scalar_chain_used`` and
        ``scalar_chain_estimate``.

    Returns
    -------
    str
        The Fortran source.
    """
    chain_opts = opts.replace(scalar_chain=ScalarChainMode.ON)
    structured = _chain_structure(ctx, chain_opts)
    estimate = (structured or {}).get("_scalar_estimate")
    use = (
        estimate is not None
        and estimate[1] <= _PROPAGATION_SLACK * estimate[0]
    )
    if use:
        used_opts, emitted = chain_opts, structured
    else:
        used_opts = opts.replace(scalar_chain=ScalarChainMode.OFF)
        emitted, _ = _plain_structure(ctx, used_opts, structured)
    code = _emit_code(ctx, emitted, stats, True, **used_opts.to_legacy())
    if stats is not None:
        stats["scalar_chain_used"] = use
        stats["scalar_chain_estimate"] = estimate
    return code


def _generate_cost(
    ctx: _EmitContext, opts: CodegenOptions, stats: Stats | None
) -> str:
    """Emit with ``scalar_chain="cost"``: both paths, keep the cheaper.

    Keeps the path whose scalar block (definitions and outputs without
    loops) has fewer operations; the loops are the same by construction.

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    opts : CodegenOptions
        Options with ``scalar_chain`` set to ``COST``.
    stats : dict or None
        Receives the statistics of the chosen path and
        ``scalar_chain_used``.

    Returns
    -------
    str
        The Fortran source of the chosen path.
    """
    chain_opts = opts.replace(scalar_chain=ScalarChainMode.ON)
    structured = _chain_structure(ctx, chain_opts)
    stats_sc: Stats = {}
    code_sc = _emit_code(
        ctx, structured, stats_sc, True, **chain_opts.to_legacy()
    )
    stats_sc.pop(_STRUCTURE_STATS_KEY, None)
    plain_opts = opts.replace(scalar_chain=ScalarChainMode.OFF)
    plain, propagated = _plain_structure(ctx, plain_opts, structured)
    stats_plain: Stats = {}
    code_plain = _emit_code(
        ctx, plain, stats_plain, True, **plain_opts.to_legacy()
    )
    use = (
        propagated
        and stats_sc["scalar_operation_count"]
        < stats_plain["scalar_operation_count"]
    )
    chosen_code, chosen_stats = (
        (code_sc, stats_sc) if use else (code_plain, stats_plain)
    )
    if stats is not None:
        stats.update(chosen_stats)
        stats["scalar_chain_used"] = use
    return chosen_code


def _to_fortran_impl(
    diff: Any,
    derivatives: Any = None,
    name: str | None = None,
    subroutine_name: str | None = None,
    integer_args: Iterable[str] | None = None,
    module_symbols: Iterable[str] | None = None,
    symmetric_params: Iterable[str] | None = (),
    stats: Stats | None = None,
    *,
    opts: CodegenOptions,
) -> str:
    """Generate the subroutine from resolved options (see ``to_fortran``).

    Dispatches on ``opts.scalar_chain``: off and on make one structure and
    one emission, ``estimate`` and ``cost`` go through their own stage.

    Parameters
    ----------
    diff : DiffPlz
        Instance holding the expression and derivatives.
    derivatives, name, subroutine_name, integer_args, module_symbols
        As in :func:`to_fortran`.
    symmetric_params : iterable, optional
        Parameters declared symmetric.
    stats : dict, optional
        Receives generation statistics.
    opts : CodegenOptions
        Validated behaviour options.

    Returns
    -------
    str
        The Fortran source.
    """
    ctx = _resolve_context(
        diff,
        derivatives,
        name,
        subroutine_name,
        integer_args,
        module_symbols,
        symmetric_params,
    )
    if opts.scalar_chain in _ESTIMATE_OR_COST and not (
        opts.chain_rule and not opts.optional_derivatives
    ):
        opts = opts.replace(scalar_chain=ScalarChainMode.OFF)
    if opts.scalar_chain is ScalarChainMode.ESTIMATE:
        return _generate_estimate(ctx, opts, stats)
    if opts.scalar_chain is ScalarChainMode.COST:
        return _generate_cost(ctx, opts, stats)
    structured = _chain_structure(ctx, opts)
    return _emit_code(
        ctx,
        structured,
        stats,
        opts.scalar_chain is ScalarChainMode.ON,
        **opts.to_legacy(),
    )


@dataclasses.dataclass(eq=False)
class _EmitState:
    """Mutable state shared by the passes of one emission.

    The passes mutate it in place.  ``eq=False`` keeps the identity
    semantics: ``namer``, ``arrays`` and the lists of definitions are
    shared mutable objects that the helpers update and compare by
    identity.

    Attributes
    ----------
    opts : CodegenOptions
        Behaviour options of this emission.
    structured : dict or None
        Result of the chain-rule step.
    stats : dict or None
        Receives the statistics.
    count_scalar : bool
        Whether to report ``scalar_operation_count``.
    resolver : _PlaceholderResolver
        Placeholder resolver, one per emission.
    namer : _Namer
        Temporary-name generator, one per emission.
    selected : list of str
        Derivative attributes to emit.
    integer_args : list of str
        Integer dummy arguments.
    module_symbols : set of str
        Names provided by the enclosing module.
    symmetric_params : tuple
        Symmetric parameters (grows with the pair functions).
    scalar_chain_defs : sequence
        Scalar-chain definitions (empty without ``scalar_chain``).
    scalar_chain_active : bool
        Whether the structure carries scalar-chain nodes or definitions.
    sc_temp_names : set of str
        Names of the scalar-chain temporaries.
    outputs : list of tuple
        ``(attr, out_name, expr)`` of every output.
    share_sums, scalar_cse : bool
        Resolved ``share_sums`` and ``scalar_cse``.
    staged : list of tuple
        Outputs while they are canonicalised.
    prepared : list of tuple
        ``(attr, out_name, expr, free_order)`` of every output.
    all_exprs : list
        The output expressions (rewritten by the sharing passes).
    component_defs : list of tuple
        ``(name, body, rank, sym)`` of the component arrays.
    scans, params, placeholder_args : list
        Atom scans and sorted argument names of the signature.
    param_arrays : dict
        Rank of every array argument.
    needs_nc : bool
        Whether ``nc`` is an argument.
    inputs, output_decls : list of str
        Declarations of the dummy arguments.
    arrays : dict
        Rank of every temporary array.
    component_array_defs, array_defs, sum_defs, cse_defs : list
        Definition pools.
    reserved : set of str
        Names that temporaries must not take.
    residuals : list
        Output expressions after sharing.
    scalar_chain_definitions : list
        Definitions of the scalar-chain scalars.
    rendered : list of tuple
        ``(attr, out_name, residual, free_order)`` of every output.
    definitions : list
        Ordered definitions to render.
    lowrank_outs : set of str
        Outputs made with the low-rank form.
    out_names : dict
        Output name of every attribute.
    consumers : dict
        Outputs that read each temporary.
    body : list of str
        Rendered body lines.
    temps, loop_indices : list of str
        Names declared as scalars and loop indices.
    text : str
        The Fortran source.
    """

    opts: CodegenOptions
    structured: Structure | None
    stats: Stats | None
    count_scalar: bool
    resolver: _PlaceholderResolver
    namer: _Namer
    selected: list[str]
    integer_args: list[str]
    module_symbols: set[str]
    symmetric_params: tuple[str, ...]
    scalar_chain_defs: Sequence[ScalarDef] = ()
    scalar_chain_active: bool = False
    sc_temp_names: set[str] = dataclasses.field(default_factory=set)
    outputs: list[Output] = dataclasses.field(default_factory=list)
    share_sums: bool = False
    scalar_cse: bool = False
    staged: Staged = dataclasses.field(default_factory=list)
    prepared: list[PreparedOutput] = dataclasses.field(default_factory=list)
    all_exprs: list[sp.Expr] = dataclasses.field(default_factory=list)
    component_defs: list[ComponentDef] = dataclasses.field(
        default_factory=list
    )
    scans: list[Any] = dataclasses.field(default_factory=list)
    param_arrays: dict[str, int] = dataclasses.field(default_factory=dict)
    params: list[str] = dataclasses.field(default_factory=list)
    placeholder_args: list[str] = dataclasses.field(default_factory=list)
    needs_nc: bool = False
    inputs: list[str] = dataclasses.field(default_factory=list)
    output_decls: list[str] = dataclasses.field(default_factory=list)
    arrays: dict[str, int] = dataclasses.field(default_factory=dict)
    component_array_defs: list[TempDef] = dataclasses.field(
        default_factory=list
    )
    array_defs: list[TempDef] = dataclasses.field(default_factory=list)
    sum_defs: list[TempDef] = dataclasses.field(default_factory=list)
    cse_defs: list[TempDef] = dataclasses.field(default_factory=list)
    reserved: set[str] = dataclasses.field(default_factory=set)
    residuals: list[sp.Expr] = dataclasses.field(default_factory=list)
    scalar_chain_definitions: list[TempDef] = dataclasses.field(
        default_factory=list
    )
    rendered: list[PreparedOutput] = dataclasses.field(default_factory=list)
    definitions: list[TempDef] = dataclasses.field(default_factory=list)
    lowrank_outs: set[str] = dataclasses.field(default_factory=set)
    out_names: dict[str, str] = dataclasses.field(default_factory=dict)
    consumers: dict[str, set[str]] = dataclasses.field(default_factory=dict)
    body: list[str] = dataclasses.field(default_factory=list)
    temps: list[str] = dataclasses.field(default_factory=list)
    loop_indices: list[str] = dataclasses.field(default_factory=list)
    text: str = ""


class Pass(NamedTuple):
    """One stage of the emission.

    Attributes
    ----------
    name : str
        Stage name.
    enabled : callable
        Option-level gate: ``enabled(opts)`` tells whether the stage runs.
        Gates that also depend on data stay inside ``run``.
    run : callable
        ``run(ctx, state)`` mutates ``state`` in place.
    """

    name: str
    enabled: Callable[[CodegenOptions], bool]
    run: Callable[[_EmitContext, _EmitState], None]


def _always(opts: CodegenOptions) -> bool:
    """Gate of the stages without an option condition."""
    return True


def _option_or(value: bool | None, fallback: bool) -> bool:
    """Return ``fallback`` when ``value`` is ``None``, else ``value``."""
    return fallback if value is None else value


def _legacy_options(arguments: Mapping[str, Any]) -> CodegenOptions:
    """Build the options from the keyword arguments of ``_emit_code``.

    Parameters
    ----------
    arguments : mapping
        Local variables of ``_emit_code``; only the option names are used.

    Returns
    -------
    CodegenOptions
        The coerced options.
    """
    names = {field.name for field in dataclasses.fields(CodegenOptions)}
    return CodegenOptions.from_legacy(
        **{name: arguments[name] for name in names}
    )


def _new_state(
    ctx: _EmitContext,
    structured: Structure | None,
    stats: Stats | None,
    count_scalar: bool,
    opts: CodegenOptions,
) -> _EmitState:
    """Create the state of one emission from its inputs.

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    structured : dict or None
        Result of the chain-rule step.
    stats : dict or None
        Receives the statistics.
    count_scalar : bool
        Whether to report ``scalar_operation_count``.
    opts : CodegenOptions
        Behaviour options.

    Returns
    -------
    _EmitState
        A state with fresh resolver and namer.
    """
    return _EmitState(
        opts=opts,
        structured=structured,
        stats=stats,
        count_scalar=count_scalar,
        resolver=_PlaceholderResolver(strict=opts.strict),
        namer=_Namer(),
        selected=list(ctx.selected),
        integer_args=list(ctx.integer_args),
        module_symbols=set(ctx.module_symbols),
        symmetric_params=ctx.symmetric_params,
        scalar_chain_defs=(structured or {}).get("_scalar_defs", ()),
        # an outer function that is a node itself has nodes but no scalars
        scalar_chain_active=bool(
            (structured or {}).get("_scalar_defs")
            or (structured or {}).get("_scalar_nodes")
        ),
    )


def _derivative(
    ctx: _EmitContext, structured: Structure | None, attr: str
) -> sp.Expr:
    """Return the derivative ``attr`` from the structure or the instance."""
    if structured is not None:
        return structured[attr]
    return getattr(ctx.diff, attr)


def _scalar_chain_outputs(state: _EmitState) -> None:
    """Add the scalar-chain pseudo-outputs and resolve its definitions.

    Parameters
    ----------
    state : _EmitState
        Updates ``outputs``, ``scalar_chain_defs`` and ``sc_temp_names``;
        draws one name per node and scalar from the namer.
    """
    structured = cast(Structure, state.structured)
    resolver = state.resolver
    namer = state.namer
    outputs = state.outputs
    scalar_chain_defs = state.scalar_chain_defs
    # the summation nodes read by the scalars of the outer function travel
    # with the outputs (one pseudo-output each, so the summation passes see
    # them as before); the scalars become definitions after the scalar CSE
    sc_nodes = structured.get("_scalar_nodes", ())
    sc_names = {
        symbol: namer("t")
        for symbol, _ in list(sc_nodes) + list(scalar_chain_defs)
    }
    outputs = [
        (attr, out_name, expr.xreplace(sc_names))
        for attr, out_name, expr in outputs
    ]
    state.sc_temp_names = {str(name) for name in sc_names.values()}
    for symbol, body in sc_nodes:
        outputs.append(
            (_SCALAR_CHAIN, str(sc_names[symbol]), resolver.resolve(body))
        )
    # the outputs without scalar_chain ride along (then dropped), so that
    # the summation passes see the same expressions as without the option
    # (first, so the shared summations come in the same order)
    plain = structured.get("_plain", {})
    outputs = [
        (_SCALAR_PLAIN, "_plain_" + attr, resolver.resolve(plain[attr]))
        for attr in ["value"] + list(state.selected)
        if attr in plain
    ] + outputs
    state.outputs = outputs
    state.scalar_chain_defs = [
        (sc_names[symbol], resolver.resolve(body.xreplace(sc_names)))
        for symbol, body in scalar_chain_defs
    ]


def _resolve_outputs(ctx: _EmitContext, state: _EmitState) -> None:
    """Resolve the value and the derivatives into ``outputs`` (S1-S2).

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Receives ``outputs`` and, with ``scalar_chain``, the scalar-chain
        pseudo-outputs and names.
    """
    structured = state.structured
    resolver = state.resolver
    value_expr = (
        structured["value"] if structured is not None else ctx.diff.expression
    )
    outputs = [("value", ctx.base, resolver.resolve(value_expr))]
    for attr in state.selected:
        outputs.append(
            (
                attr,
                "d" + ctx.base + _DERIV_SUFFIX[attr],
                resolver.resolve(_derivative(ctx, structured, attr)),
            )
        )
    state.outputs = outputs
    if state.scalar_chain_active:
        _scalar_chain_outputs(state)


def _stage_outputs(state: _EmitState) -> None:
    """Canonicalise every output expression into ``staged`` (S4)."""
    opts = state.opts
    symmetric_params = state.symmetric_params
    staged: Staged = []
    for attr, out_name, expr in state.outputs:
        expr = _kron_to_piecewise(expr)
        expr = expr.xreplace({nc: sp.Symbol("nc")})
        if opts.float_canon:
            expr = _canonical_floats(expr)
        if opts.param_symmetry and symmetric_params:
            expr = _canonical_symmetric(expr, symmetric_params, opts.sym_rank3)
        if opts.sep_sums is not SepSumsMode.OFF:
            expr = _separate_sums(
                expr, selective=opts.sep_sums is SepSumsMode.COST
            )
        staged.append((attr, out_name, expr))
    state.staged = staged


def _merge_dummies(state: _EmitState) -> None:
    """Merge the dummy images of symmetric parameters (S5)."""
    # arrays of rank >= 2 read inside sums with free indices by some output
    staged = state.staged
    shared_bases: set[str] = set()
    for _attr, _out, expr in staged:
        for total in expr.atoms(sp.Sum):
            if _free_indices(total):
                shared_bases |= {
                    str(x.base)
                    for x in total.function.atoms(sp.Indexed)
                    if len(x.indices) >= 2
                }
    state.staged = [
        (
            attr,
            out_name,
            _merge_dummy_images(
                expr,
                state.symmetric_params,
                frozenset(shared_bases),
                sym_rank3=state.opts.sym_rank3,
            ),
        )
        for attr, out_name, expr in staged
    ]


def _prepare_outputs(ctx: _EmitContext, state: _EmitState) -> None:
    """Fold symmetric sums and order the free indices (S7)."""
    opts = state.opts
    prepared: list[PreparedOutput] = []
    for attr, out_name, expr in state.staged:
        if opts.param_symmetry and state.symmetric_params:
            expr = _fold_symmetric_sums(
                expr, state.symmetric_params, sym_rank3=opts.sym_rank3
            )
        free_order = _order_free(_free_indices(expr), ctx.diff.indexes)
        prepared.append((attr, out_name, expr, free_order))
    state.prepared = prepared
    state.all_exprs = [expr for _, _, expr, _ in prepared]


def _canonicalise_outputs(ctx: _EmitContext, state: _EmitState) -> None:
    """Canonicalise, merge, nest and prepare the outputs (S3-S7).

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Reads ``outputs``; writes ``share_sums``, ``scalar_cse``,
        ``symmetric_params``, ``prepared`` and ``all_exprs``.
    """
    opts = state.opts
    state.share_sums = _option_or(opts.share_sums, opts.share)
    state.scalar_cse = _option_or(opts.scalar_cse, opts.share)
    if state.structured is not None:
        # symmetric pair functions behave as symmetric parameters
        state.symmetric_params = state.symmetric_params + tuple(
            name
            for name, _, rank, sym in state.structured.get(
                "_component_defs", ()
            )
            if rank == 2 and sym
        )
    _stage_outputs(state)
    if opts.dummy_canon:
        _merge_dummies(state)
    if opts.nest_sums:
        state.staged = _nest_double_sums(
            state.staged,
            ctx.diff.indexes,
            state.symmetric_params,
            shared_inner=opts.nest_shared,
            general=opts.nest_general,
            cost=opts.nest_cost,
            sym_rank3=opts.sym_rank3,
        )
    _prepare_outputs(ctx, state)


def _component_definitions(state: _EmitState) -> None:
    """Canonicalise the component bodies into ``component_defs`` (S8)."""
    opts = state.opts
    component_defs: list[ComponentDef] = []
    if state.structured is not None:
        for comp in state.structured.get("_component_defs", ()):
            comp_name, comp_body, comp_rank, comp_sym = comp
            comp_body = comp_body.xreplace({nc: sp.Symbol("nc")})
            if opts.param_symmetry and state.symmetric_params:
                comp_body = _canonical_symmetric(
                    comp_body, state.symmetric_params, opts.sym_rank3
                )
            component_defs.append((comp_name, comp_body, comp_rank, comp_sym))
    state.component_defs = component_defs


def _component_arrays(state: _EmitState) -> None:
    """Declare the component arrays and build their definitions (S12)."""
    if not state.component_defs:
        return
    from thermodiff.chain import NAME_SCHEME_KEY

    opts = state.opts
    scheme = cast(Structure, state.structured)[NAME_SCHEME_KEY]
    for comp_name, comp_body, comp_rank, comp_sym in state.component_defs:
        state.arrays[comp_name] = comp_rank
        free_ = (
            [scheme.component_slot]
            if comp_rank == 1
            else list(scheme.pair_slots)
        )
        base_ = sp.IndexedBase(comp_name, shape=(nc,) * comp_rank)
        state.component_array_defs.append(
            TempDef(
                target=sp.Indexed(base_, *free_),
                free=free_,
                limits=[],
                body=(
                    _canonical_floats(comp_body)
                    if opts.float_canon
                    else comp_body
                ),
                mirror=bool(comp_sym and opts.param_symmetry),
            )
        )


def _pair_components(ctx: _EmitContext, state: _EmitState) -> None:
    """Prepare the component definitions and their arrays (S8, S12).

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Writes ``component_defs``, ``arrays`` and ``component_array_defs``.
    """
    _component_definitions(state)
    _component_arrays(state)


def _index_and_base_names(
    scans: list[Any],
) -> tuple[set[str], set[str]]:
    """Return the names of the indices and of the arrays found in ``scans``."""
    index_syms = set()
    indexed_bases = set()
    for scan in scans:
        for indexed in scan.indexed:
            index_syms.update(indexed.indices)
        for total in scan.sums:
            for limit in total.limits:
                index_syms.add(limit[0])
        index_syms.update(scan.idxs)
        indexed_bases.update(scan.bases)
    return (
        {str(idx) for idx in index_syms},
        {str(base_) for base_ in indexed_bases},
    )


def _is_declared_elsewhere(sym: sp.Symbol, state: _EmitState) -> bool:
    """Tell whether ``sym`` is not an argument scalar of the routine."""
    name = str(sym)
    if sym in (T, V, P) or name == "nc":
        return True
    if isinstance(sym, _LoopStart):
        return True
    if name in state.module_symbols or name in state.integer_args:
        return True
    return name in state.sc_temp_names


def _parameter_scalars(state: _EmitState) -> set[str]:
    """Return the names of the scalar parameters found in ``scans``."""
    # NOTE: ``n`` is compared by name because ``expr`` has been rebuilt by
    # the ``nc`` replacement, which also rewrites the shape of the ``n``
    # IndexedBase.
    index_names, base_names = _index_and_base_names(state.scans)
    param_scalars = set()
    for scan in state.scans:
        for sym in scan.symbols:
            name = str(sym)
            if name in index_names or name in base_names:
                continue
            if _is_declared_elsewhere(sym, state):
                continue
            param_scalars.add(name)
    return param_scalars


def _parameter_arrays(state: _EmitState, uses_n: bool) -> dict[str, int]:
    """Return the rank of every array argument found in ``scans``."""
    n_name = str(n)
    placeholder_names = set(state.resolver.ranks)
    component_names = {c[0] for c in state.component_defs}
    array_ranks: dict[str, int] = {}
    for scan in state.scans:
        for indexed in scan.indexed:
            base_name = str(indexed.base)
            rank = len(indexed.indices)
            array_ranks[base_name] = max(array_ranks.get(base_name, 0), rank)
    param_arrays = {
        base_name: rank
        for base_name, rank in array_ranks.items()
        if base_name != n_name
        and base_name not in placeholder_names
        and base_name not in component_names
    }
    if uses_n:
        param_arrays[n_name] = 1
    return param_arrays


def _scan_signature(ctx: _EmitContext, state: _EmitState) -> None:
    """Gather the arguments of the routine from the PRE-CSE expressions (S9).

    Must run before the sharing passes: after CSE the temporaries would
    appear as parameters.

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Writes ``scans``, ``param_arrays``, ``params``,
        ``placeholder_args`` and ``needs_nc``.
    """
    scan_exprs = (
        state.all_exprs
        + [c[1] for c in state.component_defs]
        + [body for _, body in state.scalar_chain_defs]
    )
    placeholder_names = set(state.resolver.ranks)
    # ---- gather parameters --------------------------------------------
    scans = [_scan_atoms(expr) for expr in scan_exprs]
    state.scans = scans
    n_name = str(n)
    uses_n = any(
        n_name in {str(base_) for base_ in scan.bases} for scan in scans
    )
    param_scalars = _parameter_scalars(state)
    state.param_arrays = _parameter_arrays(state, uses_n)
    state.params = sorted(param_scalars - placeholder_names)
    state.placeholder_args = sorted(state.resolver.ranks)
    state.needs_nc = (
        uses_n
        or bool(state.param_arrays)
        or any(free for _, _, _, free in state.prepared)
        or any(sp.Symbol("nc") in scan.symbols for scan in scans)
    )
    # Backstop for names built by the generator (placeholder derivatives
    # such as ``dgdT``) that the user names did not predict.
    _case_clash_renames(
        {out_name for _, out_name, _, _ in state.prepared}
        | set(state.params)
        | set(state.param_arrays)
        | set(state.placeholder_args)
        | set(state.module_symbols)
        | set(state.integer_args),
        (),
    )


def _input_declarations(state: _EmitState) -> None:
    """Declare the dummy arguments in ``inputs`` (S10)."""
    inputs = []
    if state.needs_nc:
        inputs.append("integer, intent(in) :: nc")
    if state.integer_args:
        inputs.append(
            "integer, intent(in) :: " + ", ".join(state.integer_args)
        )
    for var, label in ((V, "V"), (P, "P"), (T, "T")):
        if any(var in scan.symbols for scan in state.scans):
            inputs.append(f"real(8), intent(in) :: {label}")
    for param in state.params:
        inputs.append(f"real(8), intent(in) :: {param}")
    in_array = (
        "real(8), contiguous, intent(in)"
        if state.opts.contiguous_args
        else "real(8), intent(in)"
    )
    for arr in sorted(state.param_arrays):
        dims = ", ".join([":"] * state.param_arrays[arr])
        inputs.append(f"{in_array} :: {arr}({dims})")
    for arg in state.placeholder_args:
        rank = state.resolver.ranks[arg]
        if rank == 0:
            inputs.append(f"real(8), intent(in) :: {arg}")
        else:
            dims = ", ".join([":"] * rank)
            inputs.append(f"{in_array} :: {arg}({dims})")
    state.inputs = inputs


def _declare_signature(ctx: _EmitContext, state: _EmitState) -> None:
    """Declare the input and output arguments (S10, S11).

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Writes ``inputs`` and ``output_decls``.
    """
    _input_declarations(state)
    # ---- output declarations ------------------------------------------
    state.output_decls = [
        _declare_output(
            out_name,
            len(free_order),
            optional=state.opts.optional_derivatives,
            contiguous=state.opts.contiguous_args,
        )
        for _attr, out_name, _, free_order in state.prepared
        if _attr not in (_SCALAR_CHAIN, _SCALAR_PLAIN)
    ]


def _array_cse_enabled(opts: CodegenOptions) -> bool:
    """Gate of the array CSE."""
    return opts.array_cse


def _param_symmetry_enabled(opts: CodegenOptions) -> bool:
    """Gate of the passes that use the symmetric parameters."""
    return opts.param_symmetry


def _fuse_temps_enabled(opts: CodegenOptions) -> bool:
    """Gate of the temporary fusion."""
    return opts.fuse_temps


def _hoist_invariants_enabled(opts: CodegenOptions) -> bool:
    """Gate of the invariant hoisting."""
    return opts.hoist_invariants


def _array_cse(ctx: _EmitContext, state: _EmitState) -> None:
    """Share the common subexpressions into arrays (S13).

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Rewrites ``all_exprs``; writes ``array_defs`` and adds the
        temporaries to ``arrays``.  Draws names from the namer.
    """
    state.all_exprs, state.array_defs = _indexed_cse(
        state.all_exprs,
        state.namer,
        state.arrays,
        expensive_twice=state.opts.cse_expensive,
        # with scalar_chain, only the plain outputs count (as without it)
        counted=(
            [entry[0] == _SCALAR_PLAIN for entry in state.prepared]
            if state.scalar_chain_active
            else None
        ),
    )


def _mirror_array_temps(ctx: _EmitContext, state: _EmitState) -> None:
    """Flag the symmetric array temporaries as mirrored (S14).

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Updates ``array_defs`` in place and, with ``nest_shared``,
        ``symmetric_params``.
    """
    opts = state.opts
    if not state.symmetric_params:
        return
    for definition in state.array_defs:
        free = definition.free
        if len(free) != 2 or definition.limits:
            continue
        body = _canonical_symmetric(
            definition.body, state.symmetric_params, opts.sym_rank3
        )
        swapped_equal = (
            _swap(
                body,
                free[0],
                free[1],
                state.symmetric_params,
                opts.sym_rank3,
            )
            == body
        )
        if swapped_equal:
            definition.mirror = True
            if opts.nest_shared:
                # a symmetric temporary behaves as a symmetric parameter
                # for the later passes (sum sharing modulo renaming)
                state.symmetric_params = state.symmetric_params + (
                    _temp_name(definition.target),
                )


def _reserved_names(ctx: _EmitContext, state: _EmitState) -> None:
    """Collect the names that temporaries must not take (S15).

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Reads the POST-CSE ``all_exprs``; writes ``reserved``.
    """
    reserved = (
        set(state.module_symbols)
        | set(state.integer_args)
        | {out_name for _, out_name, _, _ in state.prepared}
    )
    for expr in state.all_exprs:
        scan = _scan_atoms(expr)
        reserved.update(str(sym) for sym in scan.symbols)
        reserved.update(str(base_) for base_ in scan.bases)
    state.reserved = reserved


def _fuse_temps(ctx: _EmitContext, state: _EmitState) -> None:
    """Fuse the temporaries that are read once (S16).

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Rewrites ``all_exprs``, re-splits the two definition pools and
        pops the fused names from ``arrays``.
    """
    opts = state.opts
    pool = state.component_array_defs + state.array_defs
    pool, state.all_exprs, fused = _fuse_temporaries(
        pool,
        state.all_exprs,
        dedup=opts.nest_shared,
        guard=opts.fuse_guard,
        keep_libm=opts.keep_libm_tables,
    )
    state.component_array_defs = [
        d for d in pool if d in state.component_array_defs
    ]
    state.array_defs = [d for d in pool if d not in state.component_array_defs]
    for name in fused:
        state.arrays.pop(name, None)


def _hoist_array_temps(ctx: _EmitContext, state: _EmitState) -> None:
    """Hoist the loop invariants of the array temporaries (S17).

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Rewrites ``all_exprs``; appends the hoisted definitions to
        ``array_defs``.  Draws names from the namer.
    """
    state.all_exprs, hoist_defs, _ = _hoist_invariants(
        state.all_exprs,
        state.namer,
        state.arrays,
        reserved=state.reserved,
    )
    state.array_defs = state.array_defs + hoist_defs


def _share_sum_temps(ctx: _EmitContext, state: _EmitState) -> None:
    """Share the summations, or pass the expressions through (S18).

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Writes ``residuals`` and ``sum_defs``.  Draws names from the namer.
    """
    opts = state.opts
    if state.share_sums:
        state.residuals, state.sum_defs = _share_sums(
            state.all_exprs,
            state.namer,
            ctx.diff.indexes,
            state.arrays,
            alpha=opts.sum_alpha,
            free_alpha=opts.sum_free_alpha,
            symmetric_params=(
                state.symmetric_params if opts.param_symmetry else ()
            ),
            transpose=opts.sum_transpose,
            sym_rank3=opts.sym_rank3,
        )
    else:
        state.residuals = list(state.all_exprs)
        state.sum_defs = []


def _hoist_sum_temps(ctx: _EmitContext, state: _EmitState) -> None:
    """Hoist the loop invariants of the shared sums (S18b).

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Rewrites ``sum_defs``.  Draws names from the namer.
    """
    if not state.sum_defs:
        return
    _, _, state.sum_defs = _hoist_invariants(
        [], state.namer, state.arrays, state.sum_defs, reserved=state.reserved
    )


def _scalar_cse_enabled(opts: CodegenOptions) -> bool:
    """Gate of the scalar CSE: ``scalar_cse``, or ``share`` when unset."""
    return _option_or(opts.scalar_cse, opts.share)


def _scalar_cse_pass(ctx: _EmitContext, state: _EmitState) -> None:
    """Share the common scalar subexpressions of the residuals (S19).

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Rewrites ``residuals``; writes ``cse_defs``.  Draws names from the
        namer.
    """
    if state.opts.cse_backend is CseBackend.SYMPY:
        cse = _sympy_scalar_cse
    else:
        cse = _common_scalar_cse
    state.residuals, state.cse_defs = cse(
        state.residuals, state.namer, state.arrays
    )


def _scalar_chain_scalars(state: _EmitState) -> dict[sp.Symbol, sp.Symbol]:
    """Make the scalar definitions of the chain; return the alias map."""
    alias: dict[sp.Symbol, sp.Symbol] = {}
    for (attr, out_name, _, _), residual in zip(
        state.prepared, state.residuals
    ):
        if attr != _SCALAR_CHAIN:
            continue
        if isinstance(residual, sp.Symbol):
            alias[sp.Symbol(out_name)] = residual
            state.namer.created[:] = [
                x for x in state.namer.created if str(x) != out_name
            ]
        else:
            state.scalar_chain_definitions.append(
                TempDef(
                    target=sp.Symbol(out_name),
                    free=[],
                    limits=[],
                    body=residual,
                )
            )
    for target, body in state.scalar_chain_defs:
        body = (
            _kron_to_piecewise(body)
            .xreplace({nc: sp.Symbol("nc")})
            .xreplace(alias)
        )
        if state.opts.float_canon:
            body = _canonical_floats(body)
        state.scalar_chain_definitions.append(
            TempDef(target=target, free=[], limits=[], body=body)
        )
    return alias


def _drop_plain_only(state: _EmitState) -> None:
    """Drop the temporaries that only the plain outputs read."""
    namer = state.namer
    pools = [
        state.component_array_defs,
        state.array_defs,
        state.sum_defs,
        state.cse_defs,
    ]
    while True:
        read = set()
        for expr in (
            state.residuals
            + [d.body for pool in pools for d in pool]
            + [d.body for d in state.scalar_chain_definitions]
        ):
            read |= {str(x) for x in expr.free_symbols} | {
                str(x.base) for x in expr.atoms(sp.Indexed)
            }
        gone = {_temp_name(d.target) for pool in pools for d in pool} - read
        if not gone:
            break
        pools = [
            [d for d in pool if _temp_name(d.target) not in gone]
            for pool in pools
        ]
        namer.created[:] = [x for x in namer.created if str(x) not in gone]
        for name in gone:
            state.arrays.pop(name, None)
    (
        state.component_array_defs,
        state.array_defs,
        state.sum_defs,
        state.cse_defs,
    ) = pools


def _alias_scalar_chain(ctx: _EmitContext, state: _EmitState) -> None:
    """Turn the scalar-chain outputs into definitions and prune (S20a).

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Writes ``scalar_chain_definitions``; prunes ``prepared``,
        ``residuals``, the pools, ``arrays`` and ``namer.created``.
    """
    if not state.scalar_chain_active:
        return
    keep = [
        k
        for k, entry in enumerate(state.prepared)
        if entry[0] not in (_SCALAR_CHAIN, _SCALAR_PLAIN)
    ]
    alias = _scalar_chain_scalars(state)
    state.prepared = [state.prepared[k] for k in keep]
    state.residuals = [state.residuals[k] for k in keep]
    state.residuals = [
        residual.xreplace(alias) for residual in state.residuals
    ]
    # drop what only the plain outputs read
    _drop_plain_only(state)


def _rendered_outputs(ctx: _EmitContext, state: _EmitState) -> None:
    """Pair every residual with its output name and free indices (S20b).

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Writes ``rendered``.
    """
    state.rendered = [
        (
            attr,
            out_name,
            residual,
            _order_free(_free_indices(residual), ctx.diff.indexes),
        )
        for (attr, out_name, _, _), residual in zip(
            state.prepared, state.residuals
        )
    ]


def _drop_unread_scalars(state: _EmitState) -> None:
    """Drop the scalars that nothing reads, then order the definitions."""
    definitions = state.definitions
    while True:
        read = set()
        for expr in state.residuals + [
            d.body for d in definitions + state.scalar_chain_definitions
        ]:
            read |= {str(x) for x in expr.free_symbols}
        unread = [
            d
            for d in state.scalar_chain_definitions
            if str(d.target) not in read
        ]
        if not unread:
            break
        gone = {str(d.target) for d in unread}
        state.scalar_chain_definitions = [
            d
            for d in state.scalar_chain_definitions
            if str(d.target) not in gone
        ]
        state.namer.created[:] = [
            x for x in state.namer.created if str(x) not in gone
        ]
    state.definitions = _dependency_order(
        definitions + state.scalar_chain_definitions
    )


def _definitions_list(ctx: _EmitContext, state: _EmitState) -> None:
    """Join the pools into the ordered list of definitions (S21).

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Writes ``definitions``; with scalar-chain definitions, drops the
        unread ones and orders by dependency.
    """
    state.definitions = (
        state.component_array_defs
        + state.array_defs
        + state.sum_defs
        + state.cse_defs
    )
    if state.scalar_chain_definitions:
        # drop the scalars nothing reads
        _drop_unread_scalars(state)


def _gate(
    name: str,
    *,
    unless_optional: bool = False,
    with_param_symmetry: bool = False,
) -> Callable[[CodegenOptions], bool]:
    """Build the option-level gate of a stage.

    Parameters
    ----------
    name : str
        Name of the ``CodegenOptions`` flag the stage depends on.
    unless_optional : bool, optional
        Also require ``optional_derivatives`` to be off.
    with_param_symmetry : bool, optional
        Also require ``param_symmetry``.

    Returns
    -------
    callable
        ``enabled(opts)`` for a :class:`Pass`.

    Raises
    ------
    ValueError
        If ``name`` is not an option.
    """
    if name not in {
        field.name for field in dataclasses.fields(CodegenOptions)
    }:
        raise ValueError(f"unknown option {name!r}")

    def enabled(opts: CodegenOptions) -> bool:
        if unless_optional and opts.optional_derivatives:
            return False
        if with_param_symmetry and not opts.param_symmetry:
            return False
        return bool(getattr(opts, name))

    return enabled


def _lowrank_enabled(opts: CodegenOptions) -> bool:
    """Gate of the low-rank Hessian: on and without optional outputs."""
    return (
        opts.lowrank_hess is not LowrankHessMode.OFF
        and not opts.optional_derivatives
    )


def _rendered_residuals(state: _EmitState) -> list[sp.Expr]:
    """Return the residual of every rendered output."""
    return [residual for _, _, residual, _ in state.rendered]


def _replace_residuals(state: _EmitState, residuals: list[sp.Expr]) -> None:
    """Set ``residuals`` and update ``rendered`` in lock-step."""
    state.residuals = residuals
    state.rendered = [
        (attr, out_name, residual, free)
        for (attr, out_name, _, free), residual in zip(
            state.rendered, residuals
        )
    ]


def _lowrank_outputs(ctx: _EmitContext, state: _EmitState) -> None:
    """Write the bilinear outputs in low-rank form (S22).

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Appends to ``definitions``; updates ``rendered`` and ``residuals``
        in place; fills ``lowrank_outs``.  Draws names from the namer.
    """
    rendered = state.rendered
    for position, (attr, out_name, residual, free) in enumerate(rendered):
        if len(free) != 2 or residual.has(sp.Sum):
            continue
        result = _lowrank_bilinear(residual, free, state.namer, state.arrays)
        if result is None:
            continue
        residual, new_defs = result
        state.definitions = state.definitions + new_defs
        rendered[position] = (attr, out_name, residual, free)
        state.residuals[position] = residual
        state.lowrank_outs.add(attr)


def _hoist_loop_pass(ctx: _EmitContext, state: _EmitState) -> None:
    """Hoist the loop invariants of definitions and outputs (S23).

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Rewrites ``definitions``, ``rendered`` and ``residuals``.  Draws
        names from the namer.
    """
    state.definitions, new_residuals = _hoist_loop_invariants(
        state.definitions,
        [(residual, free) for _, _, residual, free in state.rendered],
        state.namer,
        state.arrays,
        reserved=state.reserved,
    )
    _replace_residuals(state, new_residuals)


def _drop_pair_tables_pass(ctx: _EmitContext, state: _EmitState) -> None:
    """Drop the pair tables that can be recomputed (S24).

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Rewrites ``definitions``; pops dropped names from ``arrays``.
        Draws names from the namer.
    """
    state.definitions = _drop_pair_tables(
        state.definitions,
        _rendered_residuals(state),
        state.namer,
        state.arrays,
    )


def _share_elementwise_pass(ctx: _EmitContext, state: _EmitState) -> None:
    """Share the elementwise definitions (S25).

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Rewrites ``definitions``, ``residuals`` and ``rendered``.  Draws
        names from the namer.
    """
    state.definitions, residuals = _share_elementwise(
        state.definitions,
        _rendered_residuals(state),
        state.namer,
        state.arrays,
    )
    _replace_residuals(state, residuals)


def _split_affine_pass(ctx: _EmitContext, state: _EmitState) -> None:
    """Split the affine sums into table and scalar parts (S26).

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Rewrites ``definitions``, ``residuals`` and ``rendered``.  Draws
        names from the namer.
    """
    state.definitions, residuals = _split_affine(
        state.definitions,
        _rendered_residuals(state),
        state.namer,
        state.arrays,
    )
    _replace_residuals(state, residuals)


def _mirror_sums(ctx: _EmitContext, state: _EmitState) -> None:
    """Flag the symmetric two-index sums as mirrored (S27).

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Sets ``mirror_sum`` on the definitions in place.
    """
    symmetric_params = state.symmetric_params
    sym_rank3 = state.opts.sym_rank3
    for definition in state.definitions:
        free_ = definition.free
        if len(free_) != 2 or not definition.limits:
            continue
        body_ = _canonical_symmetric(
            definition.body, symmetric_params, sym_rank3
        )
        if body_.has(sp.Piecewise) or any(
            isinstance(x, _LoopStart) for x in body_.atoms(sp.Symbol)
        ):
            continue
        swapped_equal = (
            _swap(body_, free_[0], free_[1], symmetric_params, sym_rank3)
            == body_
        )
        if swapped_equal:
            definition.mirror_sum = True


def _column_layouts(ctx: _EmitContext, state: _EmitState) -> None:
    """Choose column sums and column-major order per definition (S28).

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Sets ``colsum`` and ``colmajor`` on the definitions in place.
    """
    for definition in state.definitions:
        if state.opts.table_layout and _prefers_column_sum(
            definition, state.symmetric_params
        ):
            definition.colsum = True
            continue
        if definition.mirror_sum:
            continue
        if not definition.symv and _prefers_column_order(definition):
            definition.colmajor = True


def _fuse_loop_nests_pass(ctx: _EmitContext, state: _EmitState) -> None:
    """Fuse the loop nests with the same ranges (S29).

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Rewrites ``definitions``.
    """
    state.definitions = _fuse_loop_nests(state.definitions, state.namer)


def _symv_parts_pass(ctx: _EmitContext, state: _EmitState) -> None:
    """Split the symmetric matrix-vector definitions (S30).

    Runs after the nest fusion, which must not see ``symv``.

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Sets ``symv`` and ``symv_acc`` in place.  Draws one name per
        definition with parts from the namer.
    """
    for definition in state.definitions:
        parts = _symv_parts(
            definition,
            state.symmetric_params,
            sym_rank3=state.opts.sym_rank3,
        )
        if parts is not None:
            definition.symv = parts
            definition.symv_acc = state.namer("t")


def _fuse_siblings_pass(ctx: _EmitContext, state: _EmitState) -> None:
    """Fuse the sibling tables (S31).

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Rewrites ``definitions``.  Draws names from the namer.
    """
    state.definitions = _fuse_sibling_tables(state.definitions, state.namer)


def _diagonal_pass(ctx: _EmitContext, state: _EmitState) -> None:
    """Read the diagonals from the tables (S32).

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Rewrites ``definitions``.
    """
    state.definitions = _diagonal_from_tables(
        state.definitions,
        state.symmetric_params,
        sym_rank3=state.opts.sym_rank3,
    )


def _permute_subscripts_pass(ctx: _EmitContext, state: _EmitState) -> None:
    """Permute the symmetric subscripts of the fused members (S33).

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Rewrites ``definitions``.
    """
    state.definitions = _permute_symmetric_subscripts(
        state.definitions, state.symmetric_params
    )


def _component_maps_pass(ctx: _EmitContext, state: _EmitState) -> None:
    """Fuse the component maps (S34).

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Rewrites ``definitions``.  Draws names from the namer lazily.
    """
    state.definitions = _fuse_component_maps(state.definitions, state.namer)


def _outer_loops_pass(ctx: _EmitContext, state: _EmitState) -> None:
    """Fuse the outer loops (S35).

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Rewrites ``definitions``.
    """
    opts = state.opts
    state.definitions = _fuse_outer_loops(
        state.definitions,
        libm_guard=opts.fuse_libm_guard,
        libm_reductions=opts.fuse_libm_guard and opts.drop_pair_tables,
    )


def _split_accumulators_pass(ctx: _EmitContext, state: _EmitState) -> None:
    """Split the reductions into two accumulators (S36).

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Rewrites ``definitions``.  Draws names from the namer.

    Raises
    ------
    ValueError
        If ``split_acc`` is not 2.
    """
    if state.opts.split_acc != _REQUIRED_ACCUMULATORS:
        raise ValueError("split_acc supports 2 accumulators only")
    state.definitions = _split_accumulators(state.definitions, state.namer)


def _scalar_operation_count(state: _EmitState) -> int:
    """Count the operations of the scalar block (no loops)."""
    return sum(  # type: ignore[no-any-return]  # sympy is Any
        sp.count_ops(residual, visual=False)
        for (_a, _o, _r, free_), residual in zip(
            state.rendered, state.residuals
        )
        if not free_
    ) + sum(
        sp.count_ops(definition.body, visual=False)
        for definition in state.definitions
        if not definition.free and not definition.limits
    )


def _collect_stats(ctx: _EmitContext, state: _EmitState) -> None:
    """Fill ``stats`` with the operation counts (S37).

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Updates ``state.stats`` when it is not ``None``.
    """
    stats = state.stats
    if stats is None:
        return
    stats["operation_count"] = sum(
        sp.count_ops(residual, visual=False) for residual in state.residuals
    ) + sum(
        sp.count_ops(definition.body, visual=False)
        for definition in state.definitions
    )
    stats["operation_count_tool"] = "sympy.count_ops(visual=False)"
    stats["n_temporaries"] = len(state.definitions)
    if state.count_scalar:
        stats["scalar_operation_count"] = _scalar_operation_count(state)
        if state.opts.scalar_chain is ScalarChainMode.ON:
            stats[_STRUCTURE_STATS_KEY] = state.structured


def _consumer_graph(state: _EmitState) -> None:
    """Find the outputs that read each temporary (S38)."""
    definitions = state.definitions
    state.out_names = {
        attr: out_name for attr, out_name, _, _ in state.rendered
    }
    temp_names = {_temp_name(definition.target) for definition in definitions}
    consumers: dict[str, set[str]] = {name: set() for name in temp_names}
    for attr, _out_name, residual, _free in state.rendered:
        for name in _referenced_temp_names(residual, temp_names):
            consumers[name].add(attr)
    for definition in reversed(definitions):
        name = _temp_name(definition.target)
        for dependency in _referenced_temp_names(definition.body, temp_names):
            consumers[dependency] |= consumers[name]
    state.consumers = consumers


def _render_outputs(state: _EmitState) -> list[str]:
    """Render the output assignments, in output order."""
    opts = state.opts
    lines = []
    for attr, out_name, expr, free_order in state.rendered:
        if opts.optional_derivatives:
            condition = f"present({out_name})"
        else:
            condition = ""
        mirror = (
            opts.hessian_symmetry
            and attr == "dnidnj"
            and (
                attr not in state.lowrank_outs
                or opts.lowrank_hess is LowrankHessMode.SYM
            )
        )
        column = (
            attr in state.lowrank_outs
            and opts.lowrank_hess is not LowrankHessMode.SYM
        )
        lines.extend(
            _guarded(
                _render_output(
                    out_name,
                    expr,
                    free_order,
                    state.namer,
                    mirror=mirror,
                    column=column,
                ),
                condition,
            )
        )
        lines.append("")
    return lines


def _render_bodies(ctx: _EmitContext, state: _EmitState) -> None:
    """Render the definitions and the outputs (S38, S39).

    The last stage that draws names from the namer (render-time
    temporaries of the expanded sums).

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Writes ``out_names``, ``consumers`` and ``body``.
    """
    _consumer_graph(state)
    body = []
    for definition in state.definitions:
        if state.opts.optional_derivatives:
            condition = _guard(
                state.consumers[_temp_name(definition.target)],
                state.out_names,
            )
        else:
            condition = ""
        body.extend(_guarded(_render_definition(definition), condition))
        body.append("")
    body.extend(_render_outputs(state))
    state.body = body


def _temp_declarations(ctx: _EmitContext, state: _EmitState) -> None:
    """Collect the scalar temporaries and the loop indices (S40).

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Reads ``namer.created``; writes ``temps`` and ``loop_indices``.
    """
    rendered = state.rendered
    definitions = state.definitions
    state.temps = sorted(
        {str(t) for t in state.namer.created if str(t) not in state.arrays}
        | {"res"}
    )
    state.loop_indices = sorted(
        {str(idx) for _, _, _, free in rendered for idx in free}
        | {str(idx) for definition in definitions for idx in definition.free}
        | {
            str(limit[0])
            for definition in definitions
            for limit in definition.limits
        }
        | {
            str(limit[0])
            for _attr, _out, residual, _free in rendered
            for total in residual.atoms(sp.Sum)
            for limit in total.limits
        }
    )


def _argument_names(ctx: _EmitContext, state: _EmitState) -> list[str]:
    """Return the dummy argument names in declaration order."""
    arg_names: list[str] = []
    if state.needs_nc:
        arg_names.append("nc")
    arg_names.extend(state.integer_args)
    for var, label in ((V, "V"), (P, "P"), (T, "T")):
        if any(var in scan.symbols for scan in state.scans):
            arg_names.append(label)
    arg_names.extend(state.params)
    arg_names.extend(sorted(state.param_arrays))
    arg_names.extend(state.placeholder_args)
    arg_names.extend(out_name for _, out_name, _, _ in state.prepared)
    return arg_names


def _array_declarations(state: _EmitState) -> list[str]:
    """Return the declaration lines of the temporary arrays, by rank."""
    arrays = state.arrays
    array_pad = state.opts.array_pad
    lines = []
    for rank in sorted(set(arrays.values())):
        dims = ", ".join(["nc"] * rank)
        if array_pad and rank >= 2:
            # padded leading dimension: avoids cache-set aliasing between the
            # many (nc, nc) temporaries for some values of nc
            dims = ", ".join([f"nc + {int(array_pad)}"] + ["nc"] * (rank - 1))
        names = [
            f"{name}({dims})"
            for name in sorted(arrays)
            if arrays[name] == rank
        ]
        for line in _wrap_name_list("real(8) :: ", names):
            lines.append(_INDENT + line)
    return lines


def _assemble_text(ctx: _EmitContext, state: _EmitState) -> None:
    """Assemble the Fortran source of the subroutine (S41).

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Reads everything the earlier passes produced; writes ``text``.
    """
    arg_names = _argument_names(ctx, state)
    sub_name = ctx.sub_name
    while sub_name in arg_names:
        sub_name += "_s"

    lines = _wrap_signature(sub_name, arg_names)
    lines.append(f"{_INDENT}implicit none")
    for decl in state.inputs:
        lines.append(_INDENT + decl)
    for decl in state.output_decls:
        lines.append(_INDENT + decl)

    for line in _wrap_name_list("real(8) :: ", state.temps):
        lines.append(_INDENT + line)
    lines.extend(_array_declarations(state))
    for line in _wrap_name_list("integer :: ", state.loop_indices):
        lines.append(_INDENT + line)

    lines.append("")
    for line in state.body:
        lines.append(_INDENT + line if line else "")

    lines.append(f"end subroutine {sub_name}")
    lines.append("")

    state.text = "\n".join(lines)


_PASSES: tuple[Pass, ...] = (
    Pass("resolve_outputs", _always, _resolve_outputs),
    Pass("canonicalise_outputs", _always, _canonicalise_outputs),
    Pass("pair_components", _always, _pair_components),
    Pass("scan_signature", _always, _scan_signature),
    Pass("declare_signature", _always, _declare_signature),
    Pass("array_cse", _array_cse_enabled, _array_cse),
    Pass("mirror_array_temps", _param_symmetry_enabled, _mirror_array_temps),
    Pass("reserved_names", _always, _reserved_names),
    Pass("fuse_temps", _fuse_temps_enabled, _fuse_temps),
    Pass("hoist_array_temps", _hoist_invariants_enabled, _hoist_array_temps),
    Pass("share_sum_temps", _always, _share_sum_temps),
    Pass("hoist_sum_temps", _hoist_invariants_enabled, _hoist_sum_temps),
    Pass("scalar_cse", _scalar_cse_enabled, _scalar_cse_pass),
    Pass("alias_scalar_chain", _always, _alias_scalar_chain),
    Pass("rendered_outputs", _always, _rendered_outputs),
    Pass("definitions_list", _always, _definitions_list),
    Pass("lowrank_outputs", _lowrank_enabled, _lowrank_outputs),
    Pass("hoist_loops", _gate("hoist_loops"), _hoist_loop_pass),
    Pass(
        "drop_pair_tables",
        _gate("drop_pair_tables", unless_optional=True),
        _drop_pair_tables_pass,
    ),
    Pass(
        "share_elementwise",
        _gate("share_elementwise", unless_optional=True),
        _share_elementwise_pass,
    ),
    Pass(
        "split_affine",
        _gate("split_affine", unless_optional=True),
        _split_affine_pass,
    ),
    Pass(
        "mirror_sums",
        _gate("sym_sum2", with_param_symmetry=True),
        _mirror_sums,
    ),
    Pass("column_layouts", _gate("col_major"), _column_layouts),
    Pass(
        "fuse_loop_nests",
        _gate("fuse_loops", unless_optional=True),
        _fuse_loop_nests_pass,
    ),
    Pass(
        "symv_parts",
        _gate("sym_matvec", with_param_symmetry=True),
        _symv_parts_pass,
    ),
    Pass(
        "fuse_siblings",
        _gate("sib_fuse", unless_optional=True),
        _fuse_siblings_pass,
    ),
    Pass(
        "diagonal_tables",
        _gate("diag_table", unless_optional=True),
        _diagonal_pass,
    ),
    Pass(
        "permute_subscripts",
        _gate("sym_perm", with_param_symmetry=True),
        _permute_subscripts_pass,
    ),
    Pass(
        "component_maps",
        _gate("comp_fuse", unless_optional=True),
        _component_maps_pass,
    ),
    Pass(
        "outer_loops",
        _gate("fuse_outer", unless_optional=True),
        _outer_loops_pass,
    ),
    Pass(
        "split_accumulators",
        _gate("split_acc", unless_optional=True),
        _split_accumulators_pass,
    ),
    Pass("collect_stats", _always, _collect_stats),
    Pass("render_bodies", _always, _render_bodies),
    Pass("temp_declarations", _always, _temp_declarations),
    Pass("assemble_text", _always, _assemble_text),
)


def _emit_code(
    ctx: _EmitContext,
    structured: Structure | None,
    stats: Stats | None,
    count_scalar: bool,
    *,
    strict: bool = False,
    optional_derivatives: bool = False,
    array_cse: bool = True,
    share: bool = True,
    share_sums: bool | None = None,
    scalar_cse: bool | None = None,
    param_symmetry: bool = False,
    hessian_symmetry: bool = False,
    cse_backend: CseBackend | str = "thermodiff",
    sum_alpha: bool = False,
    array_pad: int = 0,
    hoist_invariants: bool = False,
    chain_rule: bool = False,
    hoist_loops: bool = False,
    sum_free_alpha: bool = False,
    fuse_temps: bool = False,
    nest_sums: bool = False,
    split_sqrt: bool = False,
    split_sums: bool = False,
    nest_shared: bool = False,
    sym_matvec: bool = False,
    col_major: bool = False,
    pair_cse: bool = False,
    pair_rates: bool = False,
    sum_transpose: bool = False,
    sep_sums: SepSumsMode | bool | str = False,
    sym_rank3: bool = False,
    nest_general: bool = False,
    sym_sum2: bool = False,
    cse_expensive: bool = False,
    fuse_loops: bool = False,
    nested_nodes: bool = False,
    sym_perm: bool = False,
    fuse_guard: bool = False,
    nest_cost: bool = False,
    comp_fuse: bool = False,
    fuse_outer: bool = False,
    fuse_libm_guard: bool = False,
    keep_libm_tables: bool = False,
    diag_table: bool = False,
    sib_fuse: bool = False,
    dummy_canon: bool = False,
    float_canon: bool = False,
    contiguous_args: bool = False,
    table_layout: bool = False,
    lowrank_hess: LowrankHessMode | bool | str = False,
    drop_pair_tables: bool = False,
    share_elementwise: bool = False,
    scalar_chain: ScalarChainMode | bool | str = False,
    split_affine: bool = False,
    split_acc: int | bool = False,
) -> str:
    """Emit the subroutine from a precomputed chain-rule structure.

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    structured : dict or None
        Result of the chain-rule step (``None`` without ``chain_rule``).
    stats : dict or None
        Receives the statistics.
    count_scalar : bool
        Whether to report ``scalar_operation_count``.
    **options
        The behaviour options with the legacy ``to_fortran`` meaning
        (``CodegenOptions.to_legacy``); ``scalar_chain`` is ``True`` or
        ``False`` here.

    Returns
    -------
    str
        The Fortran source.
    """
    opts = _legacy_options(locals())
    state = _new_state(ctx, structured, stats, count_scalar, opts)
    for stage in _PASSES:
        if stage.enabled(opts):
            stage.run(ctx, state)
    return state.text


def to_fortran(
    diff: Any,
    derivatives: Any = None,
    name: str | None = None,
    subroutine_name: str | None = None,
    *,
    integer_args: Iterable[str] | None = None,
    module_symbols: Iterable[str] | None = None,
    symmetric_params: Iterable[str] | None = (),
    stats: Stats | None = None,
    options: CodegenOptions | None = None,
    **legacy: Any,
) -> str:
    """Generate a Fortran subroutine for a :class:`DiffPlz` instance.

    The behaviour options can be given as a :class:`CodegenOptions`
    (``options``), as the legacy keywords documented below, or both.  The
    resolved options are validated on every call: options that the
    generator would silently ignore for the chosen combination are
    disabled and reported in one ``UserWarning`` (see
    :meth:`CodegenOptions.validate`).  For ready-made option sets use
    ``CodegenOptions.from_preset("standard")``.

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
    array_cse : bool, optional
        When ``True`` (the default), index-dependent sub-expressions that
        several outputs share are materialised into array temporaries
        computed a single time (see :func:`_indexed_cse`).  Only
        sub-expressions of rank up to two are materialised, because a
        larger temporary costs more than it saves.  Disable the pass to
        obtain the plain, fully inlined expressions.
    share : bool, optional
        When ``True`` (the default), summations and common scalar
        sub-expressions shared by several outputs are computed a single
        time as internal temporaries (:func:`_share_sums` and
        :func:`_common_scalar_cse`).  Set it to ``False`` to disable both
        sharing passes: every output is then emitted from its own fully
        inlined expressions and repeated summations are re-evaluated per
        output.  The benchmark variant "V1 none" is
        ``array_cse=False, share=False``; "V2 partial" is
        ``array_cse=False``; "V3 full" is ``array_cse=True``.  The flag
        only changes the generated body: the signature, the declaration
        block and the output order are identical for every setting.
    stats : dict, optional
        When given, it is filled in place with generation statistics:
        ``operation_count`` (``sympy.count_ops`` summed over the final
        output expressions and over the body of every shared temporary),
        ``operation_count_tool`` (the exact tool/options used) and
        ``n_temporaries`` (number of shared temporaries emitted).  The
        count is a symbolic-operation count of the final expressions, not
        a runtime estimate.
    share_sums, scalar_cse : bool, optional
        Fine-grained control of the two passes enabled by ``share``:
        :func:`_share_sums` and the scalar common sub-expression pass.
        ``None`` (the default) inherits the value of ``share``.  Used by the
        ablation study.
    options : CodegenOptions, optional
        Behaviour options.  Defaults to ``CodegenOptions()``, that is the
        defaults of the behaviour keywords.  A keyword passed explicitly in
        the same call overrides the matching field of ``options``.
    **legacy
        Any behaviour option by name (``strict``, ``optional_derivatives``,
        ``chain_rule``, ...; see the entries of this section and the fields
        of :class:`CodegenOptions`).  An unknown name raises ``TypeError``.
    symmetric_params : iterable of str, optional
        Names of rank-2 parameters known to be symmetric (declared with the
        ``symmetric`` attribute in a ``.model`` file).
    param_symmetry : bool, optional
        Exploit ``symmetric_params``: canonicalise their subscripts and
        fold symmetric double summations into a triangular loop (see
        :func:`_fold_symmetric_sums`).  Off by default.
    hessian_symmetry : bool, optional
        Compute only the upper triangle of the composition Hessian
        ``dfdndn`` (``j >= i``) and mirror it.  Exact by Schwarz' theorem.
        Off by default.
    cse_backend : {"thermodiff", "sympy"}, optional
        Algorithm of the scalar common sub-expression pass.
    sum_alpha : bool, optional
        Share summations that are equal up to the names of their bound
        indices (alpha equivalence), not only syntactically identical ones.
        Off by default.
    array_pad : int, optional
        Extra rows added to the leading dimension of rank-2 array
        temporaries (``t(nc + pad, nc)``).  Zero (the default) keeps
        ``t(nc, nc)``.
    hoist_invariants : bool, optional
        Move costly sub-expressions that do not depend on every bound index
        of their summation out of the loop, into temporaries of lower rank
        (see :func:`_hoist_invariants`).  Off by default.
    chain_rule : bool, optional
        Differentiate by the chain rule over the summation nodes of the
        function, as a hand-written implementation does (see
        :mod:`thermodiff.chain`).  Falls back to the plain derivatives when
        the function does not fit the pattern.  Off by default.
    hoist_loops : bool, optional
        Group the terms of every loop body over free indices by their
        index-dependent factors and compute the loop invariants once (see
        :func:`_hoist_loop_invariants`).  Off by default.
    sum_free_alpha : bool, optional
        Share summations equal up to the names of their bound *and free*
        indices (``sum_k x(i,k)`` and ``sum_l x(j,l)``).  Off by default.
    fuse_temps : bool, optional
        Do not materialise cheap rank-2 array temporaries that are only
        read inside summations: their body is inserted into the summation
        (see :func:`_fuse_temporaries`).  Off by default.
    nest_sums : bool, optional
        Rewrite double sums as dot products with vector sums that other
        outputs already need (see :func:`_nest_double_sums`).  Off by
        default.
    split_sqrt : bool, optional
        With ``chain_rule``: split ``sqrt(c*A_a*A_b)`` into
        ``sqrt(c)*sqrt(A_a)*sqrt(A_b)`` for per-component temperature
        functions.  Assumes they are non-negative (true for the attractive
        parameters of cubic equations).  Off by default.
    split_sums : bool, optional
        With ``chain_rule``: expand summation bodies and split sums of terms
        into sums, so that each term can become a dot product with a vector
        sum.  Off by default.
    nest_shared : bool, optional
        With ``nest_sums``: also nest double sums whose inner sum is shared
        by several double sums (not only by an output), and count equal
        summations once when deciding whether to fuse a temporary.  Off by
        default.
    sym_matvec : bool, optional
        With ``param_symmetry``: emit the vector sums ``v(f) = sum_b
        P(b,f) R(b)`` with a symmetric ``P`` as a symmetric matrix-vector
        product that reads only the upper triangle.  Off by default.
    col_major : bool, optional
        Order the loops of vector sums and rank-2 tables so that the arrays
        they read are traversed along columns (see
        :func:`_prefers_column_order`).  Off by default.
    pair_cse, pair_rates : bool, optional
        With ``chain_rule``: write pair functions in terms of earlier ones
        (``tau*exp(-c tau)`` as ``tau*M``), and write the temperature
        derivatives of pair functions with a cheap pair-dependent
        logarithmic rate in place (``M' = M*r_ab``).  Off by default.
    sum_transpose : bool, optional
        With ``sum_free_alpha``: reuse a rank-2 summation that is the
        transpose of another one.  Off by default.
    sep_sums : bool, optional
        Distribute products over sums in summation bodies, split the sums
        and move out the factors that do not depend on the bound index (see
        :func:`_separate_sums`).  Off by default.
    sym_rank3 : bool, optional
        With ``param_symmetry``: also canonicalise symmetric parameters of
        rank three or more (fully symmetric tensors).  Off by default.
    nest_general : bool, optional
        With ``nest_sums``: also nest sums over three or more indices, and
        use rank-2 summations as inner sums.  Off by default.
    sym_sum2 : bool, optional
        With ``param_symmetry``: a rank-2 summation whose body is symmetric
        in its two free indices is computed on the upper triangle and
        mirrored.  Off by default.
    cse_expensive : bool, optional
        The indexed CSE extracts a costly single-operation sub-expression
        (``log``, ``exp``, a root) already when it is used twice.  Off by
        default.
    fuse_loops : bool, optional
        Merge consecutive vector sums with the same loop structure into one
        loop nest (see :func:`_fuse_loop_nests`); only without optional
        outputs.  Off by default.
    nested_nodes : bool, optional
        With ``chain_rule``: index-free summations nested inside a node
        become nodes too, and derivatives are composed through them (see
        :func:`thermodiff.chain._nest_inner_nodes`).  Off by default.
    sym_perm : bool, optional
        Rewrite the subscripts of ``symmetric`` parameters inside loop nests
        so that the innermost loop index comes first (contiguous access).
        Exact by symmetry; applied after every other pass.  Off by default.
    fuse_guard : bool, optional
        With ``fuse_temps``: never inline an expensive rank-2 temporary
        (transcendental body) that two or more different summations read,
        nested ones included.  Off by default.
    nest_cost : bool, optional
        With ``nest_sums`` and ``nest_general``: nest every sum over three or
        more indices into an inner rank-2 table, even when no other output
        shares it.  Off by default.
    comp_fuse : bool, optional
        Fuse consecutive per-component maps (``cmp0``, ``cmp0_T``, ...) into
        one loop with a per-element common sub-expression elimination.
        Only without ``optional_derivatives``.  Off by default.
    dummy_canon : bool, optional
        Merge the terms of a summation over two or three indices that are
        images of each other under a renaming of the bound indices (e.g.
        ``(b_k + b_l + b_m)`` with a symmetric tensor).  Exact.  Off by
        default.
    float_canon : bool, optional
        Round every floating-point constant to its double value with a single
        representation before the common sub-expression passes.  Off by
        default.
    contiguous_args : bool, optional
        Declare every assumed-shape array argument (parameters, mole numbers
        and outputs) ``contiguous``, so the compiler can drop the stride
        arithmetic and the multi-versioned loops.  A caller that passes a
        non-contiguous section gets a copy-in/copy-out (still correct).
        Off by default.
    fuse_outer : bool, optional
        Fuse consecutive per-component maps and scalar reductions that share
        the outer loop ``1..nc`` into one loop.  Only without
        ``optional_derivatives``.  Off by default.
    fuse_libm_guard : bool, optional
        With ``fuse_outer``, never fuse a double reduction into the outer loop
        of a map that calls ``exp``, ``log``, ``sqrt`` or a real power, so the
        map stays vectorisable.  Off by default.
    keep_libm_tables : bool, optional
        With ``fuse_temps``, never inline a rank-2 temporary that calls a libm
        routine into a summation; it stays a table computed by its own map
        loop.  Off by default.
    table_layout : bool, optional
        With ``chain_rule``, store a non-symmetric pair function in the
        subscript order of the parameters it reads (``G(l, k)`` for
        ``exp(-c_lk tau_lk)``) instead of the alphabetical order of the
        summation indices, so the table is written contiguously and the sums
        that read it take the contiguous inner index; with ``col_major`` a
        rank-2 sum that then reads its tables with the bound index last
        (``sum_k G(i, k) H(j, k)``) is emitted in ``jki`` order.  Off by
        default.
    lowrank_hess : bool or "sym", optional
        Assemble a rank-2 output (the composition Hessian) whose residual has
        separable terms ``c F(i) G(j)`` in rank-k form: the terms are grouped
        by the factor over one index, ``sum_l r_l(i) G_l(j)``, with the
        coefficient vectors ``r_l`` computed once (O(nc)); terms over one
        index and pair terms are kept.  Applied only when it lowers the
        per-element operation count.  ``True`` emits the full square in
        column order without mirror; ``"sym"`` keeps the triangle and mirror
        of ``hessian_symmetry``.  Only without ``optional_derivatives``.
        Off by default.
    drop_pair_tables : bool, optional
        Do not materialise a rank-2 table read only by scalar reductions (one
        double, triangular reduction and its diagonal sums) whose body calls
        no libm routine: its body is inlined there.  In the scalar double
        reductions the factors over the outer index are applied once per
        outer iteration and the products over the inner index only become
        vectors.  With ``fuse_libm_guard`` a single reduction that calls a
        libm routine does not share the outer loop of a double reduction
        either.  Only without ``optional_derivatives``.  Off by default.
    share_elementwise : bool, optional
        A costly rank-1 sub-expression (``log``, ``exp``, a root of a
        per-component expression) read in two or more places -- the body of a
        scalar reduction over ``k`` and an output over ``i`` -- is computed
        once as a vector that every place reads.  Occurrences are compared up
        to the index name and to the spelling of their scalar prefactors
        (``x/y`` and ``(1/y)*x``); differently spelt scalar temporaries of the
        shared occurrences are merged.  Only without
        ``optional_derivatives``.  Off by default.
    scalar_chain : bool, optional
        With ``chain_rule``, compute the partial derivatives of the outer
        function ``F(S_1, ..., S_m, V, T)`` by forward second-order
        propagation over its expression DAG, followed by one common
        sub-expression elimination, instead of differentiating the expanded
        ``F`` (see :func:`thermodiff.chain._scalar_chain`).  They become
        scalar temporaries computed after the reductions they read, and the
        outputs read them.  Falls back to the plain path when ``F`` is not
        smooth (``Piecewise``, ``Max`` ...).  ``"cost"`` generates the kernel
        both ways and keeps the one with the lower ``operation_count``
        (``stats["scalar_chain_used"]`` tells which).  With ``"estimate"``
        the choice is made before the array pipeline, from the operation
        counts of the plain partials of ``F`` (after one CSE) and of the
        propagated definitions, so the pipeline runs once.  Only without
        ``optional_derivatives``.  Off by default.
    split_affine : bool, optional
        In a summation with free indices, ``sum_m (c - P(m, k, l)) w(m)``
        with ``c`` and ``w`` independent of the free indices becomes
        ``c*S - sum_m P w``: the scalar ``S = sum_m w(m)`` is computed once
        and the readers of the table read ``S - t``, so the innermost body is
        one multiply-add; products of factors over the bound index only
        become vectors (see :func:`_split_affine`).  Applied only when the
        body gets cheaper; only without ``optional_derivatives``.  Off by
        default.
    split_acc : int or bool, optional
        ``2``: in the inner loop of a symmetric rank-2 table filled by a sum
        that reads a parameter of rank three (the O(N^3) nest of CMR), use two
        scalar partial accumulators per sum (even and odd ``m``, one remainder
        iteration, then their sum), so that gcc vectorises the reduction
        without ``-ffast-math`` (reportes/brechas/ronda3-cmr-o3.md).  Changes
        the summation order.  Only without ``optional_derivatives``.  Off by
        default.

    Returns
    -------
    str
        The free-form Fortran source code.

    Warns
    -----
    UserWarning
        Once per call, when validation disables ignored options.

    Raises
    ------
    TypeError
        For an unknown keyword.
    ValueError
        For an invalid option value (see
        :meth:`CodegenOptions.validate`).
    """
    base = CodegenOptions() if options is None else options
    resolved = base.replace(**legacy) if legacy else base
    normalised, _ignored = resolved.validate(stacklevel=_external_stacklevel())
    return _to_fortran_impl(
        diff,
        derivatives=derivatives,
        name=name,
        subroutine_name=subroutine_name,
        integer_args=integer_args,
        module_symbols=module_symbols,
        symmetric_params=symmetric_params,
        stats=stats,
        opts=normalised,
    )


_CONTEXT_KEYWORDS = (
    "integer_args",
    "module_symbols",
    "symmetric_params",
    "stats",
    "options",
)
_PACKAGE_DIRECTORY = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


def _external_stacklevel() -> int:
    """Return the stack level of the first frame outside the package.

    Called from :func:`to_fortran`, level 1 is that function; the result is
    the level, relative to it, of the user code that called it (also when
    the call goes through :meth:`DiffPlz.to_fortran`).

    Returns
    -------
    int
        Level to hand to ``CodegenOptions.validate(stacklevel=...)``.
    """
    frame: FrameType | None = sys._getframe(1)
    level = 1
    while frame is not None and os.path.abspath(
        frame.f_code.co_filename
    ).startswith(_PACKAGE_DIRECTORY):
        frame = frame.f_back
        level += 1
    return level


def _public_signature() -> inspect.Signature:
    """Build the signature advertised by :func:`to_fortran`.

    The leading parameters of the implementation stay positional, the
    context keywords and ``options`` come next, then every behaviour
    option (the keyword-only parameters of the emission stage), all
    keyword-only and with their defaults.

    Returns
    -------
    inspect.Signature
        Signature of the public entry point.
    """
    parameters = inspect.signature(_to_fortran_impl).parameters
    option_parameters = inspect.signature(_emit_code).parameters
    keyword_only = inspect.Parameter.KEYWORD_ONLY
    leading = [
        parameters[name].replace(kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in ("diff", "derivatives", "name", "subroutine_name")
    ]
    context = [
        parameters[name].replace(kind=keyword_only)
        for name in _CONTEXT_KEYWORDS
        if name in parameters
    ]
    context.append(inspect.Parameter("options", keyword_only, default=None))
    behaviour = [
        parameter
        for parameter in option_parameters.values()
        if parameter.kind is keyword_only
    ]
    return inspect.Signature(leading + context + behaviour)


# the explicit signature is read by ``inspect.signature`` and ``help``
to_fortran.__signature__ = _public_signature()  # type: ignore[attr-defined]
