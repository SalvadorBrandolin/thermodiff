"""The ``to_fortran`` pipeline of the Fortran code generator."""

import dataclasses
import inspect
import os
import sys
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import sympy as sp

from thermodiff.codegen import CodegenOptions, ScalarChainMode, SepSumsMode
from thermodiff.codegen.common import (
    _DERIV_SUFFIX,
    _INDENT,
    _LoopStart,
    _Namer,
    _canonical_floats,
    _free_indices,
    _kron_to_piecewise,
    _order_free,
    _referenced_temp_names,
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
_ESTIMATE_OR_COST = (ScalarChainMode.COST, ScalarChainMode.ESTIMATE)
# ``stats`` entry through which the structure leaks (``scalar_chain`` on)
_STRUCTURE_STATS_KEY = "_structured"


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

    diff: object
    selected: tuple
    base: str
    sub_name: str
    integer_args: tuple
    module_symbols: frozenset
    symmetric_params: tuple


def _resolve_context(
    diff,
    derivatives,
    name,
    subroutine_name,
    integer_args,
    module_symbols,
    symmetric_params,
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
    return _EmitContext(
        diff=diff,
        selected=tuple(_select_derivatives(diff, derivatives)),
        base=base,
        sub_name=sub_name,
        integer_args=tuple(integer_args or ()),
        module_symbols=frozenset(module_symbols or ()),
        symmetric_params=tuple(symmetric_params or ()),
    )


def _chain_structure(ctx: _EmitContext, opts: CodegenOptions):
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
    return chain_rule_derivatives(
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


def _plain_structure(
    ctx: _EmitContext, opts: CodegenOptions, structured
) -> tuple:
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


def _generate_estimate(ctx: _EmitContext, opts: CodegenOptions, stats):
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


def _generate_cost(ctx: _EmitContext, opts: CodegenOptions, stats):
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
    stats_sc = {}
    code_sc = _emit_code(
        ctx, structured, stats_sc, True, **chain_opts.to_legacy()
    )
    stats_sc.pop(_STRUCTURE_STATS_KEY, None)
    plain_opts = opts.replace(scalar_chain=ScalarChainMode.OFF)
    plain, propagated = _plain_structure(ctx, plain_opts, structured)
    stats_plain = {}
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
    diff,
    derivatives=None,
    name=None,
    subroutine_name=None,
    integer_args=None,
    module_symbols=None,
    symmetric_params=(),
    stats=None,
    *,
    opts,
):
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
    structured: dict | None
    stats: dict | None
    count_scalar: bool
    resolver: _PlaceholderResolver
    namer: _Namer
    selected: list
    integer_args: list
    module_symbols: set
    symmetric_params: tuple
    scalar_chain_defs: Any = ()
    sc_temp_names: set = dataclasses.field(default_factory=set)
    outputs: list = dataclasses.field(default_factory=list)
    share_sums: bool = False
    scalar_cse: bool = False
    staged: list = dataclasses.field(default_factory=list)
    prepared: list = dataclasses.field(default_factory=list)
    all_exprs: list = dataclasses.field(default_factory=list)
    component_defs: list = dataclasses.field(default_factory=list)
    scans: list = dataclasses.field(default_factory=list)
    param_arrays: dict = dataclasses.field(default_factory=dict)
    params: list = dataclasses.field(default_factory=list)
    placeholder_args: list = dataclasses.field(default_factory=list)
    needs_nc: bool = False
    inputs: list = dataclasses.field(default_factory=list)
    output_decls: list = dataclasses.field(default_factory=list)
    arrays: dict = dataclasses.field(default_factory=dict)
    component_array_defs: list = dataclasses.field(default_factory=list)
    array_defs: list = dataclasses.field(default_factory=list)
    sum_defs: list = dataclasses.field(default_factory=list)
    cse_defs: list = dataclasses.field(default_factory=list)
    reserved: set = dataclasses.field(default_factory=set)
    residuals: list = dataclasses.field(default_factory=list)
    scalar_chain_definitions: list = dataclasses.field(default_factory=list)
    rendered: list = dataclasses.field(default_factory=list)
    definitions: list = dataclasses.field(default_factory=list)
    lowrank_outs: set = dataclasses.field(default_factory=set)
    out_names: dict = dataclasses.field(default_factory=dict)
    consumers: dict = dataclasses.field(default_factory=dict)
    body: list = dataclasses.field(default_factory=list)
    temps: list = dataclasses.field(default_factory=list)
    loop_indices: list = dataclasses.field(default_factory=list)
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
    structured: dict | None,
    stats: dict | None,
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
    )


def _derivative(ctx: _EmitContext, structured: dict | None, attr: str):
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
    structured = state.structured
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
    if state.scalar_chain_defs:
        _scalar_chain_outputs(state)


def _stage_outputs(state: _EmitState) -> None:
    """Canonicalise every output expression into ``staged`` (S4)."""
    opts = state.opts
    symmetric_params = state.symmetric_params
    staged = []
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
    shared_bases = set()
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
    prepared = []
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
    component_defs = []
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
    scheme = state.structured[NAME_SCHEME_KEY]
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


def _parameter_scalars(state: _EmitState) -> set:
    """Return the names of the scalar parameters found in ``scans``."""
    # NOTE: ``n`` is compared by name because ``expr`` has been rebuilt by
    # the ``nc`` replacement, which also rewrites the shape of the ``n``
    # IndexedBase.
    index_syms = set()
    indexed_bases = set()
    for scan in state.scans:
        for indexed in scan.indexed:
            index_syms.update(indexed.indices)
        for total in scan.sums:
            for limit in total.limits:
                index_syms.add(limit[0])
        index_syms.update(scan.idxs)
        indexed_bases.update(scan.bases)

    index_names = {str(idx) for idx in index_syms}
    base_names = {str(base_) for base_ in indexed_bases}

    param_scalars = set()
    for scan in state.scans:
        for sym in scan.symbols:
            name = str(sym)
            if name in index_names or name in base_names:
                continue
            if sym in (T, V, P) or name == "nc":
                continue
            if isinstance(sym, _LoopStart):
                continue
            if name in state.module_symbols or name in state.integer_args:
                continue
            if name in state.sc_temp_names:
                continue
            param_scalars.add(name)
    return param_scalars


def _parameter_arrays(state: _EmitState, uses_n: bool) -> dict:
    """Return the rank of every array argument found in ``scans``."""
    n_name = str(n)
    placeholder_names = set(state.resolver.ranks)
    component_names = {c[0] for c in state.component_defs}
    array_ranks = {}
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


def _legacy_tail(ctx: _EmitContext, state: _EmitState) -> None:
    """Run the stages not yet extracted (S13-S41); temporary."""
    diff = ctx.diff
    structured = state.structured
    stats = state.stats
    count_scalar = state.count_scalar
    namer = state.namer
    integer_args = state.integer_args
    module_symbols = state.module_symbols
    symmetric_params = state.symmetric_params
    scalar_chain_defs = state.scalar_chain_defs
    share_sums = state.share_sums
    scalar_cse = state.scalar_cse
    prepared = state.prepared
    all_exprs = state.all_exprs
    scans = state.scans
    params = state.params
    param_arrays = state.param_arrays
    placeholder_args = state.placeholder_args
    needs_nc = state.needs_nc
    inputs = state.inputs
    output_decls = state.output_decls
    arrays = state.arrays
    component_array_defs = state.component_array_defs
    sub_name = ctx.sub_name
    legacy = state.opts.to_legacy()
    array_cse = legacy["array_cse"]
    cse_expensive = legacy["cse_expensive"]
    param_symmetry = legacy["param_symmetry"]
    sym_rank3 = legacy["sym_rank3"]
    nest_shared = legacy["nest_shared"]
    fuse_temps = legacy["fuse_temps"]
    fuse_guard = legacy["fuse_guard"]
    keep_libm_tables = legacy["keep_libm_tables"]
    hoist_invariants = legacy["hoist_invariants"]
    sum_alpha = legacy["sum_alpha"]
    sum_free_alpha = legacy["sum_free_alpha"]
    sum_transpose = legacy["sum_transpose"]
    cse_backend = legacy["cse_backend"]
    float_canon = legacy["float_canon"]
    optional_derivatives = legacy["optional_derivatives"]
    lowrank_hess = legacy["lowrank_hess"]
    hoist_loops = legacy["hoist_loops"]
    drop_pair_tables = legacy["drop_pair_tables"]
    share_elementwise = legacy["share_elementwise"]
    split_affine = legacy["split_affine"]
    sym_sum2 = legacy["sym_sum2"]
    col_major = legacy["col_major"]
    table_layout = legacy["table_layout"]
    fuse_loops = legacy["fuse_loops"]
    sym_matvec = legacy["sym_matvec"]
    sib_fuse = legacy["sib_fuse"]
    diag_table = legacy["diag_table"]
    sym_perm = legacy["sym_perm"]
    comp_fuse = legacy["comp_fuse"]
    fuse_outer = legacy["fuse_outer"]
    fuse_libm_guard = legacy["fuse_libm_guard"]
    split_acc = legacy["split_acc"]
    hessian_symmetry = legacy["hessian_symmetry"]
    array_pad = legacy["array_pad"]
    scalar_chain = legacy["scalar_chain"]

    if array_cse:
        all_exprs, array_defs = _indexed_cse(
            all_exprs,
            namer,
            arrays,
            expensive_twice=cse_expensive,
            # with scalar_chain, only the plain outputs count (as without it)
            counted=(
                [entry[0] == _SCALAR_PLAIN for entry in prepared]
                if scalar_chain_defs
                else None
            ),
        )
    else:
        array_defs = []
    if param_symmetry and symmetric_params:
        for definition in array_defs:
            free = definition.free
            if len(free) != 2 or definition.limits:
                continue
            body = _canonical_symmetric(
                definition.body, symmetric_params, sym_rank3
            )
            swapped_equal = (
                _swap(body, free[0], free[1], symmetric_params, sym_rank3)
                == body
            )
            if swapped_equal:
                definition.mirror = True
                if nest_shared:
                    # a symmetric temporary behaves as a symmetric parameter
                    # for the later passes (sum sharing modulo renaming)
                    symmetric_params = symmetric_params + (
                        _temp_name(definition.target),
                    )
    reserved = (
        set(module_symbols)
        | set(integer_args)
        | {out_name for _, out_name, _, _ in prepared}
    )
    for expr in all_exprs:
        scan = _scan_atoms(expr)
        reserved.update(str(sym) for sym in scan.symbols)
        reserved.update(str(base_) for base_ in scan.bases)
    if fuse_temps:
        pool = component_array_defs + array_defs
        pool, all_exprs, fused = _fuse_temporaries(
            pool,
            all_exprs,
            dedup=nest_shared,
            guard=fuse_guard,
            keep_libm=keep_libm_tables,
        )
        component_array_defs = [d for d in pool if d in component_array_defs]
        array_defs = [d for d in pool if d not in component_array_defs]
        for name in fused:
            arrays.pop(name, None)
    if hoist_invariants:
        all_exprs, hoist_defs, _ = _hoist_invariants(
            all_exprs, namer, arrays, reserved=reserved
        )
        array_defs = array_defs + hoist_defs
    if share_sums:
        residuals, sum_defs = _share_sums(
            all_exprs,
            namer,
            diff.indexes,
            arrays,
            alpha=sum_alpha,
            free_alpha=sum_free_alpha,
            symmetric_params=symmetric_params if param_symmetry else (),
            transpose=sum_transpose,
            sym_rank3=sym_rank3,
        )
    else:
        residuals = list(all_exprs)
        sum_defs = []
    if hoist_invariants and sum_defs:
        _, _, sum_defs = _hoist_invariants(
            [], namer, arrays, sum_defs, reserved=reserved
        )
    if scalar_cse:
        if cse_backend == "sympy":
            residuals, cse_defs = _sympy_scalar_cse(residuals, namer, arrays)
        else:
            residuals, cse_defs = _common_scalar_cse(residuals, namer, arrays)
    else:
        cse_defs = []
    scalar_chain_definitions = []
    if scalar_chain_defs:
        keep = [
            k
            for k, entry in enumerate(prepared)
            if entry[0] not in (_SCALAR_CHAIN, _SCALAR_PLAIN)
        ]
        alias = {}
        for (attr, out_name, _, _), residual in zip(prepared, residuals):
            if attr != _SCALAR_CHAIN:
                continue
            if isinstance(residual, sp.Symbol):
                alias[sp.Symbol(out_name)] = residual
                namer.created[:] = [
                    x for x in namer.created if str(x) != out_name
                ]
            else:
                scalar_chain_definitions.append(
                    TempDef(
                        target=sp.Symbol(out_name),
                        free=[],
                        limits=[],
                        body=residual,
                    )
                )
        for target, body in scalar_chain_defs:
            body = (
                _kron_to_piecewise(body)
                .xreplace({nc: sp.Symbol("nc")})
                .xreplace(alias)
            )
            if float_canon:
                body = _canonical_floats(body)
            scalar_chain_definitions.append(
                TempDef(target=target, free=[], limits=[], body=body)
            )
        prepared = [prepared[k] for k in keep]
        residuals = [residuals[k] for k in keep]
        residuals = [residual.xreplace(alias) for residual in residuals]
        # drop what only the plain outputs read
        pools = [component_array_defs, array_defs, sum_defs, cse_defs]
        while True:
            read = set()
            for e in (
                residuals
                + [d.body for pool in pools for d in pool]
                + [d.body for d in scalar_chain_definitions]
            ):
                read |= {str(x) for x in e.free_symbols} | {
                    str(x.base) for x in e.atoms(sp.Indexed)
                }
            gone = {
                _temp_name(d.target) for pool in pools for d in pool
            } - read
            if not gone:
                break
            pools = [
                [d for d in pool if _temp_name(d.target) not in gone]
                for pool in pools
            ]
            namer.created[:] = [x for x in namer.created if str(x) not in gone]
            for name in gone:
                arrays.pop(name, None)
        component_array_defs, array_defs, sum_defs, cse_defs = pools
    rendered = [
        (
            attr,
            out_name,
            residual,
            _order_free(_free_indices(residual), diff.indexes),
        )
        for (attr, out_name, _, _), residual in zip(prepared, residuals)
    ]

    definitions = component_array_defs + array_defs + sum_defs + cse_defs
    if scalar_chain_definitions:
        # drop the scalars nothing reads
        while True:
            read = set()
            for e in residuals + [
                d.body for d in definitions + scalar_chain_definitions
            ]:
                read |= {str(x) for x in e.free_symbols}
            unread = [
                d
                for d in scalar_chain_definitions
                if str(d.target) not in read
            ]
            if not unread:
                break
            gone = {str(d.target) for d in unread}
            scalar_chain_definitions = [
                d
                for d in scalar_chain_definitions
                if str(d.target) not in gone
            ]
            namer.created[:] = [x for x in namer.created if str(x) not in gone]
        definitions = _dependency_order(definitions + scalar_chain_definitions)
    lowrank_outs = set()
    if lowrank_hess and not optional_derivatives:
        for position, (attr, out_name, residual, free) in enumerate(rendered):
            if len(free) != 2 or residual.has(sp.Sum):
                continue
            result = _lowrank_bilinear(residual, free, namer, arrays)
            if result is None:
                continue
            residual, new_defs = result
            definitions = definitions + new_defs
            rendered[position] = (attr, out_name, residual, free)
            residuals[position] = residual
            lowrank_outs.add(attr)
    if hoist_loops:
        definitions, new_residuals = _hoist_loop_invariants(
            definitions,
            [(residual, free) for _, _, residual, free in rendered],
            namer,
            arrays,
            reserved=reserved,
        )
        rendered = [
            (attr, out_name, residual, free)
            for (attr, out_name, _, free), residual in zip(
                rendered, new_residuals
            )
        ]
        residuals = new_residuals
    if drop_pair_tables and not optional_derivatives:
        definitions = _drop_pair_tables(
            definitions,
            [residual for _, _, residual, _ in rendered],
            namer,
            arrays,
        )
    if share_elementwise and not optional_derivatives:
        definitions, residuals = _share_elementwise(
            definitions,
            [residual for _, _, residual, _ in rendered],
            namer,
            arrays,
        )
        rendered = [
            (attr, out_name, residual, free)
            for (attr, out_name, _, free), residual in zip(rendered, residuals)
        ]
    if split_affine and not optional_derivatives:
        definitions, residuals = _split_affine(
            definitions,
            [residual for _, _, residual, _ in rendered],
            namer,
            arrays,
        )
        rendered = [
            (attr, out_name, residual, free)
            for (attr, out_name, _, free), residual in zip(rendered, residuals)
        ]
    if sym_sum2 and param_symmetry:
        for definition in definitions:
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
    if col_major:
        for definition in definitions:
            if table_layout and _prefers_column_sum(
                definition, symmetric_params
            ):
                definition.colsum = True
                continue
            if definition.mirror_sum:
                continue
            if not definition.symv and _prefers_column_order(definition):
                definition.colmajor = True
    if fuse_loops and not optional_derivatives:
        definitions = _fuse_loop_nests(definitions, namer)
    if sym_matvec and param_symmetry:
        for definition in definitions:
            parts = _symv_parts(
                definition, symmetric_params, sym_rank3=sym_rank3
            )
            if parts is not None:
                definition.symv = parts
                definition.symv_acc = namer("t")
    if sib_fuse and not optional_derivatives:
        definitions = _fuse_sibling_tables(definitions, namer)
    if diag_table and not optional_derivatives:
        definitions = _diagonal_from_tables(
            definitions, symmetric_params, sym_rank3=sym_rank3
        )
    if sym_perm and param_symmetry:
        definitions = _permute_symmetric_subscripts(
            definitions, symmetric_params
        )
    if comp_fuse and not optional_derivatives:
        definitions = _fuse_component_maps(definitions, namer)
    if fuse_outer and not optional_derivatives:
        definitions = _fuse_outer_loops(
            definitions,
            libm_guard=fuse_libm_guard,
            libm_reductions=fuse_libm_guard and drop_pair_tables,
        )
    if split_acc and not optional_derivatives:
        if int(split_acc) != 2:
            raise ValueError("split_acc supports 2 accumulators only")
        definitions = _split_accumulators(definitions, namer)
    if stats is not None:
        stats["operation_count"] = sum(
            sp.count_ops(residual, visual=False) for residual in residuals
        ) + sum(
            sp.count_ops(definition.body, visual=False)
            for definition in definitions
        )
        stats["operation_count_tool"] = "sympy.count_ops(visual=False)"
        stats["n_temporaries"] = len(definitions)
    if stats is not None and count_scalar:
        stats["scalar_operation_count"] = sum(
            sp.count_ops(residual, visual=False)
            for (_a, _o, _r, free_), residual in zip(rendered, residuals)
            if not free_
        ) + sum(
            sp.count_ops(definition.body, visual=False)
            for definition in definitions
            if not definition.free and not definition.limits
        )
        if scalar_chain is True:
            stats[_STRUCTURE_STATS_KEY] = structured
    out_names = {attr: out_name for attr, out_name, _, _ in rendered}

    temp_names = {_temp_name(definition.target) for definition in definitions}
    consumers = {name: set() for name in temp_names}
    for attr, _out_name, residual, _free in rendered:
        for name in _referenced_temp_names(residual, temp_names):
            consumers[name].add(attr)
    for definition in reversed(definitions):
        name = _temp_name(definition.target)
        for dependency in _referenced_temp_names(definition.body, temp_names):
            consumers[dependency] |= consumers[name]

    body = []
    for definition in definitions:
        if optional_derivatives:
            condition = _guard(
                consumers[_temp_name(definition.target)], out_names
            )
        else:
            condition = ""
        body.extend(_guarded(_render_definition(definition), condition))
        body.append("")
    for attr, out_name, expr, free_order in rendered:
        if optional_derivatives:
            condition = f"present({out_name})"
        else:
            condition = ""
        mirror = (
            hessian_symmetry
            and attr == "dnidnj"
            and (attr not in lowrank_outs or lowrank_hess == "sym")
        )
        column = attr in lowrank_outs and lowrank_hess != "sym"
        body.extend(
            _guarded(
                _render_output(
                    out_name,
                    expr,
                    free_order,
                    namer,
                    mirror=mirror,
                    column=column,
                ),
                condition,
            )
        )
        body.append("")

    temps = sorted(
        {str(t) for t in namer.created if str(t) not in arrays} | {"res"}
    )
    loop_indices = sorted(
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

    # ---- assemble ------------------------------------------------------
    arg_names = []
    if needs_nc:
        arg_names.append("nc")
    arg_names.extend(integer_args)
    for var, label in ((V, "V"), (P, "P"), (T, "T")):
        if any(var in scan.symbols for scan in scans):
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
    for line in _wrap_name_list("integer :: ", loop_indices):
        lines.append(_INDENT + line)

    lines.append("")
    for line in body:
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
    Pass("legacy_tail", _always, _legacy_tail),
)


def _emit_code(
    ctx,
    structured,
    stats,
    count_scalar,
    *,
    strict=False,
    optional_derivatives=False,
    array_cse=True,
    share=True,
    share_sums=None,
    scalar_cse=None,
    param_symmetry=False,
    hessian_symmetry=False,
    cse_backend="thermodiff",
    sum_alpha=False,
    array_pad=0,
    hoist_invariants=False,
    chain_rule=False,
    hoist_loops=False,
    sum_free_alpha=False,
    fuse_temps=False,
    nest_sums=False,
    split_sqrt=False,
    split_sums=False,
    nest_shared=False,
    sym_matvec=False,
    col_major=False,
    pair_cse=False,
    pair_rates=False,
    sum_transpose=False,
    sep_sums=False,
    sym_rank3=False,
    nest_general=False,
    sym_sum2=False,
    cse_expensive=False,
    fuse_loops=False,
    nested_nodes=False,
    sym_perm=False,
    fuse_guard=False,
    nest_cost=False,
    comp_fuse=False,
    fuse_outer=False,
    fuse_libm_guard=False,
    keep_libm_tables=False,
    diag_table=False,
    sib_fuse=False,
    dummy_canon=False,
    float_canon=False,
    contiguous_args=False,
    table_layout=False,
    lowrank_hess=False,
    drop_pair_tables=False,
    share_elementwise=False,
    scalar_chain=False,
    split_affine=False,
    split_acc=False,
):
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
    diff,
    derivatives=None,
    name=None,
    subroutine_name=None,
    *,
    integer_args=None,
    module_symbols=None,
    symmetric_params=(),
    stats=None,
    options=None,
    **legacy,
):
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
    frame = sys._getframe(1)
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


to_fortran.__signature__ = _public_signature()
