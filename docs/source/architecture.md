# Code generator architecture

This page describes how a `.model` file (or a `DiffPlz` object) becomes a
Fortran module, and how the code generator in `thermodiff/codegen/` is
organised.  It is written for contributors who want to change the
generator, for example to add a pass.  The guiding rule of the
`refactor/perf-arch` work is that a change must not alter the generated
Fortran unless the change says so: see Invariants.

## Overview

```text
 .model text
     |
     v
 thermodiff/model/parser.py      read_sections -> RawModel
     |
     v
 thermodiff/model/ir.py          build_ir -> ModelIR (declarations, inlined
     |                           definitions, target function)
     v
 thermodiff/model/emit.py        build_diff -> DiffPlz
     |   (compile_model drives these three steps; emit_module also resolves
     |    the options with resolve_codegen_options)
     v
 thermodiff/diffplz.py           DiffPlz: symbolic derivatives
     |
     v
 thermodiff/fortran.py           to_fortran  (facade; implementation in
     |                           thermodiff/codegen/pipeline.py)
     |
     |  chain_rule option on:  thermodiff/chain.py
     |     chain_rule_derivatives(...) -> structured derivatives
     |     (NameScheme names the generated component/pair arrays)
     v
 pass pipeline                   _emit_code runs the passes of _PASSES over
     |                           one _EmitState, in order
     v
 thermodiff/codegen/render.py    definitions -> Fortran statements
     |
     v
 Fortran subroutine text  ->  emit_module wraps it in a module  ->  .f90
```

`compile_model(text, filename, derivatives, *, preset, options,
**overrides)` in `thermodiff/model/compile.py` chains `read_sections`,
`build_ir` and `emit_module`.  `emit_module` builds the `DiffPlz` with
`build_diff`, resolves the options with `resolve_codegen_options` and calls
`to_fortran`.

`to_fortran` can also be called directly on any `DiffPlz` object, without a
`.model` file; `DiffPlz.to_fortran` goes through the same function.

The yaeos wrapper path is separate from the pipeline.
`thermodiff/yaeos_model.py` does not run any pass: `yaeos_model_module`
takes the kernels that were already generated (name, argument list and
output set of each workload, see `kernel_interface`) and emits a Fortran
module with a yaeos model type that dispatches, with a `select case` on the
set of optional outputs that are present, to the kernel that computes
exactly that output set.

## Module map

All modules live in `thermodiff/codegen/` unless stated otherwise.  The
private helpers of every module are imported by `thermodiff/fortran.py` and
by the tests.

`thermodiff/fortran.py`
: Public facade.  Its `__all__` is only `["to_fortran"]`.  It re-exports the
  private helpers of the codegen modules (with `noqa: F401`) so that code
  and tests written before the split keep importing them from
  `thermodiff.fortran`.  It contains no logic.

`common.py`
: Helpers shared by the other modules: `_Namer` (sequential temporary
  names `t1`, `t2`, ...), `_sanitize`, free and bound index analysis
  (`_free_indices`, `_index_set`, `_order_free`), `_new_shared_temp`,
  `_print_expr` (SymPy to Fortran text), the `_DERIV_SUFFIX` table, the
  indentation constant `_INDENT` and the `_LoopStart` symbol.

`ir.py`
: The typed definition records (`TempDef` and its four subclasses), see
  Definition records.

`options.py`
: `CodegenOptions`, the option enums, the presets, validation and
  `resolve_codegen_options`, see Options.

`resolve.py`
: Placeholder resolution.  `_PlaceholderResolver` maps the internal
  functions (and their derivatives) of an expression to input arguments;
  `_scan_atoms` collects the atoms that become arguments;
  `_select_derivatives` resolves the user's derivative selection.

`symmetry.py`
: Symmetry helpers: `_canonical_symmetric` and `_swap` decide whether a
  summation body is symmetric under the exchange of two indices,
  `_fold_symmetric_sums` and the `_merge_*` helpers merge symmetric terms,
  and `_symv_parts` / `_render_symv` split and render the symmetric
  matrix-vector products.

`hoisting.py`
: Invariant and affine hoisting: `_hoist_invariants` (loop invariants of
  the array temporaries and of the shared sums), `_hoist_loop_invariants`
  (definitions and outputs), `_split_affine` and `_lowrank_bilinear`.

`cse.py`
: Common-subexpression search: `_indexed_cse` (array temporaries),
  `_share_sums` (summations), `_common_scalar_cse` and
  `_sympy_scalar_cse` (scalar sub-expressions, selected by `cse_backend`).

`loops.py`
: Loop fusion and table rewrites: temporary fusion, nest fusion
  (`_fuse_loop_nests`, `_fuse_sibling_tables`, `_fuse_component_maps`,
  `_fuse_outer_loops`), `_drop_pair_tables`, `_share_elementwise`,
  `_diagonal_from_tables`, `_permute_symmetric_subscripts`,
  `_split_accumulators`, the column-order choices and `_nest_double_sums`.
  It is the largest module.

`render.py`
: Turns definitions into Fortran lines.  `_render_definition` is a
  `functools.singledispatch` function over the records of `ir.py`.  It
  also holds `_expand_sums` (explicit `do` loops for the summations of an
  expression), output declaration and rendering, the dependency ordering
  of the definitions (`_dependency_order`), the optional-argument guards
  (`_guard`, `_guarded`) and the line wrapping of the signature.

`pipeline.py`
: The orchestration.  It defines `_EmitContext` (immutable inputs),
  `_EmitState` (mutable state of one emission), `Pass`, the `_PASSES`
  tuple, the pass functions themselves (most of them thin adapters over
  the modules above), `_emit_code` (the loop over the passes) and the
  public `to_fortran`.

`thermodiff/chain.py`
: The chain-rule step, used only when `chain_rule` is on.
  `chain_rule_derivatives` differentiates structurally over the summation
  nodes (`summation_nodes`, `factor_sums`) and abstracts component and pair
  functions into arrays.  `NameScheme` carries the slot index names
  (`ic`, `ia`, `ib`) and the array prefixes (`cmp`, `mat`) of one run;
  `DEFAULT_NAME_SCHEME` is used unless the model clashes with them.

## Options

`CodegenOptions` (`options.py`) is a frozen dataclass that groups the
behaviour options of `to_fortran`.  Each field has the name and default of
the keyword of the same name.  Per-call context (the `DiffPlz`, the
derivative selection, names, `integer_args`, `module_symbols`,
`symmetric_params`, `stats`) is not part of it.

* Values are coerced in `__post_init__`.  The enum fields accept the legacy
  values: `ScalarChainMode` (`False | True | "cost" | "estimate"`),
  `SepSumsMode` (`False | True | "cost"`), `LowrankHessMode`
  (`False | True | "sym"`) and `CseBackend` (`"thermodiff"` | `"sympy"`).
  `split_acc` is `None` or `2`.
* `from_legacy(**kwargs)` and `to_legacy()` convert to and from the keyword
  form; `from_preset(preset, **overrides)` and `replace(**changes)` build
  variants.
* `validate(stacklevel=1)` returns `(normalised, ignored_names)`.  It
  disables, in one `UserWarning`, the options that the generator would
  silently ignore for the chosen combination (the table `_RULES`: for
  example, everything that needs `chain_rule` when it is off, or the passes
  that cannot run with `optional_derivatives`).  The only error is a
  `split_acc` other than 2 while `optional_derivatives` is off.

Presets are the `Preset` enum, with the settings in `PRESET_SETTINGS`:

| Preset | Meaning |
| ------ | ------- |
| `minimal` | The legacy defaults: the output of earlier versions. |
| `standard` | The default of `compile_model`: chain rule, symmetry, nesting and sum sharing (`_STANDARD_ENABLED`). |
| `fastest` | Opt-in: the fastest generated code, slower generation and different last-bit rounding (`_FASTEST_ENABLED` plus the modes `lowrank_hess=ON`, `scalar_chain=ESTIMATE`, `sep_sums=COST`, `split_acc=2`). |

`to_fortran` itself resolves its options in this order.  The base is
`options` if it was given, else the legacy defaults.  The legacy keywords
passed by the caller are applied on top, then the result is validated with
a stack level that points at the first frame outside the package.  An
unknown keyword is a `TypeError`.  `to_fortran` does not apply a preset: it
keeps the legacy defaults, so existing callers produce the same output.

`resolve_codegen_options(preset, options, **overrides)` is the resolver of
the entry points of the model front-end (`emit_module`, and so
`compile_model` and the CLI).  `preset=None` means `"standard"` unless
`options` is given; passing both is a `ValueError`; an override of `None`
means "not given".  The result is already normalised, and only the ignored
options that the caller set explicitly are reported.

Note on `"fastest"` with `optional_derivatives=True`: the preset is meant
for kernels specialised to one derivative set.  With optional outputs the
passes listed in `_RULES` for `optional_derivatives` are inactive;
`resolve_codegen_options` disables them silently and emits one extra
`UserWarning` saying that the preset applies fully only with
`optional_derivatives=False`.

## The pass pipeline

`_emit_code` is a plain loop:

```python
state = _new_state(ctx, structured, stats, count_scalar, opts)
for stage in _PASSES:
    if stage.enabled(opts):
        stage.run(ctx, state)
return state.text
```

* `_EmitContext` is a frozen dataclass with the resolved inputs (the
  `DiffPlz`, the selected derivatives, names and extra arguments).
* `_EmitState` is a mutable dataclass (`eq=False`, so identity semantics)
  with everything the passes share: `opts`, the `resolver`, the `namer`,
  the outputs in their successive forms (`outputs`, `staged`, `prepared`,
  `all_exprs`), the definition pools (`component_array_defs`,
  `array_defs`, `sum_defs`, `cse_defs`), the signature data and, at the
  end, `text`.  Passes mutate it in place.
* `Pass` is a `NamedTuple(name, enabled, run)`.  `enabled(opts)` is the
  option-level gate and returns a `bool`; gates that depend on data stay
  inside `run`.  `run(ctx, state)` returns `None`.
* `_gate(name, *, unless_optional=False, with_param_symmetry=False)` builds
  the gate of a pass that depends on one option flag; it raises
  `ValueError` if `name` is not a `CodegenOptions` field.

The docstring of each pass function carries its `S<n>` step label, which
shows the position the pass held in the original monolithic function.

### Passes in execution order

This table is the content of `_PASSES`.  Gates written as "`flag`, not
`optional_derivatives`" are `_gate(flag, unless_optional=True)`; gates with
"and `param_symmetry`" are `_gate(flag, with_param_symmetry=True)`.

| # | Pass | Runs when | What it does | Implementation |
| - | ---- | --------- | ------------ | -------------- |
| 1 | `resolve_outputs` | always | Resolve the value and the derivatives into `outputs` (S1-S2). | `pipeline` |
| 2 | `canonicalise_outputs` | always | Canonicalise, merge, nest and prepare the outputs (S3-S7). | `pipeline`, `loops` |
| 3 | `pair_components` | always | Prepare the component definitions and their arrays (S8, S12). | `pipeline` |
| 4 | `scan_signature` | always | Gather the arguments of the routine from the pre-CSE expressions (S9). | `pipeline`, `resolve` |
| 5 | `declare_signature` | always | Declare the input and output arguments (S10, S11). | `pipeline`, `render` |
| 6 | `array_cse` | `array_cse` | Share the common sub-expressions into arrays (S13). | `cse` |
| 7 | `mirror_array_temps` | `param_symmetry` | Flag the symmetric array temporaries as mirrored (S14). | `symmetry` |
| 8 | `reserved_names` | always | Collect the names that temporaries must not take (S15). | `pipeline`, `resolve` |
| 9 | `fuse_temps` | `fuse_temps` | Fuse the temporaries that are read once (S16). | `loops` |
| 10 | `hoist_array_temps` | `hoist_invariants` | Hoist the loop invariants of the array temporaries (S17). | `hoisting` |
| 11 | `share_sum_temps` | always | Share the summations, or pass the expressions through (S18). | `cse` |
| 12 | `hoist_sum_temps` | `hoist_invariants` | Hoist the loop invariants of the shared sums (S18b). | `hoisting` |
| 13 | `scalar_cse` | `scalar_cse`, or `share` when `scalar_cse` is unset | Share the common scalar sub-expressions of the residuals (S19). | `cse` |
| 14 | `alias_scalar_chain` | always | Turn the scalar-chain outputs into definitions and prune (S20a). | `pipeline` |
| 15 | `rendered_outputs` | always | Pair every residual with its output name and free indices (S20b). | `pipeline`, `common` |
| 16 | `definitions_list` | always | Join the pools into the ordered list of definitions (S21). | `pipeline` |
| 17 | `lowrank_outputs` | `lowrank_hess` not off, not `optional_derivatives` | Write the bilinear outputs in low-rank form (S22). | `hoisting` |
| 18 | `hoist_loops` | `hoist_loops` | Hoist the loop invariants of definitions and outputs (S23). | `hoisting` |
| 19 | `drop_pair_tables` | `drop_pair_tables`, not `optional_derivatives` | Drop the pair tables that can be recomputed (S24). | `loops` |
| 20 | `share_elementwise` | `share_elementwise`, not `optional_derivatives` | Share the elementwise definitions (S25). | `loops` |
| 21 | `split_affine` | `split_affine`, not `optional_derivatives` | Split the affine sums into table and scalar parts (S26). | `hoisting` |
| 22 | `mirror_sums` | `sym_sum2` and `param_symmetry` | Flag the symmetric two-index sums as mirrored (S27). | `symmetry` |
| 23 | `column_layouts` | `col_major` | Choose column sums and column-major order per definition (S28). | `loops` |
| 24 | `fuse_loop_nests` | `fuse_loops`, not `optional_derivatives` | Fuse the loop nests with the same ranges (S29). | `loops` |
| 25 | `symv_parts` | `sym_matvec` and `param_symmetry` | Split the symmetric matrix-vector definitions (S30). | `symmetry` |
| 26 | `fuse_siblings` | `sib_fuse`, not `optional_derivatives` | Fuse the sibling tables (S31). | `loops` |
| 27 | `diagonal_tables` | `diag_table`, not `optional_derivatives` | Read the diagonals from the tables (S32). | `loops` |
| 28 | `permute_subscripts` | `sym_perm` and `param_symmetry` | Permute the symmetric subscripts of the fused members (S33). | `loops` |
| 29 | `component_maps` | `comp_fuse`, not `optional_derivatives` | Fuse the component maps (S34). | `loops` |
| 30 | `outer_loops` | `fuse_outer`, not `optional_derivatives` | Fuse the outer loops (S35). | `loops` |
| 31 | `split_accumulators` | `split_acc`, not `optional_derivatives` | Split the reductions into two accumulators (S36). | `loops` |
| 32 | `collect_stats` | always | Fill `stats` with the operation counts (S37). | `pipeline` |
| 33 | `render_bodies` | always | Render the definitions and the outputs (S38, S39). | `pipeline`, `render` |
| 34 | `temp_declarations` | always | Collect the scalar temporaries and the loop indices (S40). | `pipeline` |
| 35 | `assemble_text` | always | Assemble the Fortran source of the subroutine (S41). | `pipeline`, `render` |

`split_acc` is `None` or `2`; a gate built with `_gate("split_acc")` is on
when it is `2`.  The `Implementation` column lists the modules whose
helpers the pass calls; the pass function itself is always in
`pipeline.py`.

Two properties of the order matter:

* `scan_signature` runs before every sharing pass.  After the CSE the
  temporaries would appear as parameters.
* `render_bodies` is the last stage that draws names from the namer
  (render-time temporaries of the expanded sums), and the sharing, hoisting
  and fusion passes draw names in the order of the table.
  `tests/codegen/test_pipeline_passes.py` checks both.

With `scalar_chain="cost"`, `to_fortran` runs `_emit_code` twice (with and
without the scalar chain) through `_generate_cost` and keeps the output
with the lower `scalar_operation_count`; each run has its own state.

## Definition records

`ir.py` holds the typed records of the temporaries.  A `TempDef` is one
temporary: `target` is assigned `body` inside loops over the `free` indices
and the reduction `limits` (tuples `(index, low, high)`).  The optional
annotations are `mirror`, `mirror_sum`, `colsum`, `colmajor`, `symv`,
`symv_acc`, `factored` and `acc`; `None` means absent, any other value
(`False` included) means present.  The records replaced string-keyed dicts
and keep their semantics: they are mutable, compare by value and are
unhashable.  `present_keys()` reports the present annotations under the old
key names and `copy(**changes)` returns a shallow copy.

Four subclasses (keyword-only fields) wrap other definitions:

| Record | Added fields | Produced by |
| ------ | ------------ | ----------- |
| `FusedNest` | `fused`: member definitions in one loop nest | `fuse_loop_nests` |
| `ComponentFusedMap` | `steps`, `members` (`(target, body)` pairs) | `component_maps` |
| `OuterFusedNest` | `members` (definition records), `outer` index | `outer_loops` |
| `SiblingNest` | `sibling`, `accs`, `accumulators`, `accs2` | `fuse_siblings`, `split_accumulators` |

`render._render_definition` is a `functools.singledispatch` function.  The
base handler covers `TempDef` and tests the annotations in a fixed order
(`symv`, `colmajor`, `colsum`, `mirror_sum`, `mirror`, `factored`,
`limits`) and falls back to the plain loop nest.  `SiblingNest`,
`OuterFusedNest`, `ComponentFusedMap` and `FusedNest` have registered
handlers.  A new record type needs a new registered handler.

## Invariants

Every change to the generator must respect these rules.

* **Byte-identical output.**  `tools/check_golden.py --tier
  fast|full|complete -j N` regenerates a tier and compares it with the
  stored golden; exit status 1 on any missing, extra or different file.
  `fast` is the quick check before a commit, `full` skips the slow cases,
  `complete` is everything.  Do not edit the golden to make a change pass:
  find out why the output changed.
* **Determinism.**  Generation must give the same text for the same input.
  The golden records the `PYTHONHASHSEED` it was written with; run the
  generator with the same value.  Keep the existing ordering
  (`sp.default_sort_key`, dict insertion order); never iterate a plain
  `set` of SymPy objects where the order reaches the output.
* **Namer order and count.**  `_Namer` numbers the temporaries by call
  order, so adding, removing or reordering a call changes every later
  `tN`.  `tests/codegen/test_pipeline_pins.py` pins the SHA-256 of the
  Fortran text and of the names the namer created.
* **Identity versus value equality.**  `_EmitState` is `eq=False` and the
  helpers compare pools, `namer` and `arrays` by identity.  The records of
  `ir.py` compare by value and are unhashable.  Do not replace one by the
  other, and do not use records as dict keys or set members.
* **Per-call caches.**  Caches are local to a call and passed explicitly.
  There are no module-level caches on SymPy expressions (no
  `functools.lru_cache` on them).
* **No global state.**  Generation is re-entrant across interleaved calls
  and threads; `tests/test_reentrancy.py` generates the same cases alone,
  interleaved, after a failed call and concurrently, and requires the same
  text.  Options are immutable; do not add mutable module globals.
* **Function size and complexity.**
  `tests/codegen/test_function_lengths.py` limits functions to fifty code
  lines and bounds the cyclomatic complexity (10).  Split a function that
  grows instead of raising the limit.
* **Strict typing.**  `tox -e typecheck` runs `mypy` in strict mode over
  `thermodiff/codegen`.  New code needs complete annotations.
* **Style.**  Black (line length 79), flake8 and NumPy-convention docstrings
  (`tox -e style,docstyle`).  Do not mix refactoring, optimisation and
  formatting in one commit.

## How to add a new pass

The steps below add a pass `my_pass`, gated by an option `my_option`.

1. **Option.**  Add the field to `CodegenOptions` in `options.py`, with a
   default that keeps today's output (normally `False`).  If it takes a
   mode instead of a flag, add an enum and a coercer in `_COERCERS`.  Add a
   keyword-only parameter with the same name and default to `_emit_code` in
   `pipeline.py`: `to_fortran`'s public signature is built from those
   parameters, and `_emit_code` receives the options as
   `**opts.to_legacy()`.
2. **Validation.**  If the option has no effect in some combinations (for
   example, it needs `chain_rule` or cannot run with
   `optional_derivatives`), add its name to the matching `_Rule` in
   `_RULES`, so `validate()` disables and reports it.
3. **Presets.**  Decide for each preset.  `minimal` stays empty.  Add the
   name to `_STANDARD_ENABLED` only if the pass is meant for the default;
   add it to `_FASTEST_ENABLED` if it belongs to the fastest set.  A change
   to `standard` changes the output of every `.model` compile, so say so in
   the commit.
4. **Pass function.**  Write the transformation as a function with explicit
   inputs and outputs in the module that owns the technique (`cse`,
   `loops`, `hoisting`, `symmetry`), without module state, and a thin
   adapter in `pipeline.py` that reads and writes `_EmitState`.  The
   adapter does not hold state of its own and draws names only from
   `state.namer`.
5. **Register it in `_PASSES`** at the right position.  Place it after the
   passes whose output it reads and before those that must not see its
   result.  If it calls the namer, its position among the namer-using
   stages fixes the numbering of every later temporary.
6. **Tests.**  Update `EXPECTED_ORDER` and the gate expectations in
   `tests/codegen/test_pipeline_passes.py`, add unit tests for the pure
   function, and run the pins (`tests/codegen/test_pipeline_pins.py`).
   With the option off, the golden must not change; with it on, add or
   extend a golden tier and explain the expected difference.
7. **Docs.**  Add the pass to the table above, in the position it has in
   `_PASSES`.

Skeleton, with the signatures of the real code:

```python
# thermodiff/codegen/options.py -- field of CodegenOptions
    my_option: bool = False

# thermodiff/codegen/pipeline.py -- keyword-only parameter of _emit_code
#     my_option: bool = False,


def _my_pass(ctx: _EmitContext, state: _EmitState) -> None:
    """Rewrite the definitions with the new technique (S42).

    Parameters
    ----------
    ctx : _EmitContext
        Resolved inputs.
    state : _EmitState
        Emission state; ``definitions`` is rewritten in place.
    """
    state.definitions = rewrite_definitions(state.definitions, state.namer)


_PASSES: tuple[Pass, ...] = (
    ...,
    Pass(
        "my_pass",
        _gate("my_option", unless_optional=True),
        _my_pass,
    ),
    ...,
)
```

`rewrite_definitions` stands for the pure function in `loops.py` (or the
module that fits); it takes the pool and the `namer` explicitly and returns
the new pool.  The attribute names of `_EmitState` are listed in its
docstring.

## Benchmarks and profiling

`benchmarks/td_bench` holds the generation variants and the derivative
workloads.  `registry.py` is the single source of truth for the variants: an
ordered tuple of `stage(key, base, **delta)` entries, each naming its parent
and only the settings it changes, plus the immutable alias map; to add a
variant, append a `stage`.  `hooks.py` builds the public tables (`VARIANTS`,
`EXTENDED_VARIANTS`, `STAGE_VARIANTS`, `WORKLOADS`) from it and exposes
`generate_workload`, which generates the Fortran for one workload and
variant and returns timing and size metadata.  The directory is not part of
the installed package.  `tools/profile_case.py <model> <variant>
[--workload W] [--top N] [--out file.prof] [--warm]` profiles the
generation of one case with `cProfile`, in a fresh child process by default
so that the SymPy cache starts empty.  The measurement protocol and the
recorded timings are in `docs/dev/perf_log.md`; profiles are used to find
hotspots only, never as timings.
