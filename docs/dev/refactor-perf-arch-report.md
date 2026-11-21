# Refactor report: `refactor/perf-arch`

Branch `refactor/perf-arch`, from `improvements` @ `afebcf9`
(183 commits including this report, linear history, no merges, not pushed). Written by the
coordinator from `.orchestration/STATE.md` (the task ledger); the plan was
`ORCHESTRATOR.md`.

Goal: make the Fortran code generator faster and give it a maintainable
architecture **without changing the generated Fortran**, except where a
change is declared below.

## 1. Outcome against the acceptance criteria

| Criterion (ORCHESTRATOR.md §5) | Result |
|---|---|
| `check_golden --tier complete` passes, except approved differences listed here | **Met.** 581 cases byte-identical at every merge (fast tier after every commit, full tier before every merge, complete tier at each checkpoint and at G3). No golden file was regenerated. |
| `pytest tests/` green (+ `-m thesis` where external models exist) | **Met.** 1565 passed, 1 skipped (G3 on `5a61381`). The `thesis` marker is registered but no test carries it (decision in Wave 2), so the thesis tier collects 0 tests. |
| `tox -e style,docstyle,coverage` green | **Met.** style 0 findings (588 at the base), docstyle 0 (42 at the base), coverage 98 % (threshold 90 %). Also `tox -e typecheck` (new) and `tox -e docs` (Sphinx `-W`) green. |
| Backend (`codegen/**`, `chain.py`): no function over 50 lines or complexity 10 | **Met**, and enforced by `tests/codegen/test_function_lengths.py` (code lines, docstrings excluded; McCabe ≤ 10). At the base, `fortran.py` alone was 4371 lines and `_emit_code` had complexity 128. |
| No mutable globals | **Met.** The `_SYM_RANK3` flag and chain.py's name globals are gone (explicit `sym_rank3` argument, per-call `NameScheme`); `tests/test_reentrancy.py` proves interleaved calls do not interact. Module-level constants remain (tables, immutable mappings). |
| No magic-key dictionaries | **Mostly met.** Definitions are typed records (`codegen/ir.py`). Remaining string keys: the `structured` dict that `chain_rule_derivatives` returns (`"value"`, `"_plain"`, `"_component_defs"`, 3 sites in `pipeline.py`) and the public `stats` output dict (documented keys, part of the API). See §6. |
| `to_fortran` takes one options object, old kwargs still accepted | **Met.** `to_fortran(diff, options=CodegenOptions(...))`; the 58 legacy keyword arguments still work through `CodegenOptions.from_legacy` with an explicit signature. |
| Profiled slow case ≥ 70 % faster | **Met.** Slow case (pcsaft e3 w6, cold): 192.7 s → 57.7 s median, **−70.1 %**. |
| Linear history of atomic commits, no push | **Met.** Every task merged by fast-forward after its gate; one logical change per commit (`Task:`/`Step:` trailers). |

## 2. Performance

Cold fresh processes, one generation each, `generation_s`; final = medians of the T8.6
session (`docs/dev/perf_log.md`, section "Summary" and "Final measurement").

| Case | Wave 0 baseline | Checkpoint 1 (`331ad50`) | Final (`5a61381`) | Final vs baseline |
|---|---|---|---|---|
| slow pcsaft e3 w6 | 192.7 s | 57.6 s | 57.7 s [57.3, 57.7] | **−70.1 %** |
| medium pcsaft e2 w5d | 66.2 s (in-process) | 21.4 s | 21.3 s | ≈ −68 % (not like-for-like) |
| p32 pcsaft p32 w6 | 127.6 s | 62.3 s | 62.0 s | −51.4 % |
| peak RSS, slow case | 114.6 MiB | 138.3 MiB | 144.3 MiB | +26 % |

All the speed came from Wave 1. Waves 2-9 (options, pipeline, module split, typed records,
decompositions, typing) cost nothing measurable: measured against `331ad50` in the same
session, interleaved, the final code is −1.7 % (slow), −0.3 % (medium) and −2.8 % (p32), all
within run-to-run noise, and peak RSS +0.7 %. The RSS increase against the baseline comes
from the exact per-call memos of T1.9 (freed after each call).

Where the time went (Wave 1, all exact, output byte-identical):

| Task | Change | Effect (slow case, cold) |
|---|---|---|
| T1.1 | Kronecker handling: `subs` → `xreplace` | −6.8 % (medium −12.2 %) |
| T1.2 | `_indexed_cse`: per-call memo and reuse | −6.1 % net (one intermediate commit regressed +18 %, recovered in the same task) |
| T1.7a | `_free_indices`: exact per-call memo (it was 40 % of the profile, 1.85 M calls) | −36 % |
| T1.9 | Per-subtree memo of pre-order entries in both CSE passes | −39.5 % (RSS +20 %, accepted: per call, freed after) |
| T1.10 | Skip no-op `xreplace` at expression level, lazy rank | −2.7 % (p32 −15 %) |
| T1.11 | Merge the repeated `atoms` walks in `to_fortran` | brought the cold median to 57.6 s |

Considered and not done: T1.3 (batch CSE extraction; would renumber temporaries), T1.4a/b,
T1.5, T1.6 (chain.py and `_print_expr` were < 12 % and 0.1 % of the measured profiles),
step 1.8 (disk cache of differentiation, D7), and the pruned-`xreplace` candidates B and C
from scout T1.S2 (estimated 3-6 s and 2-3 s; not needed once the target was met).

## 3. What changed (architecture)

- **Options (Wave 3).** `CodegenOptions` (frozen dataclass, enums, `validate()`): option
  combinations that were silently ignored now raise a `UserWarning` naming the inert options
  and are disabled explicitly; `ValueError` only where the old code already failed (D2).
- **Presets (Wave 3, declared output change, see §4).** `"minimal"`, `"standard"`,
  `"fastest"`; `--preset` on the CLI and a preset selector in the GUI.
- **No global state (Wave 4).** Fixed a real bug on the way: `_SYM_RANK3` leaked into the
  next `chain_rule` generation after a failed call (T4.1).
- **Module split (Wave 5).** `thermodiff/fortran.py` is a 63-line facade over
  `thermodiff/codegen/{common, ir, options, resolve, symmetry, hoisting, cse, loops,
  render, pipeline}.py` (acyclic import graph).
- **Typed definitions (Wave 6).** `TempDef` and subclasses `FusedNest`,
  `ComponentFusedMap`, `OuterFusedNest`, `SiblingNest`. Deviation D16: they are mutable
  dataclasses with value equality, not the frozen hierarchy of the plan, because the code
  relies on identity checks, in-place mutation and namer order that a frozen conversion would
  break.
- **Pass pipeline (Wave 7).** `_emit_code` (complexity 128) became 35 named passes,
  `Pass(name, enabled, run)` over one `_EmitState`; every backend function was then decomposed
  to complexity ≤ 10 and ≤ 50 lines (extra round D17, 18 more functions).
- **Benchmarks (Wave 8).** `thermodiff/bench_hooks.py` moved out of the package to
  `benchmarks/td_bench/`, with one declarative variant registry (118 keys and 56 aliases
  frozen by `tests/test_variant_registry.py`).
- **Quality (Wave 9).** Strict mypy on `codegen/` (714 → 0 errors, `tox -e typecheck`);
  style and docstyle at 0; four tests that checked numbers through Fortran text now compile
  the kernel and compare values (T8.3); `docs/source/architecture.md` describes the pipeline
  and how to add a pass.
- **Tooling and tests.** Golden checker with fast (112) / complete (581) / full tiers
  (`tools/check_golden.py`, external golden in `../golden_ext`); pytest-xdist and a `slow`
  marker (full suite ≈ 4-8 min with `-n 8`, was ≈ 20 min serial); 587 → 1566 tests, most of
  them pins written before each refactor.

## 4. Declared output differences (approved by the human)

1. **Default preset at the entry points (D13).** `compile_model`, `emit_module`, the CLI and
   the GUI now default to `preset="standard"` (the e8 variant: sum sharing, CSE, symmetry,
   invariant hoisting, chain rule) instead of the old minimal settings. The generated Fortran
   of these entry points therefore changes. Validated numerically: `standard` kernels compiled
   with gfortran match `minimal` (test `972f818`). `"fastest"` (p33) is opt-in (it adds
   gfortran/x86 micro-optimisations and can differ in the last bit). Old behaviour:
   `preset="minimal"`.
   **`to_fortran` called directly is unchanged**: it keeps the neutral low-level defaults
   (= "minimal"); only `resolve_codegen_options`, used by the entry points above, applies
   "standard". This is why the golden (which calls `to_fortran` with explicit variants) stays
   byte-identical.
2. **New warnings.** Combinations whose options are inert now emit a `UserWarning` (D2), and
   `"fastest"` with `optional_derivatives=True` notes which of its options are inactive
   (12 names: 11 gated options plus the `fuse_libm_guard` cascade; T3.2/T3.6).
   No generated code changes.
3. **Public signature metadata.** `inspect.signature(to_fortran)` now carries type
   annotations (names, kinds and defaults unchanged).

No other output difference: every merge kept the 581 golden cases byte-identical.

## 5. Decisions not to do something, and why

- Frozen definition records with `replace` (plan Wave 6): deferred (D16, see §3).
- Disk cache of differentiation (step 1.8): not done (D7).
- T1.3 batch CSE: would renumber temporaries; unnecessary after T1.9.
- `thesis` marker applied to no test: no test depends on files outside the repository;
  marking `test_extensions` would have removed 162 tests from the gates.
- Literal-Fortran asserts (T8.3): the plan allowed deleting the brittle ones. The coordinator
  chose to add numeric checks first and delete little: 2 asserts removed, 6 relaxed, 4 numeric
  tests added; "output unchanged" text comparisons were kept on purpose.
- E203 is ignored in flake8 (D19), as Black recommends; no other global ignore.

## 6. Known debt and candidate cleanups

- `chain_rule_derivatives` still returns a string-keyed dict (`"value"`, `"_plain"`,
  `"_component_defs"`); a small typed record would finish the "no magic keys" goal.
- `chain.py` is outside the strict-mypy scope; one `# type: ignore[no-untyped-call]` in each
  of `hoisting.py`, `loops.py` and `pipeline.py` marks calls into it.
- `_split_affine` rejected-split path: the namer number and `arrays` entry survive
  `del defs[k:position]`. Probably unreachable; to remove the side effect safely, count hits on
  that line during `check_golden --tier complete` first.
- Repeated literals: `"nc"` and the `_free{pos}` prefix across `cse.py`/`loops.py`; the
  `_orientations_*` 4-tuples should be a NamedTuple; `_REQUIRED_ACCUMULATORS` duplicates
  `options._REQUIRED_SPLIT_ACC`; `scheme=` should become required in chain.py helpers.
- `tests/codegen/test_loops_fusion.py` is 3821 lines of pins; fine as a safety net, heavy to
  maintain.
- `thermodiff/model/__main__.py` is not covered (4 statements).
- RSS: the per-call memos of T1.9 raised peak memory on the slow case from 114.6 to
  144.3 MiB (+26 %). Acceptable for a generator; revisit if large models hit memory limits.

## 7. How this was verified

- Gates: G1 (writer, before each commit), G2 (gatekeeper before each merge: full suite with
  slow tests + golden full), G3 at checkpoints (golden complete, coverage, tox).
- Determinism: output independent of `PYTHONHASHSEED` (checked with 3 seeds); runs use
  `PYTHONHASHSEED=0`.
- Reviews: a reviewer agent checked every architecture task for behaviour risks (identity
  checks, value-equality list operations, in-place mutation, namer order, late-binding
  closures); the coordinator also compared annotation-stripped ASTs for the typing tasks.
- Checkpoints: CP0 `401daf6`, CP1 `3d2d072` (code `331ad50`), CP2 `7042675`, CP3 `2bcc7f2`,
  G3 `5a61381`.
