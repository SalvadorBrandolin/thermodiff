# Spec: autotuning mode ("greedy") for generated kernels

Status: specification only, nothing implemented. Not a priority; kept here
so the work can start from an agreed design. Written against
`docs/compiler-flags` (stacked on the `feat/fused-chain-sparse` work).

Evidence labels: **[M]** measured (source given), **[R]** read in the
source, **[S]** suggestion, **[Q]** open question for the maintainer.
Measurement sources named V6, V7, V8, W21, W22 and "max_nc sweep" are the
benchmark tasks behind PRs #23, #24 and #26; their tables are summarised in
those PR descriptions and in `docs/source/fortran.rst`.

---

## 1. Motivation

The fastest code-generator settings depend on the model, the output set,
`nc` and the compiler flags; no fixed preset wins everywhere:

- `-fstack-arrays` vs heap temporaries: 0.90x median, ~0.5x at small `nc`
  [M: V7].
- `work_array`: 0.80x median at `nc = 2` without the flag, but 1.05-1.16x
  on some PR/PR76 kernels at `nc = 50`, up to 2.46x with the flag, and
  0.65-0.75x on the full UNIFAC output set [M: V7, V8].
- `max_nc`: a few percent faster when close to the real maximum `nc`,
  harmful when large [M: max_nc sweep; `docs/source/fortran.rst`,
  "Compiler flags and storage of temporaries"].
- `sparse_parameters`: 1.15-1.8x faster on sparse UNIFAC data, 0.80-0.94x on
  dense data [M: W22].
- `chain_rule` / `work_array` decide whether UNIFAC Ge+Gen is 0.64-0.80x or
  1.25-1.30x of hand-written yaeos [M: yaeos session, PR #24 comment].

The user should not have to run these campaigns by hand. An autotuning mode
generates a curated set of candidates, checks them numerically, times them
on the user's machine with the user's compiler and flags, keeps the fastest
and writes a report.

## 2. Scope

In: one `.model` file (or a `DiffPlz` plus `to_fortran` keywords), one or
more output sets, Fortran kernels compiled by an external compiler
(gfortran first; `FC` and flags configurable).
Out (v1): GPU or other backends, tuning the compiler flags themselves (the
user gives one flag set, or a short list to compare: §4.4), tuning inside
yaeos (yaeos consumes the result).

## 3. User interface [S]

Python:

```python
from thermodiff.tune import tune_model

result = tune_model(
    "unifac.model",
    workloads={"lng": ["dn"], "full": ["dT", "dT2", "dn", "dndT", "dndn"]},
    nc=(2, 10, 50),                 # sizes to optimise for (weights: §6)
    inputs="unifac_inputs.json",    # representative values (§5); optional
    compiler="gfortran",
    flags=("-O3", "-funroll-loops"),
    max_nc=None,                    # allow the max_nc candidate only if given
    budget_s=900,                   # wall-clock limit for the whole run
    jobs=4,                         # parallel generation/compilation
    core=None,                      # pin timings (Linux taskset), or None
    report="unifac_tuning.md",
)
result.options["full"]   # CodegenOptions of the winner per workload
result.code              # Fortran module with the winning kernels
result.save("unifac_tuned.json")   # pinned choice, see §7
```

CLI: `thermodiff-model tune unifac.model --workload lng=dn
--workload full=dT,dT2,dn,dndT,dndn --nc 2,10,50 --inputs inputs.json
--flags="-O3 -funroll-loops" --report tuning.md --out unifac.f90
--save unifac_tuned.json`.

Generation from a saved choice, with no timing:
`compile_model(..., tuned="unifac_tuned.json")` and
`thermodiff-model generate --tuned unifac_tuned.json` (the file names the
model hash; a mismatch warns, §7).

`preset="greedy"` [Q]: the maintainer suggested the name. Presets are static
option sets today (`PRESET_SETTINGS`), so "greedy" should be accepted by the
CLI and `compile_model` as a spelling of "run the tuner with defaults", not
added to `PRESET_SETTINGS`. Confirm the name.

## 4. Candidate set

### 4.1 Base candidates [S]

Per workload, the cross product of:

| Axis | Values |
|---|---|
| preset | `standard`, `fastest` |
| storage | default, `work_array`, `max_nc=N` (only if the user gives `N`) |
| `chain_rule` | preset value, flipped (only when the model has summation nodes it applies to) |
| `optional_derivatives` | False (one kernel per workload); True only if the user asks for a single all-optional kernel |
| `sparse_parameters` | off, on (only if the model declares sparse parameters) |

That is 2 x 2-3 x 1-2 x 1-2 = 4 to 24 candidates per workload. Axes that
cannot change the output for a model are pruned before generation: compare
the generated text, and time each distinct text once (identical text =
identical kernel).

### 4.2 Extended candidates [S]

Opt-in (`extended=True`): individual toggles of the low-level loop options
of `fastest` (`fuse_*`, `split_acc`, `keep_libm_tables` ...), one at a time
from the best base candidate (coordinate descent, not the full cross
product: `fastest` has ~44 settings).

### 4.3 Validity

Candidate combinations that `CodegenOptions.validate()` would downgrade with
a warning are skipped (they duplicate another candidate).

### 4.4 Compiler flag sets [S]

`flags` may be a list of flag sets (`[("-O3",), ("-O3", "-fstack-arrays")]`);
each candidate is then compiled and timed once per flag set, and the report
shows the best kernel per flag set. The tuner never picks flags for the user
silently: the result records the flag set the winner was timed with, and
generation from a saved choice warns if the caller's flags differ (when
known).

## 5. Inputs: representative values

Timing and the numerical check need values for every input argument.

- A JSON file maps parameter and variable names to values, per `nc`
  (explicit arrays) or as generators:
  `{"T": 350.0, "V": {"expr": "2*sum(n)"}, "n": {"expr": "0.3+0.1*i"},
  "tc": {"expr": "400+20*i"}, "v": {"sparse": {"groups_per_component": [1, 3],
  "ng": 10, "value": [1, 4]}}}`. Expressions are evaluated in Fortran (the
  driver) with `i`, `j`, `k` and `nc` in scope, as in
  `tests/model_harness.driver` [R].
- Without a file, defaults from `tests/model_harness.driver` (positive,
  smooth values). The report warns that the defaults may not be
  representative; if any output is NaN or Inf the candidate is discarded
  and the report says the inputs are out of domain (seen with CMR and the
  defaults [M: eb18435 pin validation]).
- `.model` extension [Q]: a `tune` section with the same content, so the
  values travel with the model.

## 6. Measurement protocol

Same method as the campaign measurements (V6-V8):

- One executable per candidate and flag set: the kernel plus a generated
  driver that sets the inputs, calls the kernel `reps` times (varying `T`
  by 1e-9 per call so the call cannot be hoisted), and prints the time per
  call and a checksum of the outputs. The checksum must be written so the
  compiler cannot drop the calls: a nrtl kernel timed at 0.03 ns per call
  showed that the calls had been optimised away [M: V8]. Reject a candidate
  whose time per call is below a sanity floor (e.g. 1 ns).
- `reps` calibrated per (workload, `nc`) so one sample lasts ~40 ms.
- Samples: interleave all candidates within each sample (rotating the
  order), at least 7 samples; keep min and median.
- Pinning: `taskset -c <core>` on Linux when `core` is given; otherwise
  unpinned, and the report says so. Record `os.getloadavg()` before and
  after; warn above 1.0 per core in use.
- Score per workload: the geometric mean over the requested `nc` of
  min-time ratios against the `standard` default candidate, with optional
  weights per `nc` (`nc={2: 1, 10: 2, 50: 1}`). Ties within 2% go to the
  simpler candidate (fewer non-default options), so noise never adds
  options.
- Generation runs in parallel (`jobs`), timing serially. Generation of
  heavy models takes minutes per candidate [M: PC-SAFT fastest]; cache
  generated text by (model hash, options) under a cache directory
  (`$XDG_CACHE_HOME/thermodiff/tune` or `cache_dir=`).
- `budget_s`: stop launching new candidates when exceeded; report what was
  not measured.

## 7. Correctness and outputs

- Reference: the `standard` default candidate, compiled with the same
  flags. Every candidate's outputs must match it at `rel = 1e-12`, with an
  absolute floor of 1e-13 times the largest value of each output (as in
  `tests/test_sparse_parameters.py` [R]); otherwise the candidate is
  discarded and listed in the report with its worst difference.
- Result file (`result.save`): JSON with the thermodiff version, the model
  file hash, the workloads, and per workload the winning `CodegenOptions`
  (as a dict), plus the compiler version, flags, machine (`platform.uname`),
  date and score. Loading it with a different model hash or thermodiff
  version warns but proceeds (the options are still valid inputs).
- Report (Markdown; HTML optional): a summary table (winner per workload,
  speed-up over `standard` and over `fastest` per `nc`), then the full
  candidate x `nc` table (min/median time, ratio, status: ok / discarded
  with reason), the environment (compiler, flags, CPU, load, pinning), and
  the input values used.

## 8. Code layout [S]

- `thermodiff/tune/` (new package, strict mypy like `codegen/`):
  `candidates.py` (§4), `driver.py` (Fortran driver generation, promoted
  and generalised from `tests/model_harness.driver` and the campaign
  scripts: rank-3 parameters, continuation lines in signatures, extra
  dimensions such as `ng` — the campaign drivers failed on these [M: V8]),
  `bench.py` (§6), `report.py`, `api.py` (`tune_model`, `TuneResult`).
- CLI subcommand in `thermodiff/model/cli.py`.
- No new runtime dependencies (numpy is already one; the compiler is
  external and found as in `tests/fortran_tools.find_gfortran` [R]).
- Tests: candidate pruning and validity (no compiler); an end-to-end tune of
  a tiny model with 2-3 candidates and `nc=(2,)` marked `slow` and skipped
  without gfortran; result save/load round trip; report rendering; the
  driver on rank-1/2/3 parameters and an extra dimension.

## 9. Open questions [Q]

1. Name: `preset="greedy"` vs a `tune` entry point only (§3).
2. Default `nc` and weights when the user gives none (suggestion: 2, 10, 50
   with equal weights).
3. Should the tuner also compare against a hand-written reference kernel if
   the user provides one (yaeos), and put the ratio in the report?
4. Keep the `.model` `tune` section for input values (§5)?
5. Is the workload dispatcher part of the timed unit (end-to-end call cost)
   or only the fixed kernels? The fused-chain measurements showed the
   dispatcher costs 7-15 ns [M: W21].
