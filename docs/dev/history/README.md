# Project history

A knowledge base of what was improved, measured, decided, found broken or
tried and rejected in thermodiff, with the evidence. One file per finding;
this page is the index. Read the **Open** and **Dead ends** lists before
starting work in an area.

## How to add an entry

1. Copy `_template.md` to `YYYY-MM-DD-short-slug.md` (date of the finding,
   lower-case slug with hyphens). Dated names do not collide between
   parallel branches.
2. Fill the front matter:
   - `type`: `improvement` (made something better), `bug` (something wrong:
     symptom, root cause, fix), `measurement` (a benchmark that informs a
     decision), `decision` (a choice between options and why), `dead-end`
     (tried and rejected, so nobody retries it blindly), `gotcha` (a tooling,
     CI or environment trap).
   - `status`: `open`, `resolved`, `wontfix`, or
     `superseded-by: <file>.md`.
   - `area` and `links` (PRs, issues, commits, docs pages).
3. Write the sections. Give numbers with their conditions (model, workload,
   nc, compiler and flags, pinning, samples, min or median) and label claims
   **[M]** measured, **[R]** read in the source, **[S]** suggestion.
4. Add a row to the table below (newest first) and, if it applies, to
   **Open** or **Dead ends**, in the same commit as the entry.
   `tests/test_dev_history.py` checks that every entry is indexed with the
   type and status of its front matter.

Never rewrite the evidence of an old entry: when it stops being true, add a
new entry and set the old one's status to `superseded-by: <new file>`.
Generation-time measurements belong in `../perf_log.md`, designs and specs
in their own files under `docs/dev/`, user guidance in `docs/source/`; an
entry summarises and links them.

## Open

- [Why is work_array 25-35% faster on the full UNIFAC output set?](2026-11-25-work-array-unifac-full.md) (measurement)
- [Harness default inputs give NaN for the CMR model](2026-11-25-cmr-nan-default-inputs.md) (gotcha)
- [Autotuning ("greedy") mode specified, not implemented](2026-11-25-autotune-spec.md) (decision)

## Dead ends

- [work_array in the fastest preset rejected (up to 16% slower on PR kernels)](2026-11-25-work-array-not-in-fastest.md)

## All entries

| Date | Entry | Type | Status | Area |
|---|---|---|---|---|
| 2026-11-25 | [yaeos wrapper broke when a workload did not request f](2026-11-25-yaeos-wrapper-scratch-outputs.md) | bug | resolved | yaeos |
| 2026-11-25 | [Why is work_array 25-35% faster on the full UNIFAC output set?](2026-11-25-work-array-unifac-full.md) | measurement | open | codegen/storage, codegen/loops |
| 2026-11-25 | [work_array in the fastest preset rejected (up to 16% slower on PR kernels)](2026-11-25-work-array-not-in-fastest.md) | dead-end | wontfix | codegen/storage, presets |
| 2026-11-25 | [Placeholder, case-clash and extra-dimension bugs fixed (issues #6-#17)](2026-11-25-upstream-issues-6-17.md) | bug | resolved | codegen, chain, resolve, model, diffplz |
| 2026-11-25 | [Sparse-parameter guards speed up sparse UNIFAC 1.15-1.8x, slow dense data](2026-11-25-sparse-parameter-guards.md) | improvement | resolved | codegen/sparse, options, model |
| 2026-11-25 | [Linux CI fails in setup-fortran when the ubuntu-toolchain-r PPA answers 503](2026-11-25-setup-fortran-ppa-503.md) | gotcha | resolved | CI |
| 2026-11-25 | [share_elementwise now tables libm calls over rank-2 temporaries](2026-11-25-rank2-libm-table.md) | improvement | resolved | codegen/loops |
| 2026-11-25 | [max_nc helps a few percent near the real maximum nc and hurts when large](2026-11-25-max-nc-sweep.md) | measurement | resolved | codegen/storage |
| 2026-11-25 | [work_array and max_nc stay as opt-in options, off in every preset](2026-11-25-keep-work-array-and-max-nc-opt-in.md) | decision | resolved | codegen/storage, options, presets |
| 2026-11-25 | [False expression-argument warning on B(n) silently disabled nested_nodes](2026-11-25-indexedbase-false-warning.md) | bug | resolved | resolve, options, codegen/loops |
| 2026-11-25 | [gfortran puts kernel temporaries on the heap; -fstack-arrays is ~10% faster](2026-11-25-gfortran-heap-temporaries.md) | measurement | resolved | codegen/storage, docs, yaeos |
| 2026-11-25 | [Fused scalar-core + chain workload kernels help only behind a dispatcher](2026-11-25-fused-core-chain-kernels.md) | measurement | resolved | codegen/core_chain, workload, yaeos |
| 2026-11-25 | [Harness default inputs give NaN for the CMR model](2026-11-25-cmr-nan-default-inputs.md) | gotcha | open | tests, benchmarks |
| 2026-11-25 | [chain_factor_sums enabled in the standard and fastest presets](2026-11-25-chain-factor-sums-default.md) | improvement | resolved | codegen/chain, options, presets |
| 2026-11-25 | [Kernel benchmarks can be optimised away (0.03 ns per call)](2026-11-25-benchmark-calls-optimised-away.md) | gotcha | resolved | benchmarks |
| 2026-11-25 | [Autotuning ("greedy") mode specified, not implemented](2026-11-25-autotune-spec.md) | decision | open | codegen, benchmarks, tools |
| 2026-11-25 | [Derivatives of max, min and abs printed invalid or wrong Fortran](2026-11-25-abs-max-min-derivatives.md) | bug | resolved | codegen/common, resolve, render |
| 2026-11-21 | [refactor/perf-arch made generation 70% faster and split fortran.py into codegen/](2026-11-21-refactor-perf-arch.md) | improvement | resolved | codegen, chain, options, tests |
