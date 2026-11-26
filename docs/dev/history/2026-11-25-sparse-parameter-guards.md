---
title: Sparse-parameter guards speed up sparse UNIFAC 1.15-1.8x, slow dense data
date: 2026-11-25
type: improvement
status: resolved
area: [codegen/sparse, options, model]
links: [PR #24, docs/source/fortran.rst]
---

## Summary
`sparse_parameters=("v",)` (or `v nc ng sparse` in a `.model` file) enables
the `sparse_guards` pass: expensive reduction terms carrying `v` go under
`if (v(l,k) /= 0)`, tables read only by such terms are skipped where `v = 0`,
and tables of the form `f·v` are zero-filled there. It pays on sparse group
matrices and costs 6-20 % on dense ones, so no preset enables it.

## Context
Suggested by the yaeos session: UNIFAC group-count matrices have 1-3 nonzero
groups per component out of ng.

## Evidence
UNIFAC, ng = 10, `optional_derivatives=False`, gfortran `-O2 -fstack-arrays`,
9 interleaved samples, one pinned core; speed-up = unguarded / guarded time.

| v density | ge | Ge + Gen | full set |
|---|---|---|---|
| 1-3 groups per component (nc 5-20) | 1.56-1.84x | 1.24-1.44x | 1.15-1.23x |
| ~50 % nonzero | 1.11-1.31x | 1.04-1.15x | 1.01-1.06x |
| dense | 0.92-0.93x | 0.89-0.91x | 0.91-0.92x (standard; 0.80-0.94x over both presets) |

- **[M]** With the guards, yaeos's UNIFAC built from thermodiff kernels beat
  the hand-written one everywhere (worst case 1.25x faster; yaeos session).
- **[R]** Guards are emitted only when the skipped work has a transcendental
  call, a non-integer power or a nested loop (an operation-count guard, rule 7
  of AGENTS.md).
- **[R]** Where a table is zero-filled the result is 0 even if the unguarded
  `f` would be NaN or Inf; documented.

## Outcome
`thermodiff/codegen/sparse.py`. Tests (`tests/test_sparse_parameters.py`)
fill every local array with NaN on entry, so reading a skipped entry fails;
guarded vs unguarded at rtol 1e-13 plus an absolute floor of 1e-13 times the
largest value of each output.

## Follow-ups
Idea not pursued: emit both kernels and choose at run time by the density of
`v` (one count per call, or once per model setup).
