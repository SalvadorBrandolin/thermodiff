---
title: gfortran puts kernel temporaries on the heap; -fstack-arrays is ~10% faster
date: 2026-11-25
type: measurement
status: resolved
area: [codegen/storage, docs, yaeos]
links: [PR #26, issue #10, docs/source/fortran.rst]
---

## Summary
Generated kernels keep intermediate arrays as automatic arrays. gfortran
allocates those on the heap unless `-fstack-arrays` (or `-Ofast`) is given,
so each call pays malloc/free per temporary. On the `fastest` kernels the
flag is 10 % faster at the median and up to 2x at small nc, with a small
stack footprint. yaeos builds with fpm's release flags
`-O3 -funroll-loops`, which do not include it.

## Context
Task V7 asked whether the flag alone makes the `work_array` and `max_nc`
options (issue #10) unnecessary.

## Evidence
Kernels: `fastest` golden cases of PC-SAFT, CMR, PR76, NRTL, UNIQUAC, and
UNIFAC (ng = 10; Ge, Ge+Gen, full); gfortran `-O3 -funroll-loops`; nc = 2,
5, 10, 20, 50; variants interleaved, >= 7 samples, one pinned core.
Full table: 215 cases in the V7 timing log of the orchestration ledger
(not in the repository); summary:

- **[M]** stack / heap (B/A, min): median 0.90; 0.46-0.70 for the larger
  PC-SAFT workloads at nc = 2; about 0.95-1.0 at nc = 50. A few CMR kernels
  are 2-3 % slower at nc >= 20 (noise level).
- **[M]** Largest per-call automatic-array footprint at nc = 100: about
  420 KB over the measured models (8 MB default main-thread stack).
- **[M]** The flag was *not* enough to replace the other options: in 139 of
  215 cases the stack build was more than 2 % slower than the better of
  `work_array` and `max_nc=64` (typically 2-10 %; UNIFAC full 1.25-1.35x
  slower than `work_array`, see
  [2026-11-25-work-array-unifac-full.md](2026-11-25-work-array-unifac-full.md)).

## Outcome
Documentation recommendation, no code change: `docs/source/fortran.rst`,
"Compiler flags and storage of temporaries" (also `ulimit -s` /
`OMP_STACKSIZE`; ifx/ifort use the stack by default). See
[2026-11-25-keep-work-array-and-max-nc-opt-in.md](2026-11-25-keep-work-array-and-max-nc-opt-in.md).
