---
title: max_nc helps a few percent near the real maximum nc and hurts when large
date: 2026-11-25
type: measurement
status: resolved
area: [codegen/storage]
links: [PR #26, docs/source/fortran.rst]
---

## Summary
`max_nc=N` gives every temporary the compile-time extent N along nc. With N
of 10 to 100 kernels are typically 3-5 % faster than the stack build or
`work_array`; at N = 1000 the gain is gone, and N = 10000 breaks some
kernels (link failure, or 100x slower UNIFAC calls). No value is a safe
default.

## Context
Maintainer request: check N = 10, 100, 1000, 10000.

## Evidence
Same kernels and protocol as
[2026-11-25-gfortran-heap-temporaries.md](2026-11-25-gfortran-heap-temporaries.md)
(gfortran `-O3 -funroll-loops`, nc = 2-50, only nc <= N, 7 samples, one
pinned core). Median ratio of min times over all cases:

| N | vs heap (default) | vs `-fstack-arrays` | vs `work_array` |
|---|---|---|---|
| 10 | 0.74 | 0.95 | 0.94 |
| 100 | 0.86 | 0.97 | 0.96 |
| 1000 | 0.93 | 1.00 | 0.99 |
| 10000 | 0.93 | 1.01 | 1.00 |

- **[M]** Slower cases (ratio vs `work_array` / vs stack, min times), almost
  all at nc = 2: N = 10, up to 1.32 / 1.32 (NRTL g1, PR76 w1, w5a) and
  UNIFAC full 1.17-1.28 / 0.97-1.10; N = 100, PC-SAFT w5b 1.24 / 1.46,
  UNIFAC full 1.31-1.44 / 0.99-1.35, and one NRTL kernel (g4) at 2.4 / 2.6
  that was not re-measured.
- **[M]** N = 1000: one NRTL sample at 57x did not reproduce on a re-run
  (treat single huge outliers as noise until re-measured).
- **[M]** N = 10000: UNIFAC full takes ~110 us per call at nc = 2 (100x),
  because extents along `ng` stay automatic and every call allocates
  N x ng arrays; NRTL no longer links (`relocation truncated`: static data
  above 2 GB).
- **[R]** gfortran makes local arrays above 64 KB static (a rank-2 temporary
  with N above about 90), so such a kernel is not thread safe; the generator
  warns. N = 1000 needs 8-40 MB per kernel.

## Outcome
No default change; trade-offs documented in `docs/source/fortran.rst`. The
sweep script was a scratch file and is not kept.
