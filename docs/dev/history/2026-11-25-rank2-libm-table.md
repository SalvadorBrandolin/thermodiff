---
title: share_elementwise now tables libm calls over rank-2 temporaries
date: 2026-11-25
type: improvement
status: resolved
area: [codegen/loops]
links: [PR #21]
---

## Summary
In `fastest`, `share_elementwise` turned a one-argument libm call read at
several sites into a table only for rank-1 temporaries. It now also does it
for rank-2 temporaries. The UNIFAC `lng` kernel computed
`log(t24(l, k))` in two separate nests: 2·nc·ng logarithms per call, now
nc·ng.

## Context
Found by reading the UNIFAC kernels generated for yaeos.

## Evidence
- **[R]** Generated UNIFAC `lng` kernel, before and after: one log table
  instead of two calls per entry.
- **[M]** 581 golden cases identical (no golden case has the pattern).

## Outcome
Enabled wherever `share_elementwise` is on (`fastest`). Tested numerically
in `tests/test_shared_pair_libm.py`.
