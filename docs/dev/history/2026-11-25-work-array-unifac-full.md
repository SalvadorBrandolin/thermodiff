---
title: Why is work_array 25-35% faster on the full UNIFAC output set?
date: 2026-11-25
type: measurement
status: open
area: [codegen/storage, codegen/loops]
links: [PR #26]
---

## Summary
On the full UNIFAC output set (Ge and all first and second derivatives),
`work_array` beats both the default heap temporaries and `-fstack-arrays` by
25-35 % for nc >= 5, with or without the flag. On every other kernel it is
neutral or slower once the flag is used. The cause is not known; it may
point to a cheaper fix in the generator for all users.

## Context
Seen in the V7 storage comparison.

## Evidence
UNIFAC (ng = 10, `chain_rule` on, `fastest`), gfortran `-O3 -funroll-loops`,
one pinned core, us per call (min):

| nc | heap | stack | work_array | max_nc=64 | work_array + stack |
|---|---|---|---|---|---|
| 2 | 1.13 | 0.935 | 0.864 | 1.20 | 0.813 |
| 10 | 3.96 | 3.59 | 2.71 | 3.86 | 2.68 |
| 50 | 28.5 | 28.4 | 21.0 | 28.6 | 21.2 |

- **[S]** Hypotheses: the internal `<name>_body` procedure sees explicit-shape
  dummies, so gfortran knows the strides and vectorises loops it does not
  vectorise on automatic arrays; or packing all temporaries in one array
  improves locality for the rank-3 (nc x nc x ng) temporaries.

## Follow-ups
Compare `-fopt-info-vec` output of the two builds, and try explicit-shape
dummies alone (without the single work array). If the first hypothesis
holds, the generator could emit explicit-shape internals by default.
