---
title: work_array in the fastest preset rejected (up to 16% slower on PR kernels)
date: 2026-11-25
type: dead-end
status: wontfix
area: [codegen/storage, presets]
links: [PR #26]
---

## Summary
Enabling `work_array` in `fastest` would speed up most kernels at small nc
but makes several cubic-EoS kernels reproducibly 3-16 % slower at nc = 50,
and with `-fstack-arrays` it can be much slower. Not enabled.

## Context
Task V8, after the yaeos session suggested `work_array` for UNIFAC-like
models. Rule: proceed only if no case is more than 2 % slower.

## Evidence
80 of 83 `fastest` (p33) golden cases change with the option. gfortran
`-O3 -march=native -funroll-loops`, no `-fstack-arrays`, 9 samples, one
pinned core; ratio = on/off (min); values agree to 8.4e-15.

| nc | median | min | max |
|---|---|---|---|
| 2 | 0.795 | 0.413 | 1.123 |
| 5 | 0.868 | 0.702 | 1.210 |
| 20 | 0.976 | 0.650 | 1.110 |
| 50 | 1.002 | 0.744 | 1.172 |

- **[M]** Re-measured (11 samples) and still slower: pr w5b nc = 50 1.16x,
  pr76 w3 nc = 50 1.13-1.14x, pr w5d nc = 50 1.09x, pr w5c 1.05x, srk w5b
  1.04x, uniquac g0 nc = 2 1.04x.
- **[M]** With `-fstack-arrays`: pr w5b 2.46x at nc = 2 and 1.33x at nc = 50;
  pr w5d 1.05-1.24x; pcsaft w5a neutral.

## Outcome
No change. See
[2026-11-25-keep-work-array-and-max-nc-opt-in.md](2026-11-25-keep-work-array-and-max-nc-opt-in.md).
