---
title: work_array and max_nc stay as opt-in options, off in every preset
date: 2026-11-25
type: decision
status: resolved
area: [codegen/storage, options, presets]
links: [PR #26, PR #27, issue #10]
---

## Summary
Neither removing the storage options nor enabling them by default survived
measurement. Both stay, off in every preset; `-fstack-arrays` is a
documentation recommendation; choosing per model is left to a future
autotuning mode.

## Context
The maintainer first preferred removing both options if `-fstack-arrays`
was enough ("unnecessary boilerplate"), then asked to keep both and enable
them in `fastest`. Measurements decided against both.

## Evidence
- **[M]** Removal rule (stack build within 2 % of the best option
  everywhere) failed in 139 of 215 cases:
  [2026-11-25-gfortran-heap-temporaries.md](2026-11-25-gfortran-heap-temporaries.md).
- **[M]** `work_array` in `fastest` failed its rule (no case > 2 % slower):
  [2026-11-25-work-array-not-in-fastest.md](2026-11-25-work-array-not-in-fastest.md).
- **[M]** `max_nc` has no safe default value:
  [2026-11-25-max-nc-sweep.md](2026-11-25-max-nc-sweep.md).

## Outcome
Maintainer decision: "keep both off as default, and give an extra option
'greedy' that tries all possible combinations ..., benchs them and keeps the
fastest". Docs in PR #26; the autotuner is only specified
(`docs/dev/autotune-spec.md`, PR #27), not a priority.
