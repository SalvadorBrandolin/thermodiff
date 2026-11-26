---
title: Harness default inputs give NaN for the CMR model
date: 2026-11-25
type: gotcha
status: open
area: [tests, benchmarks]
links: [commit eb18435, tests/model_harness.py]
---

## Summary
The generic driver in `tests/model_harness.py` fills inputs with positive,
smooth default values. For the CMR model these are outside the physical
domain and every output is NaN, so a numerical comparison "old kernel vs new
kernel" passes vacuously or fails for the wrong reason.

## Context
Found while validating the six text-pin updates of the `chain_factor_sums`
change (commit eb18435).

## Evidence
- **[M]** CMR kernels run with the default driver inputs: NaN outputs; with
  physical inputs (set by hand) old and new kernels agree to 3.2e-15.

## Outcome
Validated with hand-set physical inputs. Not fixed in the harness.

## Follow-ups
Any numerical check should assert that outputs are finite before comparing.
Per-model input values (the `tune` section idea in
`docs/dev/autotune-spec.md`) would fix this for benchmarks too.
