---
title: chain_factor_sums enabled in the standard and fastest presets
date: 2026-11-25
type: improvement
status: resolved
area: [codegen/chain, options, presets]
links: [PR #21, PR #23, commit 34e2908, commit eb18435]
---

## Summary
With `chain_rule`, factors that depend only on the free indices stayed inside
reductions after `fuse_temps` (e.g. `qs(i)*qs(j)*w(k)*tau(i,k)*tau(j,k)`:
four multiplications per iteration instead of two). `chain_factor_sums`
factors them out. Added opt-in in PR #21, then enabled in `standard` and
`fastest` in PR #23 after measuring that it slows nothing.

## Context
Seen in UNIQUAC kernels with a placeholder `tau(l, k, T)` generated for
yaeos. Maintainer rule: "If there is no negative impact on performance I do
not see why it should not be a default".

## Evidence
gfortran `-O3 -march=native -funroll-loops`, one pinned core, interleaved
A/B, >= 9 samples; ratio = on/off by min time; +-2 % is noise.

| Cases | nc = 2 | nc = 5 | nc = 20 | nc = 50 |
|---|---|---|---|---|
| PC-SAFT, the 29 golden cases it changes | 0.93-1.02 | 0.93-1.01 | 0.90-1.02 | 0.85-1.00 |
| UNIQUAC-like | 1.04-1.06 | 0.90-0.98 | 0.95-0.96 | 0.86-0.96 |
| NRTL-like | 0.99-1.01 | 0.99-1.00 | 1.00 | 1.00 |

- **[M]** UNIQUAC-like at nc = 2: about +10 ns on a 150 ns call; faster at
  every larger size.
- **[M]** Old and new PC-SAFT kernels agree to 2.3e-15 (golden) and 3.2e-15
  (text pins).

## Outcome
- Declared output change: 29 PC-SAFT external golden cases regenerated;
  `tests/data/variant_settings.json` gains the setting at bench stage c6m
  (e8), so standard == e8 and fastest == p33 still hold.
- Six text-pin digests in `tests/codegen` updated; each old digest reproduces
  with `chain_factor_sums=False` (the procedure to follow for any pin
  update).
- `to_fortran` called directly and `CodegenOptions()` keep it off.
