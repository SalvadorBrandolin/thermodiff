---
title: Placeholder, case-clash and extra-dimension bugs fixed (issues #6-#17)
date: 2026-11-25
type: bug
status: resolved
area: [codegen, chain, resolve, model, diffplz]
links: [PR #18, issue #6, issue #7, issue #8, issue #16, issue #17]
---

## Summary
Issues #6-#17 of the upstream tracker were fixed together in PR #18: wrong or
non-compiling code for indexed placeholders under `chain_rule`, Fortran
case-insensitivity clashes, sums over `i`/`j`, and extra dimensions such as
`ng`. The same PR added the features asked for in #9-#15 (real kind,
`work_array`/`max_nc`, workload kernels, `.model` functions, scalar
variables, symmetric placeholders, fixed-`nc` copies).

## Context
The issues were reported while building yaeos models (UNIFAC, scalar-core
cubic EoS) from thermodiff kernels.

## Evidence
- **[R]** #6: chain-rule component definitions bypassed the placeholder
  resolver, so `tau(l, k, T)` was not resolved. Side bug: with `fastest`, an
  outer function that is itself a chain node leaked `_S0` into the signature
  and never computed `f`.
- **[R]** #7: Fortran is case-insensitive; `r` and `R` became one variable.
  One rename map now covers signature, declarations and body (`r` -> `r_1`);
  clashes among outputs, module symbols or thermodynamic variables raise
  `ValueError`.
- **[R]** #16: bound indices named `i`/`j` collided with the output indices;
  `DiffPlz` renames them, a free `i`/`j` raises.
- **[R]** #17: passes assumed every index ran over `nc`; `codegen/extents.py`
  now carries the range of each index, and `nc`-only passes skip the others.
- **[R]** `from thermodiff import *` failed: `install_in_colab` was in
  `__all__` but undefined.
- **[M]** About 320 new tests, most compiling with `-fcheck=bounds`; each bug
  has a regression test that fails without its fix. 581 golden cases
  identical except #8.

## Outcome
- Declared output change (#8): placeholder derivatives are named by their
  variables (`dBdn`, `dBdndn`), so CSE and symmetry see equal copies;
  `canonical_placeholders=False` restores the old names.
- Review follow-ups in the same PR: dispatcher fallback when a rank-2 actual
  is larger than the explicit extent; generated helper names avoid user
  names.
