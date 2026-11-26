---
title: yaeos wrapper broke when a workload did not request f
date: 2026-11-25
type: bug
status: resolved
area: [yaeos]
links: [PR #22, issue #20]
---

## Summary
`yaeos_model_module` passed nothing for mandatory kernel outputs that a
workload did not request (for example `f` in an `lng` workload), so the
wrapper did not compile. Separately, an output-like argument from another
model family (`dfdV2` in a Ge kernel) was silently treated as a model
parameter.

## Context
Found while wiring activity-coefficient-only workloads in yaeos.

## Evidence
- **[M]** A test compiles and runs the wrapper against a minimal `GeModel`
  stub: the `lng` workload (only `dfdn`) returns the same `Gen` as the full
  workload.
- **[R]** Wrappers whose workloads request every output are byte-identical.

## Outcome
Each unrequested mandatory output gets a local scratch actual
(`<name>_scratch_`; scalars, or automatic arrays sized by `size(n)`).
Foreign output-like arguments raise `ValueError`.
