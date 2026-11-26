---
title: Fused scalar-core + chain workload kernels help only behind a dispatcher
date: 2026-11-25
type: measurement
status: resolved
area: [codegen/core_chain, workload, yaeos]
links: [PR #24, PR #21, docs/source/fortran.rst]
---

## Summary
`scalar_chain_workload_module` (opt-in) emits, per workload, one subroutine
that computes the scalar-core derivatives as locals and applies the chain
rule for exactly its outputs. The fused kernel is 0.87-1.04x of calling core
and chain kernels directly; the gain comes from replacing a dispatched core
plus an optional chain routine. A caller that already knows its output set
should call the fixed kernels directly.

## Context
Suggested by the yaeos session after PR #21 added `scalar_chain_routine`
(scalar-core design of issue #13).

## Evidence
Cubic scalar core (B, D, E, S), `fastest`, gfortran -O3, 9 interleaved
samples, one pinned core, ns per call (min). Other agents loaded the machine:
compare within a row only.

| workload | nc | fused, dispatched | core dispatcher + optional chain | separate fixed, direct | fused, direct |
|---|---|---|---|---|---|
| fug (dv, dni) | 2 | 33 | 37 | 26 | 26 |
| fug | 50 | 51 | 52 | 42 | 44 |
| jac (7 outputs) | 2 | 78 | 97 | 75 | 63 |
| jac | 50 | 143 | 167 | 140 | 128 |

- **[M]** The dispatcher (19 `present()` tests, `select case`, forwarding)
  costs 7 ns for fug and 15 ns for jac.
- **[M]** yaeos session: 1-6 % whole-call gain in its models.

## Outcome
Opt-in entry point in `thermodiff/codegen/core_chain_workload.py`; correctness
against the separate path at rel 1e-13 in
`tests/test_core_chain_workload.py`. Caller contracts (`f` must be present to
reach a workload kernel; every input of the selected workload must be
present) are in `docs/source/fortran.rst`.
