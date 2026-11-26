---
title: Kernel benchmarks can be optimised away (0.03 ns per call)
date: 2026-11-25
type: gotcha
status: resolved
area: [benchmarks]
links: [docs/dev/autotune-spec.md]
---

## Summary
In one kernel timing run an NRTL kernel built with `-fstack-arrays` reported
0.03 ns per call, i.e. ratios of thousands against the other build. The
compiler had removed the calls because the driver did not use their results
in a way it could not see through.

## Evidence
- **[M]** nrtl p33 g2 with `-fstack-arrays`: ratios 2537-366756 between two
  builds of the same kernel; the per-call time was below one clock cycle.

## Outcome
Rules for any timing driver (used since, and written into the autotuning
spec): vary an input on every call (e.g. `T` by 1e-9) so the call cannot be
hoisted, accumulate a checksum of all outputs and print it, and reject a
result below a sanity floor (about 1 ns per call). Treat any ratio far
outside 0.3-3x as an artefact until re-measured.
