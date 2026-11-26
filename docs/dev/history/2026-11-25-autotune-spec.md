---
title: Autotuning ("greedy") mode specified, not implemented
date: 2026-11-25
type: decision
status: open
area: [codegen, benchmarks, tools]
links: [PR #27, docs/dev/autotune-spec.md]
---

## Summary
Because no fixed preset is fastest for every model, output set, nc and
compiler flag set (see the storage, sparse and chain-rule entries of
2026-11-25), the maintainer proposed a mode that generates the candidate
option sets, benchmarks them on the user's machine, keeps the fastest and
writes a report. It is specified in `docs/dev/autotune-spec.md`; not a
priority.

## Outcome
Spec only. Its section 9 lists the open questions (name `preset="greedy"`
vs a `tune` entry point, default nc and weights, comparison against a
hand-written kernel, a `.model` `tune` section, whether the dispatcher is
timed).
