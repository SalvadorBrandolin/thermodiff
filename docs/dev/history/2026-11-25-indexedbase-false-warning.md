---
title: False expression-argument warning on B(n) silently disabled nested_nodes
date: 2026-11-25
type: bug
status: resolved
area: [resolve, options, codegen/loops]
links: [PR #21, commit e1b3d15, issue #13]
---

## Summary
The check added for #13 (a placeholder with an expression argument) treated
the mole-number vector `n`, an `IndexedBase`, as an expression. It warned on
the documented `B(n)` / `D(n, T)` pattern and, under `fastest`, also turned
off the `nested_nodes` pass for those models without saying so.

## Context
Found by the yaeos session while generating scalar-core cubic models with the
PR #18 branch.

## Evidence
- **[R]** The warning path and the pass gate shared the same predicate, so a
  false positive changed generated code, not just the log.

## Outcome
The predicate accepts `IndexedBase` arguments. A related check in the same
PR: a `variables=` name that clashes by case with `n`/`T`/`V`/`P` (e.g. `N`)
raises an error that explains Fortran's case-insensitivity.

## Follow-ups
Lesson: a warning that also disables something needs a test showing the
documented pattern stays warning-free and keeps the pass on.
