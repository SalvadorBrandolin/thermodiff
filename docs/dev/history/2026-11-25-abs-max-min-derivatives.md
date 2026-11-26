---
title: Derivatives of max, min and abs printed invalid or wrong Fortran
date: 2026-11-25
type: bug
status: resolved
area: [codegen/common, resolve, render]
links: [PR #22, issue #19]
---

## Summary
Models using `max`, `min` or `abs` produced Fortran that did not compile or
computed the wrong derivative: the Heaviside step was printed with integer
literals, `max`/`min` mixed integer and real arguments, and `abs` was
differentiated through `re(u)`/`im(u)` because model symbols are not declared
real.

## Context
Found while composing monolithic yaeos models that clip values.

## Evidence
- **[R]** SymPy differentiates `Abs(u)` of a complex symbol through `re`/`im`
  and `sign`; `fcode` has no valid output for those.
- **[R]** Declaring the symbols real would fix `abs` but changes SymPy's
  simplifications elsewhere, and with them the generated code of every model
  (rejected for that reason).
- **[M]** New tests compile and compare dT, dT2, dn, dndT with SymPy for
  `max`, `min`, `abs` in minimal, standard and fastest, on both sides of the
  kink.

## Outcome
A Fortran printer in `codegen/common.py` (also used by `render.py`) prints
the step as `merge(0.0d0, merge(0.5d0, 1.0d0, ...), ...)` (H(0) = 1/2 as in
SymPy) and `DiracDelta` as `0`; `max`/`min` promote integer literals; the
resolver rewrites `re(u) -> u`, `im(u) -> 0` and `Derivative(sign(u)) -> 0`
before printing.

## Follow-ups
Limitation kept: exactly at the kink, `abs` follows SymPy (sign(0) = 0, delta
terms dropped).
