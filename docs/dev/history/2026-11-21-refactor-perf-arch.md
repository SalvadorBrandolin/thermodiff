---
title: refactor/perf-arch made generation 70% faster and split fortran.py into codegen/
date: 2026-11-21
type: improvement
status: resolved
area: [codegen, chain, options, tests]
links: [docs/dev/refactor-perf-arch-report.md, docs/dev/perf_log.md, docs/source/architecture.md]
---

## Summary
The `refactor/perf-arch` work made code generation about 3.3x faster on the
slowest profiled case and replaced the 4371-line `fortran.py` by the
`codegen/` package (typed definition records, one options object, a table of
passes), with the generated Fortran byte-identical except for declared
changes. The full report is `docs/dev/refactor-perf-arch-report.md`; this
entry records the points later work keeps tripping over.

## Context
Generation of heavy models (PC-SAFT with second derivatives) took minutes,
and `_emit_code` had McCabe complexity 128, which made every pass risky to
change.

## Evidence
- **[M]** pcsaft e3 w6, cold fresh processes: 192.7 s -> 57.7 s median
  (-70.1 %); protocol and per-change numbers in `docs/dev/perf_log.md`.
- **[M]** 581 golden cases byte-identical at every merge; no golden file was
  regenerated.
- **[R]** Bug found on the way: the `_SYM_RANK3` module flag was set after
  `chain_rule_derivatives` had already read it, so a call saw the previous
  call's value and output could depend on generation order inside one
  process. Fixed by passing `sym_rank3` explicitly;
  `tests/test_reentrancy.py` checks interleaved calls.

## Outcome
- Declared output changes: the entry points (`compile_model`, `emit_module`,
  CLI, GUI) default to `preset="standard"`; `to_fortran` called directly
  keeps the minimal defaults; inert option combinations warn.
- Enforced since: functions in `codegen/` and `chain.py` stay within 50 code
  lines and complexity 10 (`tests/codegen/test_function_lengths.py`); no
  module-level caches on SymPy expressions.

## Follow-ups
Remaining debt is listed in section 6 of the report (string-keyed dict
returned by `chain_rule_derivatives`, `chain.py` outside strict mypy,
repeated literals, very large pin files).
