# Developer notes

Working documents for thermodiff developers. They are not part of the Sphinx
documentation (`docs/source/`) and are not shipped in the package.

| File | What it is |
|---|---|
| `perf_log.md` | Generation-time measurements: protocol, baselines, per-change results and profiles. Add new measurements here. |
| `refactor-perf-arch-report.md` | Report of the `refactor/perf-arch` work (performance and architecture of the code generator): acceptance criteria, before/after timings, declared output changes, decisions not taken, remaining debt. |
| `autotune-spec.md` | Specification of an autotuning ("greedy") mode that generates candidate kernels, checks and times them on the user's machine and keeps the fastest (nothing implemented yet). |
| `third-derivatives-plan.md` | Proposal for third derivatives in the code generator (nothing implemented yet), with open questions. |

The code generator design itself is documented in
`docs/source/architecture.md`; the agent and contributor guide is
`AGENTS.md` at the repository root.
