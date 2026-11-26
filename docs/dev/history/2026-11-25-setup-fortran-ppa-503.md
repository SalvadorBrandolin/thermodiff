---
title: Linux CI fails in setup-fortran when the ubuntu-toolchain-r PPA answers 503
date: 2026-11-25
type: gotcha
status: resolved
area: [CI]
links: [PR #25, PR #18, PR #22, .github/workflows/ci.yml]
---

## Summary
`fortran-lang/setup-fortran` installs gcc on Linux from the
`ubuntu-toolchain-r` PPA. When the PPA answers `503 Service Unavailable`,
`test-linux` fails before any test runs (seen on PRs #18 and #22; macOS and
Windows were fine). A second trap: tox passes `PATH` to the tests but not
`FC`, so a compiler exported only as `FC` is invisible to them.

## Evidence
- **[M]** CI logs of PRs #18 and #22: the `Set up gfortran` step fails with
  503 from the PPA.
- **[R]** `tox.ini` has no `passenv`: tox 4 passes `PATH` by default but not
  `FC`. `tests/fortran_tools.py` honours `FC`, then looks for `gfortran` on
  `PATH`.

## Outcome
Linux job: `continue-on-error` on setup-fortran; a fallback step removes the
PPA source, uses the runner's `gfortran-14/13/12` or installs Ubuntu's
`gfortran` with retries, links it as `gfortran` into a directory on
`GITHUB_PATH` and sets `FC`; a `Check gfortran` step compiles and runs a tiny
program so the job fails loudly instead of skipping the Fortran tests. Both
branches were run locally from the parsed workflow with `sudo` stubbed.

## Follow-ups
Failed jobs on PRs opened before this fix need a rerun once it is merged.
