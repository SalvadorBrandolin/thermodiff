# td_bench

Benchmark and golden generation hooks for thermodiff. `hooks.py` defines the
generation variants (`VARIANTS`, `EXTENDED_VARIANTS`, `STAGE_VARIANTS`), the
derivative workloads (`WORKLOADS`) and `generate_workload`, which generates
the Fortran for one (workload, variant) pair and returns timing and size
metadata.

`registry.py` is the single source of truth for the variants: one ordered
tuple of `stage(key, base, **delta)` entries (each names its parent and only
the settings it changes) plus the immutable alias map (`e*`/`p*` to the
legacy `a*`/`c*`/`it*` keys). `hooks.py` builds the public tables once from
it; nothing is mutated at import time. To add a variant, append a `stage`.

Usage:

- `tools/` scripts (`golden.py`, `check_golden.py`, `profile_case.py`) put the
  repository root on `sys.path` and run `from benchmarks.td_bench import ...`.
- The tests import it the same way; `pythonpath = ["."]` in
  `[tool.pytest.ini_options]` makes it importable under pytest and xdist.

This directory is not part of the installed package: `import thermodiff` does
not provide it, and there is no compatibility shim at
`thermodiff.bench_hooks`.
