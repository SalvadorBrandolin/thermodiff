# Performance log

Purpose: record the serial wall time of symbolic generation (`parse`, `diff`,
`emit`) for representative cases, so that every change in the
`refactor/perf-arch` branch can be compared against the same measurement
protocol. Generation is measured with `thermodiff.bench_hooks.generate_workload`
and `time.perf_counter`, without any profiler. Profiles (cProfile) are kept
separate and are used only to locate hotspots; their times are inflated and
must not be used as absolute timings.

## How to measure

Run from the root of a worktree of the commit under test, with the shared
virtualenv (`PY=/home/ruther/docs/research/thermo/algos/thermodiff/.venv/bin/python`):

```bash
export PYTHONHASHSEED=0 PYTHONPATH=$PWD
# timing: one process per repetition set, no -j (bench.py, listed below)
taskset -c 0-4,6-10 $PY bench.py <model> <variant> <workload> <reps> <tag>
# profile (separate run, never used for timings)
taskset -c 0-4,6-10 $PY tools/profile_case.py <model> <variant> --workload <W> \
    --top 30 --out <file>.prof
# golden checks (gate and wall time)
taskset -c 0-4,6-10 $PY tools/check_golden.py --tier fast -j 8
taskset -c 0-4,6-10 $PY tools/check_golden.py --tier complete -j 8
```

`bench.py` (timing driver, prints one JSON line per repetition):

```python
import json, os, sys, tempfile, time
import thermodiff
from thermodiff.bench_hooks import generate_workload

model, variant, workload, reps, tag = sys.argv[1:6]
print(json.dumps({"thermodiff": thermodiff.__file__, "tag": tag}), flush=True)
out_dir = tempfile.gettempdir()
for rep in range(1, int(reps) + 1):
    start = time.perf_counter()
    res = generate_workload(model, workload, variant, out_dir, write=False)
    wall = time.perf_counter() - start
    print(json.dumps({"tag": tag, "rep": rep, "variant": variant,
        "workload": workload, "parse_s": res["parse_time_s"],
        "diff_s": res["diff_time_s"], "emit_s": res["emit_time_s"],
        "generation_s": res["generation_time_s"], "wall_s": wall}), flush=True)
```

Model paths: `tests/data/models/<name>/<name>.model` (pcsaft, srk, ...).
Variant `e0` has no indexed CSE (`array_cse=False`); `e2`/`e3` and `p22`/`p33`
do. Workloads: `w5c`, `w5d`, `w6` as listed in `thermodiff/bench_hooks.py`.

Measurement rules: slow case 1 repetition (repeat only if two candidates are
within 5 %); medium case 3 repetitions reported as min / median / max; smoke
case 5 repetitions reported as min / median. The first repetition of each
process pays cold-start cost in the `diff` phase (SymPy caches); it is kept in
the ranges and is visible in the per-repetition data.

## Machine notes (Wave 0 baseline)

- CPU: AMD Ryzen 5 5600X (1 socket, 6 cores, SMT, 12 logical CPUs).
- Cores used: logical CPUs 0-4 and 6-10 (`taskset -c 0-4,6-10`).
- Python 3.14.8 (`.venv`), SymPy 1.14.0. Kernel Linux 7.2.9-arch1-1.
- Load at start (uptime, 21:45): 1.88 / 2.11 / 3.90 (1, 5, 15 min).
- **Campaign running: YES, for the whole measurement window (21:46 to 23:11).**
  `campaign.py` (another project, pid 352450,
  `bench_v2/scripts/campaign.py --campaign met5 --epochs 5`) was running. Its
  CPU affinity is `0-4,6-10`, not CPU 5 alone as expected, so it shares every
  core measured here. All numbers below are flagged "campaign running" and are
  expected to be inflated and noisy relative to a quiet machine.
- Load during the golden `complete` run (-j 8): 1-minute load up to 8.6.

## Baseline (commit aae26e2)

Phase times are seconds from `generate_workload`. For multi-repetition rows the
phases are those of the minimum-total repetition.

| case | role | parse (s) | diff (s) | emit (s) | total (s) | repetitions | commit |
|------|------|-----------|----------|----------|-----------|-------------|--------|
| pcsaft e3 w6 | slow | 0.126 | 34.201 | 158.910 | 193.238 | 1 | aae26e2 |
| pcsaft e2 w6 | slow (check: within 5.3 % of e3, measured) | 0.128 | 34.338 | 148.487 | 182.953 | 1 | aae26e2 |
| pcsaft e2 w5d | medium | 0.087 | 10.014 | 54.063 | 64.164 (min), median 64.168, max 69.378 | 3 | aae26e2 |
| srk e0 w3 | smoke | 0.00573 | 0.00791 | 0.06119 | 0.07484 (min), median 0.07544, max 0.29798 | 5 | aae26e2 |

Notes on the rows:

- Slow case, extra sample: the unprofiled first generation inside the
  `profile_case.py` run gave parse 0.130 / diff 34.238 / emit 163.693 /
  generation 198.061 s, so the single slow timing has about 2.5 % spread.
- Medium case repetitions (total s): 69.378 (rep 1, cold `diff` 15.17 s), 64.164,
  64.168. The `diff` phase of the first repetition is about 5 s slower than the
  later ones in the same process.
- Smoke case repetitions (total s): 0.29798 (rep 1, cold), 0.07544, 0.07521,
  0.07633, 0.07484.
- Golden (commit aae26e2, campaign running): `--tier fast -j 8` OK, 112 cases
  identical, about 11 s wall. `--tier complete -j 8` OK, 581 cases identical,
  exit 0, about 575 s wall (T0.4 measured about 650 s with a noisier machine).
- Profiled totals: slow case 967.65 s (cProfile, about 5.0x the unprofiled
  total); medium case 334.20 s (about 5.2x).

### Case selection (medium case screening)

Medium case: 30-120 s serial generation AND the largest `_indexed_cse` share.
Screens with cProfile (share = `_indexed_cse` cumtime / profiled total):

| candidate | serial gen (s) | `_indexed_cse` share | outcome |
|-----------|----------------|----------------------|---------|
| pcsaft e0 w6 | 73.9 | 0 (no indexed CSE in e0) | rejected |
| pcsaft e0 w5c | 32.9 | 0 (no indexed CSE in e0) | rejected |
| pcsaft e0 w5d | 31.3 | 0 (no indexed CSE in e0) | rejected |
| pcsaft p22 w6 | 146.9 | 13.3 % | out of window |
| pcsaft e2 w5c | 78.3 | 22.2 % | in window, not chosen |
| pcsaft e2 w5d | 69.7 | 28.5 % | **chosen** |
| pcsaft e3 w5c | 84.7 (timing only, not profiled) | not measured | in window, not profiled |

The first three screens were chosen without checking that e0 has no indexed
CSE; the two e2 candidates were screened afterwards. Six cProfile screens were
run in total, above the 2-4 candidate guidance. Profile of the chosen case:
`.orchestration/scratch/T0.5/medium.prof` (not committed).

## Perfil base

### Slow case: pcsaft e3 w6, top 30 by cumtime (profiled total 967.65 s)

Run with cProfile, so times are inflated. `generate` and `generate_workload`
are the harness entry points.

| # | function (file:line) | ncalls | cumtime (s) | % of profiled total |
|---|----------------------|--------|-------------|---------------------|
| 1 | `generate (tools/profile_case.py:31)` | 1 | 967.65 | 100.0 % |
| 2 | `generate_workload (thermodiff/bench_hooks.py:1097)` | 1 | 967.65 | 100.0 % |
| 3 | `to_fortran (thermodiff/fortran.py:3910)` | 1 | 867.08 | 89.6 % |
| 4 | `__next__ (sympy/core/traversal.py:160)` | 259607556 | 576.75 | 59.6 % |
| 5 | `<built-in method builtins.next>` | 259607556 | 537.01 | 55.5 % |
| 6 | `atoms (sympy/core/basic.py:601)` | 5108601 | 513.11 | 53.0 % |
| 7 | `_preorder_traversal (sympy/core/traversal.py:117)` | 2280130252 | 492.52 | 50.9 % |
| 8 | `_common_scalar_cse (thermodiff/fortran.py:615)` | 1 | 448.34 | 46.3 % |
| 9 | `_free_indices (thermodiff/fortran.py:100)` | 1853883 | 392.64 | 40.6 % |
| 10 | `_indexed_cse (thermodiff/fortran.py:711)` | 1 | 298.08 | 30.8 % |
| 11 | `wrapper (sympy/core/cache.py:69)` | 8590110 | 154.61 | 16.0 % |
| 12 | `xreplace (sympy/core/basic.py:1299)` | 543920 | 136.19 | 14.1 % |
| 13 | `_xreplace (sympy/core/basic.py:1364)` | 38502513 | 136.00 | 14.1 % |
| 14 | `build_diff (thermodiff/model/emit.py:80)` | 1 | 100.12 | 10.3 % |
| 15 | `__init__ (thermodiff/diffplz.py:140)` | 1 | 99.96 | 10.3 % |
| 16 | `_compute (thermodiff/diffplz.py:170)` | 15 | 99.93 | 10.3 % |
| 17 | `diff_ni (thermodiff/diffplz.py:221)` | 4 | 98.37 | 10.2 % |
| 18 | `handle_sum_kronecker (thermodiff/core/kronecker_handling.py:15)` | 4 | 86.62 | 9.0 % |
| 19 | `subs (sympy/core/basic.py:965)` | 4232 | 83.26 | 8.6 % |
| 20 | `_subs (sympy/core/basic.py:1176)` | 10440 | 83.16 | 8.6 % |
| 21 | `fallback (sympy/core/basic.py:1248)` | 8448 | 83.15 | 8.6 % |
| 22 | `is_same (sympy/core/basic.py:2153)` | 48234 | 82.39 | 8.5 % |
| 23 | `<built-in method builtins.isinstance>` | 651288781 | 65.70 | 6.8 % |
| 24 | `__eq__ (sympy/core/basic.py:497)` | 35996646 | 57.82 | 6.0 % |
| 25 | `diff_dnidnj (thermodiff/diffplz.py:251)` | 1 | 47.42 | 4.9 % |
| 26 | `canonical (thermodiff/fortran.py:768)` | 309010 | 46.32 | 4.8 % |
| 27 | `_is_hoistable (thermodiff/fortran.py:698)` | 42617597 | 45.51 | 4.7 % |
| 28 | `postorder_traversal (sympy/core/traversal.py:250)` | 223487908 | 38.25 | 4.0 % |
| 29 | `args (sympy/core/basic.py:906)` | 341694647 | 33.37 | 3.4 % |
| 30 | `<built-in method builtins.hasattr>` | 275147348 | 27.53 | 2.8 % |

### Medium case: pcsaft e2 w5d, top 10 by cumtime (profiled total 334.20 s)

| # | function (file:line) | ncalls | cumtime (s) | % of profiled total |
|---|----------------------|--------|-------------|---------------------|
| 1 | `generate (tools/profile_case.py:31)` | 1 | 334.20 | 100.0 % |
| 2 | `generate_workload (thermodiff/bench_hooks.py:1097)` | 1 | 334.20 | 100.0 % |
| 3 | `to_fortran (thermodiff/fortran.py:3910)` | 1 | 282.77 | 84.6 % |
| 4 | `__next__ (sympy/core/traversal.py:160)` | 86216743 | 186.92 | 55.9 % |
| 5 | `<built-in method builtins.next>` | 86216743 | 173.85 | 52.0 % |
| 6 | `atoms (sympy/core/basic.py:601)` | 1758438 | 169.35 | 50.7 % |
| 7 | `_preorder_traversal (sympy/core/traversal.py:117)` | 748764494 | 159.60 | 47.8 % |
| 8 | `_common_scalar_cse (thermodiff/fortran.py:615)` | 1 | 146.18 | 43.7 % |
| 9 | `_free_indices (thermodiff/fortran.py:100)` | 629459 | 127.14 | 38.0 % |
| 10 | `_indexed_cse (thermodiff/fortran.py:711)` | 1 | 95.13 | 28.5 % |

## Change log

| date | commit | task | case | before | after | delta % | notes |
|------|--------|------|------|--------|-------|---------|-------|
