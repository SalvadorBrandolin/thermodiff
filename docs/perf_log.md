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

Columns follow the logging rule of the refactor. Medium and slow rows use the
same serial protocol as the baseline (`bench.py`, `taskset -c 0-4,6-10`,
`PYTHONHASHSEED=0`). Antes/Después are total generation times (s). For the
medium case they are the minimum of 3 repetitions; the slow case has 1
repetition. "Antes" is the parent commit, or the re-measured base commit when
the parent differs only by tests (no change in `thermodiff/` or `tools/`).
Machine state for every row: campaign running (pid 352450), see Wave 1 interim.

| Commit | Paso | Cambio | Caso (modelo/variante/workload) | Antes (s) | Después (s) | Δ % | Fase dominante | Notas |
|--------|------|--------|---------------------------------|-----------|-------------|-----|----------------|-------|
| 401daf6 | 0.5 (drift) | none, re-measure of the base | medium pcsaft/e2/w5d | 64.164 (logged, aae26e2) | 66.183 | +3.15 % | emit (55.7 s) | Drift check. Reps 66.183 / 67.184 (median) / 69.374 (max). Earlier logged value is 3.1 % lower, so compare rows within this session only. |
| 401daf6 | 0.5 (drift) | none, re-measure of the base | slow pcsaft/e3/w6 | 193.238 (logged) | 192.705 | −0.28 % | emit (158.1 s) | 1 rep. Drift check for the slow case. |
| 813f9a5 | 1.1 | `subs` → `xreplace` in `handle_sum_kronecker` | medium pcsaft/e2/w5d | 66.183 (401daf6) | 58.112 | −12.19 % | emit (56.0 s); diff 10.43 → 2.00 s (−80.8 %) | Reps 58.112 / 58.547 (median) / 61.031 (max). Emit unchanged (+0.6 %). Profile: `handle_sum_kronecker` 86.6 s → 5.0 s and `subs` 83.3 s → 0.1 s (slow case, see Wave 1 interim). |
| 813f9a5 | 1.1 | `subs` → `xreplace` in `handle_sum_kronecker` | slow pcsaft/e3/w6 | 192.705 (401daf6) | 179.667 | −6.77 % | emit (161.0 s); diff 34.44 → 18.54 s (−46.2 %) | 1 rep. Δ > 5 %, so no repeat. Emit +2.9 s (+1.8 %) is not attributed to the change. |
| 0fd074d | 1.2a | memoise index sets in `_indexed_cse` | medium pcsaft/e2/w5d | 58.112 (813f9a5, same code as parent 7140e50) | 55.948 | −3.72 % | emit (53.9 s) | Reps 55.948 / 56.133 (median) / 59.425 (max). Diff unchanged (2.0 s). |
| fe7a992 | 1.2b | cache hoistability and canonical form per call | medium pcsaft/e2/w5d | 55.948 (0fd074d) | 66.181 | **+18.29 % (regression)** | emit (64.1 s, was 53.9 s) | Reproduced in a repeat block: 66.362 min, emit 64.4–64.6 s (3 reps). Cause not identified; not profiled. The next commit removes it again. |
| ca6ddc4 | 1.2c | reuse counting-phase entries in `_indexed_cse` | medium pcsaft/e2/w5d | 66.181 (fe7a992) | 53.946 | −18.49 % | emit (51.9 s) | Reps 53.946 / 54.215 (median) / 55.932 (max). Net vs 0fd074d: −3.58 %; net vs 401daf6: −18.49 %. Repeat block: 54.718 min (−17.55 % vs the repeat of fe7a992, 66.362). |
| ca6ddc4 | 1.2a–c | `_indexed_cse` memo, caches, reuse (cumulative) | slow pcsaft/e3/w6 | 179.667 (813f9a5) | 168.774 | −6.06 % | emit (149.5 s); diff 18.54 → 19.14 s (+3.2 %) | 1 rep. Δ > 5 %, so no repeat. The unprofiled first call inside the profile run gave 166.669 s (diff 17.49, emit 149.05), a second sample. |
| ca6ddc4 | 1.1 + 1.2a–c (cumulative) | cumulative vs baseline | slow pcsaft/e3/w6 | 192.705 (401daf6) | 168.774 | −12.42 % | emit (149.5 s); diff 34.44 → 19.14 s (−44.4 %) | Against the logged baseline 193.238 s: −12.66 %. Emit −5.46 %. |

## Wave 1 interim

Machine state at the start of this block (`uptime`, 01:14): load 1.30 / 1.89 /
3.04. Campaign: `campaign.py` (pid 352450, `--campaign met5`, another project)
was running for the whole window. Its affinity is `0-4,6-10`, not CPU 5 alone,
so it shares every measured core. Its CPU use was 0.0 % at the start of the
block. The load was about 2 during the block. All timings above are therefore
flagged as "campaign running". The medium case moved by +3 % against the
earlier baseline (drift), and the repeat block reproduced the medium numbers of fe7a992 and ca6ddc4 within
about 1.5 %, so the relative deltas are the reliable part.

Medium repetitions were run serially in one process per commit; the first
repetition of each process carries the cold `diff` cost (about 7 to 15 s).
Scripts and raw data: `.orchestration/scratch/T1.P1/` and
`.orchestration/logs/T1.P1-*.jsonl` (not committed).

### Top 10 at ca6ddc4 (slow pcsaft e3 w6, profiled total 811.78 s)

cProfile run, `tools/profile_case.py ... --top 30`, separate from the timings.
Raw profile: `.orchestration/scratch/T1.P1/slow-ca6ddc4.prof` (not committed).

| # | function (file:line) | ncalls | cumtime (s) | % of profiled total |
|---|----------------------|--------|-------------|---------------------|
| 1 | `generate (tools/profile_case.py:31)` | 1 | 811.78 | 100.00 % |
| 2 | `generate_workload (thermodiff/bench_hooks.py:1097)` | 1 | 811.78 | 100.00 % |
| 3 | `to_fortran (thermodiff/fortran.py:3998)` | 1 | 792.96 | 97.68 % |
| 4 | `__next__ (sympy/core/traversal.py:160)` | 225194571 | 487.78 | 60.09 % |
| 5 | `atoms (sympy/core/basic.py:601)` | 3822957 | 472.63 | 58.22 % |
| 6 | `<built-in method builtins.next>` | 225194571 | 454.13 | 55.94 % |
| 7 | `_common_scalar_cse (thermodiff/fortran.py:615)` | 1 | 439.65 | 54.16 % |
| 8 | `_preorder_traversal (sympy/core/traversal.py:117)` | 1940106763 | 417.20 | 51.39 % |
| 9 | `_free_indices (thermodiff/fortran.py:100)` | 1853883 | 384.99 | 47.43 % |
| 10 | `_indexed_cse (thermodiff/fortran.py:808)` | 1 | 234.38 | 28.87 % |

Shares by function, same profile, for comparison with the baseline: `_indexed_cse`
30.8 % → 28.9 %; `_common_scalar_cse` 46.3 % → 54.2 %; `handle_sum_kronecker` 9.0 % →
0.6 %; `subs` 8.6 % → 0.02 %. The share of `_common_scalar_cse` rose because the
total fell; its absolute time fell from 448.3 s to 439.6 s only.

New hotspots (top 3 after the Wave 1 commits, excluding the harness rows and
the sympy internals):

1. `_free_indices` 47.43 % (384.99 s, 1.85 M calls, the same call count as the
   baseline; not changed by T1.1 or T1.2).
2. `_hoistable_entries` 22.24 % (180.57 s, 190 calls), with `_cached_hoistable`
   12.97 % (105.25 s, 21.9 M calls) inside it. Before, this cost was spread
   over `_is_hoistable` (4.7 %) and `canonical` (4.8 %).
3. `xreplace` / `_xreplace` 12.61 % (102.37 s, 234 253 calls), and `__eq__`
   (sympy `basic.py:497`) 11.79 % (95.71 s), which comes from the sympy
   comparisons inside the same passes.

### Lane C input: pcsaft p32 w6 at ca6ddc4

cProfile run, `tools/profile_case.py pcsaft p32 --workload w6`. Profiled total
537.33 s. The same run's unprofiled generation was 127.64 s (parse 0.126,
diff 18.70, emit 108.82). Raw profile:
`.orchestration/scratch/T1.P1/p32w6-ca6ddc4.prof` (not committed).

Variant `p32` uses `scalar_chain="estimate"`, not `"cost"` as the task
assumed. `to_fortran` was called once and `_common_scalar_cse` was called
once, so on this path the scalar pipeline does not run twice.

| function (file:line) | ncalls | cumtime (s) | % of profiled total |
|----------------------|--------|-------------|---------------------|
| `chain.py`: `chain_rule_derivatives (chain.py:1046)` | 1 | 29.75 | 5.54 % |
| `chain.py`: `assemble (chain.py:1357)` | 1 | 27.52 | 5.12 % |
| `chain.py`: `factor_sums (chain.py:71)` | 835561 | 11.96 | 2.23 % |
| `chain.py`: `total_second (chain.py:1307)` | 6 | 9.52 | 1.77 % |
| `chain.py`: `F2_plain (chain.py:1279)` | 451 | 9.26 | 1.72 % |
| `_print_expr (fortran.py:369)` | 846 | 0.50 | 0.09 % |
| `_common_scalar_cse (fortran.py:615)` | 1 | 294.55 | 54.82 % |
| `_free_indices (fortran.py:100)` | 1469520 | 188.29 | 35.04 % |
| `_indexed_cse (fortran.py:808)` | 1 | 71.93 | 13.39 % |
| `handle_sum_kronecker (core/kronecker_handling.py:15)` | 38 | 5.43 | 1.01 % |

The `chain.py` rows are nested (`assemble` is inside `chain_rule_derivatives`),
so they must not be added. The whole chain.py share is at most about 5.5 %, and
`_print_expr` is 0.09 %. `_free_indices` (35 %) and `_common_scalar_cse` (55 %)
dominate this case too.
