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
| daa02c8 | 1.7a base (drift) | none, re-measure of the ca6ddc4 code | medium pcsaft/e2/w5d | 53.946 (ca6ddc4, T1.P1) | 57.236 | +6.09 % | emit (55.1 s) | Reps 59.748 (max) / 57.242 (median) / 57.236 (min). Same code as ca6ddc4, so the +6 % is machine drift between sessions (campaign pid 352450 still running). Compare rows within this block only. |
| daa02c8 | 1.7a base (drift) | none, re-measure of the ca6ddc4 code | slow pcsaft/e3/w6 | 168.774 (ca6ddc4, T1.P1) | 168.040 | −0.43 % | emit (149.95 s); diff 17.96 s | 1 rep, same block as 8172049 (the campaign-condition check of T1.P2 spec item 2 was done: this is the same-block base). |
| fff7914 | 1.7a step 1 | `_free_indices` memo per call in `_common_scalar_cse` | medium pcsaft/e2/w5d | 57.236 (daa02c8) | 33.195 | −42.00 % | emit (31.1 s, was 55.1 s); diff 2.01 → 1.98 s | Reps 35.756 (max) / 33.510 (median) / 33.195 (min). Δ vs daa02c8 median: −41.46 %. |
| 8172049 | 1.7a step 2 | candidate eligibility and rank cache per call in `_common_scalar_cse` | medium pcsaft/e2/w5d | 57.236 (daa02c8) | 34.933 | −38.97 % | emit (32.9 s); diff 1.99 s | Reps 36.999 (max) / 34.962 (median) / 34.933 (min). Net vs daa02c8: −38.92 % (median). **Step 2 is +5.24 % (min) / +4.33 % (median) vs step 1 in the same block**, not explained; ranges overlap only partly. Not profiled separately. |
| 76e633c | 1.7a (tests + black) | none (test and style commit) | medium pcsaft/e2/w5d | 34.933 (8172049) | not measured | n/a | n/a | Spec: no timing for a test and black commit; timing equals 8172049. |
| 8172049 | 1.7a steps 1+2 (cumulative vs same-block base) | `_free_indices` memo + candidate caches | slow pcsaft/e3/w6 | 168.040 (daa02c8, same block) | 107.155 | **−36.23 %** | emit (88.92 s, was 149.95 s, −40.7 %); diff 17.96 → 18.11 s (+0.8 %) | 1 rep. Δ > 5 %, so no repeat. Parse 0.127 s. Cumulative vs the logged baseline 192.705 s (401daf6 re-measure): −44.39 %; vs 193.238 s: −44.55 %. |

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

## Wave 1 after T1.7a

Task T1.P2. Commits measured: daa02c8 (same code as ca6ddc4, drift check),
fff7914 (T1.7a step 1), 8172049 (T1.7a step 2). 76e633c is test and style only
and was not timed. Serial, `taskset -c 0-4,6-10`, `PYTHONHASHSEED=0`, one
process per repetition set, `bench.py` and `tools/profile_case.py` as in "How to
measure". The machine was run with the same protocol as the earlier waves.

Machine state (`uptime`): medium block start 03:01, load 1.58 / 2.08 / 3.17.
Slow block start 03:08, load 2.05 / 2.07 / 2.79. Profile block start 03:10,
load 2.01 / 2.04 / 2.70. Campaign: `campaign.py` (pid 352450, `--campaign met5`,
another project) still running for the whole window, affinity `0-4,6-10`, CPU
use 0.0 % at the start. **Campaign running, so the absolute times are flagged.**
Raw data: `.orchestration/logs/T1.P2-*.jsonl` (not committed).

### Medium pcsaft e2 w5d, 3 repetitions per commit (total s)

| commit | rep 1 | rep 2 | rep 3 | min | median | max | diff (min rep) | emit (min rep) |
|--------|-------|-------|-------|-----|--------|-----|----------------|----------------|
| daa02c8 | 59.748 | 57.242 | 57.236 | 57.236 | 57.242 | 59.748 | 2.005 | 55.139 |
| fff7914 | 35.756 | 33.510 | 33.195 | 33.195 | 33.510 | 35.756 | 1.977 | 31.129 |
| 8172049 | 36.999 | 34.933 | 34.962 | 34.933 | 34.962 | 36.999 | 1.992 | 32.852 |

Speed-up per commit (min and median of the same block):

| step | Δ % (min) | Δ % (median) | note |
|------|-----------|--------------|------|
| daa02c8 → fff7914 (T1.7a step 1) | −42.00 % | −41.46 % | emit 55.1 → 31.1 s |
| daa02c8 → 8172049 (steps 1+2) | −38.97 % | −38.92 % | emit 55.1 → 32.9 s |
| fff7914 → 8172049 (step 2 alone) | +5.24 % | +4.33 % | not explained; step 2 looks slower on the medium case |

The cold first repetition costs about 2.5 s more in `diff` (7.1 s vs 2.0 s), as
in the earlier waves.

### Slow pcsaft e3 w6, 1 repetition

| commit | parse | diff | emit | total (s) |
|--------|-------|------|------|-----------|
| daa02c8 (same block) | 0.128 | 17.958 | 149.953 | 168.040 |
| 8172049 | 0.127 | 18.106 | 88.921 | 107.155 |

Δ 8172049 vs daa02c8 (same block): **−36.23 %**; emit −40.7 %, diff +0.8 %. The
daa02c8 value matches ca6ddc4 (168.774 s, −0.43 %), so the campaign conditions
did not change. The block was repeated for daa02c8 because the change is large.
Cumulative vs baseline: −44.39 % against 192.705 s (401daf6 re-measure) and
−44.55 % against the logged 193.238 s. Emit is still 83 % of the slow case.

### Profile at 76e633c (cProfile, separate runs, not used for timings)

Raw profiles (not committed):
`.orchestration/scratch/T1.P2/slow-76e633c.prof` and
`.orchestration/scratch/T1.P2/p32w6-76e633c.prof`. Share is cumtime / profiled
total of `generate`. Tables computed with `.orchestration/scratch/T1.P2/analyze.py`.

#### Slow pcsaft e3 w6: top 15 thermodiff functions by cumtime (profiled total 468.21 s)

| # | function (file:line) | ncalls | cumtime (s) | % of profiled total | tottime (s) |
|---|----------------------|--------|-------------|---------------------|-------------|
| 1 | `generate_workload (thermodiff/bench_hooks.py:1097)` | 1 | 468.21 | 100.00 % | 0.00 |
| 2 | `to_fortran (thermodiff/fortran.py:4082)` | 1 | 449.51 | 96.00 % | 0.03 |
| 3 | `_indexed_cse (thermodiff/fortran.py:892)` | 1 | 239.05 | 51.06 % | 0.20 |
| 4 | `_hoistable_entries (thermodiff/fortran.py:868)` | 190 | 184.29 | 39.36 % | 6.65 |
| 5 | `_cached_hoistable (thermodiff/fortran.py:850)` | 21 875 122 | 107.44 | 22.95 % | 11.60 |
| 6 | `_common_scalar_cse (thermodiff/fortran.py:670)` | 1 | 91.27 | 19.49 % | 0.00 |
| 7 | `_count_scalar_candidates (thermodiff/fortran.py:734)` | 140 | 65.52 | 13.99 % | 3.29 |
| 8 | `_is_scalar_candidate (thermodiff/fortran.py:720)` | 9 790 204 | 35.51 | 7.58 % | 4.93 |
| 9 | `build_diff (thermodiff/model/emit.py:80)` | 1 | 18.26 | 3.90 % | 0.00 |
| 10 | `__init__ (thermodiff/diffplz.py:140)` | 1 | 18.12 | 3.87 % | 0.00 |
| 11 | `_compute (thermodiff/diffplz.py:170)` | 15 | 18.09 | 3.86 % | 0.00 |
| 12 | `diff_ni (thermodiff/diffplz.py:221)` | 4 | 16.55 | 3.53 % | 0.00 |
| 13 | `_share_sums (thermodiff/fortran.py:559)` | 1 | 16.05 | 3.43 % | 0.00 |
| 14 | `resolve (thermodiff/fortran.py:227)` | 10 | 12.05 | 2.57 % | — |
| 15 | `_free_indices (thermodiff/fortran.py:136)` | 21 348 | 9.74 | 2.08 % | 0.02 |

#### p32 w6: top 15 thermodiff functions by cumtime (profiled total 337.85 s)

| # | function (file:line) | ncalls | cumtime (s) | % of profiled total | tottime (s) |
|---|----------------------|--------|-------------|---------------------|-------------|
| 1 | `generate_workload (thermodiff/bench_hooks.py:1097)` | 1 | 337.85 | 100.00 % | 0.00 |
| 2 | `to_fortran (thermodiff/fortran.py:4082)` | 1 | 317.44 | 93.96 % | 0.03 |
| 3 | `_common_scalar_cse (thermodiff/fortran.py:670)` | 1 | 96.28 | 28.50 % | 0.01 |
| 4 | `_indexed_cse (thermodiff/fortran.py:892)` | 1 | 71.52 | 21.17 % | 0.05 |
| 5 | `_hoistable_entries (thermodiff/fortran.py:868)` | 110 | 53.20 | 15.75 % | 1.93 |
| 6 | `_count_scalar_candidates (thermodiff/fortran.py:734)` | 217 | 50.25 | 14.87 % | 2.22 |
| 7 | `_is_scalar_candidate (thermodiff/fortran.py:720)` | 5 521 462 | 32.97 | 9.76 % | 3.02 |
| 8 | `<lambda> (thermodiff/fortran.py:705)` (rank key of `max`) | 105 523 | 30.11 | 8.91 % | 0.05 |
| 9 | `_candidate_rank (thermodiff/fortran.py:750)` | 105 523 | 30.05 | 8.89 % | 0.08 |
| 10 | `_cached_hoistable (thermodiff/fortran.py:850)` | 6 509 909 | 29.94 | 8.86 % | 3.44 |
| 11 | `chain_rule_derivatives (thermodiff/chain.py:1046)` | 1 | 29.47 | 8.72 % | 0.00 |
| 12 | `assemble (thermodiff/chain.py:1357)` | 1 | 27.29 | 8.08 % | 0.00 |
| 13 | `_nest_double_sums (thermodiff/fortran.py:1248)` | 1 | 21.44 | 6.34 % | 0.00 |
| 14 | `__init__ (thermodiff/diffplz.py:140)` | 11 | 20.20 | 5.98 % | 0.00 |
| 15 | `_compute (thermodiff/diffplz.py:170)` | 141 | 20.17 | 5.97 % | 0.00 |

The `chain.py` rows (`chain_rule_derivatives`, `assemble`) are 8.7 % and 8.1 % in
p32. The baseline said `chain.py` was 5.5 % at ca6ddc4, so the share grew
because the CSE total fell.

#### Specific functions (cumtime, % of profiled total, ncalls)

| function | slow ncalls | slow cum (s) | slow % | p32 ncalls | p32 cum (s) | p32 % |
|----------|-------------|--------------|--------|------------|-------------|-------|
| `_common_scalar_cse` | 1 | 91.27 | 19.49 | 1 | 96.28 | 28.50 |
| `_free_indices` | 21 348 | 9.74 | 2.08 | 25 392 | 6.86 | 2.03 |
| `_indexed_cse` | 1 | 239.05 | 51.06 | 1 | 71.52 | 21.17 |
| `_hoistable_entries` | 190 | 184.29 | 39.36 | 110 | 53.20 | 15.75 |
| `_cached_hoistable` | 21 875 122 | 107.44 | 22.95 | 6 509 909 | 29.94 | 8.86 |
| `xreplace` (all callers) | 234 253 | 103.43 | 22.09 | 623 917 | 56.24 | 16.65 |
| `_share_sums` | 1 | 16.05 | 3.43 | 1 | 5.07 | 1.50 |
| `count_ops` | 2 930 | 4.94 | 1.06 | 10 475 | 28.38 | 8.40 |
| `default_sort_key` | 60 512 | 1.07 | 0.23 | 113 666 | 1.99 | 0.59 |
| `preorder_traversal` (`__next__`, all callers) | 62 052 495 | 175.10 | 37.40 | 39 799 900 | 104.26 | 30.86 |

The `preorder_traversal` and `_cached_hoistable` cumtimes contain each other
(the traversal loop calls the predicate), so they must not be added.
`_free_indices` went from 1 853 883 calls (aae26e2 and ca6ddc4, about 40 % of
the profile) to 21 348 calls (slow) and 25 392 calls (p32). `atoms` is still
called directly by `to_fortran` (35.3 s, 7.5 % of the slow profile); that is
outside the CSE passes.

#### Greedy iterations (from the call counts; each pass ends with one counting pass that finds no candidate)

| pass | slow pcsaft e3 w6 | p32 w6 |
|------|-------------------|--------|
| `_common_scalar_cse` temporaries created (`_new_shared_temp` calls from this caller) | 139 | 216 |
| `_common_scalar_cse` counting passes (`_count_scalar_candidates` calls) | 140 | 217 |
| `_common_scalar_cse` `xreplace` calls (= temporaries × 10 expressions) | 1 390 | 2 160 |
| `_indexed_cse` array temporaries (`_new_shared_temp` calls from this caller) | 18 | 10 |
| `_indexed_cse` counting passes (`_hoistable_entries` calls = passes × 10 expressions) | 190 | 110 |
| `_share_sums` temporaries (not a greedy loop) | 107 | 41 |
| `_common_scalar_cse` and `to_fortran` calls | 1 | 1 |

The expression count of 10 per pass is derived from the call counts. Both runs
use one pipeline (no double `to_fortran`).

#### tottime split inside `_common_scalar_cse` (cumtime of the callees, caller→callee edges)

| component | slow (s) | slow % of `_common_scalar_cse` | p32 (s) | p32 % |
|-----------|----------|--------------------------------|---------|-------|
| traversal and eligibility (`_count_scalar_candidates`: `preorder_traversal` + `_is_scalar_candidate`) | 65.52 | 71.8 | 50.25 | 52.2 |
| of which `preorder_traversal` (`__next__`) | 25.84 | 28.3 | 13.72 | 14.2 |
| of which `_is_scalar_candidate` (eligibility memo, `_free_indices`, `sp.Sum` test) | 35.51 | 38.9 | 32.97 | 34.2 |
| `xreplace` | 21.37 | 23.4 | 15.88 | 16.5 |
| key computation (`max` with key = `_candidate_rank`: `count_ops` + `default_sort_key`) | 4.38 | 4.8 | 30.14 | 31.3 |
| self time, `_new_shared_temp`, dict and list operations | ~0 | ~0 | ~0 | ~0 |

The key is cheap on the slow case (22 675 rank calls) and expensive on p32
(105 523 rank calls, about 94 % of it in `count_ops`).

#### Edges of `_cached_hoistable` (slow) and `__eq__`

`_cached_hoistable` (107.44 s cumtime, 11.6 s tottime) spends 72.02 s in sympy
`Basic.__eq__` (7 721 713 calls from the dict probes of the per-call cache). On
p32 it spends 17.40 s there. The probe cost is structural equality, not hashing.
In the whole slow profile `Basic.__eq__` is 112.98 s (24.1 %) and `__hash__` is
11.6 s. This is the main cost left in `_hoistable_entries`.

### Hotspots after T1.7a (top 3 per profile, % of profiled total)

Slow pcsaft e3 w6:
1. `_indexed_cse` 51.06 % (`_hoistable_entries` 39.36 %, of which the
   `_cached_hoistable` probes 22.95 %, with `Basic.__eq__` the largest part).
2. `xreplace` (all callers) 22.09 %; `_common_scalar_cse` 19.49 % (of which
   `_count_scalar_candidates` 13.99 %).
3. `_is_scalar_candidate` 7.58 % (`_free_indices` is now 2.08 %).

p32 w6:
1. `_common_scalar_cse` 28.50 % (rank key `_candidate_rank` 8.89 %, of which
   `count_ops` 8.40 % of the profile).
2. `_indexed_cse` 21.17 % (`_hoistable_entries` 15.75 %).
3. `xreplace` (all callers) 16.65 %; `_is_scalar_candidate` 9.76 %.

### Unprofiled p32 sample (extra, from the profile run)

`tools/profile_case.py` ran one unprofiled generation before the profile:
parse 0.127, diff 18.256, emit 72.383, generation 90.766 s (ca6ddc4 in T1.P1:
127.637 s, −28.9 %). This is one sample inside a different run and is not a
timed row; it is recorded only for the next step.

### Observations for the next wave

- `_free_indices` is no longer a hotspot (2 % in both cases).
- The per-node cache probes in `_hoistable_entries` (`Basic.__eq__`) and the
  `xreplace` passes of `_indexed_cse` and `_common_scalar_cse` are now the cost.
- Step 2 (8172049) did not show a gain on the medium case (+4.3 % median vs
  fff7914). It is a per-call key cache and its benefit is on the p32 rank key
  (`count_ops`), which this medium case does not exercise. The difference is not
  resolved by one block and was not profiled.
