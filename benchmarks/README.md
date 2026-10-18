# Benchmarks

This directory measures how much work the generated code does and how fast
it runs, across the successive generations of the code generator.

## Layout

| file | purpose |
| --- | --- |
| `generate_model.py` | builds a benchmark model and prints its Fortran |
| `compare.py` | materialises each generation, compiles and times it |
| `report.md` | the generated comparison table |

## Usage

```bash
python benchmarks/compare.py
python benchmarks/compare.py --nc 80 --reps 100 --models pr
```

`compare.py` never touches the working copy: it uses `git archive` to extract
each revision into a temporary directory and runs that revision's own
`thermodiff` to generate the Fortran.  The result is printed and written to
`report.md`.

Options:

* `--nc` number of components used by the timing driver (default `50`);
* `--reps` calls per timing run (default `200`);
* `--runs` timing runs per variant; the best one is kept (default `3`);
* `--models` comma separated list of models to benchmark.

## Models

* `toy` - `f = T*sum(n_k**2) + V*sum(n_k)`.  Small enough that the very first
  generator can already compile it; used to show that the sharing passes do
  not slow down a trivial model.
* `pr` - Peng-Robinson with a two index attractive term,
  `amix = sum_i sum_j n_i n_j sqrt(a_i a_j) (1 - k_ij)`.  This is the shape
  that real equations of state have, and the one the sharing passes target.

## Generations

| label | revision | what it adds |
| --- | --- | --- |
| `mvp` | `cf4f841` | the first code generator |
| `no-share` | `b30ddb8` | `.model` front-end, no sharing |
| `level-1` | `6bc9861` | shared sums + scalar CSE + optional outputs |
| `level-2` | working tree | index-dependent array temporaries |

`mvp` cannot compile the `pr` model: its free index analysis is not complete
yet, so it declares scalar outputs as arrays and gets the rank of `k_ij`
wrong.  The harness reports that instead of a timing.
