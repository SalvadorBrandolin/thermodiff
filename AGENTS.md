# AGENTS.md - Thermodiff

## Project Overview
Thermodiff is a lightweight Python package for symbolic differentiation and manipulation of thermodynamic expressions using SymPy. It provides utilities for:
- Indexed functions and summation handling
- Kronecker delta simplification
- Automated thermodynamic derivative computation (`DiffPlz` class)
- LaTeX output for equations
- Fortran code generation of the model and its derivatives (`to_fortran`, the
  `.model` language with `compile_model` and the `thermodiff-model` CLI), plus
  yaeos wrappers around the generated kernels

## Key Commands

### Development Setup
```bash
# Install in development mode with dev dependencies
pip install -e . -r requirements-dev.txt
```

### Testing
```bash
# Run all tests (pytest); -n uses pytest-xdist (requirements-dev.txt)
pytest tests/ -n 8

# Fast set (~30 s): skip the `slow` tests (compile-and-run, large generations)
pytest tests/ -n 8 -m "not slow"

# Run with coverage (requires 90% minimum)
pytest tests/ --cov=thermodiff --cov-report=term-missing

# Run a single test file
pytest tests/test_mole_fraction.py

# Run a single test class
pytest tests/test_mole_fraction.py::TestMoleFraction

# Run a single test
pytest tests/test_mole_fraction.py::TestMoleFraction::test_dt_is_zero
```

### Code Quality
```bash
# Lint (flake8 with plugins)
flake8 tests/ thermodiff/

# Docstyle (numpy convention)
pydocstyle thermodiff --convention=numpy

# Format (Black, line-length 79, target py310 to py313)
black tests/ thermodiff/

# Static types (strict mypy on thermodiff/codegen, config in pyproject.toml)
mypy

# Check manifest
check-manifest
```

### Documentation
```bash
# Build HTML docs
cd docs && make html
# or via tox
tox -e docs
```

### Tox Environments
```bash
# All environments
tox

# Specific Python versions
tox -e py310
tox -e py311
tox -e py312
tox -e py313

# Style checks
tox -e style
tox -e docstyle

# Coverage
tox -e coverage

# Strict mypy on the code generator
tox -e typecheck
```

## Package Structure
```
thermodiff/
├── __init__.py          # Public exports
├── chain.py             # Structured differentiation by the chain rule over summation nodes
├── derivatives.py       # Resolution of user supplied derivative selections
├── diffplz.py           # DiffPlz class (main feature)
├── fortran.py           # Public facade: to_fortran (implementation in codegen/)
├── latex_utils.py       # LaTeX formatting utilities for DiffPlz
├── thermovars.py        # Predefined symbols (P, R, T, V, i, j, k, l, m, n, nc)
├── yaeos_model.py       # yaeos model wrappers for thermodiff-generated kernels
├── codegen/             # Fortran code generator (strict mypy)
│   ├── options.py       # CodegenOptions, enums, presets, resolve_codegen_options
│   ├── pipeline.py      # to_fortran, _EmitState, the pass table _PASSES, declarations, assembly
│   ├── ir.py            # Typed definition records: TempDef, FusedNest, ComponentFusedMap, ...
│   ├── common.py        # Shared helpers: name sanitising, the namer, expression printing
│   ├── resolve.py       # Placeholder and derivative resolution
│   ├── symmetry.py      # Symmetric parameters, mirrored sums, canonical dummy indices
│   ├── hoisting.py      # Loop-invariant hoisting, low-rank and affine splitting
│   ├── cse.py           # Scalar and indexed common sub-expression elimination, shared sums
│   ├── loops.py         # Loop fusion and table passes (gfortran/x86 oriented)
│   ├── sparse.py        # Zero guards for sparse parameters (opt-in sparse_parameters)
│   ├── extents.py       # Index ranges of extra dimensions (ng, ...): extents and loop bounds
│   ├── storage.py       # Temporaries storage: work_array, max_nc declarations
│   ├── core_chain_workload.py  # scalar_chain_workload_module: fused core + chain kernels per workload
│   ├── workload.py      # workload_module: fixed-output kernels + present() dispatcher, fixed_nc copies
│   └── render.py        # Definitions to Fortran statements
├── core/
│   ├── __init__.py
│   ├── easy_sums.py     # sum_components, sum_custom
│   ├── idxfunction.py   # idx_function factory
│   └── kronecker_handling.py  # handle_free_kronecker, handle_sum_kronecker
├── gui/                 # Optional Streamlit front-end for the .model language
└── model/               # Front-end for the .model file format (parser, IR, emit, CLI)
benchmarks/td_bench/     # Benchmark hooks: variant registry (e*/p*/v* keys), workloads
tools/                   # Golden generation/check (golden.py, check_golden.py), profile_case.py
tests/codegen/           # Pins of the code generator passes, function-size limits
tests/data/              # Vendored .model files, scripts and measured kernels
docs/source/             # Sphinx docs, including architecture.md (code generator design)
docs/dev/                # Developer notes: perf log, refactor report, plans (not in the docs build)
```

## Important Conventions

### Testing
- Tests use `pytest` with fixtures (`@pytest.fixture(autouse=True)`)
- Tests are organized in classes matching tutorial examples
- LaTeX output is tested against exact expected strings
- Coverage threshold: 90% (enforced in `tox.ini`)
- Markers: `slow` (≥ 3 s: Fortran compile-and-run or large generations) and
  `thesis` (needs files outside the repository; registered, applied to no test)
- Tests that compile generated Fortran need `gfortran` (`tests/fortran_tools.py`
  finds it; honours `FC`). Prefer checking generated kernels numerically
  (compile, run, compare with SymPy or the option-off kernel at `rel=1e-12`)
  over asserting literal Fortran text; keep text asserts for structural facts
  (a loop is or is not emitted, a declaration's intent or rank)

### Code Style
- Black: line-length 79, target Python 3.10 to 3.13
- Flake8 with plugins: black, builtins, import-order, pep8-naming (`E203` is
  ignored in `tox.ini` for Black compatibility; nothing else). Import order is
  the plugin's default style: stdlib, then one group per third-party package
- Docstrings: NumPy convention (validated by pydocstyle)

### SymPy Patterns Used
- `sp.Sum` for symbolic summations
- `sp.Derivative` for unevaluated derivatives
- `sp.Piecewise` for conditional expressions (Kronecker handling)
- `sp.Function` for internal function symbols in LaTeX output
- `idx_function(r"\phi")(n[k])` creates indexed function symbols

### DiffPlz Usage Pattern
```python
from thermodiff import DiffPlz, sum_components, idx_function, n, k, l, T

phi_k = idx_function(r"\phi")(n[k])
tau_lk = idx_function(r"\tau")(l, k, T)
expr = sum_components(n[k] * phi_k * tau_lk, k)

diff = DiffPlz(expr, internal_functions=[phi_k, tau_lk], indexes=[k, l], name="f")

# Access derivatives
diff.dt      # d/dT
diff.dv      # d/dV
diff.dp      # d/dP
diff.dni     # d/dn_i
diff.dnidnj  # d²/dn_i dn_j
diff.dtdni   # d²/dT dn_i

# LaTeX output
latex = diff.latex_readable_plz()
diff.clean_plz(["dT", "dT2"])  # Factor internal functions in derivatives
```

## Default Workflow: Thermodynamic Model Differentiation & yaeos Implementation

When requested to implement or analyze a thermodynamic model specified in LaTeX, the agent must follow this multi-step workflow:

### 1. Workspace & Output Directory
- Create and organize all generated files within a directory with a descriptive name corresponding to the user request/model (e.g., `models/<model_name>/` or `<model_name>_implementation/`).

### 2. Symbolic Differentiation with `thermodiff`
- Parse the model's LaTeX mathematical formulation.
- Use `thermodiff` (`DiffPlz`, `idx_function`, `sum_components`, `sum_custom`, etc.) to obtain the required thermodynamic derivatives ($d/dT$, $d/dV$, $d/dP$, $d/dn_i$, $d^2/dn_i dn_j$, $d^2/dT dn_i$, etc.).
- Consult `thermodiff` documentation and tutorials to identify opportunities for simplification, defining intermediate/internal functions, and generating readable, simplified analytical expressions (e.g. using `.clean_plz(...)` and `.latex_readable_plz()`).
- Save the python file used to generate the derivatives.

### 3. Fortran 90 Implementation & Numerical Validation
- Implement the model and its analytical derivatives in Fortran 90.
- Perform numerical verification by comparing the analytical derivatives against numerical differentiation (e.g., central finite differences or complex-step differentiation).
- Create a Markdown report (`report.md`) containing:
  - The mathematical expressions in LaTeX for both the model and its derivatives.
  - A comparison table/summary between analytical derivatives and numerical differentiation results.
  - Discussion of any edge cases or numerical tolerance considerations.

### 4. Integration with `yaeos` Architecture
- Implement the final model following the conventions, interfaces, and design patterns of the [yaeos](https://github.com/ipqa-research/yaeos) library.
- Consult the corresponding reference implementations according to the model type:
  - **Helmholtz Energy Models (Equation of State / Residual Helmholtz)**:
    - [ar_models.f90](https://github.com/ipqa-research/yaeos/blob/main/src/models/residual_helmholtz/ar_models.f90)
    - [generic_cubic.f90](https://github.com/ipqa-research/yaeos/blob/main/src/models/residual_helmholtz/cubic/generic_cubic.f90)
  - **Alpha Functions**:
    - [alphas.f90](https://github.com/ipqa-research/yaeos/blob/main/src/models/residual_helmholtz/cubic/alphas/alphas.f90)
  - **Mixing Rules**:
    - [quadratic_mixing.f90](https://github.com/ipqa-research/yaeos/blob/main/src/models/residual_helmholtz/cubic/mixing_rules/quadratic_mixing.f90)
  - **Excess Gibbs Energy ($G^E$) Models**:
    - [ge_models.f90](https://github.com/ipqa-research/yaeos/blob/main/src/models/excess_gibbs/ge_models.f90)
    - [uniquac.f90](https://github.com/ipqa-research/yaeos/blob/main/src/models/excess_gibbs/uniquac.f90)

## Developing the Code Generator

The full design is in `docs/source/architecture.md` (pipeline diagram, module
map, options and presets, the table of passes in execution order, definition
records, invariants, how to add a pass). Read it before changing
`thermodiff/codegen/` or `thermodiff/chain.py`. Summary:

```
.model text -> model/parser.py -> model/ir.py -> model/emit.py (DiffPlz)
  -> fortran.to_fortran -> codegen/pipeline.py: 36 passes over one _EmitState
  (resolve, CSE, sharing, hoisting, symmetry, loop fusion, render) -> Fortran text
  (chain_rule on: chain.chain_rule_derivatives builds the structured derivatives first)
```

### Options and presets
- `to_fortran(diff, options=CodegenOptions(...))`. The legacy keyword
  arguments still work (`CodegenOptions.from_legacy`); new code should pass
  `options`. `CodegenOptions.validate()` warns (`UserWarning`) and disables
  options that would be inert in a combination; it raises only where the
  old code failed.
- Presets (`codegen/options.py`, `PRESET_SETTINGS`): `"minimal"` (legacy
  output), `"standard"` (= bench variant e8: sum sharing, CSE, symmetry,
  invariant hoisting, chain rule, `chain_factor_sums`) and `"fastest"`
  (= p33: adds the gfortran/x86 loop passes; slower generation, last-bit
  rounding differences).
  `compile_model`, `emit_module`, the CLI (`--preset`) and the GUI default to
  `"standard"`; **`to_fortran` called directly keeps the neutral (minimal)
  defaults**. Tests pin standard == e8 and fastest == p33.
- Kernels are fastest when specialised per output set
  (`optional_derivatives=False`, one subroutine per workload) and dispatched
  by a `present()` mask; most loop passes are disabled under
  `optional_derivatives=True`. `workload_module` (`codegen/workload.py`,
  `compile_model(workloads=..., fixed_nc=...)`, CLI `--workload`,
  `--fixed-nc`) builds that module: one fixed-output kernel per workload,
  the optional kernel as fallback, a dispatcher, and optional copies with a
  compile-time `nc`. It works on the text of `to_fortran` (it parses the
  dummy lists), so a change to the signature or declaration format must
  keep `tests/test_workload.py` green.
- Other options, all off or neutral by default (byte-identical output):
  `work_array` (one automatic array for all temporaries) and `max_nc`
  (compile-time extents, `error stop` above it) against heap allocation of
  temporaries; `real_kind`/`kind_module` (`real(pr)` and `_pr` literals, a
  final text pass in `common.apply_real_kind`). `canonical_placeholders` is
  on by default: placeholder derivatives are named by their variables
  (`dBdn`, `dBdndn`, `dDdTdn`); `False` restores `dBdi`/`dBdj`.
- Placeholders: `idx_function` objects passed as `internal_functions`
  (`.model`: the `functions` section). `idx_function(name, symmetric=(0, 1))`
  or `symmetric_params=[name]` makes the placeholder and its derivative
  inputs symmetric. A placeholder argument must be a variable (`n`, `T`,
  `V`, `P`); an expression argument triggers a warning (the caller must
  supply total derivatives). For a scalar core, differentiate with respect
  to extra symbols: `DiffPlz(..., variables=[B, D])` emits `dfdB`, `dfdBdD`,
  `dfdndB`, ...
- Names: Fortran is case-insensitive. Parameters and placeholders whose
  names differ only in case from another name are renamed (`r` -> `r_1`)
  consistently in signature and body; module symbols, outputs and the
  thermodynamic variables keep their names (a clash among them raises).
  `i` and `j` are reserved for the derivative indices: `DiffPlz` renames
  summation indices called `i`/`j` and rejects them as free indices.
- Extra dimensions: a summation bound other than `nc` (`sum_custom(...,
  m, 1, ng)`, `.model` `sum_m[1:ng]`) becomes an integer argument; its
  index carries the range (`codegen/extents.py`), so temporaries get `ng`
  extents and loops `ng` bounds. Passes that only handle `nc` skip such
  indices. One index label cannot be summed over two different ranges.

### Rules for changing the generator
1. **Generated Fortran must not change unless the change says so.** Check
   with the golden tiers:
   ```bash
   # once, on a clean main: write the external golden (../golden_ext, or $THERMODIFF_GOLDEN_EXT)
   PYTHONHASHSEED=0 python tools/golden.py --tier complete -j 8
   # after each change
   PYTHONHASHSEED=0 python tools/check_golden.py --tier fast -j 8      # 112 cases, ~15 s
   PYTHONHASHSEED=0 python tools/check_golden.py --tier complete -j 8  # 581 cases, ~10 min
   ```
   The fast tier also reads the external golden, so generate it first. An
   intended output change must be declared, validated numerically (compile
   and compare) and the golden regenerated in its own commit.
2. **Pins first.** Before refactoring a pass, add tests that pin its current
   result (see `tests/codegen/`), then change it.
3. **Order and identity matter.** Passes draw temporary names from one namer
   in table order (changing the order or number of draws renames
   temporaries); some code relies on object identity, in-place mutation and
   list identity. Definition records are mutable with value equality on
   purpose (`codegen/ir.py`).
4. **No global state.** Caches are per call and passed explicitly; never a
   module-level `functools.lru_cache` on SymPy expressions.
   `tests/test_reentrancy.py` checks interleaved calls.
5. **Limits** (enforced by `tests/codegen/test_function_lengths.py`): every
   function in `codegen/` and `chain.py` has ≤ 50 code lines and McCabe
   complexity ≤ 10.
6. **Types.** `codegen/` is strict mypy (`mypy`, `tox -e typecheck`). Do not
   add `from __future__ import annotations` (some signatures are read at
   runtime). Annotations are evaluated at import on Python 3.10-3.13 (only
   3.14 defers them): a name imported under `TYPE_CHECKING` must be quoted.
   If you develop on 3.14, run the tests once on an older Python before
   pushing.
   Lint with the plugins: `tox -e style` (plain `flake8` from a venv without
   `flake8-import-order`/`pep8-naming` misses import-order and naming
   findings that CI reports).
7. **Every transformation that can increase work needs an operation-count
   guard** (e.g. `sep_sums="cost"`, `nest_cost`, `dummy_canon` with its
   sharing criterion); always-on variants measured up to 6.7× slower kernels.

### Adding a pass or an option
Add the field to `CodegenOptions` (enum and `validate()` rule if needed), a
keyword-only parameter of `_emit_code` (the public signature of `to_fortran`
is built from it), the pass function with explicit state, its entry in
`_PASSES` at the right position, pins and a golden check, and a row in the
pass table of `docs/source/architecture.md`. Decide explicitly whether a
preset should enable it (that is a declared output change).

### Performance work
- Measure generation time with the protocol in `docs/dev/perf_log.md`
  (cold fresh processes, `benchmarks.td_bench.generate_workload`, `taskset`
  away from cores used by other benchmarks); profile with
  `tools/profile_case.py` (cProfile, cold, for locating hotspots only).
- Reference: pcsaft e3 w6 went 192.7 s → 57.7 s in the `refactor/perf-arch`
  work (`docs/dev/refactor-perf-arch-report.md`). Measure kernel run time on a
  quiet, dedicated core: heavy load on other cores distorts the numbers.
- gfortran puts automatic arrays on the heap unless `-fstack-arrays`; large
  temporaries then cost malloc/free and page faults on every call.

## Future Work

### Third derivatives
Plan: `docs/dev/third-derivatives-plan.md` (proposal, nothing implemented;
ends with open questions for the maintainer). Key points:
- yaeos does not use a third-derivative tensor; its critical-point solver
  takes finite differences of the smallest eigenvalue
  (`src/equilibria/critical.f90`). The exact replacement is the cubic form
  C(w) = Σ Ar_ijl w_i w_j w_l (T,V basis), plus a few contracted mixed terms
  for an analytic Jacobian.
- Recommended design: a directional variable s with n → n + s·w, so
  C(w) = ∂³Ar/∂s³ and the existing chain rule and passes apply with outputs
  of rank ≤ 2. The full packed N³ tensor (i ≤ j ≤ l) is a secondary option.
- The structured chain rule extends exactly (third-order formula checked
  symbolically in the plan); for quadratic mixing rules the result is low
  rank, O(N²) for the contraction.
- Hazards in today's code: `chain._rate_based_value` returns the
  second-order formula for every order ≥ 2; `DiffPlz(derivatives=None)`
  computes every attribute in `DERIVATIVE_ATTRIBUTES`, so third-order
  attributes need their own tuple. `sym_rank3` is about rank-3 *parameters*,
  not output symmetry.
- Validate with high-precision references (mpmath), finite differences of
  the second-derivative kernels and permutation symmetry.

### Other target languages (C, C++, Julia, Rust, ...)
Not implemented; this is the suggested route, based on where the Fortran
specifics live today.
- **Language-neutral already:** the model front end, `DiffPlz`, `chain.py`
  and the symbolic passes (`resolve`, `symmetry`, `hoisting`, `cse`, and most
  of `loops`) work on SymPy expressions and the typed records of
  `codegen/ir.py`.
- **Fortran-specific:** expression printing (`common._print_expr` uses SymPy
  `fcode`, `_FCODE_SETTINGS`); statement rendering (`render.py`: `do`/`end
  do`, `if`, accumulators; `symmetry._render_symv`); declarations and
  assembly in `pipeline.py` (`_input_declarations`, `_array_declarations`,
  `_assemble_text`, `intent`, `optional`/`present()` guards); the module
  wrapper in `model/emit.py`; `yaeos_model.py` (yaeos only).
- **Semantics baked in:** 1-based inclusive loops over `nc`, column-major
  arrays (`table_layout`, `col_major`, `contiguous_args` assume it),
  case-insensitive identifiers (`_sanitize` and the clash renaming), optional
  arguments, and the gfortran/x86 tuning of the low-level options
  (`keep_libm_tables`, `fuse_libm_guard`, `split_acc`, `array_pad`).
- **Suggested design:** a backend object (expression printer: SymPy has
  `ccode`, `cxxcode`, `julia_code`, `rust_code`; statement emitters for
  loop/if/assign/accumulate/declare; naming rules; index base; memory
  layout) chosen by a `CodegenOptions` field, default `"fortran"`. Move
  `render.py` and the text parts of `pipeline.py` behind it so Fortran output
  stays byte-identical (golden), mark which options are portable versus
  backend-specific (gate the latter in `validate()`), and keep one golden per
  backend. C is the easiest first target (compile with gcc, compare
  numerically against the Fortran kernels and SymPy).

## CI/CD
- GitHub Actions runs on Linux, macOS and Windows: `py310` (with style, docstyle, check-manifest, coverage, docs, typecheck), `py311`, `py312`, `py313`
- The Linux job is pinned to `ubuntu-24.04` because `fortran-lang/setup-fortran` has no gcc `latest` for ubuntu-26.04 yet; revisit when it does
- Tests must be path-portable (Windows runners use different drives for the checkout and the temp directory): use `pathlib`, not string concatenation of paths
- Coverage must pass 90% threshold
- Workflows are in `.github/workflows/` (`ci.yml`, `docs.yml`, `release.yml`); `tox.ini` defines the matrix via `[gh-actions]`

## Common Gotchas
1. **Agent guide** — this file is the primary agent guide for the repository
2. **Install** — `pip install thermodiff` from PyPI; releases are published from the upstream repository SalvadorBrandolin/thermodiff. All install commands are in the Installation section of `README.md`
3. **SymPy >= 1.14.0** required
4. **Predefined symbols** in `thermovars.py`: `P, R, T, V` (thermo vars) and `i, j, k, l, m, n, nc` (indices)
5. **Internal functions** must be passed to `DiffPlz` for proper LaTeX substitution (`clean_plz`, `latex_readable_plz`)
6. **Kronecker handling** is automatic via `handle_free_kronecker` / `handle_sum_kronecker` in core
7. **Running code** Always work on a virtual environment, prefer the environment "thermodiff" of virtualenv
8. **Determinism** — generated code does not depend on `PYTHONHASHSEED`, but use `PYTHONHASHSEED=0` for golden runs and reproducible manifests
9. **Editable install and worktrees** — if `thermodiff` is installed editable from another checkout, prefix commands with `PYTHONPATH=$PWD` and check `python -c "import thermodiff; print(thermodiff.__file__)"`
10. **Developer notes** go in `docs/dev/`, not in the repository root

## Quick Verification
```bash
# Smoke test import and basic functionality
python -c "import thermodiff as td; print(td.DiffPlz(td.T))"
```
