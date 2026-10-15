# AGENTS.md

Small SymPy-based Python library for symbolic differentiation/manipulation of
thermodynamic expressions. One installable package, `thermodiff` (no app or
server). Supports Python 3.10-3.13.

## Setup

- No lockfile; deps are not vendored. For local dev:
  `pip install -r requirements-dev.txt && pip install -e .`
- `tox` is the primary task runner and builds its own envs. CI just runs `tox -r`.

## Commands

- All checks: `tox -r` (slow; runs style, docstyle, check-manifest, docs,
  py310-py313, coverage).
- One interpreter / full test suite: `tox -e py312`
- Single test (args pass through to pytest):
  `tox -e py312 -- tests/test_mole_fraction.py::TestMoleFraction::test_dni_is_piecewise`
- Lint: `tox -e style` -> flake8 over `tests/ thermodiff/`
- Docstring style: `tox -e docstyle` -> pydocstyle, numpy convention
- Coverage: `tox -e coverage` - fails under 90% (`--fail-under=90`)
- Docs: `tox -e docs` (or `sphinx-build -b html docs/source _build`).
  Requires `pandoc` installed; deps live in `docs/requirements.txt` (furo theme).

## Conventions and gotchas

- Black config is in `pyproject.toml`: `line-length = 79` (not the default 88)
  and `target-version = ["py312"]`. There is no separate flake8 config;
  flake8-black enforces the 79-char limit.
- flake8 runs with flake8-builtins, flake8-import-order, and pep8-naming.
  The index symbol `l` is intentional and needs `# noqa: E741` (see
  `thermovars.py`).
- Public API is re-exported in `thermodiff/__init__.py`; keep `__all__` in sync.
  It currently lists `install_in_colab`, which is defined nowhere.

## Architecture

- `thermodiff/thermovars.py` - canonical SymPy symbols: `T`, `V`, `P`, `R`, `nc`,
  indices `i`/`j`/`k`/`l`/`m`, and `n` (an `IndexedBase` over `nc`).
- `thermodiff/diffplz.py` - `DiffPlz`, the main class. Computes T/V/P and
  compositional derivatives and handles Kronecker deltas.
- `thermodiff/fortran.py` - `to_fortran(diff, ...)` generates a free-form
  Fortran subroutine returning the value and its derivatives. Also exposed as
  `DiffPlz.to_fortran()`. `Sum` is expanded to `do` loops, `Piecewise` to
  `if`/`else`, and `idx_function` placeholders become extra input arguments
  (`strict=True` raises instead).
- `thermodiff/core/` - helpers: `idx_function`, `sum_components`/`sum_custom`,
  and Kronecker-delta handling.
- Domain rule: `i` and `j` are reserved by `DiffPlz` for the derivative
  variables. Expressions passed to it must use `k`, `l`, `m` instead.
- `idx_function(name)` creates an `sp.Function` subclass whose derivatives stay
  unevaluated, which is required for symbolic derivatives to render nicely.
- `DiffPlz.clean_plz()` rewrites derivatives in terms of the named function; run
  it before `latex_readable_plz()`. `latex_readable_plz()` is explicitly ad-hoc
  and fragile.

## Testing

- Only three modules under `tests/`. `test_mole_fraction.py` asserts exact LaTeX
  strings, so changes to `latex_readable_plz()` output formatting break tests.
- `tests/test_fortran.py` compiles and runs the generated Fortran with
  `gfortran` when it is available, and skips those cases otherwise.
- CI: `.github/workflows/ci.yml` runs tox on Linux/Windows/macOS x py3.10-3.13;
  `.github/workflows/docs.yml` deploys `gh-pages` from `main`.
