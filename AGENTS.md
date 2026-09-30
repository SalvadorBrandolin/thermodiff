# AGENTS.md - Thermodiff

## Project Overview
Thermodiff is a lightweight Python package for symbolic differentiation and manipulation of thermodynamic expressions using SymPy. It provides utilities for:
- Indexed functions and summation handling
- Kronecker delta simplification
- Automated thermodynamic derivative computation (`DiffPlz` class)
- LaTeX output for equations

## Key Commands

### Development Setup
```bash
# Install in development mode with dev dependencies
pip install -e . -r requirements-dev.txt
```

### Testing
```bash
# Run all tests (pytest)
pytest tests/

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

# Format (Black, line-length 79, target py312)
black tests/ thermodiff/

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
```

## Package Structure
```
thermodiff/
├── __init__.py          # Public exports
├── diffplz.py           # DiffPlz class (main feature)
├── thermovars.py        # Predefined symbols (P, R, T, V, i, j, k, l, m, n, nc)
└── core/
    ├── __init__.py
    ├── easy_sums.py     # sum_components, sum_custom
    ├── idxfunction.py   # idx_function factory
    └── kronecker_handling.py  # handle_free_kronecker, handle_sum_kronecker
```

## Important Conventions

### Testing
- Tests use `pytest` with fixtures (`@pytest.fixture(autouse=True)`)
- Tests are organized in classes matching tutorial examples
- LaTeX output is tested against exact expected strings
- Coverage threshold: 90% (enforced in `tox.ini`)

### Code Style
- Black: line-length 79, target Python 3.12
- Flake8 with plugins: black, builtins, import-order, pep8-naming
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

## CI/CD
- GitHub Actions runs: `py310` (with style, docstyle, check-manifest, coverage, docs), `py311`, `py312`, `py313`
- Coverage must pass 90% threshold
- No separate CI config file found; `tox.ini` defines the matrix via `[gh-actions]`

## Common Gotchas
1. **No `AGENTS.md` or `.github/` configs existed** — this file is the primary agent guide
2. **Install via git** — `pip install git+https://github.com/SalvadorBrandolin/thermodiff` (not on PyPI)
3. **SymPy >= 1.14.0** required
4. **Predefined symbols** in `thermovars.py`: `P, R, T, V` (thermo vars) and `i, j, k, l, m, n, nc` (indices)
5. **Internal functions** must be passed to `DiffPlz` for proper LaTeX substitution (`clean_plz`, `latex_readable_plz`)
6. **Kronecker handling** is automatic via `handle_free_kronecker` / `handle_sum_kronecker` in core
7. **Running code** Always work on a virtual environment, prefer the environment "thermodiff" of virtualenv 

## Quick Verification
```bash
# Smoke test import and basic functionality
python -c "import thermodiff as td; print(td.DiffPlz(td.T))"
```