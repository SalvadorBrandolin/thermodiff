"""Tests for DiffPlz based on the tutorial notebook examples.

Three test cases:
    1. Mole fraction (pure compositional, no T/V/P dependence).
    2. Summation of indexed functions (phi_k * tau_lk).
    3. tau_lk explicit expression (UNIQUAC-style, clean_plz +
       latex_readable_plz).
"""

import pytest

import sympy as sp

from thermodiff import (
    DiffPlz,
    T,
    idx_function,
    k,
    l,
    n,
    sum_components,
)


# =============================================================================
# Example 1 — Mole fraction
# =============================================================================
class TestMoleFraction:
    """Derivatives of x_k = n[k] / sum(n[l], l).

    The expression has no T, V, or P dependence, so those derivatives are 0.
    The compositional derivatives produce Piecewise expressions.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        nt = sum_components(n[l], l)
        expr = n[k] / nt
        self.diff = DiffPlz(
            expr, internal_functions=None, indexes=[k, l], name="x_k"
        )

    def test_dt_is_zero(self):
        assert self.diff.dt == sp.Integer(0)

    def test_dv_is_zero(self):
        assert self.diff.dv == sp.Integer(0)

    def test_dp_is_zero(self):
        assert self.diff.dp == sp.Integer(0)

    def test_dni_is_piecewise(self):
        assert self.diff.dni.is_Piecewise

    def test_dni_has_two_cases(self):
        # case i==k and otherwise
        assert len(self.diff.dni.args) == 2

    def test_dni_case_i_eq_k(self):
        """When i == k: 1/nt - n[k]/nt**2."""
        from thermodiff import i

        nt = sum_components(n[l], l)
        case_i_eq_k = self.diff.dni.args[0][0]  # expression for Eq(i, k)
        expected = -n[i] / nt**2 + 1 / nt
        assert sp.simplify(case_i_eq_k - expected) == 0

    def test_dni_case_otherwise(self):
        """When i != k: -n[k]/nt**2."""
        nt = sum_components(n[l], l)
        case_otherwise = self.diff.dni.args[1][0]
        expected = -n[k] / nt**2
        assert sp.simplify(case_otherwise - expected) == 0

    def test_dnidnj_is_piecewise(self):
        assert self.diff.dnidnj.is_Piecewise

    def test_dnidnj_has_four_cases(self):
        # i==j==k, i==k, j==k, otherwise
        assert len(self.diff.dnidnj.args) == 4

    def test_latex_readable_zeros(self):
        latex = self.diff.latex_readable_plz()
        for key in [
            "dT",
            "dV",
            "dP",
            "dT2",
            "dV2",
            "dP2",
            "dTn",
            "dVn",
            "dPn",
            "dTV",
            "dTP",
            "dVP",
        ]:
            assert (
                latex[key] == "0"
            ), f"Expected '0' for {key}, got: {latex[key]}"


# =============================================================================
# Example 2 — Summation of indexed functions
# =============================================================================
class TestSumIndexedFunctions:
    """Derivatives of sum_k( n[k] * phi_k * tau_lk ).

    phi_k depends on n (compositional), tau_lk depends only on T.
    """

    EXPECTED_LATEX = {
        "dT": r"\sum_{k=1}^{N_{c}} \frac{\partial \tau_{lk}}{\partial T} \phi_{k} {n}_{k}",  # noqa
        "dV": "0",
        "dP": "0",
        "dT2": r"\sum_{k=1}^{N_{c}} \frac{\partial^2 \tau_{lk}}{\partial T^2} \phi_{k} {n}_{k}",  # noqa
        "dV2": "0",
        "dP2": "0",
        "dn2": r"\frac{\partial \phi_{i}}{\partial n_j} \tau_{li} + \frac{\partial \phi_{j}}{\partial n_i} \tau_{lj} + \sum_{k=1}^{N_{c}} \frac{\partial^2 \phi_{k}}{\partial n_i \partial n_j} \tau_{lk} {n}_{k}",  # noqa
        "dTn": r"\frac{\partial \tau_{li}}{\partial T} \phi_{i} + \sum_{k=1}^{N_{c}} \frac{\partial \phi_{k}}{\partial n_i} \frac{\partial \tau_{lk}}{\partial T} {n}_{k}",  # noqa
        "dVn": "0",
        "dPn": "0",
        "dTV": "0",
        "dTP": "0",
        "dVP": "0",
        "dni": r"\phi_{i} \tau_{li} + \sum_{k=1}^{N_{c}} \frac{\partial \phi_{k}}{\partial n_i} \tau_{lk} {n}_{k}",  # noqa
    }

    @pytest.fixture(autouse=True)
    def setup(self):
        self.phi_k = idx_function(r"\phi")(n[k])
        self.tau_lk = idx_function(r"\tau")(l, k, T)
        expr = sum_components(n[k] * self.phi_k * self.tau_lk, k)
        self.diff = DiffPlz(
            expr,
            internal_functions=[self.phi_k, self.tau_lk],
            indexes=[k, l],
            name="f",
        )

    def test_dv_is_zero(self):
        assert self.diff.dv == sp.Integer(0)

    def test_dp_is_zero(self):
        assert self.diff.dp == sp.Integer(0)

    def test_dt_is_sum(self):
        """dT should be a Sum (tau_lk depends on T)."""
        assert self.diff.dt.has(sp.Sum)

    def test_dt_has_derivative_of_tau(self):
        """dT contains Derivative(..., T) from tau_lk."""
        assert self.diff.dt.has(sp.Derivative)

    def test_dt2_is_sum(self):
        assert self.diff.dt2.has(sp.Sum)

    def test_dni_has_two_terms(self):
        """Fix the following.

        dni = phi(n[i])*tau(l,i,T) + Sum(tau(l,k,T)*Deriv(phi(n[k]),n[i])*n[k])
        """
        # Result is a sum of two terms (Add)
        assert isinstance(self.diff.dni, sp.Add)

    def test_dni_has_sum(self):
        assert self.diff.dni.has(sp.Sum)

    def test_dnidnj_has_three_terms(self):
        """dnidnj has three terms: two free + one Sum."""
        assert isinstance(self.diff.dnidnj, sp.Add)
        assert self.diff.dnidnj.has(sp.Sum)

    def test_dtdni_has_two_terms(self):
        assert isinstance(self.diff.dtdni, sp.Add)
        assert self.diff.dtdni.has(sp.Sum)

    # --- latex_readable_plz ---

    def test_latex_dt_subscripts_first(self):
        latex = self.diff.latex_readable_plz()
        # tau should appear as \tau_{lk}, phi as \phi_{k}
        assert r"\tau_{lk}" in latex["dT"]
        assert r"\phi_{k}" in latex["dT"]

    def test_latex_dni_subscripts(self):
        latex = self.diff.latex_readable_plz()
        # free term: phi_i * tau_li
        assert r"\phi_{i}" in latex["dni"]
        assert r"\tau_{li}" in latex["dni"]
        # sum term: partial phi_k / partial n_i * tau_lk
        assert r"\tau_{lk}" in latex["dni"]

    def test_latex_dn2_subscripts(self):
        latex = self.diff.latex_readable_plz()
        assert r"\phi_{i}" in latex["dn2"]
        assert r"\phi_{j}" in latex["dn2"]
        assert r"\tau_{li}" in latex["dn2"]
        assert r"\tau_{lj}" in latex["dn2"]

    def test_latex_dt_subscripts_second(self):
        latex = self.diff.latex_readable_plz()
        assert r"\tau_{li}" in latex["dTn"]
        assert r"\phi_{i}" in latex["dTn"]

    def test_latex_dv_is_zero(self):
        latex = self.diff.latex_readable_plz()
        assert latex["dV"] == "0"

    def test_latex_dp_is_zero(self):
        latex = self.diff.latex_readable_plz()
        assert latex["dP"] == "0"

    @pytest.mark.parametrize("key", list(EXPECTED_LATEX.keys()))
    def test_latex_full_output(self, key):
        latex = self.diff.latex_readable_plz()
        assert latex[key] == self.EXPECTED_LATEX[key], (
            f"Mismatch for key '{key}':\n"
            f"  got:      {latex[key]!r}\n"
            f"  expected: {self.EXPECTED_LATEX[key]!r}"
        )


# =============================================================================
# Example 3 — tau_lk explicit (UNIQUAC-style), clean_plz + latex_readable_plz
# =============================================================================
class TestTauLkExplicit:
    """Derivatives of tau_lk = exp(a + b/T + c*ln(T) + d*T + e*T^2).

    After clean_plz, the expression should be factored in terms of tau_lk(T).
    After latex_readable_plz with tau_symbol as internal function, the (T)
    should be dropped and tau_lk should appear as a clean symbol.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.a = sp.Symbol("a_{lk}")
        self.b = sp.Symbol("b_{lk}")
        self.c = sp.Symbol("c_{lk}")
        self.d = sp.Symbol("d_{lk}")
        self.e = sp.Symbol("e_{lk}")

        self.tau_expr = sp.exp(
            self.a
            + self.b / T
            + self.c * sp.ln(T)
            + self.d * T
            + self.e * T**2
        )

        tau_symbol = sp.Function(r"\tau_{lk}")(T)

        self.diff = DiffPlz(
            self.tau_expr,
            internal_functions=[tau_symbol],
            indexes=[l, k],
            name=r"\tau_{lk}",
        )
        self.diff.clean_plz(["dT", "dT2"])

    def test_dni_is_zero(self):
        """tau_lk has no compositional dependence."""
        assert self.diff.dni == sp.Integer(0)

    def test_dv_is_zero(self):
        assert self.diff.dv == sp.Integer(0)

    def test_dp_is_zero(self):
        assert self.diff.dp == sp.Integer(0)

    def test_dt_contains_tau_symbol(self):
        """After clean_plz, dt should contain the tau_lk(T) symbol."""
        tau_sym = sp.Function(r"\tau_{lk}")(T)
        assert self.diff.dt.has(tau_sym)

    def test_dt2_contains_tau_symbol(self):
        tau_sym = sp.Function(r"\tau_{lk}")(T)
        assert self.diff.dt2.has(tau_sym)

    def test_dt_structure(self):
        """dt = (du/dT) * tau_lk where u is the exponent."""
        # du/dT = 2*e*T + d + c/T - b/T^2
        du_dt = 2 * self.e * T + self.d + self.c / T - self.b / T**2
        tau_sym = sp.Function(r"\tau_{lk}")(T)
        expected = du_dt * tau_sym
        assert sp.simplify(self.diff.dt - expected) == 0

    def test_dt2_structure(self):
        """dt2 = (d2u/dT2 + (du/dT)^2) * tau_lk."""
        du_dt = 2 * self.e * T + self.d + self.c / T - self.b / T**2
        d2u_dt2 = 2 * self.e - self.c / T**2 + 2 * self.b / T**3
        tau_sym = sp.Function(r"\tau_{lk}")(T)
        expected = (d2u_dt2 + du_dt**2) * tau_sym
        assert sp.simplify(self.diff.dt2 - expected) == 0

    def test_latex_dt_no_t_argument(self):
        """After latex_readable_plz, tau_lk should appear without (T)."""
        latex = self.diff.latex_readable_plz()
        # Should contain tau_{lk} as a clean symbol, not tau_{lk}(T)
        assert r"\tau_{lk}" in latex["dT"]
        # The (T) residual should be gone
        assert r"\left( T \right)" not in latex["dT"]

    def test_latex_dt2_no_t_argument(self):
        latex = self.diff.latex_readable_plz()
        assert r"\tau_{lk}" in latex["dT2"]
        assert r"\left( T \right)" not in latex["dT2"]

    def test_latex_dni_is_zero(self):
        latex = self.diff.latex_readable_plz()
        assert latex["dni"] == "0"
