"""Tests for the derivative selection vocabulary."""

import pytest

from thermodiff.derivatives import (
    DERIVATIVE_ATTRIBUTES,
    resolve_derivatives,
)
from thermodiff.fortran import _DERIV_SUFFIX


def test_attribute_order_matches_the_emitter():
    """The canonical order and the Fortran suffixes must agree."""
    assert tuple(_DERIV_SUFFIX) == DERIVATIVE_ATTRIBUTES


class TestNames:
    """Friendly names, aliases and separators."""

    def test_friendly_names(self):
        assert resolve_derivatives("dT") == ["dt"]
        assert resolve_derivatives("dV") == ["dv"]
        assert resolve_derivatives("dP") == ["dp"]
        assert resolve_derivatives("dn") == ["dni"]
        assert resolve_derivatives("dndn") == ["dnidnj"]
        assert resolve_derivatives("dndT") == ["dtdni"]

    def test_attribute_names_still_work(self):
        assert resolve_derivatives(["dni", "dnidnj"]) == [
            "dni",
            "dnidnj",
        ]

    def test_df_prefix_and_case_are_optional(self):
        assert resolve_derivatives("dfdndn") == ["dnidnj"]
        assert resolve_derivatives("DFDT") == ["dt"]

    def test_mixed_order_synonyms(self):
        assert resolve_derivatives("dTdn") == resolve_derivatives("dndT")
        assert resolve_derivatives("dVdn") == ["dvdni"]
        assert resolve_derivatives("dPdn") == ["dpdni"]

    def test_str_and_iterable_and_separators(self):
        expected = ["dt", "dni"]
        assert resolve_derivatives("dT, dn") == expected
        assert resolve_derivatives("dT dn") == expected
        assert resolve_derivatives(["dT", "dn"]) == expected

    def test_order_is_canonical(self):
        assert resolve_derivatives("dn,dT") == ["dt", "dni"]

    def test_empty_selection(self):
        assert resolve_derivatives("") == []
        assert resolve_derivatives([]) == []


class TestGroups:
    """The ``all``, ``none``, ``first`` and ``second`` groups."""

    declared = {"n", "V", "T"}

    def test_all_only_includes_applicable(self):
        resolved = resolve_derivatives("all", self.declared)
        assert {"dt", "dv", "dni", "dtdv"} <= set(resolved)
        # No pressure variable was declared.
        assert "dp" not in resolved and "dtdp" not in resolved

    def test_first(self):
        assert resolve_derivatives("first", self.declared) == [
            "dt",
            "dv",
            "dni",
        ]

    def test_second(self):
        resolved = resolve_derivatives("second", self.declared)
        assert {"dt2", "dv2", "dnidnj", "dtdv"} <= set(resolved)
        assert "dp2" not in resolved

    def test_none(self):
        assert resolve_derivatives("none", self.declared) == []

    def test_group_and_explicit_union(self):
        resolved = resolve_derivatives("first,dndV", self.declared)
        assert resolved == ["dt", "dv", "dni", "dvdni"]

    def test_all_without_declared_variables(self):
        assert resolve_derivatives("all") == list(DERIVATIVE_ATTRIBUTES)


class TestValidation:
    """Invalid selections raise readable errors."""

    def test_unknown_name(self):
        with pytest.raises(ValueError, match="Unknown derivatives"):
            resolve_derivatives("nope")

    def test_not_applicable(self):
        with pytest.raises(ValueError, match="requires the variable"):
            resolve_derivatives("dP", {"n", "T", "V"})

    def test_cross_term_requires_both_variables(self):
        with pytest.raises(ValueError, match="requires the variable"):
            resolve_derivatives("dTdV", {"T", "n"})
