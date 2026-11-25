"""Summations over the reserved indices ``i`` and ``j`` (issue #16)."""

import pytest
import sympy as sp

from thermodiff import DiffPlz, i, j, k, n, sum_components

X = sp.IndexedBase("x")


def _expression(index):
    return sum_components(n[index] * X[index], index) ** 2


@pytest.mark.parametrize("bound", [i, j])
def test_bound_reserved_index_matches_free_notation(bound):
    reference = DiffPlz(_expression(k), indexes=[k], name="f")
    renamed = DiffPlz(_expression(bound), indexes=[bound], name="f")
    assert renamed.expression == reference.expression
    assert renamed.indexes == reference.indexes
    for attribute in ("dt", "dni", "dnidnj", "dtdni"):
        assert getattr(renamed, attribute) == getattr(reference, attribute)


def test_fortran_matches_free_notation():
    reference = DiffPlz(_expression(k), indexes=[k], name="f")
    renamed = DiffPlz(_expression(i), indexes=[i], name="f")
    assert renamed.to_fortran() == reference.to_fortran()


def test_fresh_index_avoids_names_in_use():
    expression = sum_components(n[i] * X[i] * sum_components(n[k], k), i)
    diff = DiffPlz(expression, indexes=[k], name="f")
    assert not diff.expression.has(i)
    assert diff.expression.has(sp.Idx("l"))


def test_free_reserved_index_raises():
    with pytest.raises(ValueError, match="reserved"):
        DiffPlz(n[i] * X[i], indexes=[k], name="f")


def test_non_clashing_expression_is_untouched():
    expression = _expression(k)
    assert DiffPlz(expression, indexes=[k]).expression is expression
