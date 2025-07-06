"""thermodiff package

This package provides simple SymPy shenanigans and shortcuts for thermodynamic
expressions differentiation and manipulation.
"""

from thermodiff.core.easy_sums import SumComponents, SumCustom
from thermodiff.core.idxfunction import idx_function
from thermodiff.core.kronecker_handling import (
    handle_free_kronecker,
    handle_sum_kronecker,
)
from thermodiff.diffplz import DiffPlz
from thermodiff.thermovars import i, j, k, l, m, n, nc, P, R, T, V


__all__ = [
    # User-friendly sums:
    "SumComponents",
    "SumCustom",
    # Index Base function:
    "idx_function",
    # Kronecker handling functions:
    "handle_free_kronecker",
    "handle_sum_kronecker",
    # DiffPlz class:
    "DiffPlz",
    # Thermodynamic variables and indices:
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "nc",
    "P",
    "R",
    "T",
    "V",
]
