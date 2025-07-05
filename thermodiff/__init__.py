"""thermodiff package

This package provides simple SymPy shenanigans and shortcuts for thermodynamic
expressions differentiation and manipulation.

The project is designed for personal use and personal preferences. If some
expression is not handled the way you like you could create your own
expressions parsing functions inspired by the existing ones on the source code.

By now, this package is kind of a toy/ simple tool. There is no guarantee that
more features will be added in the future.

If you find a bad differentiation or a bug, please report it in the repository
issues section.
"""

from thermodiff.core.easy_sums import SumComponents, SumCustom
from thermodiff.core.idxfunction import IdxFunction
from thermodiff.core.kronecker_handling import (
    handle_free_kronecker,
    handle_sum_kronecker,
)
from thermodiff.diffclass import DiffClass
from thermodiff.thermovars import i, j, k, l, m, n, nc, R, T, V


__all__ = [
    # User-friendly sums:
    "SumComponents",
    "SumCustom",
    # Index Base function:
    "IdxFunction",
    # Kronecker handling functions:
    "handle_free_kronecker",
    "handle_sum_kronecker",
    # DiffClass:
    "DiffClass",
    # Thermodynamic variables and indices:
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "nc",
    "R",
    "T",
    "V",
]
