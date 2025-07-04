import sympy as sp

from thermodiff.thermovars import nc


def SumComponents(expr, idx):
    return sp.Sum(expr, (idx, 1, nc))


def SumCustom(expr, idx, start=1, end=None):
    return sp.Sum(expr, (idx, start, end))
