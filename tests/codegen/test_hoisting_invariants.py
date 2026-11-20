"""Pins of the invariant hoisting and the low-rank rewrite.

The unit tests pin the returned expressions and definitions, the namer
state and the ``arrays`` dict on small synthetic inputs; the end-to-end
tests pin the hash of the generated module on models where each pass runs
(and check that toggling its option changes the hash).
"""

import hashlib
import warnings
from pathlib import Path

import pytest
import sympy as sp

from thermodiff.codegen.common import _Namer
from thermodiff.codegen.hoisting import (
    _hoist_invariants,
    _lowrank_bilinear,
)
from thermodiff.codegen.ir import TempDef
from thermodiff.model import compile_model

I, J = sp.Idx("i"), sp.Idx("j")
NC = sp.Symbol("nc", integer=True)
T = sp.Symbol("T")
A, B, C, X, Y, AC, TT, KK = (
    sp.IndexedBase(name)
    for name in ("A", "B", "C", "x", "y", "ac", "tt", "kk")
)
V, W = sp.IndexedBase("V"), sp.IndexedBase("W")


def summation(body, *indexes):
    """Sum ``body`` over ``indexes`` from 1 to ``nc``."""
    return sp.Sum(body, *[(index, 1, NC) for index in indexes])


def render(definitions):
    """Render definitions as comparable plain tuples."""
    return [
        (
            str(item.target),
            [str(index) for index in item.free],
            [tuple(str(part) for part in lim) for lim in item.limits],
            str(item.body),
        )
        for item in definitions
    ]


def run_hoist(expressions, **kwargs):
    """Run the pass; return rewritten, hoisted, sums, names and arrays."""
    namer, arrays = _Namer(), {}
    exprs, hoisted, sums = _hoist_invariants(
        expressions, namer, arrays, **kwargs
    )
    return (
        [str(expr) for expr in exprs],
        render(hoisted),
        render(sums),
        [str(name) for name in namer.created],
        arrays,
    )


def run_lowrank(expr):
    """Run the rewrite; return the rewritten expression, definitions."""
    namer, arrays = _Namer(), {}
    result = _lowrank_bilinear(expr, (I, J), namer, arrays)
    names = [str(name) for name in namer.created]
    if result is None:
        return None, names, arrays
    return (str(result[0]), render(result[1])), names, arrays


def test_hoist_rank_one_factors_and_scalars():
    """Costly factors become rank-1 or scalar temporaries."""
    exprs = [
        summation(sp.exp(A[I]) * B[J] * C[I, J], I, J),
        summation(
            sp.sqrt(AC[I] * AC[J] * TT[I] ** KK[I] * TT[J] ** KK[J]), I, J
        ),
        summation(sp.exp(T) * X[I], I)
        + summation(sp.exp(A[I]) * sp.exp(A[J]) * X[I] * X[J], I, J),
    ]
    assert run_hoist(exprs) == (
        [
            "Sum(B[j]*C[i, j]*t1[i], (i, 1, nc), (j, 1, nc))",
            "Sum(sqrt(t2[i]*t2[j]), (i, 1, nc), (j, 1, nc))",
            "Sum(t3*x[i], (i, 1, nc)) + "
            "Sum(t4[i]*t4[j], (i, 1, nc), (j, 1, nc))",
        ],
        [
            ("t1[p]", ["p"], [], "exp(A[p])"),
            ("t2[p]", ["p"], [], "ac[p]*tt[p]**kk[p]"),
            ("t3", [], [], "exp(T)"),
            ("t4[p]", ["p"], [], "exp(A[p])*x[p]"),
        ],
        [],
        ["t1", "t2", "t3", "t4"],
        {"t1": 1, "t2": 1, "t4": 1},
    )


def test_hoist_reserved_names_skip_slot_indexes():
    """Reserved names are excluded from the slot indices (``k``, ``l``)."""
    expr = summation(
        sp.sqrt(AC[I] * AC[J] * TT[I] ** KK[I] * TT[J] ** KK[J]), I, J
    )
    assert run_hoist([expr + sp.Symbol("q")], reserved=("k", "l")) == (
        ["q + Sum(sqrt(t1[i]*t1[j]), (i, 1, nc), (j, 1, nc))"],
        [("t1[p]", ["p"], [], "ac[p]*tt[p]**kk[p]")],
        [],
        ["t1"],
        {"t1": 1},
    )


def test_hoist_sum_definitions_insert_before_their_body():
    """Hoisted temporaries precede the sum definition they came from."""
    sum_defs = [
        TempDef(
            target=sp.Symbol("s1"),
            free=[],
            limits=[(I, 1, NC)],
            body=sp.exp(T) * X[I],
        ),
        TempDef(target=sp.Symbol("s2"), free=[], limits=[], body=sp.exp(T)),
        TempDef(
            target=sp.Symbol("s3"),
            free=[],
            limits=[(I, 1, NC), (J, 1, NC)],
            body=sp.log(A[I]) * sp.log(B[J]) * C[I, J],
        ),
    ]
    assert run_hoist(
        [sp.exp(T) * summation(X[I], I)], sum_definitions=sum_defs
    ) == (
        ["exp(T)*Sum(x[i], (i, 1, nc))"],
        [],
        [
            ("t1", [], [], "exp(T)"),
            ("s1", [], [("i", "1", "nc")], "t1*x[i]"),
            ("s2", [], [], "exp(T)"),
            ("t2[p]", ["p"], [], "log(A[p])"),
            ("t3[p]", ["p"], [], "log(B[p])"),
            (
                "s3",
                [],
                [("i", "1", "nc"), ("j", "1", "nc")],
                "C[i, j]*t2[i]*t3[j]",
            ),
        ],
        ["t1", "t2", "t3"],
        {"t2": 1, "t3": 1},
    )


def test_hoist_leaves_piecewise_without_sums_alone():
    """A ``Piecewise`` that holds no sum is not rewritten."""
    expr = summation(
        sp.Piecewise((sp.exp(A[I]) * X[J], sp.Eq(I, J)), (0, True)), I, J
    )
    assert run_hoist([expr]) == (
        [
            "Sum(Piecewise((exp(A[i])*x[j], Eq(i, j)), (0, True)), "
            "(i, 1, nc), (j, 1, nc))"
        ],
        [],
        [],
        [],
        {},
    )


def test_hoist_does_not_mutate_its_inputs():
    """The input lists are copied, and untouched definitions are kept."""
    exprs = [summation(sp.exp(A[I]) * B[J], I, J)]
    plain = TempDef(target=sp.Symbol("s2"), free=[], limits=[], body=T)
    sum_defs = [plain]
    _, _, rewritten = _hoist_invariants(exprs, _Namer(), {}, sum_defs)
    assert sum_defs == [plain]
    assert rewritten[0] is plain
    assert len(exprs) == 1


LOWRANK_CASES = {
    "scalar-sum": (
        sum(sp.exp(T * n) * A[I] * B[J] * (n + 1) for n in range(1, 3)),
        (
            "B[j]*t1[i]",
            [("t1[i]", ["i"], [], "3*exp(2*T)*A[i] + 2*exp(T)*A[i]")],
        ),
        ["t1"],
    ),
    "keeps-rest": (
        sp.exp(A[I]) * B[J]
        + sp.exp(A[I]) * C[J] * T
        + X[I] * sp.exp(T)
        + C[I, J],
        (
            "exp(T)*x[i] + C[i, j] + t1[j]*t2[i]",
            [
                ("t1[j]", ["j"], [], "T*C[j] + B[j]"),
                ("t2[i]", ["i"], [], "exp(A[i])"),
            ],
        ),
        ["t1", "t2"],
    ),
    "plain-vectors": (
        V[I] * W[J] + V[I] * X[J] * T + Y[I] * W[J] + Y[I] * X[J] + C[I, J],
        (
            "C[i, j] + W[j]*t1[i] + t2[i]*x[j]",
            [
                ("t1[i]", ["i"], [], "V[i] + y[i]"),
                ("t2[i]", ["i"], [], "T*V[i] + y[i]"),
            ],
        ),
        ["t1", "t2"],
    ),
    "indexed-coefficient-reused": (
        A[I] * sp.exp(B[J]) + X[I] * sp.exp(C[J]) + Y[I] * sp.exp(B[J]) * T,
        (
            "t1[i]*t2[j] + t3[j]*x[i]",
            [
                ("t1[i]", ["i"], [], "T*y[i] + A[i]"),
                ("t2[j]", ["j"], [], "exp(B[j])"),
                ("t3[j]", ["j"], [], "exp(C[j])"),
            ],
        ),
        ["t1", "t2", "t3"],
    ),
    "single-key": (
        sp.exp(A[I]) * sp.sin(B[J]) + sp.exp(X[I]) * sp.sin(B[J]),
        (
            "t1[i]*t2[j]",
            [
                ("t1[i]", ["i"], [], "exp(A[i]) + exp(x[i])"),
                ("t2[j]", ["j"], [], "sin(B[j])"),
            ],
        ),
        ["t1", "t2"],
    ),
    "distributed": (
        (A[I] + B[J]) * (X[I] + Y[J]) * (T + 1),
        (
            "(T + 1)*A[i]*x[i] + (T + 1)*B[j]*y[j] + B[j]*t1[i] "
            "+ t2[i]*y[j]",
            [
                ("t1[i]", ["i"], [], "(T + 1)*x[i]"),
                ("t2[i]", ["i"], [], "(T + 1)*A[i]"),
            ],
        ),
        ["t1", "t2"],
    ),
}


@pytest.mark.parametrize("label", sorted(LOWRANK_CASES))
def test_lowrank_rewrites(label):
    """Pin the rewritten expression, definitions and namer draws."""
    expr, expected, names = LOWRANK_CASES[label]
    result, created, arrays = run_lowrank(expr)
    assert result == expected
    assert created == names
    assert arrays == {name: 1 for name in names}


@pytest.mark.parametrize(
    "expr",
    [
        A[I] * B[J],
        sp.Piecewise((1, sp.Eq(I, J)), (0, True)) * A[I],
    ],
)
def test_lowrank_declines_without_gain(expr):
    """No separable term, or no cheaper form: None and no names drawn."""
    assert run_lowrank(expr) == (None, [], {})


def digest(code):
    """Return the short SHA-256 of generated code."""
    return hashlib.sha256(code.encode()).hexdigest()[:16]


ROOT = Path(__file__).parents[2]
MODELS = {
    "pr": ROOT / "examples" / "pr.model",
    "srk": ROOT / "tests" / "data" / "models" / "srk" / "srk.model",
    "uniquac": ROOT
    / "tests"
    / "data"
    / "models"
    / "uniquac"
    / "uniquac.model",
}

#: ``(model, preset, overrides)`` -> hash; overrides toggle the pass off
#: (``lowrank_hess`` / ``hoist_invariants``) and must change the hash.
END_TO_END = [
    ("uniquac", "standard", {}, "2a9eafa7820b6c13"),
    ("uniquac", "standard", {"hoist_invariants": False}, "637419a2b715bf14"),
    ("uniquac", "fastest", {}, "6b6fb31dd9328db5"),
    ("uniquac", "fastest", {"hoist_invariants": False}, "88997b18fad9838f"),
    ("uniquac", "fastest", {"lowrank_hess": False}, "18cd999927fb533a"),
    ("pr", "fastest", {}, "25a8b3fbfb4552df"),
    ("pr", "fastest", {"lowrank_hess": False}, "e54d1f52650cff62"),
    ("srk", "fastest", {}, "9659190a766c0a09"),
    ("srk", "fastest", {"lowrank_hess": False}, "3dc7cbed45b2a2ca"),
]


@pytest.mark.slow
@pytest.mark.parametrize(
    ("model", "preset", "overrides", "digest_value"),
    END_TO_END,
    ids=lambda value: (
        str(value) if not isinstance(value, dict) else repr(value)
    ),
)
def test_generated_module_hash(model, preset, overrides, digest_value):
    """Pin the module where the hoisting and low-rank passes run."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        code = compile_model(
            MODELS[model].read_text(),
            preset=preset,
            optional_derivatives=False,
            **overrides,
        )
    assert digest(code) == digest_value
