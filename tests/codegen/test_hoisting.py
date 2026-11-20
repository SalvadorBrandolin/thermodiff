"""Pins of the loop-invariant and affine hoisting passes.

The unit tests pin the returned definitions, the namer state and the
``arrays`` dict on small synthetic inputs; the end-to-end tests pin the
hash of the generated module for the presets that run each pass.
"""

import hashlib
from pathlib import Path

import pytest
import sympy as sp

from thermodiff.codegen.common import _Namer
from thermodiff.codegen.hoisting import (
    _hoist_loop_invariants,
    _split_affine,
)
from thermodiff.codegen.ir import TempDef
from thermodiff.model import compile_model
from thermodiff.model.examples import EXAMPLES

I, J, K, M = (sp.Idx(name) for name in "ijkm")
NC = sp.Symbol("nc", integer=True)
T, V, R, X = sp.symbols("T V R X")
XV, YV, MV, HV, GV = (sp.IndexedBase(name) for name in "xymHG")
LV, AV, BV, TV = (sp.IndexedBase(name) for name in "LabT")


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


def test_hoist_loop_invariants_pins_definitions_and_namer():
    namer, arrays = _Namer(), {}
    definitions = [
        TempDef(
            target=HV[I, J],
            free=[I, J],
            limits=[],
            body=sp.exp(T / V) * XV[I] * YV[J]
            + sp.log(T * R) * MV[I, J]
            + sp.sqrt(V) * XV[I] * sp.exp(YV[J] / T),
        ),
        TempDef(
            target=GV[I],
            free=[I],
            limits=[],
            body=sp.sin(T) * sp.cos(V) * XV[I] + sp.exp(T / V) * XV[I] ** 2,
        ),
        TempDef(target=sp.Symbol("s"), free=[], limits=[], body=T * V),
    ]
    outputs = [
        (
            sp.sin(T) * sp.cos(V) * XV[I] * YV[J] + sp.exp(R / V) * HV[I, J],
            [I, J],
        ),
        (sp.exp(T / V) * XV[I], [I]),
        (T * V, []),
    ]
    result, new_outputs = _hoist_loop_invariants(
        definitions, outputs, namer, arrays, reserved=("q",)
    )
    assert render(result) == [
        ("t1", [], [], "log(R*T)"),
        ("t2", [], [], "sqrt(V)"),
        ("t3[p]", ["p"], [], "exp(y[p]/T)"),
        ("t4", [], [], "exp(T/V)"),
        (
            "H[i, j]",
            ["i", "j"],
            [],
            "t1*m[i, j] + t2*t3[j]*x[i] + t4*x[i]*y[j]",
        ),
        ("t5", [], [], "sin(T)*cos(V)"),
        ("G[i]", ["i"], [], "t4*x[i]**2 + t5*x[i]"),
        ("s", [], [], "T*V"),
        ("t6", [], [], "exp(R/V)"),
    ]
    assert [str(out) for out in new_outputs] == [
        "t5*x[i]*y[j] + t6*H[i, j]",
        "t4*x[i]",
        "T*V",
    ]
    assert [str(name) for name in namer.created] == [
        "t1", "t2", "t3", "t4", "t5", "t6",
    ]  # fmt: skip
    assert arrays == {"t3": 1}
    assert result[4] is not definitions[0]


def run_split(body, outputs, extra=()):
    """Run ``_split_affine`` on one sum ``T(k)`` plus ``extra`` records."""
    namer, arrays = _Namer(), {}
    table = TempDef(target=TV[K], free=[K], limits=[(M, 1, NC)], body=body)
    definitions = [table, *extra]
    result, new_outputs = _split_affine(definitions, outputs, namer, arrays)
    return result, new_outputs, namer, arrays, definitions


def test_split_affine_scalar_split():
    body = (1 - LV[M, K]) * AV[M]
    result, outs, namer, arrays, _ = run_split(body, [TV[K] * 2])
    assert render(result) == [
        ("t1", [], [("m", "1", "nc")], "a[m]"),
        ("T[k]", ["k"], [("m", "1", "nc")], "L[m, k]*a[m]"),
    ]
    assert [str(out) for out in outs] == ["2*t1 - 2*T[k]"]
    assert [str(name) for name in namer.created] == ["t1"]
    assert arrays == {}


def test_split_affine_vector_split():
    body = (1 - LV[M, K]) * AV[M] * BV[M]
    result, outs, namer, arrays, _ = run_split(body, [TV[K] * 2])
    assert render(result) == [
        ("t1[m]", ["m"], [], "a[m]*b[m]"),
        ("t2", [], [("m", "1", "nc")], "t1[m]"),
        ("T[k]", ["k"], [("m", "1", "nc")], "L[m, k]*t1[m]"),
    ]
    assert [str(out) for out in outs] == ["2*t2 - 2*T[k]"]
    assert [str(name) for name in namer.created] == ["t1", "t2"]
    assert arrays == {"t1": 1}


def test_split_affine_reuses_an_existing_vector():
    vector = TempDef(
        target=sp.IndexedBase("w")[I], free=[I], limits=[], body=AV[I] * BV[I]
    )
    body = (1 - LV[M, K]) * AV[M] * BV[M]
    result, _, namer, arrays, _ = run_split(body, [TV[K]], extra=[vector])
    assert render(result) == [
        ("t1", [], [("m", "1", "nc")], "w[m]"),
        ("T[k]", ["k"], [("m", "1", "nc")], "L[m, k]*w[m]"),
        ("w[i]", ["i"], [], "a[i]*b[i]"),
    ]
    assert [str(name) for name in namer.created] == ["t1"]
    assert arrays == {}


def test_split_affine_keeps_identity_of_untouched_records():
    reader = TempDef(target=sp.Symbol("u"), free=[], limits=[], body=X)
    result, _, _, _, definitions = run_split(
        (1 - LV[M, K]) * AV[M], [TV[K]], extra=[reader]
    )
    assert result[-1] is not reader  # copied with a re-read body
    assert result[-1] == reader
    assert definitions[0] not in result


def test_split_affine_rejected_split_keeps_namer_and_arrays(monkeypatch):
    """A rejected split consumes a namer number and keeps the array name."""
    real_count_ops = sp.count_ops

    def penalise_vector(expr, *args, **kwargs):
        cost = real_count_ops(expr, *args, **kwargs)
        has_vector = isinstance(expr, sp.Basic) and any(
            str(base) == "t1" for base in expr.atoms(sp.IndexedBase)
        )
        return cost + 100 if has_vector else cost

    monkeypatch.setattr(sp, "count_ops", penalise_vector)
    body = (1 - LV[M, K]) * AV[M] * BV[M]
    result, outs, namer, arrays, definitions = run_split(body, [TV[K]])
    assert render(result) == render(definitions)
    assert result[0] is definitions[0]
    assert [str(out) for out in outs] == ["T[k]"]
    assert [str(name) for name in namer.created] == ["t1"]
    assert arrays == {"t1": 1}


def digest(code):
    """Return the short SHA-256 of generated code."""
    return hashlib.sha256(code.encode()).hexdigest()[:16]


CMR_MODEL = Path(__file__).parents[1] / "data" / "models" / "cmr" / "cmr.model"
PR_PRESSURE = EXAMPLES["Peng-Robinson (pressure)"]


@pytest.mark.parametrize(
    ("label", "digest_value"),
    [
        ("pr-standard", "dabaa4e42240b70e"),
        ("cmr-standard", "484ef4b6a0b80adb"),
        ("cmr-fastest", "ac3a42baf2b39c72"),
    ],
)
def test_generated_module_hash(label, digest_value):
    """Pin hoist_loops (standard) and split_affine (fastest) end to end."""
    cases = {
        "pr-standard": (PR_PRESSURE, {"preset": "standard"}),
        "cmr-standard": (CMR_MODEL.read_text(), {"preset": "standard"}),
        "cmr-fastest": (
            CMR_MODEL.read_text(),
            {"preset": "fastest", "optional_derivatives": False},
        ),
    }
    text, options = cases[label]
    assert digest(compile_model(text, **options)) == digest_value
