"""Pin ``_fuse_outer_loops`` on synthetic definitions and on two models.

The unit cases pin the returned list (record types, targets, member order,
outer index) for maps, reductions, factored members, component-fused maps,
a libm clash and independent scalars; the end-to-end cases pin the SHA-256
of the Fortran text of models where the pass fires.
"""

import hashlib
import warnings
from pathlib import Path

import pytest
import sympy as sp

from thermodiff.codegen import CodegenOptions
from thermodiff.codegen.common import _LoopStart
from thermodiff.codegen.ir import (
    ComponentFusedMap,
    OuterFusedNest,
    TempDef,
)
from thermodiff.codegen.loops import _fuse_outer_loops
from thermodiff.fortran import to_fortran
from thermodiff.model.emit import build_diff
from thermodiff.model.ir import build_ir
from thermodiff.model.parser import read_sections

ROOT = Path(__file__).resolve().parent.parent.parent
NC = sp.Symbol("nc", integer=True)
IDX_I = sp.Idx("i", NC)
IDX_J = sp.Idx("j", NC)
IDX_K = sp.Idx("k", NC)
PARAM = sp.IndexedBase("a")
ACC = sp.Symbol("acc")


def array(name):
    """Return the indexed base of a temporary."""
    return sp.IndexedBase(name)


def make_map(name, index, body):
    """Return a per-component map ``name(index) = body``."""
    return TempDef(
        target=array(name)[index], free=[index], limits=[], body=body
    )


def make_reduction(name, index, body, inner=None):
    """Return a scalar reduction over ``1..nc`` (and an inner loop)."""
    limits = [(index, 1, NC)]
    if inner is not None:
        limits.append((inner, _LoopStart(index), NC))
    return TempDef(target=sp.Symbol(name), free=[], limits=limits, body=body)


def make_scalar(name, body):
    """Return an independent scalar definition."""
    return TempDef(target=sp.Symbol(name), free=[], limits=[], body=body)


def summarise(definitions):
    """Return ``(type, target, members, outer)`` for each definition."""
    rows = []
    for definition in definitions:
        members = []
        outer = None
        if isinstance(definition, OuterFusedNest):
            members = [
                (str(member.target), str(member.free), str(member.limits))
                for member in definition.members
            ]
            outer = str(definition.outer)
        rows.append(
            (type(definition).__name__, str(definition.target), members, outer)
        )
    return rows


def compfused(name, index):
    """Return a component-fused map with one step."""
    target = array(name)[index]
    value = PARAM[index] * 2
    return ComponentFusedMap(
        target=target,
        free=[index],
        limits=[],
        body=sp.S.Zero,
        steps=[("map", target, value)],
        members=[(target, value)],
    )


def case_maps():
    """Two maps, the second over another index name."""
    return [
        make_map("t1", IDX_I, PARAM[IDX_I] ** 2),
        make_map("t2", IDX_J, array("t1")[IDX_J] + 1),
    ], {}


def case_map_and_reduction():
    """A reduction reads the map at the current outer index."""
    return [
        make_map("t1", IDX_I, PARAM[IDX_I] * 3),
        make_reduction("t2", IDX_J, array("t1")[IDX_J]),
    ], {}


def case_reads_other_index():
    """The second map reads the first at a different index: not fused."""
    return [
        make_map("t1", IDX_I, PARAM[IDX_I]),
        make_map("t2", IDX_J, array("t1")[IDX_K]),
    ], {}


def case_reads_accumulator():
    """A map that reads an accumulator of the group ends the group."""
    return [
        make_reduction("t1", IDX_I, PARAM[IDX_I]),
        make_map("t2", IDX_J, sp.Symbol("t1") * PARAM[IDX_J]),
    ], {}


def case_factored_member():
    """A reduction annotated ``factored`` still joins the group."""
    factored = make_reduction("t2", IDX_J, PARAM[IDX_J] ** 2)
    factored.factored = True
    factored.acc = ACC
    return [make_map("t1", IDX_I, PARAM[IDX_I] + 1), factored], {}


def case_component_fused():
    """A component-fused map joins a map, renamed to the group index."""
    return [make_map("t1", IDX_I, PARAM[IDX_I]), compfused("t2", IDX_J)], {}


def case_triangular():
    """A triangular reduction after a map, and one after a reduction."""
    return [
        make_map("t1", IDX_I, PARAM[IDX_I]),
        make_reduction("t2", IDX_J, array("t1")[IDX_J], inner=IDX_K),
        make_reduction("t3", IDX_I, PARAM[IDX_I]),
    ], {}


def case_libm_clash():
    """With ``libm_guard`` a double reduction and a libm map separate."""
    return [
        make_map("t1", IDX_I, sp.exp(PARAM[IDX_I])),
        make_reduction("t2", IDX_J, PARAM[IDX_J], inner=IDX_K),
    ], {"libm_guard": True}


def case_libm_reduction():
    """With ``libm_reductions`` a libm reduction separates from a double."""
    return [
        make_reduction("t1", IDX_I, PARAM[IDX_I], inner=IDX_K),
        make_reduction("t2", IDX_J, sp.log(PARAM[IDX_J])),
    ], {"libm_guard": True, "libm_reductions": True}


def case_independent_scalar():
    """An independent scalar between two maps moves ahead of the group."""
    return [
        make_map("t1", IDX_I, PARAM[IDX_I]),
        make_scalar("t2", sp.Symbol("R") * 2),
        make_map("t3", IDX_I, array("t1")[IDX_I] + 1),
        make_scalar("t4", sp.Symbol("t3")),
        make_map("t5", IDX_I, PARAM[IDX_I]),
    ], {}


def case_non_plain_ends_group():
    """A mirrored definition ends the group and is emitted as is."""
    mirrored = make_map("t2", IDX_I, PARAM[IDX_I])
    mirrored.mirror = True
    return [
        make_map("t1", IDX_I, PARAM[IDX_I]),
        mirrored,
        make_map("t3", IDX_I, PARAM[IDX_I]),
        make_map("t4", IDX_I, PARAM[IDX_I]),
    ], {}


CASES = {
    "maps": case_maps,
    "map_and_reduction": case_map_and_reduction,
    "reads_other_index": case_reads_other_index,
    "reads_accumulator": case_reads_accumulator,
    "factored_member": case_factored_member,
    "component_fused": case_component_fused,
    "triangular": case_triangular,
    "libm_clash": case_libm_clash,
    "libm_reduction": case_libm_reduction,
    "independent_scalar": case_independent_scalar,
    "non_plain_ends_group": case_non_plain_ends_group,
}

EXPECTED = {
    "component_fused": [
        (
            "OuterFusedNest",
            "t1[i]",
            [("t1[i]", "[i]", "[]"), ("t2[j]", "[i]", "[]")],
            "i",
        )
    ],
    "factored_member": [
        (
            "OuterFusedNest",
            "t1[i]",
            [("t1[i]", "[i]", "[]"), ("t2", "[]", "[(i, 1, nc)]")],
            "i",
        )
    ],
    "independent_scalar": [
        ("TempDef", "t2", [], None),
        (
            "OuterFusedNest",
            "t1[i]",
            [("t1[i]", "[i]", "[]"), ("t3[i]", "[i]", "[]")],
            "i",
        ),
        ("TempDef", "t4", [], None),
        ("TempDef", "t5[i]", [], None),
    ],
    "libm_clash": [
        ("TempDef", "t1[i]", [], None),
        ("TempDef", "t2", [], None),
    ],
    "libm_reduction": [
        ("TempDef", "t1", [], None),
        ("TempDef", "t2", [], None),
    ],
    "map_and_reduction": [
        (
            "OuterFusedNest",
            "t1[i]",
            [("t1[i]", "[i]", "[]"), ("t2", "[]", "[(i, 1, nc)]")],
            "i",
        )
    ],
    "maps": [
        (
            "OuterFusedNest",
            "t1[i]",
            [("t1[i]", "[i]", "[]"), ("t2[i]", "[i]", "[]")],
            "i",
        )
    ],
    "non_plain_ends_group": [
        ("TempDef", "t1[i]", [], None),
        ("TempDef", "t2[i]", [], None),
        (
            "OuterFusedNest",
            "t3[i]",
            [("t3[i]", "[i]", "[]"), ("t4[i]", "[i]", "[]")],
            "i",
        ),
    ],
    "reads_accumulator": [
        ("TempDef", "t1", [], None),
        ("TempDef", "t2[j]", [], None),
    ],
    "reads_other_index": [
        ("TempDef", "t1[i]", [], None),
        ("TempDef", "t2[j]", [], None),
    ],
    "triangular": [
        (
            "OuterFusedNest",
            "t1[i]",
            [
                ("t1[i]", "[i]", "[]"),
                ("t2", "[]", "[(i, 1, nc), (k, i + 1, nc)]"),
                ("t3", "[]", "[(i, 1, nc)]"),
            ],
            "i",
        )
    ],
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_fuse_outer_loops_result_is_pinned(case):
    definitions, options = CASES[case]()
    assert summarise(_fuse_outer_loops(definitions, **options)) == (
        EXPECTED[case]
    )


def generate_code(model, derivatives, **overrides):
    """Return the Fortran text of a model with the ``fastest`` preset."""
    path = ROOT / model
    ir = build_ir(
        read_sections(path.read_text(encoding="utf-8"), str(path)), str(path)
    )
    diff, _ = build_diff(ir, derivatives=derivatives)
    options = CodegenOptions.from_preset(
        "fastest", optional_derivatives=False, **overrides
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return to_fortran(
            diff,
            derivatives=derivatives,
            name="f",
            subroutine_name="routine",
            integer_args=ir.dimensions,
            module_symbols=ir.imported,
            symmetric_params=getattr(ir, "symmetric", ()),
            options=options,
        )


def digest(code):
    """Return the SHA-256 of a text."""
    return hashlib.sha256(code.encode()).hexdigest()


WORKLOAD_W6 = "dT,dV,dn,dT2,dV2,dndn,dTdV,dndT,dndV"
MODEL_PINS = {
    "pr": (
        "examples/pr.model",
        "cadccd18b3a2c2e9638881086e42ef6497f286f549a06d0248a3784c56816d45",
        "b788b6e485adfe890e5860d18654836c1c9ec4c15a3ce4f76e9958db0437ddef",
    ),
    "cmr": (
        "tests/data/models/cmr/cmr.model",
        "7a1a1f67a123b19a8c64c8cf0aa150ab599990f956b3b1f0e48e166ecaa58744",
        "5c1361a01d6fdeb7aa61572458388700225baef5003be87e190faf49601f27df",
    ),
}


@pytest.mark.parametrize("name", sorted(MODEL_PINS))
def test_outer_fusion_text_is_pinned(name):
    model, fused, unfused = MODEL_PINS[name]
    assert digest(generate_code(model, WORKLOAD_W6)) == fused
    code = generate_code(model, WORKLOAD_W6, fuse_outer=False)
    assert digest(code) == unfused
