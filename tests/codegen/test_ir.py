"""Tests for the typed definition records and their mapping protocol."""

import pytest

import sympy as sp

from thermodiff.codegen.ir import (
    ComponentFusedMap,
    FusedNest,
    OuterFusedNest,
    SiblingNest,
    TempDef,
)

BASE_KEYS = {"target", "free", "limits", "body"}
IDX = sp.Idx("i", 3)
ARRAY = sp.IndexedBase("a")
TARGET = ARRAY[IDX]
BODY = sp.Symbol("x") + 1


def make(**annotations):
    """Build a base record with the given annotations."""
    return TempDef(TARGET, [IDX], [], BODY, **annotations)


def make_fused():
    """Build a fused wrapper around one member."""
    return FusedNest(
        target=TARGET, free=[IDX], limits=[], body=BODY, fused=[make()]
    )


def test_present_keys_of_a_plain_record():
    assert make().present_keys() == BASE_KEYS


def test_none_annotation_is_absent_and_false_is_present():
    assert "mirror" not in make().present_keys()
    assert "mirror" in make(mirror=False).present_keys()
    assert make(mirror=False).present_keys() == BASE_KEYS | {"mirror"}


def test_every_annotation_name_is_reported():
    record = make(
        mirror=True,
        mirror_sum=True,
        colsum=True,
        colmajor=True,
        symv=(BODY, BODY),
        symv_acc=sp.Symbol("t1"),
        factored=True,
        acc=sp.Symbol("t2"),
    )
    assert record.present_keys() == BASE_KEYS | {
        "mirror",
        "mirror_sum",
        "colsum",
        "colmajor",
        "symv",
        "symv_acc",
        "factored",
        "acc",
    }


def test_copy_replaces_fields_by_attribute_name_and_shares_lists():
    record = make(mirror=True)
    other = record.copy(body=sp.Integer(2), factored=True)
    assert other is not record
    assert other.body == 2 and other.factored is True
    assert other.mirror is True
    assert other.free is record.free
    assert record.body == BODY and record.factored is None
    assert type(other) is TempDef


def test_copy_rejects_old_key_names():
    with pytest.raises(TypeError):
        make().copy(split_acc=2)
    with pytest.raises(TypeError):
        make().copy(compfused=True)


def test_copy_of_wrappers_keeps_the_class():
    nest = make_fused()
    assert type(nest.copy(target=ARRAY[IDX + 1])) is FusedNest
    sibling = SiblingNest(
        target=TARGET,
        free=[],
        limits=[],
        body=BODY,
        sibling=[make()],
        accs=[sp.Symbol("t1")],
    )
    split = sibling.copy(accumulators=2, accs2=[sp.Symbol("t2")])
    assert split.accumulators == 2
    assert split.present_keys() == BASE_KEYS | {
        "sibling",
        "accs",
        "split_acc",
        "accs2",
    }
    assert sibling.present_keys() == BASE_KEYS | {"sibling", "accs"}


def test_equality_follows_dict_value_equality():
    assert make() == make()
    assert make(mirror=False) != make()
    assert make(mirror=True) != make(mirror=False)
    assert make() != make().copy(body=sp.Integer(3))
    assert make() != make_fused()
    assert make_fused() == make_fused()


def test_records_are_unhashable_like_dicts():
    with pytest.raises(TypeError):
        hash(make())
    with pytest.raises(TypeError):
        hash(make_fused())


def test_list_operations_use_value_equality():
    first, second = make(), make().copy(target=ARRAY[IDX + 1])
    pool = [first, second]
    assert make() in pool
    pool.remove(make())
    assert pool == [second]


def test_annotations_are_assignable_attributes():
    record = make()
    record.mirror_sum = True
    assert record.present_keys() == BASE_KEYS | {"mirror_sum"}


def test_wrapper_present_keys_use_the_old_names():
    assert make_fused().present_keys() == BASE_KEYS | {"fused"}
    component = ComponentFusedMap(
        target=TARGET,
        free=[IDX],
        limits=[],
        body=BODY,
        steps=[("local", TARGET, BODY)],
        members=[(TARGET, BODY)],
    )
    assert component.present_keys() == BASE_KEYS | {
        "compfused",
        "steps",
        "members",
    }
    outer = OuterFusedNest(
        target=TARGET,
        free=[],
        limits=[],
        body=sp.S.Zero,
        members=[make()],
        outer=IDX,
    )
    assert outer.present_keys() == BASE_KEYS | {"outerfused", "outer"}
    assert outer.members == [make()]
    assert outer.outer == IDX
    assert "members" not in outer.present_keys()


def test_fused_nest_keeps_the_symv_fields():
    nest = make_fused()
    nest.symv = (BODY, BODY)
    nest.symv_acc = sp.Symbol("t3")
    assert nest.present_keys() == BASE_KEYS | {"fused", "symv", "symv_acc"}


def test_flag_set_gates():
    loops_plain = BASE_KEYS | {"mirror", "factored", "acc"}
    loops_lone = BASE_KEYS | {"mirror_sum"}
    hoisting = BASE_KEYS | {"mirror"}
    assert make().present_keys() <= hoisting
    assert make(mirror=False).present_keys() <= hoisting
    assert make(mirror=False).present_keys() <= loops_plain
    assert not make(mirror_sum=True).present_keys() <= hoisting
    assert make(mirror_sum=True).present_keys() <= loops_lone
    assert not make(mirror=False).present_keys() <= loops_lone
    assert not make(colmajor=True).present_keys() <= loops_plain
    assert not make_fused().present_keys() <= loops_plain
    factored = make(factored=True, acc=sp.Symbol("t"))
    assert factored.present_keys() <= loops_plain
