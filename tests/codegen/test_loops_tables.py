"""Pin ``_drop_pair_tables``, ``_share_elementwise`` and the diagonal pass.

The unit cases run each function on small synthetic definitions and pin the
returned records (type, target, free indexes, limits, body, flags), the
arrays and namer side effects, and the inputs the function mutates.  The
end-to-end cases pin the SHA-256 of the Fortran text of models where the
passes fire, with each pass toggled off to show that it matters.
"""

import hashlib
import warnings
from pathlib import Path

import pytest

import sympy as sp

from thermodiff.codegen import CodegenOptions
from thermodiff.codegen.common import _LoopStart, _Namer
from thermodiff.codegen.ir import SiblingNest, TempDef
from thermodiff.codegen.loops import (
    _diagonal_from_tables,
    _drop_pair_tables,
    _share_elementwise,
)
from thermodiff.fortran import to_fortran
from thermodiff.model.emit import build_diff
from thermodiff.model.ir import build_ir
from thermodiff.model.parser import read_sections

ROOT = Path(__file__).resolve().parent.parent.parent
NC = sp.Symbol("nc", integer=True)
IDX_I = sp.Idx("i", NC)
IDX_J = sp.Idx("j", NC)
IDX_K = sp.Idx("k", NC)
IDX_M = sp.Idx("m", NC)
PARAM = sp.IndexedBase("a")
OTHER = sp.IndexedBase("b")
RANK3 = sp.IndexedBase("c")
SYMMETRIC = sp.IndexedBase("s")
SCALE = sp.Symbol("R")


def array(name):
    """Return the indexed base of a temporary."""
    return sp.IndexedBase(name)


def make_table(name, body, first=IDX_I, second=IDX_J):
    """Return a rank-2 table ``name(first, second) = body``."""
    return TempDef(
        target=array(name)[first, second],
        free=[first, second],
        limits=[],
        body=body,
    )


def make_scalar(name, body, limits=()):
    """Return a scalar definition, a reduction when ``limits`` is given."""
    return TempDef(
        target=sp.Symbol(name), free=[], limits=list(limits), body=body
    )


def make_map(name, index, body):
    """Return a per-component map ``name(index) = body``."""
    return TempDef(
        target=array(name)[index], free=[index], limits=[], body=body
    )


def full(index):
    """Return the limits ``1..nc`` of ``index``."""
    return (index, 1, NC)


def triangle(index, outer):
    """Return the limits of ``index`` from the outer index to ``nc``."""
    return (index, _LoopStart(outer), NC)


def flags(definition):
    """Return the sorted annotation names present on a definition."""
    return sorted(definition.present_keys() - {"target", "free", "limits"})


def summarise(definitions):
    """Return one comparable row per definition."""
    return [
        (
            type(definition).__name__,
            str(definition.target),
            [str(index) for index in definition.free],
            [str(limit) for limit in definition.limits],
            str(definition.body),
            flags(definition),
            str(definition.acc) if definition.acc is not None else None,
        )
        for definition in definitions
    ]


def fresh_namer(count):
    """Return a namer that already created ``t1..t<count>``."""
    namer = _Namer()
    for _ in range(count):
        namer("t")
    return namer


# ---------------------------------------------------------------- drop pair
def double_sum(name, body):
    """Return a scalar double reduction over ``i`` and ``j``."""
    return make_scalar(name, body, [full(IDX_I), full(IDX_J)])


def table_ij():
    """Return the pair table ``t1(i, j) = a(i)*b(j) + R``."""
    return make_table("t1", PARAM[IDX_I] * OTHER[IDX_J] + SCALE)


def pair_read(name="t1"):
    """Return the read ``t1(i, j)``."""
    return array(name)[IDX_I, IDX_J]


def case_drop_inline():
    """A table read by one double and one single reduction is inlined."""
    table = table_ij()
    return (
        [
            table,
            double_sum("t2", pair_read() * PARAM[IDX_I]),
            make_scalar("t3", array("t1")[IDX_K, IDX_K], [full(IDX_K)]),
        ],
        [],
        {"t1": 2, "t9": 1},
        3,
    )


def case_drop_triangular():
    """A mirrored table read by a triangular double reduction is inlined."""
    table = table_ij()
    table.mirror = True
    double = make_scalar(
        "t2",
        pair_read(),
        [full(IDX_I), triangle(IDX_J, IDX_I)],
    )
    return [table, double], [], {"t1": 2}, 2


def case_drop_mirror_not_triangular():
    """A mirrored table read by a full double reduction is kept."""
    table = table_ij()
    table.mirror = True
    return [table, double_sum("t2", pair_read())], [], {"t1": 2}, 2


def case_drop_read_by_output():
    """A table read by an output is kept."""
    return (
        [table_ij(), double_sum("t2", pair_read())],
        [pair_read() + 1],
        {"t1": 2},
        2,
    )


def case_drop_libm_body():
    """A table whose body calls libm is kept."""
    table = make_table("t1", sp.exp(PARAM[IDX_I] * OTHER[IDX_J]))
    return [table, double_sum("t2", pair_read())], [], {"t1": 2}, 2


def case_drop_two_doubles():
    """A table read by two double reductions is kept."""
    return (
        [
            table_ij(),
            double_sum("t2", pair_read()),
            double_sum("t3", pair_read() * 2),
        ],
        [],
        {"t1": 2},
        3,
    )


def case_drop_chain():
    """Dropping a table exposes a second table that reads it."""
    inner = table_ij()
    outer = make_table("t2", pair_read() * PARAM[IDX_I])
    return (
        [inner, outer, double_sum("t3", array("t2")[IDX_I, IDX_J])],
        [],
        {"t1": 2, "t2": 2},
        3,
    )


def case_factor_outer_and_inner():
    """Outer factors move out; a product over the inner index is a vector."""
    body = PARAM[IDX_I] * OTHER[IDX_J] * RANK3[IDX_J] * SCALE
    return [double_sum("t1", body)], [], {}, 1


def case_factor_shared_vector():
    """A second reduction reuses the vector of the same inner product."""
    first = PARAM[IDX_I] * OTHER[IDX_J] * RANK3[IDX_J]
    second = OTHER[IDX_J] * RANK3[IDX_J] * PARAM[IDX_I] * SCALE
    return (
        [double_sum("t1", first), double_sum("t2", second)],
        [],
        {},
        2,
    )


def case_factor_outer_vector():
    """The outer product reads a vector created by an earlier reduction."""
    first = PARAM[IDX_I] * OTHER[IDX_J] * RANK3[IDX_J]
    second = OTHER[IDX_I] * RANK3[IDX_I] * SCALE * PARAM[IDX_J]
    return (
        [double_sum("t1", first), double_sum("t2", second)],
        [],
        {},
        2,
    )


def case_factor_untouched():
    """Piecewise bodies and bodies without both factor kinds stay."""
    inner_only = double_sum("t1", OTHER[IDX_J] * RANK3[IDX_J])
    guarded = double_sum(
        "t2",
        sp.Piecewise((PARAM[IDX_I], PARAM[IDX_J] > 0), (0, True)) * SCALE,
    )
    plain = double_sum("t3", PARAM[IDX_I] + OTHER[IDX_J])
    return [inner_only, guarded, plain], [], {}, 3


def case_factor_vector_position():
    """A new vector goes right after the definition it reads."""
    reads = make_map("t1", IDX_J, PARAM[IDX_J] + 1)
    body = PARAM[IDX_I] * array("t1")[IDX_J] * OTHER[IDX_J]
    return (
        [make_scalar("t2", SCALE * 2), reads, double_sum("t3", body)],
        [],
        {"t1": 1},
        3,
    )


DROP_CASES = {
    "inline": case_drop_inline,
    "triangular": case_drop_triangular,
    "mirror_not_triangular": case_drop_mirror_not_triangular,
    "read_by_output": case_drop_read_by_output,
    "libm_body": case_drop_libm_body,
    "two_doubles": case_drop_two_doubles,
    "chain": case_drop_chain,
    "factor_outer_and_inner": case_factor_outer_and_inner,
    "factor_shared_vector": case_factor_shared_vector,
    "factor_outer_vector": case_factor_outer_vector,
    "factor_untouched": case_factor_untouched,
    "factor_vector_position": case_factor_vector_position,
}


def run_drop(case):
    """Run ``_drop_pair_tables`` and return everything observable."""
    definitions, outputs, arrays, created = DROP_CASES[case]()
    namer = fresh_namer(created)
    result = _drop_pair_tables(definitions, outputs, namer, arrays)
    return {
        "result": summarise(result),
        "inputs": summarise(definitions),
        "arrays": dict(arrays),
        "created": [str(symbol) for symbol in namer.created],
    }


# --------------------------------------------------------------- share elem
def log_of(scalar, index):
    """Return ``log(scalar * r(index))``."""
    return sp.log(scalar * PARAM[index])


def case_share_two_readers():
    """A reduction and an output share a log; spellings are merged."""
    ratio, alias = sp.Symbol("t1"), sp.Symbol("t2")
    return (
        [
            make_scalar("t1", SCALE * 2),
            make_scalar("t2", SCALE * 2),
            make_scalar("t3", log_of(ratio, IDX_K), [full(IDX_K)]),
        ],
        [log_of(alias, IDX_I)],
        {},
        3,
        40,
    )


def case_share_reductions_only():
    """Two reductions over the same libm vector."""
    ratio = sp.Symbol("t1")
    return (
        [
            make_scalar("t1", SCALE / 3),
            make_scalar("t2", log_of(ratio, IDX_K), [full(IDX_K)]),
            make_scalar("t3", log_of(ratio, IDX_J) * 2, [full(IDX_J)]),
        ],
        [],
        {},
        3,
        40,
    )


def case_share_maps_do_not_count():
    """Per-component maps do not count as readers."""
    ratio = sp.Symbol("t1")
    return (
        [
            make_scalar("t1", SCALE / 3),
            make_map("t2", IDX_K, log_of(ratio, IDX_K)),
            make_scalar("t3", log_of(ratio, IDX_J), [full(IDX_J)]),
        ],
        [],
        {},
        3,
        40,
    )


def case_share_max_ops():
    """With a small ``max_ops`` the scalars are not expanded."""
    first, second = sp.Symbol("t1"), sp.Symbol("t2")
    return (
        [
            make_scalar("t1", SCALE * 2 + 1),
            make_scalar("t2", SCALE * 2 + 1),
            make_scalar("t3", log_of(first, IDX_K), [full(IDX_K)]),
        ],
        [log_of(second, IDX_I)],
        {},
        3,
        1,
    )


def case_share_not_editable():
    """A mirrored definition is not rewritten."""
    ratio = sp.Symbol("t1")
    mirrored = make_scalar("t3", log_of(ratio, IDX_K), [full(IDX_K)])
    mirrored.mirror_sum = True
    return (
        [make_scalar("t1", SCALE / 3), mirrored],
        [log_of(ratio, IDX_I)],
        {},
        3,
        40,
    )


def case_share_two_keys():
    """The most read key goes first, then the other one."""
    ratio = sp.Symbol("t1")
    return (
        [
            make_scalar("t1", SCALE / 3),
            make_scalar("t2", log_of(ratio, IDX_K), [full(IDX_K)]),
            make_scalar(
                "t3",
                (log_of(ratio, IDX_K) + sp.exp(PARAM[IDX_K])),
                [full(IDX_K)],
            ),
        ],
        [sp.exp(PARAM[IDX_I]) + log_of(ratio, IDX_I)],
        {},
        3,
        40,
    )


def case_share_nothing():
    """No libm vector is read twice: the lists come back equal."""
    return (
        [
            make_scalar("t1", SCALE),
            make_scalar("t2", PARAM[IDX_K], [full(IDX_K)]),
        ],
        [sp.exp(PARAM[IDX_I])],
        {},
        2,
        40,
    )


SHARE_CASES = {
    "two_readers": case_share_two_readers,
    "reductions_only": case_share_reductions_only,
    "maps_do_not_count": case_share_maps_do_not_count,
    "max_ops": case_share_max_ops,
    "not_editable": case_share_not_editable,
    "two_keys": case_share_two_keys,
    "nothing": case_share_nothing,
}


def run_share(case):
    """Run ``_share_elementwise`` and return everything observable."""
    definitions, outputs, arrays, created, max_ops = SHARE_CASES[case]()
    namer = fresh_namer(created)
    result, outs = _share_elementwise(
        definitions, outputs, namer, arrays, max_ops=max_ops
    )
    return {
        "result": summarise(result),
        "outputs": [str(expression) for expression in outs],
        "inputs": summarise(definitions),
        "arrays": dict(arrays),
        "created": [str(symbol) for symbol in namer.created],
    }


# ----------------------------------------------------------------- diagonal
def mirrored_table(name="t1", body=None, limits=None):
    """Return ``t(a, b) = sum_m B(a, b, m)`` filled as a triangle."""
    body = RANK3[IDX_I, IDX_J, IDX_M] if body is None else body
    table = make_table(name, body)
    table.limits = [full(IDX_M)] if limits is None else limits
    table.mirror_sum = True
    return table


def diagonal_sum(name, body=None, limits=None):
    """Return ``d(k) = sum_m B(k, k, m)`` as a rank-1 reduction."""
    body = RANK3[IDX_K, IDX_K, IDX_M] if body is None else body
    return TempDef(
        target=array(name)[IDX_K],
        free=[IDX_K],
        limits=[full(IDX_M)] if limits is None else limits,
        body=body,
    )


def sibling_host(tables):
    """Return a sibling nest hosting ``tables``."""
    return SiblingNest(
        target=tables[0].target,
        free=list(tables[0].free),
        limits=list(tables[0].limits),
        body=tables[0].body,
        sibling=tables,
        accs=[sp.Symbol("acc%d" % n) for n, _ in enumerate(tables)],
    )


def case_diag_replaced():
    """The diagonal sum becomes a read of the table diagonal."""
    return [mirrored_table(), diagonal_sum("t2")], (), False


def case_diag_before_table():
    """A diagonal defined before the table moves right after it."""
    return [diagonal_sum("t2"), mirrored_table()], (), False


def case_diag_before_table_used():
    """A consumer between the diagonal and the table keeps everything."""
    consumer = make_scalar("t3", array("t2")[IDX_K], [full(IDX_K)])
    return [diagonal_sum("t2"), consumer, mirrored_table()], (), False


def case_diag_sibling_host():
    """The table lives in a sibling nest: the map follows the nest."""
    other = mirrored_table(
        "t5", OTHER[IDX_I, IDX_J] * RANK3[IDX_I, IDX_J, IDX_M]
    )
    host = sibling_host([mirrored_table(), other])
    return [host, diagonal_sum("t2")], (), False


def case_diag_other_limits():
    """Different summation limits: kept."""
    other = diagonal_sum("t2", limits=[(IDX_M, 2, NC)])
    return [mirrored_table(), other], (), False


def case_diag_different_body():
    """A diagonal with another body: kept."""
    other = diagonal_sum(
        "t2", OTHER[IDX_K, IDX_K] * RANK3[IDX_K, IDX_K, IDX_M]
    )
    return [mirrored_table(), other], (), False


def case_diag_reads_itself():
    """The table body reads the diagonal being replaced: kept."""
    table = mirrored_table(
        body=RANK3[IDX_I, IDX_J, IDX_M] * array("t2")[IDX_I]
    )
    diagonal = diagonal_sum(
        "t2", RANK3[IDX_K, IDX_K, IDX_M] * array("t2")[IDX_K]
    )
    return [table, diagonal], (), False


def case_diag_inlined_map():
    """A map read in the bodies is inlined before comparing."""
    table = mirrored_table(
        body=RANK3[IDX_I, IDX_J, IDX_M] * array("t7")[IDX_I]
    )
    diagonal = diagonal_sum(
        "t2", RANK3[IDX_K, IDX_K, IDX_M] * (PARAM[IDX_K] + 1)
    )
    return (
        [make_map("t7", IDX_I, PARAM[IDX_I] + 1), table, diagonal],
        (),
        False,
    )


def case_diag_symmetric_params():
    """Transposed symmetric parameters compare equal after sorting."""
    table = mirrored_table(
        body=SYMMETRIC[IDX_I, IDX_J] * RANK3[IDX_I, IDX_J, IDX_M]
    )
    diagonal = diagonal_sum(
        "t2", SYMMETRIC[IDX_K, IDX_K] * RANK3[IDX_K, IDX_K, IDX_M]
    )
    return [table, diagonal], ("s",), False


def case_diag_two_diagonals():
    """Each diagonal is looked up independently."""
    return (
        [
            mirrored_table(),
            diagonal_sum("t2"),
            diagonal_sum("t3", OTHER[IDX_K] * RANK3[IDX_K, IDX_K, IDX_M]),
        ],
        (),
        False,
    )


DIAG_CASES = {
    "replaced": case_diag_replaced,
    "before_table": case_diag_before_table,
    "before_table_used": case_diag_before_table_used,
    "sibling_host": case_diag_sibling_host,
    "other_limits": case_diag_other_limits,
    "different_body": case_diag_different_body,
    "reads_itself": case_diag_reads_itself,
    "inlined_map": case_diag_inlined_map,
    "symmetric_params": case_diag_symmetric_params,
    "two_diagonals": case_diag_two_diagonals,
}


def run_diagonal(case):
    """Run ``_diagonal_from_tables`` and return result and inputs."""
    definitions, symmetric, sym_rank3 = DIAG_CASES[case]()
    snapshot = list(definitions)
    result = _diagonal_from_tables(definitions, symmetric, sym_rank3=sym_rank3)
    return {
        "result": summarise(result),
        "same_objects": [any(r is s for s in snapshot) for r in result],
        "inputs": summarise(definitions),
    }


# ------------------------------------------------------------ end to end
def generate_code(model, derivatives, preset, **overrides):
    """Return the Fortran text of a model with a preset."""
    path = ROOT / model
    ir = build_ir(
        read_sections(path.read_text(encoding="utf-8"), str(path)), str(path)
    )
    diff, _ = build_diff(ir, derivatives=derivatives)
    options = CodegenOptions.from_preset(
        preset, optional_derivatives=False, **overrides
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


EXPECTED_DROP = {
    "chain": {
        "arrays": {},
        "created": ["t1", "t2", "t3", "t4"],
        "inputs": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                [],
                "R + a[i]*b[j]",
                ["body"],
                None,
            ),
            (
                "TempDef",
                "t2[i, j]",
                ["i", "j"],
                [],
                "a[i]*t1[i, j]",
                ["body"],
                None,
            ),
            (
                "TempDef",
                "t3",
                [],
                ["(i, 1, nc)", "(j, 1, nc)"],
                "(R + a[i]*b[j])*a[i]",
                ["body"],
                None,
            ),
        ],
        "result": [
            (
                "TempDef",
                "t3",
                [],
                ["(i, 1, nc)", "(j, 1, nc)"],
                "(R + a[i]*b[j])*a[i]",
                ["acc", "body", "factored"],
                "t4",
            )
        ],
    },
    "factor_outer_and_inner": {
        "arrays": {"t2": 1},
        "created": ["t1", "t2", "t3"],
        "inputs": [
            (
                "TempDef",
                "t1",
                [],
                ["(i, 1, nc)", "(j, 1, nc)"],
                "R*a[i]*b[j]*c[j]",
                ["body"],
                None,
            )
        ],
        "result": [
            ("TempDef", "t2[i]", ["i"], [], "b[i]*c[i]", ["body"], None),
            (
                "TempDef",
                "t1",
                [],
                ["(i, 1, nc)", "(j, 1, nc)"],
                "R*a[i]*t2[j]",
                ["acc", "body", "factored"],
                "t3",
            ),
        ],
    },
    "factor_outer_vector": {
        "arrays": {"t3": 1},
        "created": ["t1", "t2", "t3", "t4", "t5"],
        "inputs": [
            (
                "TempDef",
                "t1",
                [],
                ["(i, 1, nc)", "(j, 1, nc)"],
                "a[i]*b[j]*c[j]",
                ["body"],
                None,
            ),
            (
                "TempDef",
                "t2",
                [],
                ["(i, 1, nc)", "(j, 1, nc)"],
                "R*a[j]*b[i]*c[i]",
                ["body"],
                None,
            ),
        ],
        "result": [
            ("TempDef", "t3[i]", ["i"], [], "b[i]*c[i]", ["body"], None),
            (
                "TempDef",
                "t1",
                [],
                ["(i, 1, nc)", "(j, 1, nc)"],
                "a[i]*t3[j]",
                ["acc", "body", "factored"],
                "t4",
            ),
            (
                "TempDef",
                "t2",
                [],
                ["(i, 1, nc)", "(j, 1, nc)"],
                "R*a[j]*t3[i]",
                ["acc", "body", "factored"],
                "t5",
            ),
        ],
    },
    "factor_shared_vector": {
        "arrays": {"t3": 1},
        "created": ["t1", "t2", "t3", "t4", "t5"],
        "inputs": [
            (
                "TempDef",
                "t1",
                [],
                ["(i, 1, nc)", "(j, 1, nc)"],
                "a[i]*b[j]*c[j]",
                ["body"],
                None,
            ),
            (
                "TempDef",
                "t2",
                [],
                ["(i, 1, nc)", "(j, 1, nc)"],
                "R*a[i]*b[j]*c[j]",
                ["body"],
                None,
            ),
        ],
        "result": [
            ("TempDef", "t3[i]", ["i"], [], "b[i]*c[i]", ["body"], None),
            (
                "TempDef",
                "t1",
                [],
                ["(i, 1, nc)", "(j, 1, nc)"],
                "a[i]*t3[j]",
                ["acc", "body", "factored"],
                "t4",
            ),
            (
                "TempDef",
                "t2",
                [],
                ["(i, 1, nc)", "(j, 1, nc)"],
                "R*a[i]*t3[j]",
                ["acc", "body", "factored"],
                "t5",
            ),
        ],
    },
    "factor_untouched": {
        "arrays": {},
        "created": ["t1", "t2", "t3"],
        "inputs": [
            (
                "TempDef",
                "t1",
                [],
                ["(i, 1, nc)", "(j, 1, nc)"],
                "b[j]*c[j]",
                ["body"],
                None,
            ),
            (
                "TempDef",
                "t2",
                [],
                ["(i, 1, nc)", "(j, 1, nc)"],
                "R*Piecewise((a[i], a[j] > 0), (0, True))",
                ["body"],
                None,
            ),
            (
                "TempDef",
                "t3",
                [],
                ["(i, 1, nc)", "(j, 1, nc)"],
                "a[i] + b[j]",
                ["body"],
                None,
            ),
        ],
        "result": [
            (
                "TempDef",
                "t1",
                [],
                ["(i, 1, nc)", "(j, 1, nc)"],
                "b[j]*c[j]",
                ["body"],
                None,
            ),
            (
                "TempDef",
                "t2",
                [],
                ["(i, 1, nc)", "(j, 1, nc)"],
                "R*Piecewise((a[i], a[j] > 0), (0, True))",
                ["body"],
                None,
            ),
            (
                "TempDef",
                "t3",
                [],
                ["(i, 1, nc)", "(j, 1, nc)"],
                "a[i] + b[j]",
                ["body"],
                None,
            ),
        ],
    },
    "factor_vector_position": {
        "arrays": {"t1": 1, "t4": 1},
        "created": ["t1", "t2", "t3", "t4", "t5"],
        "inputs": [
            ("TempDef", "t2", [], [], "2*R", ["body"], None),
            ("TempDef", "t1[j]", ["j"], [], "a[j] + 1", ["body"], None),
            (
                "TempDef",
                "t3",
                [],
                ["(i, 1, nc)", "(j, 1, nc)"],
                "a[i]*b[j]*t1[j]",
                ["body"],
                None,
            ),
        ],
        "result": [
            ("TempDef", "t2", [], [], "2*R", ["body"], None),
            ("TempDef", "t1[j]", ["j"], [], "a[j] + 1", ["body"], None),
            ("TempDef", "t4[i]", ["i"], [], "b[i]*t1[i]", ["body"], None),
            (
                "TempDef",
                "t3",
                [],
                ["(i, 1, nc)", "(j, 1, nc)"],
                "a[i]*t4[j]",
                ["acc", "body", "factored"],
                "t5",
            ),
        ],
    },
    "inline": {
        "arrays": {"t9": 1},
        "created": ["t1", "t2", "t3", "t4"],
        "inputs": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                [],
                "R + a[i]*b[j]",
                ["body"],
                None,
            ),
            (
                "TempDef",
                "t2",
                [],
                ["(i, 1, nc)", "(j, 1, nc)"],
                "(R + a[i]*b[j])*a[i]",
                ["body"],
                None,
            ),
            (
                "TempDef",
                "t3",
                [],
                ["(k, 1, nc)"],
                "R + a[k]*b[k]",
                ["body"],
                None,
            ),
        ],
        "result": [
            (
                "TempDef",
                "t2",
                [],
                ["(i, 1, nc)", "(j, 1, nc)"],
                "(R + a[i]*b[j])*a[i]",
                ["acc", "body", "factored"],
                "t4",
            ),
            (
                "TempDef",
                "t3",
                [],
                ["(k, 1, nc)"],
                "R + a[k]*b[k]",
                ["body"],
                None,
            ),
        ],
    },
    "libm_body": {
        "arrays": {"t1": 2},
        "created": ["t1", "t2"],
        "inputs": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                [],
                "exp(a[i]*b[j])",
                ["body"],
                None,
            ),
            (
                "TempDef",
                "t2",
                [],
                ["(i, 1, nc)", "(j, 1, nc)"],
                "t1[i, j]",
                ["body"],
                None,
            ),
        ],
        "result": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                [],
                "exp(a[i]*b[j])",
                ["body"],
                None,
            ),
            (
                "TempDef",
                "t2",
                [],
                ["(i, 1, nc)", "(j, 1, nc)"],
                "t1[i, j]",
                ["body"],
                None,
            ),
        ],
    },
    "mirror_not_triangular": {
        "arrays": {"t1": 2},
        "created": ["t1", "t2"],
        "inputs": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                [],
                "R + a[i]*b[j]",
                ["body", "mirror"],
                None,
            ),
            (
                "TempDef",
                "t2",
                [],
                ["(i, 1, nc)", "(j, 1, nc)"],
                "t1[i, j]",
                ["body"],
                None,
            ),
        ],
        "result": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                [],
                "R + a[i]*b[j]",
                ["body", "mirror"],
                None,
            ),
            (
                "TempDef",
                "t2",
                [],
                ["(i, 1, nc)", "(j, 1, nc)"],
                "t1[i, j]",
                ["body"],
                None,
            ),
        ],
    },
    "read_by_output": {
        "arrays": {"t1": 2},
        "created": ["t1", "t2"],
        "inputs": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                [],
                "R + a[i]*b[j]",
                ["body"],
                None,
            ),
            (
                "TempDef",
                "t2",
                [],
                ["(i, 1, nc)", "(j, 1, nc)"],
                "t1[i, j]",
                ["body"],
                None,
            ),
        ],
        "result": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                [],
                "R + a[i]*b[j]",
                ["body"],
                None,
            ),
            (
                "TempDef",
                "t2",
                [],
                ["(i, 1, nc)", "(j, 1, nc)"],
                "t1[i, j]",
                ["body"],
                None,
            ),
        ],
    },
    "triangular": {
        "arrays": {},
        "created": ["t1", "t2"],
        "inputs": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                [],
                "R + a[i]*b[j]",
                ["body", "mirror"],
                None,
            ),
            (
                "TempDef",
                "t2",
                [],
                ["(i, 1, nc)", "(j, i + 1, nc)"],
                "R + a[i]*b[j]",
                ["body"],
                None,
            ),
        ],
        "result": [
            (
                "TempDef",
                "t2",
                [],
                ["(i, 1, nc)", "(j, i + 1, nc)"],
                "R + a[i]*b[j]",
                ["body"],
                None,
            )
        ],
    },
    "two_doubles": {
        "arrays": {"t1": 2},
        "created": ["t1", "t2", "t3", "t4"],
        "inputs": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                [],
                "R + a[i]*b[j]",
                ["body"],
                None,
            ),
            (
                "TempDef",
                "t2",
                [],
                ["(i, 1, nc)", "(j, 1, nc)"],
                "t1[i, j]",
                ["body"],
                None,
            ),
            (
                "TempDef",
                "t3",
                [],
                ["(i, 1, nc)", "(j, 1, nc)"],
                "2*t1[i, j]",
                ["body"],
                None,
            ),
        ],
        "result": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                [],
                "R + a[i]*b[j]",
                ["body"],
                None,
            ),
            (
                "TempDef",
                "t2",
                [],
                ["(i, 1, nc)", "(j, 1, nc)"],
                "t1[i, j]",
                ["body"],
                None,
            ),
            (
                "TempDef",
                "t3",
                [],
                ["(i, 1, nc)", "(j, 1, nc)"],
                "2*t1[i, j]",
                ["acc", "body", "factored"],
                "t4",
            ),
        ],
    },
}


EXPECTED_SHARE = {
    "maps_do_not_count": {
        "arrays": {},
        "created": ["t1", "t2", "t3"],
        "inputs": [
            ("TempDef", "t1", [], [], "R/3", ["body"], None),
            ("TempDef", "t2[k]", ["k"], [], "log(t1*a[k])", ["body"], None),
            (
                "TempDef",
                "t3",
                [],
                ["(j, 1, nc)"],
                "log(t1*a[j])",
                ["body"],
                None,
            ),
        ],
        "outputs": [],
        "result": [
            ("TempDef", "t1", [], [], "R/3", ["body"], None),
            ("TempDef", "t2[k]", ["k"], [], "log(t1*a[k])", ["body"], None),
            (
                "TempDef",
                "t3",
                [],
                ["(j, 1, nc)"],
                "log(t1*a[j])",
                ["body"],
                None,
            ),
        ],
    },
    "max_ops": {
        "arrays": {},
        "created": ["t1", "t2", "t3"],
        "inputs": [
            ("TempDef", "t1", [], [], "2*R + 1", ["body"], None),
            ("TempDef", "t2", [], [], "2*R + 1", ["body"], None),
            (
                "TempDef",
                "t3",
                [],
                ["(k, 1, nc)"],
                "log(t1*a[k])",
                ["body"],
                None,
            ),
        ],
        "outputs": ["log(t2*a[i])"],
        "result": [
            ("TempDef", "t1", [], [], "2*R + 1", ["body"], None),
            ("TempDef", "t2", [], [], "2*R + 1", ["body"], None),
            (
                "TempDef",
                "t3",
                [],
                ["(k, 1, nc)"],
                "log(t1*a[k])",
                ["body"],
                None,
            ),
        ],
    },
    "not_editable": {
        "arrays": {},
        "created": ["t1", "t2", "t3"],
        "inputs": [
            ("TempDef", "t1", [], [], "R/3", ["body"], None),
            (
                "TempDef",
                "t3",
                [],
                ["(k, 1, nc)"],
                "log(t1*a[k])",
                ["body", "mirror_sum"],
                None,
            ),
        ],
        "outputs": ["log(t1*a[i])"],
        "result": [
            ("TempDef", "t1", [], [], "R/3", ["body"], None),
            (
                "TempDef",
                "t3",
                [],
                ["(k, 1, nc)"],
                "log(t1*a[k])",
                ["body", "mirror_sum"],
                None,
            ),
        ],
    },
    "nothing": {
        "arrays": {},
        "created": ["t1", "t2"],
        "inputs": [
            ("TempDef", "t1", [], [], "R", ["body"], None),
            ("TempDef", "t2", [], ["(k, 1, nc)"], "a[k]", ["body"], None),
        ],
        "outputs": ["exp(a[i])"],
        "result": [
            ("TempDef", "t1", [], [], "R", ["body"], None),
            ("TempDef", "t2", [], ["(k, 1, nc)"], "a[k]", ["body"], None),
        ],
    },
    "reductions_only": {
        "arrays": {"t4": 1},
        "created": ["t1", "t2", "t3", "t4"],
        "inputs": [
            ("TempDef", "t1", [], [], "R/3", ["body"], None),
            (
                "TempDef",
                "t2",
                [],
                ["(k, 1, nc)"],
                "log(t1*a[k])",
                ["body"],
                None,
            ),
            (
                "TempDef",
                "t3",
                [],
                ["(j, 1, nc)"],
                "2*log(t1*a[j])",
                ["body"],
                None,
            ),
        ],
        "outputs": [],
        "result": [
            ("TempDef", "t1", [], [], "R/3", ["body"], None),
            ("TempDef", "t4[k]", ["k"], [], "log(t1*a[k])", ["body"], None),
            ("TempDef", "t2", [], ["(k, 1, nc)"], "t4[k]", ["body"], None),
            ("TempDef", "t3", [], ["(j, 1, nc)"], "2*t4[j]", ["body"], None),
        ],
    },
    "two_keys": {
        "arrays": {"t4": 1, "t5": 1},
        "created": ["t1", "t2", "t3", "t4", "t5"],
        "inputs": [
            ("TempDef", "t1", [], [], "R/3", ["body"], None),
            (
                "TempDef",
                "t2",
                [],
                ["(k, 1, nc)"],
                "log(t1*a[k])",
                ["body"],
                None,
            ),
            (
                "TempDef",
                "t3",
                [],
                ["(k, 1, nc)"],
                "exp(a[k]) + log(t1*a[k])",
                ["body"],
                None,
            ),
        ],
        "outputs": ["t5[i]"],
        "result": [
            ("TempDef", "t1", [], [], "R/3", ["body"], None),
            ("TempDef", "t4[k]", ["k"], [], "log(t1*a[k])", ["body"], None),
            ("TempDef", "t2", [], ["(k, 1, nc)"], "t4[k]", ["body"], None),
            (
                "TempDef",
                "t5[k]",
                ["k"],
                [],
                "exp(a[k]) + t4[k]",
                ["body"],
                None,
            ),
            ("TempDef", "t3", [], ["(k, 1, nc)"], "t5[k]", ["body"], None),
        ],
    },
    "two_readers": {
        "arrays": {"t4": 1},
        "created": ["t1", "t3", "t4"],
        "inputs": [
            ("TempDef", "t1", [], [], "2*R", ["body"], None),
            ("TempDef", "t2", [], [], "2*R", ["body"], None),
            (
                "TempDef",
                "t3",
                [],
                ["(k, 1, nc)"],
                "log(t1*a[k])",
                ["body"],
                None,
            ),
        ],
        "outputs": ["t4[i]"],
        "result": [
            ("TempDef", "t1", [], [], "2*R", ["body"], None),
            ("TempDef", "t4[k]", ["k"], [], "log(t1*a[k])", ["body"], None),
            ("TempDef", "t3", [], ["(k, 1, nc)"], "t4[k]", ["body"], None),
        ],
    },
}


EXPECTED_DIAGONAL = {
    "before_table": {
        "inputs": [
            (
                "TempDef",
                "t2[k]",
                ["k"],
                ["(m, 1, nc)"],
                "c[k, k, m]",
                ["body"],
                None,
            ),
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "c[i, j, m]",
                ["body", "mirror_sum"],
                None,
            ),
        ],
        "result": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "c[i, j, m]",
                ["body", "mirror_sum"],
                None,
            ),
            ("TempDef", "t2[k]", ["k"], [], "t1[k, k]", ["body"], None),
        ],
        "same_objects": [True, False],
    },
    "before_table_used": {
        "inputs": [
            (
                "TempDef",
                "t2[k]",
                ["k"],
                ["(m, 1, nc)"],
                "c[k, k, m]",
                ["body"],
                None,
            ),
            ("TempDef", "t3", [], ["(k, 1, nc)"], "t2[k]", ["body"], None),
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "c[i, j, m]",
                ["body", "mirror_sum"],
                None,
            ),
        ],
        "result": [
            (
                "TempDef",
                "t2[k]",
                ["k"],
                ["(m, 1, nc)"],
                "c[k, k, m]",
                ["body"],
                None,
            ),
            ("TempDef", "t3", [], ["(k, 1, nc)"], "t2[k]", ["body"], None),
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "c[i, j, m]",
                ["body", "mirror_sum"],
                None,
            ),
        ],
        "same_objects": [True, True, True],
    },
    "different_body": {
        "inputs": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "c[i, j, m]",
                ["body", "mirror_sum"],
                None,
            ),
            (
                "TempDef",
                "t2[k]",
                ["k"],
                ["(m, 1, nc)"],
                "b[k, k]*c[k, k, m]",
                ["body"],
                None,
            ),
        ],
        "result": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "c[i, j, m]",
                ["body", "mirror_sum"],
                None,
            ),
            (
                "TempDef",
                "t2[k]",
                ["k"],
                ["(m, 1, nc)"],
                "b[k, k]*c[k, k, m]",
                ["body"],
                None,
            ),
        ],
        "same_objects": [True, True],
    },
    "inlined_map": {
        "inputs": [
            ("TempDef", "t7[i]", ["i"], [], "a[i] + 1", ["body"], None),
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "c[i, j, m]*t7[i]",
                ["body", "mirror_sum"],
                None,
            ),
            (
                "TempDef",
                "t2[k]",
                ["k"],
                ["(m, 1, nc)"],
                "(a[k] + 1)*c[k, k, m]",
                ["body"],
                None,
            ),
        ],
        "result": [
            ("TempDef", "t7[i]", ["i"], [], "a[i] + 1", ["body"], None),
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "c[i, j, m]*t7[i]",
                ["body", "mirror_sum"],
                None,
            ),
            ("TempDef", "t2[k]", ["k"], [], "t1[k, k]", ["body"], None),
        ],
        "same_objects": [True, True, False],
    },
    "other_limits": {
        "inputs": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "c[i, j, m]",
                ["body", "mirror_sum"],
                None,
            ),
            (
                "TempDef",
                "t2[k]",
                ["k"],
                ["(m, 2, nc)"],
                "c[k, k, m]",
                ["body"],
                None,
            ),
        ],
        "result": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "c[i, j, m]",
                ["body", "mirror_sum"],
                None,
            ),
            (
                "TempDef",
                "t2[k]",
                ["k"],
                ["(m, 2, nc)"],
                "c[k, k, m]",
                ["body"],
                None,
            ),
        ],
        "same_objects": [True, True],
    },
    "reads_itself": {
        "inputs": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "c[i, j, m]*t2[i]",
                ["body", "mirror_sum"],
                None,
            ),
            (
                "TempDef",
                "t2[k]",
                ["k"],
                ["(m, 1, nc)"],
                "c[k, k, m]*t2[k]",
                ["body"],
                None,
            ),
        ],
        "result": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "c[i, j, m]*t2[i]",
                ["body", "mirror_sum"],
                None,
            ),
            (
                "TempDef",
                "t2[k]",
                ["k"],
                ["(m, 1, nc)"],
                "c[k, k, m]*t2[k]",
                ["body"],
                None,
            ),
        ],
        "same_objects": [True, True],
    },
    "replaced": {
        "inputs": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "c[i, j, m]",
                ["body", "mirror_sum"],
                None,
            ),
            (
                "TempDef",
                "t2[k]",
                ["k"],
                ["(m, 1, nc)"],
                "c[k, k, m]",
                ["body"],
                None,
            ),
        ],
        "result": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "c[i, j, m]",
                ["body", "mirror_sum"],
                None,
            ),
            ("TempDef", "t2[k]", ["k"], [], "t1[k, k]", ["body"], None),
        ],
        "same_objects": [True, False],
    },
    "sibling_host": {
        "inputs": [
            (
                "SiblingNest",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "c[i, j, m]",
                ["accs", "body", "sibling"],
                None,
            ),
            (
                "TempDef",
                "t2[k]",
                ["k"],
                ["(m, 1, nc)"],
                "c[k, k, m]",
                ["body"],
                None,
            ),
        ],
        "result": [
            (
                "SiblingNest",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "c[i, j, m]",
                ["accs", "body", "sibling"],
                None,
            ),
            ("TempDef", "t2[k]", ["k"], [], "t1[k, k]", ["body"], None),
        ],
        "same_objects": [True, False],
    },
    "symmetric_params": {
        "inputs": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "c[i, j, m]*s[i, j]",
                ["body", "mirror_sum"],
                None,
            ),
            (
                "TempDef",
                "t2[k]",
                ["k"],
                ["(m, 1, nc)"],
                "c[k, k, m]*s[k, k]",
                ["body"],
                None,
            ),
        ],
        "result": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "c[i, j, m]*s[i, j]",
                ["body", "mirror_sum"],
                None,
            ),
            ("TempDef", "t2[k]", ["k"], [], "t1[k, k]", ["body"], None),
        ],
        "same_objects": [True, False],
    },
    "two_diagonals": {
        "inputs": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "c[i, j, m]",
                ["body", "mirror_sum"],
                None,
            ),
            (
                "TempDef",
                "t2[k]",
                ["k"],
                ["(m, 1, nc)"],
                "c[k, k, m]",
                ["body"],
                None,
            ),
            (
                "TempDef",
                "t3[k]",
                ["k"],
                ["(m, 1, nc)"],
                "b[k]*c[k, k, m]",
                ["body"],
                None,
            ),
        ],
        "result": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "c[i, j, m]",
                ["body", "mirror_sum"],
                None,
            ),
            ("TempDef", "t2[k]", ["k"], [], "t1[k, k]", ["body"], None),
            (
                "TempDef",
                "t3[k]",
                ["k"],
                ["(m, 1, nc)"],
                "b[k]*c[k, k, m]",
                ["body"],
                None,
            ),
        ],
        "same_objects": [True, False, True],
    },
}


@pytest.mark.parametrize("case", sorted(DROP_CASES))
def test_drop_pair_tables_is_pinned(case):
    assert run_drop(case) == EXPECTED_DROP[case]


@pytest.mark.parametrize("case", sorted(SHARE_CASES))
def test_share_elementwise_is_pinned(case):
    assert run_share(case) == EXPECTED_SHARE[case]


@pytest.mark.parametrize("case", sorted(DIAG_CASES))
def test_diagonal_from_tables_is_pinned(case):
    assert run_diagonal(case) == EXPECTED_DIAGONAL[case]


PCSAFT = "tests/data/models/pcsaft/pcsaft.model"
CMR = "tests/data/models/cmr/cmr.model"
# (model, derivatives, option) -> (fastest, fastest without the option,
# standard); the option is the one that makes the target pass fire
MODEL_PINS = {
    "drop_pair_tables": (
        PCSAFT,
        "none",
        "drop_pair_tables",
        "f3cb8558d08c055ca32fea5b2155235c070b3a5d4496ad542cfabc034be1f415",
        "5904acdb110531985b1fdd3a4971615a72aa4bd80d5a2f5c5e9851a88bc5dc2f",
        "a0bb02d68ce42a730c4cf5bed43b081f3f5198ad93e409eb8d17405daa1150af",
    ),
    "share_elementwise": (
        PCSAFT,
        "dn",
        "share_elementwise",
        "6dcf2193a0eefea839f4fcfae32cb14b1eeca999d6b8a18890870242abff16fa",
        "8bb73c43762c7c1fff616ca0956c0f8333602155a52c155564d1ffd4dfe6834a",
        "97fb6bf6f72785b7c75ae33a83cfc229274b834b01d7969e39818d52468fec7d",
    ),
    "diag_table": (
        CMR,
        "dn",
        "diag_table",
        "1f3d932aed3e2dbd387c035b7e82d25dd3aebfd8c5d42bec9066f77bffa58c39",
        "53b3ebb9b822258a7c69b60a0fc9b5e9f554fe5928523dae9fc25979070dd4b9",
        "811b66da262cc939ac00716cf78c063b2c7e120d4024b433fe18e0d5798d6181",
    ),
    "cmr_drop_pair_tables": (
        CMR,
        "dn",
        "drop_pair_tables",
        "1f3d932aed3e2dbd387c035b7e82d25dd3aebfd8c5d42bec9066f77bffa58c39",
        "3f162aece97ce3ec53711bed9684c3f06ed748a5c41d9af9a7e3509acddb19fc",
        "811b66da262cc939ac00716cf78c063b2c7e120d4024b433fe18e0d5798d6181",
    ),
}


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(MODEL_PINS))
def test_table_passes_text_is_pinned(name):
    model, derivatives, option, fastest, without, standard = MODEL_PINS[name]
    assert digest(generate_code(model, derivatives, "fastest")) == fastest
    code = generate_code(model, derivatives, "fastest", **{option: False})
    assert digest(code) == without
    assert without != fastest
    assert digest(generate_code(model, derivatives, "standard")) == standard
