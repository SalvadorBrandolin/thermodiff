"""Pin the fusion and nesting passes of ``thermodiff.codegen.loops``.

Covers ``_nest_double_sums``, ``_fuse_temporaries``, ``_fuse_loop_nests``,
``_permute_symmetric_subscripts``, ``_fuse_component_maps`` and
``_fuse_sibling_tables``.  The unit cases run each function on small
synthetic inputs and pin the returned values, the mutated inputs, the object
identities and the names drawn from the namer.  The end-to-end cases pin the
SHA-256 of the Fortran text of models where the passes fire, with the option
that enables each pass turned off to show that it matters.
"""

import hashlib
import warnings
from pathlib import Path

import pytest

import sympy as sp

from thermodiff.codegen import CodegenOptions
from thermodiff.codegen.common import _LoopStart, _Namer
from thermodiff.codegen.ir import (
    ComponentFusedMap,
    FusedNest,
    SiblingNest,
    TempDef,
)
from thermodiff.codegen.loops import (
    _fuse_component_maps,
    _fuse_loop_nests,
    _fuse_sibling_tables,
    _fuse_temporaries,
    _nest_double_sums,
    _permute_symmetric_subscripts,
)
from thermodiff.fortran import to_fortran
from thermodiff.model.emit import build_diff
from thermodiff.model.ir import build_ir
from thermodiff.model.parser import read_sections

ROOT = Path(__file__).resolve().parent.parent.parent
NC = sp.Symbol("nc", integer=True)
IDX_I = sp.Idx("i")
IDX_J = sp.Idx("j")
IDX_K = sp.Idx("k")
IDX_L = sp.Idx("l")
IDX_M = sp.Idx("m")
IDX_P = sp.Idx("p")
WEIGHT = sp.IndexedBase("n")
MATRIX = sp.IndexedBase("mm")
SCALE = sp.IndexedBase("w")
OTHER = sp.IndexedBase("u")
SYM2 = sp.IndexedBase("s")
SYM3 = sp.IndexedBase("c3")
PLAIN3 = sp.IndexedBase("d3")
PLAIN4 = sp.IndexedBase("d4")
TEMP = sp.Symbol("T")
RATE = sp.Symbol("R")


def array(name):
    """Return the indexed base of a temporary."""
    return sp.IndexedBase(name)


def full(index):
    """Return the limits ``1..nc`` of ``index``."""
    return (index, 1, NC)


def fresh_namer(count):
    """Return a namer that already created ``t1..t<count>``."""
    namer = _Namer()
    for _ in range(count):
        namer("t")
    return namer


def make_table(name, body, first=IDX_I, second=IDX_J):
    """Return a rank-2 table ``name(first, second) = body``."""
    return TempDef(
        target=array(name)[first, second],
        free=[first, second],
        limits=[],
        body=body,
    )


def make_map(name, index, body):
    """Return a per-component map ``name(index) = body``."""
    return TempDef(
        target=array(name)[index], free=[index], limits=[], body=body
    )


def make_scalar(name, body):
    """Return a scalar definition."""
    return TempDef(target=sp.Symbol(name), free=[], limits=[], body=body)


def make_vector_sum(name, free, bound, body, **flags):
    """Return ``name(free) = sum_bound body`` over ``1..nc``."""
    return TempDef(
        target=array(name)[free],
        free=[free],
        limits=[full(bound)],
        body=body,
        **flags,
    )


def make_mirror_table(name, free, bound, body):
    """Return a symmetric rank-2 table filled by a sum."""
    return TempDef(
        target=array(name)[free[0], free[1]],
        free=list(free),
        limits=[full(bound)],
        body=body,
        mirror_sum=True,
    )


def double_sum(body, first=IDX_K, second=IDX_L):
    """Return ``sum_first sum_second body``."""
    return sp.Sum(body, full(first), full(second))


def flags(definition):
    """Return the sorted annotation names present on a definition."""
    return sorted(definition.present_keys() - {"target", "free", "limits"})


def members_of(definition):
    """Return the nested members of a wrapper record, if it has any."""
    if isinstance(definition, FusedNest):
        return summarise(definition.fused)
    if isinstance(definition, SiblingNest):
        return summarise(definition.sibling)
    return None


def steps_of(definition):
    """Return the steps and members of a component-fused map."""
    if not isinstance(definition, ComponentFusedMap):
        return None
    return (
        [
            (kind, str(sym), str(value))
            for kind, sym, value in definition.steps
        ],
        [(str(target), str(body)) for target, body in definition.members],
    )


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
            members_of(definition),
            steps_of(definition),
            [str(acc) for acc in getattr(definition, "accs", None) or []],
        )
        for definition in definitions
    ]


# ------------------------------------------------------------ nest sums
def nest_vector_and_double():
    """A double sum nests onto a vector sum an output already needs."""
    vector = sp.Sum(WEIGHT[IDX_L] * MATRIX[IDX_K, IDX_L], full(IDX_L))
    double = double_sum(WEIGHT[IDX_K] * WEIGHT[IDX_L] * MATRIX[IDX_K, IDX_L])
    staged = [("dn", "out1", vector), ("dnn", "out2", double)]
    return staged, {}


def nest_no_vector():
    """A double sum with no matching vector is left alone."""
    double = double_sum(WEIGHT[IDX_K] * WEIGHT[IDX_L] * MATRIX[IDX_K, IDX_L])
    return [("dnn", "out1", double)], {}


def nest_shared_inner():
    """Two double sums share an inner sum: it becomes a vector."""
    first = double_sum(
        WEIGHT[IDX_K] * SCALE[IDX_K] * WEIGHT[IDX_L] * MATRIX[IDX_K, IDX_L]
    )
    second = double_sum(
        WEIGHT[IDX_K] * OTHER[IDX_K] * WEIGHT[IDX_L] * MATRIX[IDX_K, IDX_L]
    )
    staged = [("a", "out1", first), ("b", "out2", second)]
    return staged, {"shared_inner": True}


def nest_shared_inner_off():
    """The same two double sums, without ``shared_inner``."""
    staged, _ = nest_shared_inner()
    return staged, {}


def nest_unequal_ranges():
    """Bound indices with different ranges are not nested."""
    vector = sp.Sum(WEIGHT[IDX_L] * MATRIX[IDX_K, IDX_L], full(IDX_L))
    double = sp.Sum(
        WEIGHT[IDX_K] * WEIGHT[IDX_L] * MATRIX[IDX_K, IDX_L],
        full(IDX_K),
        (IDX_L, 2, NC),
    )
    return [("a", "out1", vector), ("b", "out2", double)], {}


def nest_not_a_product():
    """A sum whose body is not a product is not nested."""
    vector = sp.Sum(WEIGHT[IDX_L] * MATRIX[IDX_K, IDX_L], full(IDX_L))
    double = double_sum(WEIGHT[IDX_K] + MATRIX[IDX_K, IDX_L])
    return [("a", "out1", vector), ("b", "out2", double)], {}


def nest_symmetric():
    """Transposed subscripts of a symmetric parameter find the vector."""
    vector = sp.Sum(WEIGHT[IDX_L] * SYM2[IDX_K, IDX_L], full(IDX_L))
    double = double_sum(WEIGHT[IDX_K] * WEIGHT[IDX_L] * SYM2[IDX_L, IDX_K])
    staged = [("a", "out1", vector), ("b", "out2", double)]
    return staged, {"symmetric_params": ("s",)}


def triple_body():
    """Return the body of a sum over three indices."""
    return (
        SCALE[IDX_K]
        * OTHER[IDX_L]
        * WEIGHT[IDX_M]
        * 2
        * PLAIN3[IDX_K, IDX_L, IDX_M]
    )


def triple_sum(body):
    """Return a sum over ``k``, ``l`` and ``m``."""
    return sp.Sum(body, full(IDX_K), full(IDX_L), full(IDX_M))


def nest_general_vector():
    """A triple sum nests onto a rank-2 table an output needs."""
    table = sp.Sum(WEIGHT[IDX_M] * PLAIN3[IDX_K, IDX_L, IDX_M], full(IDX_M))
    staged = [("a", "out1", table), ("b", "out2", triple_sum(triple_body()))]
    return staged, {"general": True, "sym_rank3": True}


def nest_general_cost():
    """With ``cost`` a triple sum nests with no vector to share."""
    staged = [("a", "out1", triple_sum(triple_body()))]
    return staged, {"general": True, "cost": True}


def nest_general_no_cost():
    """Without ``cost`` a lone triple sum stays."""
    staged = [("a", "out1", triple_sum(triple_body()))]
    return staged, {"general": True}


def nest_general_pair_cost():
    """General nesting of a double sum, ordered by cost."""
    double = double_sum(WEIGHT[IDX_K] * WEIGHT[IDX_L] * MATRIX[IDX_K, IDX_L])
    return [("a", "out1", double)], {"general": True, "cost": True}


def nest_general_unequal_ranges():
    """A triple sum over different ranges has no nesting."""
    body = triple_body()
    total = sp.Sum(body, full(IDX_K), full(IDX_L), (IDX_M, 2, NC))
    return [("a", "out1", total)], {"general": True, "cost": True}


def nest_general_not_product():
    """A triple sum whose body is a sum has no nesting."""
    total = triple_sum(WEIGHT[IDX_K] + MATRIX[IDX_L, IDX_M])
    return [("a", "out1", total)], {"general": True, "cost": True}


def nest_general_no_outer_factors():
    """Bodies with no factor on the outer indices are not nested."""
    body = SCALE[IDX_K] * OTHER[IDX_L]
    return [("a", "out1", triple_sum(body))], {"general": True, "cost": True}


def nest_general_inner_too_wide():
    """An inner sum with three free indices is not a candidate."""
    body = SCALE[IDX_K] * PLAIN4[IDX_K, IDX_L, IDX_M, IDX_P]
    total = sp.Sum(body, full(IDX_K), full(IDX_L), full(IDX_M), full(IDX_P))
    return [("a", "out1", total)], {"general": True, "cost": True}


def nest_pair_no_outer_factors():
    """A double sum with no factor on one index is not nested."""
    double = double_sum(SCALE[IDX_K] * OTHER[IDX_K])
    return [("a", "out1", double)], {"shared_inner": True}


def nest_general_symmetric_rank3():
    """A symmetric rank-3 parameter with ``sym_rank3``."""
    table = sp.Sum(WEIGHT[IDX_M] * SYM3[IDX_K, IDX_L, IDX_M], full(IDX_M))
    body = (
        SCALE[IDX_K] * OTHER[IDX_L] * WEIGHT[IDX_M] * SYM3[IDX_M, IDX_L, IDX_K]
    )
    return (
        [("a", "out1", table), ("b", "out2", triple_sum(body))],
        {
            "general": True,
            "sym_rank3": True,
            "symmetric_params": ("c3",),
            "shared_inner": True,
        },
    )


def nest_outer_factor_hoisted():
    """Factors free of the bound indices leave the sums first."""
    vector = sp.Sum(WEIGHT[IDX_L] * MATRIX[IDX_K, IDX_L], full(IDX_L))
    double = double_sum(
        RATE * WEIGHT[IDX_K] * WEIGHT[IDX_L] * MATRIX[IDX_K, IDX_L]
    )
    return [("a", "out1", vector), ("b", "out2", double + RATE)], {}


NEST_CASES = {
    "vector_and_double": nest_vector_and_double,
    "no_vector": nest_no_vector,
    "shared_inner": nest_shared_inner,
    "shared_inner_off": nest_shared_inner_off,
    "unequal_ranges": nest_unequal_ranges,
    "not_a_product": nest_not_a_product,
    "symmetric": nest_symmetric,
    "general_vector": nest_general_vector,
    "general_cost": nest_general_cost,
    "general_no_cost": nest_general_no_cost,
    "general_pair_cost": nest_general_pair_cost,
    "general_symmetric_rank3": nest_general_symmetric_rank3,
    "general_unequal_ranges": nest_general_unequal_ranges,
    "general_no_outer_factors": nest_general_no_outer_factors,
    "general_inner_too_wide": nest_general_inner_too_wide,
    "pair_no_outer_factors": nest_pair_no_outer_factors,
    "general_not_product": nest_general_not_product,
    "outer_factor_hoisted": nest_outer_factor_hoisted,
}


def run_nest(case):
    """Run ``_nest_double_sums`` and return everything observable."""
    staged, options = NEST_CASES[case]()
    options = dict(options)
    options.setdefault("sym_rank3", False)
    result = _nest_double_sums(staged, [IDX_I, IDX_J], **options)
    return {
        "result": [(a, n, str(e)) for a, n, e in result],
        "inputs": [(a, n, str(e)) for a, n, e in staged],
    }


# ------------------------------------------------------ fuse temporaries
def read_sum(name, outer=IDX_I, inner=IDX_K, start=1):
    """Return ``sum_inner name(outer, inner) * w(inner)``."""
    return sp.Sum(array(name)[outer, inner] * SCALE[inner], (inner, start, NC))


def cheap_body():
    """Return a cheap body of a rank-2 temporary."""
    return WEIGHT[IDX_J] * MATRIX[IDX_I, IDX_J]


def costly_body():
    """Return a costly body (a libm call) of a rank-2 temporary."""
    return sp.exp(MATRIX[IDX_I, IDX_J] / TEMP)


def temp_inline_cheap():
    """A cheap table read inside two sums is inlined."""
    other_rank = make_map("t9", IDX_I, WEIGHT[IDX_I])
    with_limits = make_vector_sum("t8", IDX_I, IDX_J, WEIGHT[IDX_J])
    defs = [
        other_rank,
        with_limits,
        make_table("t1", cheap_body()),
    ]
    exprs = [read_sum("t1"), read_sum("t1", IDX_L, IDX_M) + RATE]
    return defs, exprs, {}


def temp_outside_costly():
    """A non-trivial table that is also read outside a sum is kept."""
    body = cheap_body() + RATE * SCALE[IDX_I] * OTHER[IDX_J]
    exprs = [read_sum("t1"), array("t1")[IDX_K, IDX_L] + 1]
    return [make_table("t1", body)], exprs, {}


def temp_outside_trivial():
    """A trivial table is inlined even when read outside a sum."""
    exprs = [read_sum("t1"), array("t1")[IDX_K, IDX_L] + 1]
    return [make_table("t1", cheap_body())], exprs, {}


def temp_costly_two_sums():
    """A costly table read by two different sums is kept (weight 2)."""
    exprs = [read_sum("t1"), read_sum("t1", IDX_L, IDX_M)]
    return [make_table("t1", costly_body())], exprs, {}


def temp_costly_one_sum():
    """A costly table read by one sum is inlined."""
    return [make_table("t1", costly_body())], [read_sum("t1")], {}


def temp_costly_dedup():
    """Two equal sums count once with ``dedup``."""
    exprs = [read_sum("t1"), read_sum("t1") * 2]
    return [make_table("t1", costly_body())], exprs, {"dedup": True}


def temp_costly_no_dedup():
    """The same two equal sums count twice without ``dedup``."""
    exprs = [read_sum("t1"), read_sum("t1") * 2]
    return [make_table("t1", costly_body())], exprs, {}


def folded_reads(first_weight, second_weight):
    """Return two folded sums that read ``t1`` with different weights."""
    start = _LoopStart(IDX_I)
    return [
        sp.Sum(
            array("t1")[IDX_I, IDX_K] * first_weight[IDX_K], (IDX_K, start, NC)
        ),
        sp.Sum(
            array("t1")[IDX_I, IDX_K] * second_weight[IDX_K],
            (IDX_K, start, NC),
        ),
    ]


def temp_guard_folded():
    """``guard`` keeps a costly table read by two sums of weight one half."""
    exprs = folded_reads(SCALE, OTHER)
    return [make_table("t1", costly_body())], exprs, {"guard": True}


def temp_guard_off_folded():
    """Without ``guard`` the same two folded sums inline the table."""
    exprs = folded_reads(SCALE, OTHER)
    return [make_table("t1", costly_body())], exprs, {}


def temp_triangle_weight():
    """A folded (triangular) sum weighs one half."""
    start = _LoopStart(IDX_I)
    diagonal = sp.Sum(array("t1")[IDX_K, IDX_K], full(IDX_K))
    exprs = [read_sum("t1", IDX_I, IDX_K, start), diagonal]
    return [make_table("t1", costly_body())], exprs, {}


def temp_diagonal_reads():
    """Sums that read only the diagonal weigh nothing."""
    inner = sp.Sum(array("t1")[IDX_K, IDX_K] * SCALE[IDX_K], full(IDX_K))
    exprs = [inner, inner + 1, read_sum("t1")]
    return [make_table("t1", costly_body())], exprs, {}


def temp_nested_sum_weight():
    """A sum holding another sum is skipped when weighing reads."""
    inner = sp.Sum(array("t1")[IDX_K, IDX_L] * SCALE[IDX_L], full(IDX_L))
    outer = sp.Sum(inner * WEIGHT[IDX_K], full(IDX_K))
    return [make_table("t1", costly_body())], [outer], {"guard": True}


def temp_libm_vector_only():
    """``keep_libm`` keeps a pair libm table read as a matrix-vector."""
    return (
        [make_table("t1", costly_body())],
        [read_sum("t1")],
        {"keep_libm": True},
    )


def temp_libm_scalar_double():
    """``keep_libm`` lets a scalar double sum read the table inline."""
    double = sp.Sum(
        array("t1")[IDX_K, IDX_L] * SCALE[IDX_L], full(IDX_K), full(IDX_L)
    )
    return (
        [make_table("t1", costly_body())],
        [double],
        {"keep_libm": True},
    )


def temp_libm_nested():
    """``keep_libm`` skips the outer sum that only holds a nested read."""
    inner = sp.Sum(array("t1")[IDX_K, IDX_L] * SCALE[IDX_L], full(IDX_L))
    outer = sp.Sum(inner * WEIGHT[IDX_K], full(IDX_K))
    return (
        [make_table("t1", costly_body())],
        [outer],
        {"keep_libm": True},
    )


def temp_libm_separable():
    """A separable libm argument is not a pair libm call."""
    body = sp.sqrt(SCALE[IDX_I]) * sp.sqrt(SCALE[IDX_J])
    return (
        [make_table("t1", body)],
        [read_sum("t1")],
        {"keep_libm": True},
    )


def temp_read_by_other():
    """A table read by another table is inlined into its body."""
    reader = make_table("t2", array("t1")[IDX_J, IDX_I] * RATE)
    defs = [make_table("t1", cheap_body()), reader]
    return defs, [read_sum("t2")], {}


def temp_read_on_diagonal():
    """Another temporary reads it on the diagonal: it is inlined there."""
    reader = make_table("t2", array("t1")[IDX_I, IDX_I] * SCALE[IDX_J])
    defs = [make_table("t1", cheap_body()), reader]
    return defs, [read_sum("t1"), read_sum("t2")], {}


def temp_read_by_kept_other():
    """A reader that is itself kept holds back the table it reads."""
    body = array("t1")[IDX_J, IDX_I] * RATE + SCALE[IDX_I] * OTHER[IDX_J] * 3
    defs = [make_table("t1", cheap_body()), make_table("t2", body)]
    exprs = [read_sum("t1"), array("t2")[IDX_K, IDX_L] + 1]
    return defs, exprs, {}


def temp_chain_second_pass():
    """Inlining one table enables the table that reads it (order matters)."""
    reader = make_table("t2", array("t1")[IDX_I, IDX_J] * RATE)
    defs = [reader, make_table("t1", cheap_body())]
    return defs, [read_sum("t2")], {}


def temp_ignored_shapes():
    """Rank-1 temporaries and tables with limits are never candidates."""
    with_limits = make_table("t2", cheap_body())
    with_limits.limits = [full(IDX_M)]
    defs = [with_limits, make_map("t3", IDX_I, WEIGHT[IDX_I])]
    return defs, [read_sum("t2") + array("t3")[IDX_I]], {}


TEMP_CASES = {
    "inline_cheap": temp_inline_cheap,
    "outside_costly": temp_outside_costly,
    "outside_trivial": temp_outside_trivial,
    "costly_two_sums": temp_costly_two_sums,
    "costly_one_sum": temp_costly_one_sum,
    "costly_dedup": temp_costly_dedup,
    "costly_no_dedup": temp_costly_no_dedup,
    "guard_folded": temp_guard_folded,
    "guard_off_folded": temp_guard_off_folded,
    "triangle_weight": temp_triangle_weight,
    "diagonal_reads": temp_diagonal_reads,
    "nested_sum_weight": temp_nested_sum_weight,
    "libm_vector_only": temp_libm_vector_only,
    "libm_scalar_double": temp_libm_scalar_double,
    "libm_nested": temp_libm_nested,
    "libm_separable": temp_libm_separable,
    "read_by_other": temp_read_by_other,
    "read_on_diagonal": temp_read_on_diagonal,
    "read_by_kept_other": temp_read_by_kept_other,
    "chain_second_pass": temp_chain_second_pass,
    "ignored_shapes": temp_ignored_shapes,
}


def run_temporaries(case):
    """Run ``_fuse_temporaries`` and return everything observable."""
    definitions, exprs, options = TEMP_CASES[case]()
    result, outs, removed = _fuse_temporaries(definitions, exprs, **options)
    return {
        "result": summarise(result),
        "same_list": result is definitions,
        "exprs": [str(expr) for expr in outs],
        "removed": sorted(removed),
        "inputs": summarise(definitions),
    }


# ------------------------------------------------------------ loop nests
def row_sum(name, free, bound, body, **flags_):
    """Return a simple vector sum ``name(free) = sum_bound body``."""
    return make_vector_sum(name, free, bound, body, **flags_)


def nests_two_merge():
    """Two simple sums with other index names merge into one nest."""
    first = row_sum("t1", IDX_I, IDX_J, MATRIX[IDX_I, IDX_J] * WEIGHT[IDX_J])
    second = row_sum("t2", IDX_K, IDX_L, MATRIX[IDX_K, IDX_L] * SCALE[IDX_L])
    return [first, second]


def nests_dependent():
    """A sum reading the previous one starts a new group."""
    first = row_sum("t1", IDX_I, IDX_J, MATRIX[IDX_I, IDX_J] * WEIGHT[IDX_J])
    second = row_sum("t2", IDX_K, IDX_L, array("t1")[IDX_K] * SCALE[IDX_L])
    return [first, second]


def nests_order_mismatch():
    """A column-major sum does not join a row-major one."""
    first = row_sum("t1", IDX_I, IDX_J, MATRIX[IDX_I, IDX_J])
    second = row_sum("t2", IDX_K, IDX_L, MATRIX[IDX_K, IDX_L], colmajor=True)
    third = row_sum("t3", IDX_I, IDX_J, MATRIX[IDX_I, IDX_J] * 2)
    return [first, second, third]


def nests_not_simple():
    """Special sums are never merged and flush the group."""
    plain = row_sum("t1", IDX_I, IDX_J, MATRIX[IDX_I, IDX_J])
    mirrored = row_sum("t2", IDX_I, IDX_J, MATRIX[IDX_I, IDX_J], mirror=True)
    partial = row_sum("t3", IDX_I, IDX_J, MATRIX[IDX_I, IDX_J])
    partial.limits = [(IDX_J, 2, NC)]
    guarded = row_sum(
        "t4",
        IDX_I,
        IDX_J,
        sp.Piecewise((MATRIX[IDX_I, IDX_J], IDX_J > 1), (0, True)),
    )
    folded = row_sum("t5", IDX_I, IDX_J, MATRIX[IDX_I, IDX_J])
    folded.body = folded.body * _LoopStart(IDX_I)
    scalar = TempDef(
        target=sp.Symbol("t6"), free=[], limits=[full(IDX_J)], body=RATE
    )
    symv = row_sum("t7", IDX_I, IDX_J, MATRIX[IDX_I, IDX_J])
    symv.symv = (1, 2)
    msum = row_sum("t8", IDX_I, IDX_J, MATRIX[IDX_I, IDX_J])
    msum.mirror_sum = True
    return [plain, mirrored, partial, guarded, folded, scalar, symv, msum]


def nests_three_and_tail():
    """Three merge, a table flushes, and the next sum starts a group."""
    sums = [
        row_sum(f"t{n}", IDX_I, IDX_J, MATRIX[IDX_I, IDX_J] * n)
        for n in (1, 2, 3)
    ]
    table = make_table("t4", MATRIX[IDX_I, IDX_J])
    tail = row_sum("t5", IDX_K, IDX_L, MATRIX[IDX_K, IDX_L])
    return sums + [table, tail]


def nests_single():
    """A lone simple sum comes back as the same object."""
    return [row_sum("t1", IDX_I, IDX_J, MATRIX[IDX_I, IDX_J])]


NEST_LOOP_CASES = {
    "two_merge": nests_two_merge,
    "dependent": nests_dependent,
    "order_mismatch": nests_order_mismatch,
    "not_simple": nests_not_simple,
    "three_and_tail": nests_three_and_tail,
    "single": nests_single,
}


def run_loop_nests(case):
    """Run ``_fuse_loop_nests`` and return everything observable."""
    definitions = NEST_LOOP_CASES[case]()
    snapshot = list(definitions)
    namer = fresh_namer(0)
    result = _fuse_loop_nests(definitions, namer)
    return {
        "result": summarise(result),
        "same_objects": [any(r is s for s in snapshot) for r in result],
        "inputs": summarise(definitions),
        "created": [str(symbol) for symbol in namer.created],
    }


# ---------------------------------------------------------- permutation
def permute_plain():
    """A table sum reads a symmetric array with the bound index last."""
    body = SYM3[IDX_I, IDX_J, IDX_M] * PLAIN3[IDX_I, IDX_J, IDX_M]
    body = body * SYM2[IDX_I, IDX_J] * SYM3[IDX_I, IDX_J, IDX_P]
    return [make_mirror_table("t1", (IDX_I, IDX_J), IDX_M, body)], ("c3", "s")


def permute_fused():
    """Members of a fused nest follow its (row or column) order."""
    first = row_sum("t1", IDX_I, IDX_J, SYM2[IDX_I, IDX_J])
    second = row_sum("t2", IDX_I, IDX_J, SYM2[IDX_I, IDX_J] * 2)
    nest = FusedNest(
        target=first.target,
        free=first.free,
        limits=first.limits,
        body=first.body,
        fused=[first, second],
    )
    first_col = row_sum("t3", IDX_I, IDX_J, SYM2[IDX_I, IDX_J], colmajor=True)
    nest_col = FusedNest(
        target=first_col.target,
        free=first_col.free,
        limits=first_col.limits,
        body=first_col.body,
        fused=[first_col],
    )
    return [nest, nest_col], ("s",)


def permute_sibling():
    """Members of a sibling nest follow the triangle order."""
    body = SYM3[IDX_I, IDX_J, IDX_M]
    first = make_mirror_table("t1", (IDX_I, IDX_J), IDX_M, body)
    second = make_mirror_table("t2", (IDX_I, IDX_J), IDX_M, body * 2)
    nest = SiblingNest(
        target=first.target,
        free=first.free,
        limits=first.limits,
        body=first.body,
        sibling=[first, second],
        accs=[sp.Symbol("a1"), sp.Symbol("a2")],
    )
    return [nest], ("c3",)


def permute_orders():
    """Column sums and mirrored sums have their own loop orders."""
    body = SYM3[IDX_I, IDX_J, IDX_M]
    column = make_table("t1", body)
    column.limits = [full(IDX_M)]
    column.colsum = True
    mirrored = make_mirror_table("t2", (IDX_I, IDX_J), IDX_M, body)
    rank1 = row_sum("t3", IDX_I, IDX_J, SYM2[IDX_J, IDX_I])
    return [column, mirrored, rank1], ("c3", "s")


def permute_skipped():
    """No order, or an index outside the nest: nothing changes."""
    body = SYM3[IDX_I, IDX_J, IDX_M]
    symv = make_mirror_table("t1", (IDX_I, IDX_J), IDX_M, body)
    symv.symv = (1, 2)
    colmajor = make_mirror_table("t2", (IDX_I, IDX_J), IDX_M, body)
    colmajor.colmajor = True
    component = ComponentFusedMap(
        target=array("t3")[IDX_I],
        free=[IDX_I],
        limits=[],
        body=SYM2[IDX_J, IDX_I],
        steps=[],
        members=[],
    )
    outside = row_sum("t4", IDX_I, IDX_J, SYM3[IDX_I, IDX_J, IDX_P])
    rank1 = row_sum("t5", IDX_I, IDX_J, SYM2[IDX_I] * SCALE[IDX_J])
    return [symv, colmajor, component, outside, rank1], ("c3", "s")


def permute_no_symmetric():
    """With no symmetric parameter the list comes back untouched."""
    return [
        make_mirror_table(
            "t1", (IDX_I, IDX_J), IDX_M, SYM3[IDX_I, IDX_J, IDX_M]
        )
    ], ()


PERMUTE_CASES = {
    "plain": permute_plain,
    "fused": permute_fused,
    "sibling": permute_sibling,
    "orders": permute_orders,
    "skipped": permute_skipped,
    "no_symmetric": permute_no_symmetric,
}


def run_permute(case):
    """Run ``_permute_symmetric_subscripts`` and return what it did."""
    definitions, symmetric = PERMUTE_CASES[case]()
    result = _permute_symmetric_subscripts(definitions, symmetric)
    return {
        "result": summarise(result),
        "same_list": result is definitions,
    }


# ------------------------------------------------------ component maps
def sqrt_ratio(index):
    """Return a sub-expression shared by several maps."""
    return sp.sqrt(TEMP / SCALE[index])


def maps_shared():
    """Two maps share a transcendental sub-expression."""
    first = make_map("t1", IDX_I, sqrt_ratio(IDX_I) + WEIGHT[IDX_I])
    second = make_map("t2", IDX_I, sqrt_ratio(IDX_I) * OTHER[IDX_I])
    return [first, second], 2


def maps_renamed_index():
    """The second map uses another index name and reads the first."""
    first = make_map("t1", IDX_I, sqrt_ratio(IDX_I) + WEIGHT[IDX_I])
    second = make_map(
        "t2", IDX_K, sqrt_ratio(IDX_K) * array("t1")[IDX_K] + OTHER[IDX_K]
    )
    return [first, second], 0


def maps_read_other_element():
    """A read of an earlier member at another element flushes the group."""
    first = make_map("t1", IDX_I, sqrt_ratio(IDX_I))
    second = make_map("t2", IDX_K, array("t1")[IDX_L] * OTHER[IDX_K])
    return [first, second], 0


def maps_scalar_moves_ahead():
    """An independent scalar between two maps moves ahead of the group."""
    first = make_map("t1", IDX_I, sqrt_ratio(IDX_I))
    scalar = make_scalar("t3", RATE * 2)
    second = make_map("t2", IDX_K, sqrt_ratio(IDX_K) * 2)
    return [first, scalar, second], 3


def maps_scalar_reads_group():
    """A scalar that reads a member keeps the original order."""
    first = make_map("t1", IDX_I, sqrt_ratio(IDX_I))
    scalar = make_scalar("t3", array("t1")[1] + 2)
    second = make_map("t2", IDX_K, sqrt_ratio(IDX_K) * 2)
    return [first, scalar, second], 3


def maps_not_maps():
    """Sums, piecewise bodies and special maps are never fused."""
    first = make_map("t1", IDX_I, WEIGHT[IDX_I])
    summed = make_map("t2", IDX_I, sp.Sum(WEIGHT[IDX_J], full(IDX_J)))
    guarded = make_map(
        "t3", IDX_I, sp.Piecewise((WEIGHT[IDX_I], IDX_I > 1), (0, True))
    )
    colmajor = make_map("t4", IDX_I, WEIGHT[IDX_I])
    colmajor.colmajor = True
    mirrored = make_map("t5", IDX_I, WEIGHT[IDX_I])
    mirrored.mirror = True
    limited = make_map("t6", IDX_I, WEIGHT[IDX_I] * _LoopStart(IDX_I))
    return [first, summed, guarded, colmajor, mirrored, limited], 6


def maps_chain():
    """A chain of members: later maps read earlier ones at the element."""
    first = make_map("t1", IDX_I, sqrt_ratio(IDX_I) + WEIGHT[IDX_I])
    second = make_map("t2", IDX_J, array("t1")[IDX_J] * sqrt_ratio(IDX_J))
    third = make_map(
        "t3", IDX_K, array("t2")[IDX_K] + array("t1")[IDX_K] * OTHER[IDX_K]
    )
    return [first, second, third], 3


def maps_no_shared():
    """Without common sub-expressions the namer is not called."""
    first = make_map("t1", IDX_I, WEIGHT[IDX_I])
    second = make_map("t2", IDX_J, OTHER[IDX_J])
    return [first, second], 2


def maps_single():
    """A lone map comes back as the same object."""
    return [make_map("t1", IDX_I, sqrt_ratio(IDX_I))], 1


MAP_CASES = {
    "shared": maps_shared,
    "renamed_index": maps_renamed_index,
    "read_other_element": maps_read_other_element,
    "scalar_moves_ahead": maps_scalar_moves_ahead,
    "scalar_reads_group": maps_scalar_reads_group,
    "not_maps": maps_not_maps,
    "chain": maps_chain,
    "no_shared": maps_no_shared,
    "single": maps_single,
}


def run_component_maps(case):
    """Run ``_fuse_component_maps`` and return everything observable."""
    definitions, created = MAP_CASES[case]()
    snapshot = list(definitions)
    namer = fresh_namer(created)
    result = _fuse_component_maps(definitions, namer)
    return {
        "result": summarise(result),
        "same_objects": [any(r is s for s in snapshot) for r in result],
        "inputs": summarise(definitions),
        "created": [str(symbol) for symbol in namer.created],
    }


# ------------------------------------------------------- sibling tables
def sibling_table(name, free=(IDX_I, IDX_J), bound=IDX_M, extra=1):
    """Return a mirrored table ``name(free) = sum_bound B * extra``."""
    body = PLAIN3[free[0], free[1], bound] * WEIGHT[bound] * extra
    return make_mirror_table(name, free, bound, body)


def siblings_two():
    """Two tables with other index names merge."""
    return [
        sibling_table("t1"),
        sibling_table("t2", (IDX_K, IDX_L), IDX_P, 2),
    ], 2


def siblings_other_limits():
    """Tables over different ranges stay apart."""
    second = sibling_table("t2", (IDX_K, IDX_L), IDX_P, 2)
    second.limits = [(IDX_P, 2, NC)]
    return [sibling_table("t1"), second], 2


def siblings_reads_first():
    """A table that reads the first one starts a new group."""
    second = make_mirror_table(
        "t2",
        (IDX_I, IDX_J),
        IDX_M,
        array("t1")[IDX_I, IDX_J] * PLAIN3[IDX_I, IDX_J, IDX_M],
    )
    return [sibling_table("t1"), second], 2


def siblings_waiting():
    """A definition between two tables is emitted after the fused nest."""
    between = make_scalar("t5", RATE * 2)
    return [
        sibling_table("t1"),
        between,
        sibling_table("t2", (IDX_K, IDX_L), IDX_P, 2),
    ], 2


def siblings_waiting_read():
    """A table reading the waiting definition flushes everything."""
    between = make_scalar("t5", RATE * 2)
    second = make_mirror_table(
        "t2",
        (IDX_I, IDX_J),
        IDX_M,
        sp.Symbol("t5") * PLAIN3[IDX_I, IDX_J, IDX_M],
    )
    return [sibling_table("t1"), between, second], 2


def siblings_reader_flushes():
    """A definition reading the group closes it."""
    reader = make_scalar("t5", array("t1")[1, 2])
    named = make_scalar("t6", sp.Symbol("t1"))
    return [sibling_table("t1"), reader, sibling_table("t3"), named], 3


def siblings_three_with_targetless():
    """Three tables merge across a definition with no target."""
    anonymous = TempDef(target=None, free=[], limits=[], body=RATE)
    return [
        sibling_table("t1"),
        anonymous,
        sibling_table("t2", (IDX_K, IDX_L), IDX_P, 2),
        sibling_table("t3", (IDX_J, IDX_K), IDX_L, 3),
    ], 3


def siblings_not_tables():
    """Column sums, piecewise bodies and plain sums are not tables."""
    column = sibling_table("t1")
    column.colsum = True
    guarded = make_mirror_table(
        "t2",
        (IDX_I, IDX_J),
        IDX_M,
        sp.Piecewise((WEIGHT[IDX_M], IDX_M > 1), (0, True)),
    )
    plain = row_sum("t3", IDX_I, IDX_J, MATRIX[IDX_I, IDX_J])
    return [column, guarded, plain], 3


def siblings_lone():
    """A lone table comes back as the same object, no name drawn."""
    return [sibling_table("t1")], 1


SIBLING_CASES = {
    "two": siblings_two,
    "other_limits": siblings_other_limits,
    "reads_first": siblings_reads_first,
    "waiting": siblings_waiting,
    "waiting_read": siblings_waiting_read,
    "reader_flushes": siblings_reader_flushes,
    "three_with_targetless": siblings_three_with_targetless,
    "not_tables": siblings_not_tables,
    "lone": siblings_lone,
}


def run_siblings(case):
    """Run ``_fuse_sibling_tables`` and return everything observable."""
    definitions, created = SIBLING_CASES[case]()
    snapshot = list(definitions)
    namer = fresh_namer(created)
    result = _fuse_sibling_tables(definitions, namer)
    return {
        "result": summarise(result),
        "same_objects": [any(r is s for s in snapshot) for r in result],
        "inputs": summarise(definitions),
        "created": [str(symbol) for symbol in namer.created],
    }


# EXPECTED-BEGIN
EXPECTED_NEST = {
    "general_cost": {
        "inputs": [
            (
                "a",
                "out1",
                "Sum(2*d3[k, l, m]*n[m]*u[l]*w[k], (k, "
                "1, nc), (l, 1, nc), (m, 1, nc))",
            )
        ],
        "result": [
            (
                "a",
                "out1",
                "2*Sum(n[m]*u[l]*Sum(d3[k, l, m]*w[k], "
                "(k, 1, nc)), (l, 1, nc), (m, 1, "
                "nc))",
            )
        ],
    },
    "general_inner_too_wide": {
        "inputs": [
            (
                "a",
                "out1",
                "Sum(d4[k, l, m, p]*w[k], "
                "(k, 1, nc), (l, 1, nc), (m, "
                "1, nc), (p, 1, nc))",
            )
        ],
        "result": [
            (
                "a",
                "out1",
                "Sum(d4[k, l, m, p]*w[k], "
                "(k, 1, nc), (l, 1, nc), (m, "
                "1, nc), (p, 1, nc))",
            )
        ],
    },
    "general_no_cost": {
        "inputs": [
            (
                "a",
                "out1",
                "Sum(2*d3[k, l, m]*n[m]*u[l]*w[k], "
                "(k, 1, nc), (l, 1, nc), (m, 1, "
                "nc))",
            )
        ],
        "result": [
            (
                "a",
                "out1",
                "2*Sum(d3[k, l, m]*n[m]*u[l]*w[k], "
                "(k, 1, nc), (l, 1, nc), (m, 1, "
                "nc))",
            )
        ],
    },
    "general_no_outer_factors": {
        "inputs": [
            (
                "a",
                "out1",
                "Sum(u[l]*w[k], (k, 1, " "nc), (l, 1, nc), (m, 1, " "nc))",
            )
        ],
        "result": [
            (
                "a",
                "out1",
                "Sum(u[l]*Sum(w[k], (k, 1, "
                "nc)), (l, 1, nc), (m, 1, "
                "nc))",
            )
        ],
    },
    "general_not_product": {
        "inputs": [
            (
                "a",
                "out1",
                "Sum(mm[l, m] + n[k], (k, 1, " "nc), (l, 1, nc), (m, 1, nc))",
            )
        ],
        "result": [
            (
                "a",
                "out1",
                "Sum(mm[l, m] + n[k], (k, 1, "
                "nc), (l, 1, nc), (m, 1, "
                "nc))",
            )
        ],
    },
    "general_pair_cost": {
        "inputs": [
            ("a", "out1", "Sum(mm[k, l]*n[k]*n[l], (k, 1, " "nc), (l, 1, nc))")
        ],
        "result": [
            ("a", "out1", "Sum(mm[k, l]*n[k]*n[l], (k, 1, " "nc), (l, 1, nc))")
        ],
    },
    "general_symmetric_rank3": {
        "inputs": [
            ("a", "out1", "Sum(c3[k, l, m]*n[m], (m, " "1, nc))"),
            (
                "b",
                "out2",
                "Sum(c3[m, l, "
                "k]*n[m]*u[l]*w[k], (k, 1, "
                "nc), (l, 1, nc), (m, 1, "
                "nc))",
            ),
        ],
        "result": [
            ("a", "out1", "Sum(c3[k, l, m]*n[m], (m, " "1, nc))"),
            (
                "b",
                "out2",
                "Sum(u[l]*w[k]*Sum(c3[m, l, "
                "k]*n[m], (m, 1, nc)), (k, "
                "1, nc), (l, 1, nc))",
            ),
        ],
    },
    "general_unequal_ranges": {
        "inputs": [
            (
                "a",
                "out1",
                "Sum(2*d3[k, l, "
                "m]*n[m]*u[l]*w[k], (k, 1, "
                "nc), (l, 1, nc), (m, 2, "
                "nc))",
            )
        ],
        "result": [
            (
                "a",
                "out1",
                "2*Sum(d3[k, l, "
                "m]*n[m]*u[l]*w[k], (k, 1, "
                "nc), (l, 1, nc), (m, 2, "
                "nc))",
            )
        ],
    },
    "general_vector": {
        "inputs": [
            ("a", "out1", "Sum(d3[k, l, m]*n[m], (m, 1, nc))"),
            (
                "b",
                "out2",
                "Sum(2*d3[k, l, m]*n[m]*u[l]*w[k], "
                "(k, 1, nc), (l, 1, nc), (m, 1, "
                "nc))",
            ),
        ],
        "result": [
            ("a", "out1", "Sum(d3[k, l, m]*n[m], (m, 1, nc))"),
            (
                "b",
                "out2",
                "2*Sum(u[l]*w[k]*Sum(d3[k, l, "
                "m]*n[m], (m, 1, nc)), (k, 1, nc), "
                "(l, 1, nc))",
            ),
        ],
    },
    "no_vector": {
        "inputs": [
            (
                "dnn",
                "out1",
                "Sum(mm[k, l]*n[k]*n[l], (k, 1, nc), (l, " "1, nc))",
            )
        ],
        "result": [
            (
                "dnn",
                "out1",
                "Sum(mm[k, l]*n[k]*n[l], (k, 1, nc), (l, " "1, nc))",
            )
        ],
    },
    "not_a_product": {
        "inputs": [
            ("a", "out1", "Sum(mm[k, l]*n[l], (l, 1, nc))"),
            ("b", "out2", "Sum(mm[k, l] + n[k], (k, 1, nc), (l, " "1, nc))"),
        ],
        "result": [
            ("a", "out1", "Sum(mm[k, l]*n[l], (l, 1, nc))"),
            ("b", "out2", "Sum(mm[k, l] + n[k], (k, 1, nc), (l, " "1, nc))"),
        ],
    },
    "outer_factor_hoisted": {
        "inputs": [
            ("a", "out1", "Sum(mm[k, l]*n[l], (l, 1, " "nc))"),
            (
                "b",
                "out2",
                "R + Sum(R*mm[k, l]*n[k]*n[l], " "(k, 1, nc), (l, 1, nc))",
            ),
        ],
        "result": [
            ("a", "out1", "Sum(mm[k, l]*n[l], (l, 1, " "nc))"),
            (
                "b",
                "out2",
                "R*Sum(n[k]*Sum(mm[k, l]*n[l], "
                "(l, 1, nc)), (k, 1, nc)) + "
                "R",
            ),
        ],
    },
    "pair_no_outer_factors": {
        "inputs": [("a", "out1", "Sum(u[k]*w[k], (k, 1, nc), " "(l, 1, nc))")],
        "result": [("a", "out1", "Sum(u[k]*w[k], (k, 1, nc), " "(l, 1, nc))")],
    },
    "shared_inner": {
        "inputs": [
            (
                "a",
                "out1",
                "Sum(mm[k, l]*n[k]*n[l]*w[k], (k, 1, " "nc), (l, 1, nc))",
            ),
            (
                "b",
                "out2",
                "Sum(mm[k, l]*n[k]*n[l]*u[k], (k, 1, " "nc), (l, 1, nc))",
            ),
        ],
        "result": [
            (
                "a",
                "out1",
                "Sum(n[k]*w[k]*Sum(mm[k, l]*n[l], (l, " "1, nc)), (k, 1, nc))",
            ),
            (
                "b",
                "out2",
                "Sum(n[k]*u[k]*Sum(mm[k, l]*n[l], (l, " "1, nc)), (k, 1, nc))",
            ),
        ],
    },
    "shared_inner_off": {
        "inputs": [
            (
                "a",
                "out1",
                "Sum(mm[k, l]*n[k]*n[l]*w[k], (k, " "1, nc), (l, 1, nc))",
            ),
            (
                "b",
                "out2",
                "Sum(mm[k, l]*n[k]*n[l]*u[k], (k, " "1, nc), (l, 1, nc))",
            ),
        ],
        "result": [
            (
                "a",
                "out1",
                "Sum(mm[k, l]*n[k]*n[l]*w[k], (k, " "1, nc), (l, 1, nc))",
            ),
            (
                "b",
                "out2",
                "Sum(mm[k, l]*n[k]*n[l]*u[k], (k, " "1, nc), (l, 1, nc))",
            ),
        ],
    },
    "symmetric": {
        "inputs": [
            ("a", "out1", "Sum(n[l]*s[k, l], (l, 1, nc))"),
            ("b", "out2", "Sum(n[k]*n[l]*s[l, k], (k, 1, nc), (l, " "1, nc))"),
        ],
        "result": [
            ("a", "out1", "Sum(n[l]*s[k, l], (l, 1, nc))"),
            (
                "b",
                "out2",
                "Sum(n[k]*Sum(n[l]*s[l, k], (l, 1, nc)), " "(k, 1, nc))",
            ),
        ],
    },
    "unequal_ranges": {
        "inputs": [
            ("a", "out1", "Sum(mm[k, l]*n[l], (l, 1, nc))"),
            (
                "b",
                "out2",
                "Sum(mm[k, l]*n[k]*n[l], (k, 1, nc), " "(l, 2, nc))",
            ),
        ],
        "result": [
            ("a", "out1", "Sum(mm[k, l]*n[l], (l, 1, nc))"),
            (
                "b",
                "out2",
                "Sum(mm[k, l]*n[k]*n[l], (k, 1, nc), " "(l, 2, nc))",
            ),
        ],
    },
    "vector_and_double": {
        "inputs": [
            ("dn", "out1", "Sum(mm[k, l]*n[l], (l, 1, nc))"),
            (
                "dnn",
                "out2",
                "Sum(mm[k, l]*n[k]*n[l], (k, 1, " "nc), (l, 1, nc))",
            ),
        ],
        "result": [
            ("dn", "out1", "Sum(mm[k, l]*n[l], (l, 1, nc))"),
            (
                "dnn",
                "out2",
                "Sum(n[k]*Sum(mm[k, l]*n[l], (l, " "1, nc)), (k, 1, nc))",
            ),
        ],
    },
}


EXPECTED_TEMPORARIES = {
    "chain_second_pass": {
        "exprs": ["Sum(R*mm[i, k]*n[k]*w[k], (k, 1, " "nc))"],
        "inputs": [],
        "removed": ["t1", "t2"],
        "result": [],
        "same_list": True,
    },
    "costly_dedup": {
        "exprs": [
            "Sum(exp(mm[i, k]/T)*w[k], (k, 1, nc))",
            "2*Sum(exp(mm[i, k]/T)*w[k], (k, 1, " "nc))",
        ],
        "inputs": [],
        "removed": ["t1"],
        "result": [],
        "same_list": True,
    },
    "costly_no_dedup": {
        "exprs": [
            "Sum(t1[i, k]*w[k], (k, 1, nc))",
            "2*Sum(t1[i, k]*w[k], (k, 1, nc))",
        ],
        "inputs": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                [],
                "exp(mm[i, j]/T)",
                ["body"],
                None,
                None,
                [],
            )
        ],
        "removed": [],
        "result": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                [],
                "exp(mm[i, j]/T)",
                ["body"],
                None,
                None,
                [],
            )
        ],
        "same_list": True,
    },
    "costly_one_sum": {
        "exprs": ["Sum(exp(mm[i, k]/T)*w[k], (k, 1, " "nc))"],
        "inputs": [],
        "removed": ["t1"],
        "result": [],
        "same_list": True,
    },
    "costly_two_sums": {
        "exprs": [
            "Sum(t1[i, k]*w[k], (k, 1, nc))",
            "Sum(t1[l, m]*w[m], (m, 1, nc))",
        ],
        "inputs": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                [],
                "exp(mm[i, j]/T)",
                ["body"],
                None,
                None,
                [],
            )
        ],
        "removed": [],
        "result": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                [],
                "exp(mm[i, j]/T)",
                ["body"],
                None,
                None,
                [],
            )
        ],
        "same_list": True,
    },
    "diagonal_reads": {
        "exprs": [
            "Sum(exp(mm[k, k]/T)*w[k], (k, 1, nc))",
            "Sum(exp(mm[k, k]/T)*w[k], (k, 1, nc)) " "+ 1",
            "Sum(exp(mm[i, k]/T)*w[k], (k, 1, " "nc))",
        ],
        "inputs": [],
        "removed": ["t1"],
        "result": [],
        "same_list": True,
    },
    "guard_folded": {
        "exprs": [
            "Sum(t1[i, k]*w[k], (k, i + 1, nc))",
            "Sum(t1[i, k]*u[k], (k, i + 1, nc))",
        ],
        "inputs": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                [],
                "exp(mm[i, j]/T)",
                ["body"],
                None,
                None,
                [],
            )
        ],
        "removed": [],
        "result": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                [],
                "exp(mm[i, j]/T)",
                ["body"],
                None,
                None,
                [],
            )
        ],
        "same_list": True,
    },
    "guard_off_folded": {
        "exprs": [
            "Sum(exp(mm[i, k]/T)*w[k], (k, i + " "1, nc))",
            "Sum(exp(mm[i, k]/T)*u[k], (k, i + " "1, nc))",
        ],
        "inputs": [],
        "removed": ["t1"],
        "result": [],
        "same_list": True,
    },
    "ignored_shapes": {
        "exprs": ["t3[i] + Sum(t2[i, k]*w[k], (k, 1, " "nc))"],
        "inputs": [
            (
                "TempDef",
                "t2[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "mm[i, j]*n[j]",
                ["body"],
                None,
                None,
                [],
            ),
            ("TempDef", "t3[i]", ["i"], [], "n[i]", ["body"], None, None, []),
        ],
        "removed": [],
        "result": [
            (
                "TempDef",
                "t2[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "mm[i, j]*n[j]",
                ["body"],
                None,
                None,
                [],
            ),
            ("TempDef", "t3[i]", ["i"], [], "n[i]", ["body"], None, None, []),
        ],
        "same_list": True,
    },
    "inline_cheap": {
        "exprs": [
            "Sum(mm[i, k]*n[k]*w[k], (k, 1, nc))",
            "R + Sum(mm[l, m]*n[m]*w[m], (m, 1, " "nc))",
        ],
        "inputs": [
            ("TempDef", "t9[i]", ["i"], [], "n[i]", ["body"], None, None, []),
            (
                "TempDef",
                "t8[i]",
                ["i"],
                ["(j, 1, nc)"],
                "n[j]",
                ["body"],
                None,
                None,
                [],
            ),
        ],
        "removed": ["t1"],
        "result": [
            ("TempDef", "t9[i]", ["i"], [], "n[i]", ["body"], None, None, []),
            (
                "TempDef",
                "t8[i]",
                ["i"],
                ["(j, 1, nc)"],
                "n[j]",
                ["body"],
                None,
                None,
                [],
            ),
        ],
        "same_list": True,
    },
    "libm_nested": {
        "exprs": ["Sum(n[k]*Sum(t1[k, l]*w[l], (l, 1, nc)), " "(k, 1, nc))"],
        "inputs": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                [],
                "exp(mm[i, j]/T)",
                ["body"],
                None,
                None,
                [],
            )
        ],
        "removed": [],
        "result": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                [],
                "exp(mm[i, j]/T)",
                ["body"],
                None,
                None,
                [],
            )
        ],
        "same_list": True,
    },
    "libm_scalar_double": {
        "exprs": ["Sum(exp(mm[k, l]/T)*w[l], (k, 1, " "nc), (l, 1, nc))"],
        "inputs": [],
        "removed": ["t1"],
        "result": [],
        "same_list": True,
    },
    "libm_separable": {
        "exprs": ["Sum(sqrt(w[i])*w[k]**(3/2), (k, 1, " "nc))"],
        "inputs": [],
        "removed": ["t1"],
        "result": [],
        "same_list": True,
    },
    "libm_vector_only": {
        "exprs": ["Sum(t1[i, k]*w[k], (k, 1, nc))"],
        "inputs": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                [],
                "exp(mm[i, j]/T)",
                ["body"],
                None,
                None,
                [],
            )
        ],
        "removed": [],
        "result": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                [],
                "exp(mm[i, j]/T)",
                ["body"],
                None,
                None,
                [],
            )
        ],
        "same_list": True,
    },
    "nested_sum_weight": {
        "exprs": [
            "Sum(n[k]*Sum(exp(mm[k, l]/T)*w[l], " "(l, 1, nc)), (k, 1, nc))"
        ],
        "inputs": [],
        "removed": ["t1"],
        "result": [],
        "same_list": True,
    },
    "outside_costly": {
        "exprs": ["Sum(t1[i, k]*w[k], (k, 1, nc))", "t1[k, l] + 1"],
        "inputs": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                [],
                "R*u[j]*w[i] + mm[i, j]*n[j]",
                ["body"],
                None,
                None,
                [],
            )
        ],
        "removed": [],
        "result": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                [],
                "R*u[j]*w[i] + mm[i, j]*n[j]",
                ["body"],
                None,
                None,
                [],
            )
        ],
        "same_list": True,
    },
    "outside_trivial": {
        "exprs": ["Sum(mm[i, k]*n[k]*w[k], (k, 1, nc))", "mm[k, l]*n[l] + 1"],
        "inputs": [],
        "removed": ["t1"],
        "result": [],
        "same_list": True,
    },
    "read_by_kept_other": {
        "exprs": ["Sum(t1[i, k]*w[k], (k, 1, nc))", "t2[k, l] + 1"],
        "inputs": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                [],
                "mm[i, j]*n[j]",
                ["body"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t2[i, j]",
                ["i", "j"],
                [],
                "R*t1[j, i] + 3*u[j]*w[i]",
                ["body"],
                None,
                None,
                [],
            ),
        ],
        "removed": [],
        "result": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                [],
                "mm[i, j]*n[j]",
                ["body"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t2[i, j]",
                ["i", "j"],
                [],
                "R*t1[j, i] + 3*u[j]*w[i]",
                ["body"],
                None,
                None,
                [],
            ),
        ],
        "same_list": True,
    },
    "read_by_other": {
        "exprs": ["Sum(R*mm[k, i]*n[i]*w[k], (k, 1, nc))"],
        "inputs": [],
        "removed": ["t1", "t2"],
        "result": [],
        "same_list": True,
    },
    "read_on_diagonal": {
        "exprs": [
            "Sum(mm[i, k]*n[k]*w[k], (k, 1, nc))",
            "Sum(mm[i, i]*n[i]*w[k]**2, (k, 1, " "nc))",
        ],
        "inputs": [],
        "removed": ["t1", "t2"],
        "result": [],
        "same_list": True,
    },
    "triangle_weight": {
        "exprs": [
            "Sum(exp(mm[i, k]/T)*w[k], (k, i + 1, " "nc))",
            "Sum(exp(mm[k, k]/T), (k, 1, nc))",
        ],
        "inputs": [],
        "removed": ["t1"],
        "result": [],
        "same_list": True,
    },
}


EXPECTED_LOOP_NESTS = {
    "dependent": {
        "created": [],
        "inputs": [
            (
                "TempDef",
                "t1[i]",
                ["i"],
                ["(j, 1, nc)"],
                "mm[i, j]*n[j]",
                ["body"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t2[k]",
                ["k"],
                ["(l, 1, nc)"],
                "t1[k]*w[l]",
                ["body"],
                None,
                None,
                [],
            ),
        ],
        "result": [
            (
                "TempDef",
                "t1[i]",
                ["i"],
                ["(j, 1, nc)"],
                "mm[i, j]*n[j]",
                ["body"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t2[k]",
                ["k"],
                ["(l, 1, nc)"],
                "t1[k]*w[l]",
                ["body"],
                None,
                None,
                [],
            ),
        ],
        "same_objects": [True, True],
    },
    "not_simple": {
        "created": [],
        "inputs": [
            (
                "TempDef",
                "t1[i]",
                ["i"],
                ["(j, 1, nc)"],
                "mm[i, j]",
                ["body"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t2[i]",
                ["i"],
                ["(j, 1, nc)"],
                "mm[i, j]",
                ["body", "mirror"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t3[i]",
                ["i"],
                ["(j, 2, nc)"],
                "mm[i, j]",
                ["body"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t4[i]",
                ["i"],
                ["(j, 1, nc)"],
                "Piecewise((mm[i, j], j > 1), (0, True))",
                ["body"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t5[i]",
                ["i"],
                ["(j, 1, nc)"],
                "i + 1*mm[i, j]",
                ["body"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t6",
                [],
                ["(j, 1, nc)"],
                "R",
                ["body"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t7[i]",
                ["i"],
                ["(j, 1, nc)"],
                "mm[i, j]",
                ["body", "symv"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t8[i]",
                ["i"],
                ["(j, 1, nc)"],
                "mm[i, j]",
                ["body", "mirror_sum"],
                None,
                None,
                [],
            ),
        ],
        "result": [
            (
                "TempDef",
                "t1[i]",
                ["i"],
                ["(j, 1, nc)"],
                "mm[i, j]",
                ["body"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t2[i]",
                ["i"],
                ["(j, 1, nc)"],
                "mm[i, j]",
                ["body", "mirror"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t3[i]",
                ["i"],
                ["(j, 2, nc)"],
                "mm[i, j]",
                ["body"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t4[i]",
                ["i"],
                ["(j, 1, nc)"],
                "Piecewise((mm[i, j], j > 1), (0, True))",
                ["body"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t5[i]",
                ["i"],
                ["(j, 1, nc)"],
                "i + 1*mm[i, j]",
                ["body"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t6",
                [],
                ["(j, 1, nc)"],
                "R",
                ["body"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t7[i]",
                ["i"],
                ["(j, 1, nc)"],
                "mm[i, j]",
                ["body", "symv"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t8[i]",
                ["i"],
                ["(j, 1, nc)"],
                "mm[i, j]",
                ["body", "mirror_sum"],
                None,
                None,
                [],
            ),
        ],
        "same_objects": [True, True, True, True, True, True, True, True],
    },
    "order_mismatch": {
        "created": [],
        "inputs": [
            (
                "TempDef",
                "t1[i]",
                ["i"],
                ["(j, 1, nc)"],
                "mm[i, j]",
                ["body"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t2[k]",
                ["k"],
                ["(l, 1, nc)"],
                "mm[k, l]",
                ["body", "colmajor"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t3[i]",
                ["i"],
                ["(j, 1, nc)"],
                "2*mm[i, j]",
                ["body"],
                None,
                None,
                [],
            ),
        ],
        "result": [
            (
                "TempDef",
                "t1[i]",
                ["i"],
                ["(j, 1, nc)"],
                "mm[i, j]",
                ["body"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t2[k]",
                ["k"],
                ["(l, 1, nc)"],
                "mm[k, l]",
                ["body", "colmajor"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t3[i]",
                ["i"],
                ["(j, 1, nc)"],
                "2*mm[i, j]",
                ["body"],
                None,
                None,
                [],
            ),
        ],
        "same_objects": [True, True, True],
    },
    "single": {
        "created": [],
        "inputs": [
            (
                "TempDef",
                "t1[i]",
                ["i"],
                ["(j, 1, nc)"],
                "mm[i, j]",
                ["body"],
                None,
                None,
                [],
            )
        ],
        "result": [
            (
                "TempDef",
                "t1[i]",
                ["i"],
                ["(j, 1, nc)"],
                "mm[i, j]",
                ["body"],
                None,
                None,
                [],
            )
        ],
        "same_objects": [True],
    },
    "three_and_tail": {
        "created": [],
        "inputs": [
            (
                "TempDef",
                "t1[i]",
                ["i"],
                ["(j, 1, nc)"],
                "mm[i, j]",
                ["body"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t2[i]",
                ["i"],
                ["(j, 1, nc)"],
                "2*mm[i, j]",
                ["body"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t3[i]",
                ["i"],
                ["(j, 1, nc)"],
                "3*mm[i, j]",
                ["body"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t4[i, j]",
                ["i", "j"],
                [],
                "mm[i, j]",
                ["body"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t5[k]",
                ["k"],
                ["(l, 1, nc)"],
                "mm[k, l]",
                ["body"],
                None,
                None,
                [],
            ),
        ],
        "result": [
            (
                "FusedNest",
                "t1[i]",
                ["i"],
                ["(j, 1, nc)"],
                "mm[i, j]",
                ["body", "fused"],
                [
                    (
                        "TempDef",
                        "t1[i]",
                        ["i"],
                        ["(j, 1, nc)"],
                        "mm[i, j]",
                        ["body"],
                        None,
                        None,
                        [],
                    ),
                    (
                        "TempDef",
                        "t2[i]",
                        ["i"],
                        ["(j, 1, nc)"],
                        "2*mm[i, j]",
                        ["body"],
                        None,
                        None,
                        [],
                    ),
                    (
                        "TempDef",
                        "t3[i]",
                        ["i"],
                        ["(j, 1, nc)"],
                        "3*mm[i, j]",
                        ["body"],
                        None,
                        None,
                        [],
                    ),
                ],
                None,
                [],
            ),
            (
                "TempDef",
                "t4[i, j]",
                ["i", "j"],
                [],
                "mm[i, j]",
                ["body"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t5[k]",
                ["k"],
                ["(l, 1, nc)"],
                "mm[k, l]",
                ["body"],
                None,
                None,
                [],
            ),
        ],
        "same_objects": [False, True, True],
    },
    "two_merge": {
        "created": [],
        "inputs": [
            (
                "TempDef",
                "t1[i]",
                ["i"],
                ["(j, 1, nc)"],
                "mm[i, j]*n[j]",
                ["body"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t2[k]",
                ["k"],
                ["(l, 1, nc)"],
                "mm[k, l]*w[l]",
                ["body"],
                None,
                None,
                [],
            ),
        ],
        "result": [
            (
                "FusedNest",
                "t1[i]",
                ["i"],
                ["(j, 1, nc)"],
                "mm[i, j]*n[j]",
                ["body", "fused"],
                [
                    (
                        "TempDef",
                        "t1[i]",
                        ["i"],
                        ["(j, 1, nc)"],
                        "mm[i, j]*n[j]",
                        ["body"],
                        None,
                        None,
                        [],
                    ),
                    (
                        "TempDef",
                        "t2[i]",
                        ["i"],
                        ["(j, 1, nc)"],
                        "mm[i, j]*w[j]",
                        ["body"],
                        None,
                        None,
                        [],
                    ),
                ],
                None,
                [],
            )
        ],
        "same_objects": [False],
    },
}


EXPECTED_PERMUTE = {
    "fused": {
        "result": [
            (
                "FusedNest",
                "t1[i]",
                ["i"],
                ["(j, 1, nc)"],
                "s[i, j]",
                ["body", "fused"],
                [
                    (
                        "TempDef",
                        "t1[i]",
                        ["i"],
                        ["(j, 1, nc)"],
                        "s[j, i]",
                        ["body"],
                        None,
                        None,
                        [],
                    ),
                    (
                        "TempDef",
                        "t2[i]",
                        ["i"],
                        ["(j, 1, nc)"],
                        "2*s[j, i]",
                        ["body"],
                        None,
                        None,
                        [],
                    ),
                ],
                None,
                [],
            ),
            (
                "FusedNest",
                "t3[i]",
                ["i"],
                ["(j, 1, nc)"],
                "s[i, j]",
                ["body", "fused"],
                [
                    (
                        "TempDef",
                        "t3[i]",
                        ["i"],
                        ["(j, 1, nc)"],
                        "s[i, j]",
                        ["body", "colmajor"],
                        None,
                        None,
                        [],
                    )
                ],
                None,
                [],
            ),
        ],
        "same_list": True,
    },
    "no_symmetric": {
        "result": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "c3[i, j, m]",
                ["body", "mirror_sum"],
                None,
                None,
                [],
            )
        ],
        "same_list": True,
    },
    "orders": {
        "result": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "c3[i, m, j]",
                ["body", "colsum"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t2[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "c3[m, i, j]",
                ["body", "mirror_sum"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t3[i]",
                ["i"],
                ["(j, 1, nc)"],
                "s[j, i]",
                ["body"],
                None,
                None,
                [],
            ),
        ],
        "same_list": True,
    },
    "plain": {
        "result": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "c3[i, j, p]*c3[m, i, j]*d3[i, j, m]*s[i, j]",
                ["body", "mirror_sum"],
                None,
                None,
                [],
            )
        ],
        "same_list": True,
    },
    "sibling": {
        "result": [
            (
                "SiblingNest",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "c3[i, j, m]",
                ["accs", "body", "sibling"],
                [
                    (
                        "TempDef",
                        "t1[i, j]",
                        ["i", "j"],
                        ["(m, 1, nc)"],
                        "c3[m, i, j]",
                        ["body", "mirror_sum"],
                        None,
                        None,
                        [],
                    ),
                    (
                        "TempDef",
                        "t2[i, j]",
                        ["i", "j"],
                        ["(m, 1, nc)"],
                        "2*c3[m, i, j]",
                        ["body", "mirror_sum"],
                        None,
                        None,
                        [],
                    ),
                ],
                None,
                ["a1", "a2"],
            )
        ],
        "same_list": True,
    },
    "skipped": {
        "result": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "c3[i, j, m]",
                ["body", "mirror_sum", "symv"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t2[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "c3[i, j, m]",
                ["body", "colmajor", "mirror_sum"],
                None,
                None,
                [],
            ),
            (
                "ComponentFusedMap",
                "t3[i]",
                ["i"],
                [],
                "s[j, i]",
                ["body", "compfused", "members", "steps"],
                None,
                ([], []),
                [],
            ),
            (
                "TempDef",
                "t4[i]",
                ["i"],
                ["(j, 1, nc)"],
                "c3[i, j, p]",
                ["body"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t5[i]",
                ["i"],
                ["(j, 1, nc)"],
                "s[i]*w[j]",
                ["body"],
                None,
                None,
                [],
            ),
        ],
        "same_list": True,
    },
}


EXPECTED_MAPS = {
    "chain": {
        "created": ["t1", "t2", "t3", "t4"],
        "inputs": [
            (
                "TempDef",
                "t1[i]",
                ["i"],
                [],
                "sqrt(T/w[i]) + n[i]",
                ["body"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t2[j]",
                ["j"],
                [],
                "sqrt(T/w[j])*t1[j]",
                ["body"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t3[k]",
                ["k"],
                [],
                "t1[k]*u[k] + t2[k]",
                ["body"],
                None,
                None,
                [],
            ),
        ],
        "result": [
            (
                "ComponentFusedMap",
                "t1[i]",
                ["i"],
                [],
                "sqrt(T/w[i]) + n[i]",
                ["body", "compfused", "members", "steps"],
                None,
                (
                    [
                        ("local", "t4", "sqrt(T/w[i])"),
                        ("member", "t1[i]", "t4 + n[i]"),
                        ("member", "t2[i]", "t4*t1[i]"),
                        ("member", "t3[i]", "t1[i]*u[i] + t2[i]"),
                    ],
                    [
                        ("t1[i]", "sqrt(T/w[i]) + n[i]"),
                        ("t2[i]", "sqrt(T/w[i])*t1[i]"),
                        ("t3[i]", "t1[i]*u[i] + t2[i]"),
                    ],
                ),
                [],
            )
        ],
        "same_objects": [False],
    },
    "no_shared": {
        "created": ["t1", "t2"],
        "inputs": [
            ("TempDef", "t1[i]", ["i"], [], "n[i]", ["body"], None, None, []),
            ("TempDef", "t2[j]", ["j"], [], "u[j]", ["body"], None, None, []),
        ],
        "result": [
            (
                "ComponentFusedMap",
                "t1[i]",
                ["i"],
                [],
                "n[i]",
                ["body", "compfused", "members", "steps"],
                None,
                (
                    [("member", "t1[i]", "n[i]"), ("member", "t2[i]", "u[i]")],
                    [("t1[i]", "n[i]"), ("t2[i]", "u[i]")],
                ),
                [],
            )
        ],
        "same_objects": [False],
    },
    "not_maps": {
        "created": ["t1", "t2", "t3", "t4", "t5", "t6"],
        "inputs": [
            ("TempDef", "t1[i]", ["i"], [], "n[i]", ["body"], None, None, []),
            (
                "TempDef",
                "t2[i]",
                ["i"],
                [],
                "Sum(n[j], (j, 1, nc))",
                ["body"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t3[i]",
                ["i"],
                [],
                "Piecewise((n[i], i > 1), (0, True))",
                ["body"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t4[i]",
                ["i"],
                [],
                "n[i]",
                ["body", "colmajor"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t5[i]",
                ["i"],
                [],
                "n[i]",
                ["body", "mirror"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t6[i]",
                ["i"],
                [],
                "i + 1*n[i]",
                ["body"],
                None,
                None,
                [],
            ),
        ],
        "result": [
            ("TempDef", "t1[i]", ["i"], [], "n[i]", ["body"], None, None, []),
            (
                "TempDef",
                "t2[i]",
                ["i"],
                [],
                "Sum(n[j], (j, 1, nc))",
                ["body"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t3[i]",
                ["i"],
                [],
                "Piecewise((n[i], i > 1), (0, True))",
                ["body"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t4[i]",
                ["i"],
                [],
                "n[i]",
                ["body", "colmajor"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t5[i]",
                ["i"],
                [],
                "n[i]",
                ["body", "mirror"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t6[i]",
                ["i"],
                [],
                "i + 1*n[i]",
                ["body"],
                None,
                None,
                [],
            ),
        ],
        "same_objects": [True, True, True, True, True, True],
    },
    "read_other_element": {
        "created": [],
        "inputs": [
            (
                "TempDef",
                "t1[i]",
                ["i"],
                [],
                "sqrt(T/w[i])",
                ["body"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t2[k]",
                ["k"],
                [],
                "t1[l]*u[k]",
                ["body"],
                None,
                None,
                [],
            ),
        ],
        "result": [
            (
                "TempDef",
                "t1[i]",
                ["i"],
                [],
                "sqrt(T/w[i])",
                ["body"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t2[k]",
                ["k"],
                [],
                "t1[l]*u[k]",
                ["body"],
                None,
                None,
                [],
            ),
        ],
        "same_objects": [True, True],
    },
    "renamed_index": {
        "created": ["t1"],
        "inputs": [
            (
                "TempDef",
                "t1[i]",
                ["i"],
                [],
                "sqrt(T/w[i]) + n[i]",
                ["body"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t2[k]",
                ["k"],
                [],
                "sqrt(T/w[k])*t1[k] + u[k]",
                ["body"],
                None,
                None,
                [],
            ),
        ],
        "result": [
            (
                "ComponentFusedMap",
                "t1[i]",
                ["i"],
                [],
                "sqrt(T/w[i]) + n[i]",
                ["body", "compfused", "members", "steps"],
                None,
                (
                    [
                        ("local", "t1", "sqrt(T/w[i])"),
                        ("member", "t1[i]", "t1 + n[i]"),
                        ("member", "t2[i]", "t1*t1[i] + u[i]"),
                    ],
                    [
                        ("t1[i]", "sqrt(T/w[i]) + n[i]"),
                        ("t2[i]", "sqrt(T/w[i])*t1[i] + u[i]"),
                    ],
                ),
                [],
            )
        ],
        "same_objects": [False],
    },
    "scalar_moves_ahead": {
        "created": ["t1", "t2", "t3", "t4"],
        "inputs": [
            (
                "TempDef",
                "t1[i]",
                ["i"],
                [],
                "sqrt(T/w[i])",
                ["body"],
                None,
                None,
                [],
            ),
            ("TempDef", "t3", [], [], "2*R", ["body"], None, None, []),
            (
                "TempDef",
                "t2[k]",
                ["k"],
                [],
                "2*sqrt(T/w[k])",
                ["body"],
                None,
                None,
                [],
            ),
        ],
        "result": [
            ("TempDef", "t3", [], [], "2*R", ["body"], None, None, []),
            (
                "ComponentFusedMap",
                "t1[i]",
                ["i"],
                [],
                "sqrt(T/w[i])",
                ["body", "compfused", "members", "steps"],
                None,
                (
                    [
                        ("local", "t4", "sqrt(T/w[i])"),
                        ("member", "t1[i]", "t4"),
                        ("member", "t2[i]", "2*t4"),
                    ],
                    [("t1[i]", "sqrt(T/w[i])"), ("t2[i]", "2*sqrt(T/w[i])")],
                ),
                [],
            ),
        ],
        "same_objects": [True, False],
    },
    "scalar_reads_group": {
        "created": ["t1", "t2", "t3"],
        "inputs": [
            (
                "TempDef",
                "t1[i]",
                ["i"],
                [],
                "sqrt(T/w[i])",
                ["body"],
                None,
                None,
                [],
            ),
            ("TempDef", "t3", [], [], "t1[1] + 2", ["body"], None, None, []),
            (
                "TempDef",
                "t2[k]",
                ["k"],
                [],
                "2*sqrt(T/w[k])",
                ["body"],
                None,
                None,
                [],
            ),
        ],
        "result": [
            (
                "TempDef",
                "t1[i]",
                ["i"],
                [],
                "sqrt(T/w[i])",
                ["body"],
                None,
                None,
                [],
            ),
            ("TempDef", "t3", [], [], "t1[1] + 2", ["body"], None, None, []),
            (
                "TempDef",
                "t2[k]",
                ["k"],
                [],
                "2*sqrt(T/w[k])",
                ["body"],
                None,
                None,
                [],
            ),
        ],
        "same_objects": [True, True, True],
    },
    "shared": {
        "created": ["t1", "t2", "t3"],
        "inputs": [
            (
                "TempDef",
                "t1[i]",
                ["i"],
                [],
                "sqrt(T/w[i]) + n[i]",
                ["body"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t2[i]",
                ["i"],
                [],
                "sqrt(T/w[i])*u[i]",
                ["body"],
                None,
                None,
                [],
            ),
        ],
        "result": [
            (
                "ComponentFusedMap",
                "t1[i]",
                ["i"],
                [],
                "sqrt(T/w[i]) + n[i]",
                ["body", "compfused", "members", "steps"],
                None,
                (
                    [
                        ("local", "t3", "sqrt(T/w[i])"),
                        ("member", "t1[i]", "t3 + n[i]"),
                        ("member", "t2[i]", "t3*u[i]"),
                    ],
                    [
                        ("t1[i]", "sqrt(T/w[i]) + n[i]"),
                        ("t2[i]", "sqrt(T/w[i])*u[i]"),
                    ],
                ),
                [],
            )
        ],
        "same_objects": [False],
    },
    "single": {
        "created": ["t1"],
        "inputs": [
            (
                "TempDef",
                "t1[i]",
                ["i"],
                [],
                "sqrt(T/w[i])",
                ["body"],
                None,
                None,
                [],
            )
        ],
        "result": [
            (
                "TempDef",
                "t1[i]",
                ["i"],
                [],
                "sqrt(T/w[i])",
                ["body"],
                None,
                None,
                [],
            )
        ],
        "same_objects": [True],
    },
}


EXPECTED_SIBLINGS = {
    "lone": {
        "created": ["t1"],
        "inputs": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "d3[i, j, m]*n[m]",
                ["body", "mirror_sum"],
                None,
                None,
                [],
            )
        ],
        "result": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "d3[i, j, m]*n[m]",
                ["body", "mirror_sum"],
                None,
                None,
                [],
            )
        ],
        "same_objects": [True],
    },
    "not_tables": {
        "created": ["t1", "t2", "t3"],
        "inputs": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "d3[i, j, m]*n[m]",
                ["body", "colsum", "mirror_sum"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t2[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "Piecewise((n[m], m > 1), (0, True))",
                ["body", "mirror_sum"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t3[i]",
                ["i"],
                ["(j, 1, nc)"],
                "mm[i, j]",
                ["body"],
                None,
                None,
                [],
            ),
        ],
        "result": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "d3[i, j, m]*n[m]",
                ["body", "colsum", "mirror_sum"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t2[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "Piecewise((n[m], m > 1), (0, True))",
                ["body", "mirror_sum"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t3[i]",
                ["i"],
                ["(j, 1, nc)"],
                "mm[i, j]",
                ["body"],
                None,
                None,
                [],
            ),
        ],
        "same_objects": [True, True, True],
    },
    "other_limits": {
        "created": ["t1", "t2"],
        "inputs": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "d3[i, j, m]*n[m]",
                ["body", "mirror_sum"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t2[k, l]",
                ["k", "l"],
                ["(p, 2, nc)"],
                "2*d3[k, l, p]*n[p]",
                ["body", "mirror_sum"],
                None,
                None,
                [],
            ),
        ],
        "result": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "d3[i, j, m]*n[m]",
                ["body", "mirror_sum"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t2[k, l]",
                ["k", "l"],
                ["(p, 2, nc)"],
                "2*d3[k, l, p]*n[p]",
                ["body", "mirror_sum"],
                None,
                None,
                [],
            ),
        ],
        "same_objects": [True, True],
    },
    "reader_flushes": {
        "created": ["t1", "t2", "t3"],
        "inputs": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "d3[i, j, m]*n[m]",
                ["body", "mirror_sum"],
                None,
                None,
                [],
            ),
            ("TempDef", "t5", [], [], "t1[1, 2]", ["body"], None, None, []),
            (
                "TempDef",
                "t3[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "d3[i, j, m]*n[m]",
                ["body", "mirror_sum"],
                None,
                None,
                [],
            ),
            ("TempDef", "t6", [], [], "t1", ["body"], None, None, []),
        ],
        "result": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "d3[i, j, m]*n[m]",
                ["body", "mirror_sum"],
                None,
                None,
                [],
            ),
            ("TempDef", "t5", [], [], "t1[1, 2]", ["body"], None, None, []),
            (
                "TempDef",
                "t3[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "d3[i, j, m]*n[m]",
                ["body", "mirror_sum"],
                None,
                None,
                [],
            ),
            ("TempDef", "t6", [], [], "t1", ["body"], None, None, []),
        ],
        "same_objects": [True, True, True, True],
    },
    "reads_first": {
        "created": ["t1", "t2"],
        "inputs": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "d3[i, j, m]*n[m]",
                ["body", "mirror_sum"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t2[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "d3[i, j, m]*t1[i, j]",
                ["body", "mirror_sum"],
                None,
                None,
                [],
            ),
        ],
        "result": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "d3[i, j, m]*n[m]",
                ["body", "mirror_sum"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t2[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "d3[i, j, m]*t1[i, j]",
                ["body", "mirror_sum"],
                None,
                None,
                [],
            ),
        ],
        "same_objects": [True, True],
    },
    "three_with_targetless": {
        "created": ["t1", "t2", "t3", "t4", "t5", "t6"],
        "inputs": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "d3[i, j, m]*n[m]",
                ["body", "mirror_sum"],
                None,
                None,
                [],
            ),
            ("TempDef", "None", [], [], "R", ["body"], None, None, []),
            (
                "TempDef",
                "t2[k, l]",
                ["k", "l"],
                ["(p, 1, nc)"],
                "2*d3[k, l, p]*n[p]",
                ["body", "mirror_sum"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t3[j, k]",
                ["j", "k"],
                ["(l, 1, nc)"],
                "3*d3[j, k, l]*n[l]",
                ["body", "mirror_sum"],
                None,
                None,
                [],
            ),
        ],
        "result": [
            (
                "SiblingNest",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "d3[i, j, m]*n[m]",
                ["accs", "body", "sibling"],
                [
                    (
                        "TempDef",
                        "t1[i, j]",
                        ["i", "j"],
                        ["(m, 1, nc)"],
                        "d3[i, j, m]*n[m]",
                        ["body", "mirror_sum"],
                        None,
                        None,
                        [],
                    ),
                    (
                        "TempDef",
                        "t2[i, j]",
                        ["i", "j"],
                        ["(m, 1, nc)"],
                        "2*d3[i, j, m]*n[m]",
                        ["body", "mirror_sum"],
                        None,
                        None,
                        [],
                    ),
                    (
                        "TempDef",
                        "t3[i, j]",
                        ["i", "j"],
                        ["(m, 1, nc)"],
                        "3*d3[i, j, m]*n[m]",
                        ["body", "mirror_sum"],
                        None,
                        None,
                        [],
                    ),
                ],
                None,
                ["t4", "t5", "t6"],
            ),
            ("TempDef", "None", [], [], "R", ["body"], None, None, []),
        ],
        "same_objects": [False, True],
    },
    "two": {
        "created": ["t1", "t2", "t3", "t4"],
        "inputs": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "d3[i, j, m]*n[m]",
                ["body", "mirror_sum"],
                None,
                None,
                [],
            ),
            (
                "TempDef",
                "t2[k, l]",
                ["k", "l"],
                ["(p, 1, nc)"],
                "2*d3[k, l, p]*n[p]",
                ["body", "mirror_sum"],
                None,
                None,
                [],
            ),
        ],
        "result": [
            (
                "SiblingNest",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "d3[i, j, m]*n[m]",
                ["accs", "body", "sibling"],
                [
                    (
                        "TempDef",
                        "t1[i, j]",
                        ["i", "j"],
                        ["(m, 1, nc)"],
                        "d3[i, j, m]*n[m]",
                        ["body", "mirror_sum"],
                        None,
                        None,
                        [],
                    ),
                    (
                        "TempDef",
                        "t2[i, j]",
                        ["i", "j"],
                        ["(m, 1, nc)"],
                        "2*d3[i, j, m]*n[m]",
                        ["body", "mirror_sum"],
                        None,
                        None,
                        [],
                    ),
                ],
                None,
                ["t3", "t4"],
            )
        ],
        "same_objects": [False],
    },
    "waiting": {
        "created": ["t1", "t2", "t3", "t4"],
        "inputs": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "d3[i, j, m]*n[m]",
                ["body", "mirror_sum"],
                None,
                None,
                [],
            ),
            ("TempDef", "t5", [], [], "2*R", ["body"], None, None, []),
            (
                "TempDef",
                "t2[k, l]",
                ["k", "l"],
                ["(p, 1, nc)"],
                "2*d3[k, l, p]*n[p]",
                ["body", "mirror_sum"],
                None,
                None,
                [],
            ),
        ],
        "result": [
            (
                "SiblingNest",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "d3[i, j, m]*n[m]",
                ["accs", "body", "sibling"],
                [
                    (
                        "TempDef",
                        "t1[i, j]",
                        ["i", "j"],
                        ["(m, 1, nc)"],
                        "d3[i, j, m]*n[m]",
                        ["body", "mirror_sum"],
                        None,
                        None,
                        [],
                    ),
                    (
                        "TempDef",
                        "t2[i, j]",
                        ["i", "j"],
                        ["(m, 1, nc)"],
                        "2*d3[i, j, m]*n[m]",
                        ["body", "mirror_sum"],
                        None,
                        None,
                        [],
                    ),
                ],
                None,
                ["t3", "t4"],
            ),
            ("TempDef", "t5", [], [], "2*R", ["body"], None, None, []),
        ],
        "same_objects": [False, True],
    },
    "waiting_read": {
        "created": ["t1", "t2"],
        "inputs": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "d3[i, j, m]*n[m]",
                ["body", "mirror_sum"],
                None,
                None,
                [],
            ),
            ("TempDef", "t5", [], [], "2*R", ["body"], None, None, []),
            (
                "TempDef",
                "t2[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "t5*d3[i, j, m]",
                ["body", "mirror_sum"],
                None,
                None,
                [],
            ),
        ],
        "result": [
            (
                "TempDef",
                "t1[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "d3[i, j, m]*n[m]",
                ["body", "mirror_sum"],
                None,
                None,
                [],
            ),
            ("TempDef", "t5", [], [], "2*R", ["body"], None, None, []),
            (
                "TempDef",
                "t2[i, j]",
                ["i", "j"],
                ["(m, 1, nc)"],
                "t5*d3[i, j, m]",
                ["body", "mirror_sum"],
                None,
                None,
                [],
            ),
        ],
        "same_objects": [True, True, True],
    },
}


MODEL_DIGESTS = {
    (
        "tests/data/models/cmr/cmr.model",
        "dn",
        "fastest",
        None,
    ): "1f3d932aed3e2dbd387c035b7e82d25dd3aebfd8c5d42bec9066f77bffa58c39",
    (
        "tests/data/models/cmr/cmr.model",
        "dn",
        "fastest",
        "fuse_temps",
    ): "3e80d0833b42cfa955a66d4cd7865d39428acde58d468d749e5989630eeb36fb",
    (
        "tests/data/models/cmr/cmr.model",
        "dn",
        "fastest",
        "nest_sums",
    ): "8d9f6a758b746fd3a2ab81caa24ca99207be2ddada90dc8ec811915044436a29",
    (
        "tests/data/models/cmr/cmr.model",
        "dn",
        "fastest",
        "fuse_loops",
    ): "9088e3e42a90eee4d6b2ca7c0a5188ef4565e0efc38283a62617544e1f267baf",
    (
        "tests/data/models/cmr/cmr.model",
        "dn",
        "fastest",
        "sib_fuse",
    ): "a94b41005dc9d32dcdef0c6922357a15b42208a3539527873d341e06c1df75b7",
    (
        "tests/data/models/cmr/cmr.model",
        "dn",
        "fastest",
        "sym_perm",
    ): "5e4bedb47efc225976f37aca19e8f93090824ad5e9f860bf388f3d5b7ac68385",
    (
        "tests/data/models/cmr/cmr.model",
        "dn",
        "standard",
        None,
    ): "811b66da262cc939ac00716cf78c063b2c7e120d4024b433fe18e0d5798d6181",
    (
        "tests/data/models/pr76/pr76_sym.model",
        "dn,dndn",
        "fastest",
        None,
    ): "8d75563be76d11f6a959d63b1934b8ac2fe18f01bb8734843b789d42ee1eae37",
    (
        "tests/data/models/pr76/pr76_sym.model",
        "dn,dndn",
        "fastest",
        "comp_fuse",
    ): "770e74ddcbc8a654940bb4248995877c9742d845be01098520444cc6f3c2d964",
    (
        "tests/data/models/pr76/pr76_sym.model",
        "dn,dndn",
        "fastest",
        "sym_perm",
    ): "087399ab69a899a9175ee5746a6664cd2684e580cae13ed17d82765d7e93b55c",
    (
        "tests/data/models/srk/srk.model",
        "dn",
        "fastest",
        None,
    ): "04ac3a517e7398d8298bcdef1c140391cccef13359320e507e88ed7a5b50d73e",
    (
        "tests/data/models/srk/srk.model",
        "dn",
        "fastest",
        "comp_fuse",
    ): "d4c761cacd5c609b9e3368ec64258932c952491ccff4d535defa6ce8fecb3d33",
    (
        "tests/data/models/nrtl/nrtl.model",
        "dndn",
        "fastest",
        None,
    ): "b53918d9f198e8c2eb1f7b99cdd1b222288a6fbb133a1ba8661ac84f64bd54a1",
    (
        "tests/data/models/nrtl/nrtl.model",
        "dndn",
        "fastest",
        "fuse_temps",
    ): "cde48a81fad65d11a876f7c223ec71a3d4011767a29cd95643858d7b4c3ad571",
    (
        "tests/data/models/nrtl/nrtl.model",
        "dndn",
        "fastest",
        "fuse_loops",
    ): "e9f95598b191a7bb7f704b71f8ef3de7c9a4f804aa17c398a826b778533d2064",
    (
        "tests/data/models/nrtl/nrtl.model",
        "dndn",
        "standard",
        None,
    ): "5c52ab85d8f0b8d627f816e48aaa27265bfc3b7a803d8e81f66ebaff52bd04fa",
    (
        "tests/data/models/nrtl/nrtl.model",
        "dT,dT2,dn,dndT,dndn",
        "fastest",
        None,
    ): "bb3648abb37f8b6a8ae88cdb2c0bd226607ac104231d516a03ef79063b50f7d8",
    (
        "tests/data/models/nrtl/nrtl.model",
        "dT,dT2,dn,dndT,dndn",
        "fastest",
        "fuse_guard",
    ): "836ee537a8167bb84668c6192ef67bd7c4259d8a5356fe646725dcb63a20d5f8",
    (
        "tests/data/models/uniquac/uniquac.model",
        "dT,dT2,dn,dndT,dndn",
        "fastest",
        None,
    ): "2a2a584df4da62bfcde27aa8d6e440e9cdcf66954614a83d7e2ff5e922edeb50",
    (
        "tests/data/models/uniquac/uniquac.model",
        "dT,dT2,dn,dndT,dndn",
        "fastest",
        "comp_fuse",
    ): "566d651c647ac2270c388111e09c48c7464085bd5734833fb439f394e953a9e8",
    (
        "tests/data/models/uniquac/uniquac.model",
        "dT,dT2,dn,dndT,dndn",
        "fastest",
        "fuse_loops",
    ): "e3ee9bd562fa33b055632cc9d412a38cddeab36ee224e06d5598f726a3e1f348",
}
# EXPECTED-END


@pytest.mark.parametrize("case", sorted(NEST_CASES))
def test_nest_double_sums_is_pinned(case):
    assert run_nest(case) == EXPECTED_NEST[case]


@pytest.mark.parametrize("case", sorted(TEMP_CASES))
def test_fuse_temporaries_is_pinned(case):
    assert run_temporaries(case) == EXPECTED_TEMPORARIES[case]


@pytest.mark.parametrize("case", sorted(NEST_LOOP_CASES))
def test_fuse_loop_nests_is_pinned(case):
    assert run_loop_nests(case) == EXPECTED_LOOP_NESTS[case]


@pytest.mark.parametrize("case", sorted(PERMUTE_CASES))
def test_permute_symmetric_subscripts_is_pinned(case):
    assert run_permute(case) == EXPECTED_PERMUTE[case]


@pytest.mark.parametrize("case", sorted(MAP_CASES))
def test_fuse_component_maps_is_pinned(case):
    assert run_component_maps(case) == EXPECTED_MAPS[case]


@pytest.mark.parametrize("case", sorted(SIBLING_CASES))
def test_fuse_sibling_tables_is_pinned(case):
    assert run_siblings(case) == EXPECTED_SIBLINGS[case]


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


CMR = "tests/data/models/cmr/cmr.model"
NRTL = "tests/data/models/nrtl/nrtl.model"
PR76 = "tests/data/models/pr76/pr76_sym.model"
SRK = "tests/data/models/srk/srk.model"
UNIQUAC = "tests/data/models/uniquac/uniquac.model"
ALL_DERIVATIVES = "dT,dT2,dn,dndT,dndn"
# (model, derivatives, preset, option turned off or None) -> text digest
MODEL_CASES = (
    (CMR, "dn", "fastest", None),
    (CMR, "dn", "fastest", "fuse_temps"),
    (CMR, "dn", "fastest", "nest_sums"),
    (CMR, "dn", "fastest", "fuse_loops"),
    (CMR, "dn", "fastest", "sib_fuse"),
    (CMR, "dn", "fastest", "sym_perm"),
    (CMR, "dn", "standard", None),
    (PR76, "dn,dndn", "fastest", None),
    (PR76, "dn,dndn", "fastest", "comp_fuse"),
    (PR76, "dn,dndn", "fastest", "sym_perm"),
    (SRK, "dn", "fastest", None),
    (SRK, "dn", "fastest", "comp_fuse"),
    (NRTL, "dndn", "fastest", None),
    (NRTL, "dndn", "fastest", "fuse_temps"),
    (NRTL, "dndn", "fastest", "fuse_loops"),
    (NRTL, "dndn", "standard", None),
    (NRTL, ALL_DERIVATIVES, "fastest", None),
    (NRTL, ALL_DERIVATIVES, "fastest", "fuse_guard"),
    (UNIQUAC, ALL_DERIVATIVES, "fastest", None),
    (UNIQUAC, ALL_DERIVATIVES, "fastest", "comp_fuse"),
    (UNIQUAC, ALL_DERIVATIVES, "fastest", "fuse_loops"),
)


@pytest.mark.slow
@pytest.mark.parametrize("case", MODEL_CASES, ids=str)
def test_fusion_passes_text_is_pinned(case):
    model, derivatives, preset, option = case
    overrides = {option: False} if option else {}
    code = generate_code(model, derivatives, preset, **overrides)
    assert digest(code) == MODEL_DIGESTS[case]


@pytest.mark.parametrize("case", [c for c in MODEL_CASES if c[3]], ids=str)
def test_each_option_changes_the_text(case):
    model, derivatives, preset, _ = case
    assert (
        MODEL_DIGESTS[case]
        != MODEL_DIGESTS[(model, derivatives, preset, None)]
    )
